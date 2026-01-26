import asyncio
import logging
import os
from functools import lru_cache
from typing import Any, Dict

import torch.multiprocessing as mp
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from .compat import compat
from .database import Database
from .trainer import Trainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hyperturing Server")

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "data/sp6/tinypajama.npy"
DB_PATH = os.path.join(BASE_DIR, "db", "hyperturing.db")
SCALING_LAWS_BUDGET = 2e18
DEFAULT_VOCAB_SIZE = 32000
CONTEXT_LENGTH = 512

# Global state for monitoring
# Map of job_id -> mp.Queue
active_queues: Dict[str, mp.Queue] = {}


@lru_cache()
def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return Database(DB_PATH)


@lru_cache()
def get_trainer():
    path = DATA_PATH
    vocab_size = DEFAULT_VOCAB_SIZE
    if not os.path.exists(path):
        logger.warning(f"Data not found at {path}, falling back to OpenWebText")
        path = "data/owt/owt_train.npy"
        vocab_size = 32000
    return Trainer(path, vocab_size=vocab_size, context_length=CONTEXT_LENGTH)


def get_job_id(api_key: str, config: Dict[str, Any]) -> str:
    """Generate a unique ID for a training job."""
    config_str = f"{config['d_model']}_{config['num_layers']}_{config['num_heads']}_{config['batch_size']}_{config['learning_rate']}_{config['train_flops']}"
    return f"{api_key}_{config_str}"


@app.get("/loss")
async def get_loss(
    d_model: int,
    num_layers: int,
    num_heads: int,
    batch_size: int,
    learning_rate: float,
    train_flops: float,  # Changed to float for safety with 1e13 notation
    api_key: str,
):
    # Validation
    if not (64 <= d_model <= 1024):
        raise HTTPException(
            status_code=404,
            detail=f"d_model must be in range [64, 1024], got {d_model}",
        )
    if not (2 <= num_layers <= 24):
        raise HTTPException(
            status_code=404,
            detail=f"num_layers must be in range [2, 24], got {num_layers}",
        )
    if not (2 <= num_heads <= 16):
        raise HTTPException(
            status_code=404,
            detail=f"num_heads must be in range [2, 16], got {num_heads}",
        )
    if batch_size not in [32, 64, 128, 256]:
        raise HTTPException(
            status_code=404,
            detail=f"batch_size must be one of {{128, 256}}, got {batch_size}",
        )
    if not (1e-4 <= learning_rate <= 1e-3):
        raise HTTPException(
            status_code=404,
            detail=f"learning_rate must be in range [1e-4, 1e-3], got {learning_rate}",
        )

    valid_flops = {
        1e13,
        3e13,
        6e13,
        1e14,
        3e14,
        6e14,
        1e15,
        3e15,
        6e15,
        1e16,
        3e16,
        6e16,
        1e17,
        3e17,
        6e17,
        1e18,
    }
    if float(train_flops) not in valid_flops:
        if not any(abs(train_flops - f) < 1.0 for f in valid_flops):
            raise HTTPException(
                status_code=404,
                detail=f"train_flops must be one of {valid_flops}, got {train_flops}",
            )

    db = get_db()
    trainer = get_trainer()
    db.add_api_key(api_key)

    config = {
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_flops": int(train_flops),
        "vocab_size": trainer.vocab_size,
    }

    job_id = get_job_id(api_key, config)

    # Check database first
    existing_loss = db.get_existing_run(api_key, config)
    if existing_loss is not None:
        total_used = db.get_total_flops(api_key)
        return {"loss": existing_loss, "total_flops_used": total_used}

    current_used = db.get_total_flops(api_key) or 0.0
    if current_used + train_flops > SCALING_LAWS_BUDGET:
        raise HTTPException(status_code=403, detail="Scaling laws budget exceeded")

    # Handle concurrency: if a job is already running, wait for it?
    # For now, we'll just run it. The mp.spawn handles its own processes.

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    active_queues[job_id] = progress_queue

    try:
        # run_in_threadpool allows the server to handle other requests (like WS) while training
        loss = await run_in_threadpool(trainer.train, config, progress_queue)

        db.record_run(api_key, config, loss)
        db.update_total_flops(api_key, float(train_flops))
        new_total_used = db.get_total_flops(api_key)
        return {"loss": loss, "total_flops_used": new_total_used}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal training error: {str(e)}"
        )
    finally:
        # Clean up queue after job finishes
        if job_id in active_queues:
            del active_queues[job_id]
        # Explicitly clear cache in the server process just in case
        compat.empty_cache()


@app.websocket("/loss_ws")
async def websocket_endpoint(websocket: WebSocket, api_key: str):
    await websocket.accept()
    logger.info(f"WebSocket connected for API key: {api_key}")

    try:
        while True:
            # We need to identify which job the user wants to monitor.
            # For simplicity, we'll monitor the "latest" or "active" job for this API key.
            # In a more complex setup, the client could send a job_id.

            # Find an active queue for this API key
            target_job_id = None
            for jid in active_queues.keys():
                if jid.startswith(api_key):
                    target_job_id = jid
                    break

            if target_job_id:
                queue = active_queues[target_job_id]
                try:
                    # Non-blocking get from queue
                    data = queue.get_nowait()
                    await websocket.send_json(data)
                except:  # noqa: E722
                    # No data yet, wait a bit
                    await asyncio.sleep(0.5)
            else:
                # No active job, just wait
                await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for API key: {api_key}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@app.get("/total_flops_used")
async def get_total_flops_used(api_key: str):
    db = get_db()
    total = db.get_total_flops(api_key)
    if total is None:
        raise HTTPException(status_code=422, detail=f"Invalid API key: {api_key}")
    return total


@app.get("/previous_runs")
async def get_previous_runs(api_key: str):
    db = get_db()
    runs = db.get_previous_runs(api_key)
    if not runs and db.get_total_flops(api_key) is None:
        raise HTTPException(status_code=422, detail=f"Invalid API key: {api_key}")
    return {"previous_runs": runs}
