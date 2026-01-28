import asyncio
import logging
import os
import signal
import subprocess
import sys
from functools import lru_cache
from typing import Any, Dict

import torch.multiprocessing as mp
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from .compat import compat
from .database import Database
from .trainer import Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hyperturing Server (Reliability Grade)")

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "data/sp6/tinypajama.npy"
DB_PATH = os.path.join(BASE_DIR, "db", "hyperturing.db")
SCALING_LAWS_BUDGET = 2e18
DEFAULT_VOCAB_SIZE = 32000
CONTEXT_LENGTH = 512

# Global state
active_queues: Dict[str, mp.Queue] = {}
training_lock = asyncio.Lock()
app_state = {"is_running": True}


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
    config_str = f"{config['d_model']}_{config['num_layers']}_{config['num_heads']}_{config['batch_size']}_{config['learning_rate']}_{config['train_flops']}"
    return f"{api_key}_{config_str}"


def get_npu_stats():
    """Extract NPU usage information by parsing npu-smi info."""
    try:
        # Run npu-smi info and capture output
        res = subprocess.check_output(["npu-smi", "info"], encoding="utf-8")
        lines = res.splitlines()

        # Searching for the memory usage line.
        # Output format has columns: NPU, Name, Health, Power, Temp, Hugepages, Chip, Bus-Id, AICore, Memory, HBM
        # We look for the line containing HBM-Usage or Memory-Usage values.
        hbm_info = "N/A"
        health = "Unknown"

        for i, line in enumerate(lines):
            if "OK" in line:
                health = "OK"
            if "/" in line and ("65536" in line or "32768" in line):
                # This is likely the line with memory usage (e.g., "3391 / 65536")
                # We can extract the HBM usage part
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 5:
                    hbm_info = parts[4]  # 5th column for HBM usage in Summary view

        return {
            "health": health,
            "hbm_usage": hbm_info,
            "lock_active": training_lock.locked(),
        }
    except Exception as e:
        logger.debug(f"Failed to parse npu-smi: {e}")
        return {"health": "error", "error": str(e)}


@app.get("/health")
async def health_check():
    """Industrial grade health check for load balancer or monitor."""
    stats = get_npu_stats()
    return {
        "status": "UP" if stats.get("health") == "OK" else "DEGRADED",
        "npu": stats,
        "database": "OK" if os.path.exists(DB_PATH) else "CRITICAL",
    }


async def guardian_task():
    """Background task to ensure system health and resource cleanup."""
    logger.info("Guardian task started.")
    while app_state["is_running"]:
        try:
            # Check for orphaned processes or unexpected memory bloat
            # On NPU/GPU, clearing cache periodically when idle can help
            if not training_lock.locked():
                compat.empty_cache()

            # Additional health checks could go here
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Error in Guardian task: {e}")
            await asyncio.sleep(10)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(guardian_task())

    # Handle signals for graceful shutdown
    def signal_handler():
        logger.info("Shutdown signal received.")
        app_state["is_running"] = False
        # In a real service, we'd wait for training to finish or kill it
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Not supported on Windows, but this is Linux
            pass


@app.get("/loss")
async def get_loss(
    d_model: int,
    num_layers: int,
    num_heads: int,
    batch_size: int,
    learning_rate: float,
    train_flops: float,
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
    if not (1e-6 <= learning_rate <= 1):
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

    # Check for existing successful run
    existing_loss = db.get_existing_run(api_key, config)
    if existing_loss is not None:
        total_used = db.get_total_flops(api_key)
        return {"loss": existing_loss, "total_flops_used": total_used}

    # Budget Check
    current_used = db.get_total_flops(api_key) or 0.0
    if current_used + train_flops > SCALING_LAWS_BUDGET:
        raise HTTPException(status_code=403, detail="Scaling laws budget exceeded")

    # Initialize Job Tracking
    run_id = db.initialize_run(api_key, config)

    # 5. Serialize Execution using a Lock
    # Only one training job can run at a time to prevent OOM
    logger.info(f"Job {job_id} (run_id: {run_id}) waiting for NPU lock...")

    async def run_training():
        async with training_lock:
            # Double check inside the lock for existing results (prevents redundant work from near-simultaneous requests)
            existing_loss = db.get_existing_run(api_key, config)
            if existing_loss is not None:
                total_used = db.get_total_flops(api_key)
                logger.info(
                    f"Job {job_id} found existing result in DB after acquiring lock. Skipping."
                )
                return {"loss": existing_loss, "total_flops_used": total_used}

            logger.info(f"Job {job_id} acquired NPU lock. Starting training.")
            db.update_run_status(run_id, "RUNNING")

            ctx = mp.get_context("spawn")
            progress_queue = ctx.Queue()
            active_queues[job_id] = progress_queue

            try:
                # Run training in worker process via mp.spawn
                loss = await run_in_threadpool(trainer.train, config, progress_queue)

                # Record success
                db.update_run_status(run_id, "SUCCESS", loss=loss)
                db.update_total_flops(api_key, float(train_flops))
                new_total_used = db.get_total_flops(api_key)
                logger.info(f"Job {job_id} completed successfully. Loss: {loss}")
                return {"loss": loss, "total_flops_used": new_total_used}

            except Exception as e:
                error_msg = str(e)
                clean_error = (
                    "NPU out of memory"
                    if "out of memory" in error_msg.lower()
                    else error_msg
                )
                logger.error(f"Job {job_id} failed: {error_msg}")
                db.update_run_status(run_id, "FAILED", error_message=clean_error)
                raise HTTPException(status_code=500, detail=clean_error)
            finally:
                if job_id in active_queues:
                    del active_queues[job_id]
                compat.empty_cache()

    # Use shield to ensure training completes even if client disconnects
    return await asyncio.shield(run_training())


@app.websocket("/loss_ws")
async def websocket_endpoint(websocket: WebSocket, api_key: str):
    await websocket.accept()
    logger.info(f"WebSocket connected for API key: {api_key}")

    try:
        while app_state["is_running"]:
            target_job_id = None
            for jid in active_queues.keys():
                if jid.startswith(api_key):
                    target_job_id = jid
                    break

            if target_job_id:
                queue = active_queues[target_job_id]
                try:
                    data = queue.get_nowait()
                    # Add system health info to WebSocket push
                    data["sys_npu"] = get_npu_stats()
                    await websocket.send_json(data)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    await asyncio.sleep(0.5)
            else:
                # Optionally send heartbeat with system stats
                await asyncio.sleep(2.0)
                try:
                    await websocket.send_json(
                        {"heartbeat": True, "sys_npu": get_npu_stats()}
                    )
                except Exception:
                    break

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
