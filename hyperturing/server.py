import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from functools import lru_cache
from multiprocessing import shared_memory
from typing import Any, Dict, Literal

import numpy as np
import torch.multiprocessing as mp
from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from .compat import compat
from .database import Database
from .trainer import Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DataSet = {
    "tss": {
        "path": "data/tss/TinyStoriesV2-GPT4-train.npy",
        "size": 10000,
    },  # vocab size
    "256": {
        "path": "data/tss/TinyStoriesV2-GPT4-train_256.npy",
        "size": 256,
    },  # only for low budget
    "owt": {
        "path": "data/owt/owt_train.npy",
        "size": 32000,
    },
    "sp6": {"path": "data/sp6/tinypajama.npy", "size": 32000},
}

DB_PATH = os.path.join(BASE_DIR, "db", "hyperturing.db")
SCALING_LAWS_BUDGET = 2e18
CONTEXT_LENGTH = 512

# Global state
active_queues: Dict[str, mp.Queue] = {}
training_lock = asyncio.Lock()
app_state = {
    "is_running": True,
    "shm_map": {},  # dataset_key -> {shm_name, shape, dtype, _ref}
}
dataset_loading_lock = asyncio.Lock()


async def ensure_dataset_loaded(dataset: str):
    """Lazy loads dataset into SharedMemory if not already present."""
    if dataset not in DataSet:
        logger.error(f"Unknown dataset requested: {dataset}")
        return

    # Check if already loaded
    if dataset in app_state["shm_map"]:
        return

    async with dataset_loading_lock:
        # Double-check inside lock
        if dataset in app_state["shm_map"]:
            return

        info = DataSet[dataset]
        data_path = info["path"]
        if not os.path.exists(data_path):
            logger.error(f"Dataset {dataset} not found at {data_path}")
            return

        try:
            logger.info(f"Lazy loading dataset {dataset} from {data_path} into RAM...")
            # Run blocking IO in threadpool to avoid blocking event loop
            raw_data = await run_in_threadpool(np.load, data_path)

            # Create SHM (must be done in main process/thread generally safe if managed correctly)
            # np.load might be heavy, so we awaited it. SHM creation is fast.
            shm = shared_memory.SharedMemory(create=True, size=raw_data.nbytes)
            shared_arr = np.ndarray(
                raw_data.shape, dtype=raw_data.dtype, buffer=shm.buf
            )
            shared_arr[:] = raw_data[:]

            app_state["shm_map"][dataset] = {
                "name": shm.name,
                "shape": raw_data.shape,
                "dtype": raw_data.dtype,
                "_ref": shm,
            }
            logger.info(
                f"Dataset {dataset} successfully loaded into SharedMemory: {shm.name}"
            )
        except Exception as e:
            logger.error(f"Failed to lazy load dataset {dataset}: {e}")
            # Ensure we don't leave broken state or leaks?
            if "shm" in locals() and "shm" not in app_state["shm_map"].values():
                shm.close()  # pyright: ignore[reportPossiblyUnboundVariable]
                shm.unlink()  # pyright: ignore[reportPossiblyUnboundVariable]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Background tasks
    asyncio.create_task(guardian_task())

    # Handle signals for graceful shutdown
    def signal_handler():
        logger.info("Shutdown signal received.")
        app_state["is_running"] = False
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop = asyncio.get_event_loop()
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass

    yield
    # Shutdown: Clean up
    app_state["is_running"] = False
    for key, info in app_state["shm_map"].items():
        try:
            info["_ref"].close()
            info["_ref"].unlink()
            logger.info(f"SharedMemory for {key} cleaned up.")
        except Exception as e:
            logger.error(f"Error cleaning up SHM for {key}: {e}")


app = FastAPI(title="Hyperturing Server", lifespan=lifespan)


@lru_cache()
def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return Database(DB_PATH)


@lru_cache()
def get_trainer(dataset):
    # Retrieve SHM info from global state if available
    shm_info = None
    if dataset in app_state["shm_map"]:
        entry = app_state["shm_map"][dataset]
        shm_info = {
            "name": entry["name"],
            "shape": entry["shape"],
            "dtype": entry["dtype"],
        }
    path = DataSet[dataset]["path"]
    vocab_size = DataSet[dataset]["size"]
    return Trainer(
        path,
        vocab_size=vocab_size,
        context_length=CONTEXT_LENGTH,
        shm_info=shm_info,
    )


def get_job_id(api_key: str, config: Dict[str, Any]) -> str:
    config_str = f"{config['d_model']}_{config['num_layers']}_{config['num_heads']}_{config['batch_size']}_{config['learning_rate']}_{config['train_flops']}"
    # Backward compatibility: only append vocab_size if it's not the default 10000
    if config.get("vocab_size", 32000) < 10000:
        config_str += f"_{config['vocab_size']}"
    return f"{api_key}_{config_str}"


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


@app.get("/loss")
async def get_loss(
    d_model: int,
    num_layers: int,
    num_heads: int,
    batch_size: int,
    learning_rate: float,
    train_flops: float,
    api_key: str,
    dataset: Literal["tss", "256", "owt", "sp6"] = "sp6",
):
    # Validation
    if not (8 <= d_model <= 1024):
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

    # Ensure data is loaded into SHM before creating trainer or running job
    await ensure_dataset_loaded(dataset)

    db = get_db()
    trainer = get_trainer(dataset)
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

    # 5. Serialize Execution using a Lock
    # Only one training job can run at a time to prevent OOM
    logger.info(f"Job {job_id} waiting for NPU lock...")

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

            # Initialize Job Tracking under lock to avoid orphaned PENDING records
            run_id = db.initialize_run(api_key, config)
            logger.info(
                f"Job {job_id} (run_id: {run_id}) acquired NPU lock. Starting training."
            )
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
