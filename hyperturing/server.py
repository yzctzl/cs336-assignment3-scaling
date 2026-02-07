import asyncio
import logging
import os
import math
from contextlib import asynccontextmanager
from functools import lru_cache
from multiprocessing import shared_memory
from typing import Any, Dict, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from .compat import DEVICE_TYPE, compat
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
active_queues: Dict[
    str, Any
] = {}  # Keeping for backward compatibility if needed, but not used for queues
dataset_loading_lock = asyncio.Lock()
app_state = {
    "is_running": True,
    "shm_map": {},  # dataset_key -> {shm_name, shape, dtype, _ref}
}


class ResourceManager:
    def __init__(self, safe_threshold: float = 0.85):
        env_threshold = float(os.getenv("TRAINING_MEMORY_THRESHOLD", str(safe_threshold)))
        self.default_threshold = max(0.1, min(0.98, env_threshold))
        self.current_threshold = safe_threshold
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Condition()
        self.last_oom_time = 0.0
        self.oom_cooldown = 60  # 60s cooldown after OOM
        self.dynamic_overhead_factor = 1.0  # Adaptive safety margin
        self.startup_grace_seconds = float(
            os.getenv("TRAINING_STARTUP_GRACE_SECONDS", "8.0")
        )
        self.device_count = compat.device_count()
        self.free_devices = set(range(self.device_count))

    def estimate_memory_gb(self, config: Dict[str, Any], group_size: int) -> float:
        """
        Estimate per-device memory usage in GB.
        Formula: Params (float16) + Gradients (float16) + Opt states (AdamW, float32)
        Plus some overhead for activations and buffers.
        N = 12 * num_layers * d_model^2
        """
        d_model = config["d_model"]
        num_layers = config["num_layers"]
        # Approx params
        n_params = 12 * num_layers * (d_model**2)
        # 2 bytes for float16 params, 2 for grads, 8 for AdamW states (2 x float32)
        # Model memory = N * (2 + 2 + 8) = 12 * N bytes
        model_mem_gb = (n_params * 12) / (1024**3)

        # Activations: very rough estimate based on batch_size and context_length
        # This is a guestimate, can be refined.
        batch_size = config["batch_size"]
        local_batch = max(1, int(math.ceil(batch_size / max(1, group_size))))
        context_length = CONTEXT_LENGTH
        # Rough activation factor: d_model * context_length * batch_size * layers * factor
        # For simplicity, let's say 2x model memory or a base overhead.
        activation_overhead = (
            local_batch * context_length * d_model * num_layers * 4 * 4
        ) / (1024**3)

        # Apply dynamic overhead factor to activation estimate
        total_est = (
            model_mem_gb + (activation_overhead * self.dynamic_overhead_factor) + 0.25
        )
        return total_est

    def _snapshot_device_memory(self, target_threshold: float) -> Dict[int, Dict[str, float]]:
        per_device: Dict[int, Dict[str, float]] = {}
        for d in range(self.device_count):
            info = compat.get_memory_info(d)
            total_mem = float(info["total"])
            allocated_mem = float(info["allocated"])
            threshold_capacity = total_mem * target_threshold
            available_mem = max(0.0, threshold_capacity - allocated_mem)
            per_device[d] = {
                "total": total_mem,
                "allocated": allocated_mem,
                "threshold_capacity": threshold_capacity,
                "available": available_mem,
            }
        return per_device

    def _pending_startup_by_device(self, now: float) -> Dict[int, float]:
        pending_by_device: Dict[int, float] = {}
        for job in self.active_jobs.values():
            if (now - float(job.get("started_at", now + 1))) >= self.startup_grace_seconds:
                continue
            est_mem = float(job.get("est_mem_per_device", job.get("est_mem", 0.0)))
            for d in job.get("devices", []):
                pending_by_device[d] = pending_by_device.get(d, 0.0) + est_mem
        return pending_by_device

    async def acquire(
        self, job_id: str, config: Dict[str, Any], group_size: int
    ) -> list[int]:
        async with self.lock:
            while True:
                # If running on accelerators, enforce device group availability.
                if self.device_count > 0:
                    if len(self.free_devices) < group_size:
                        logger.info(
                            f"Job {job_id} waiting for device group (need {group_size}, free {len(self.free_devices)})..."
                        )
                        await self.lock.wait()
                        continue

                now = asyncio.get_event_loop().time()
                # Proactive factor decay: Reduce if 10 mins have passed since last OOM
                if (
                    self.dynamic_overhead_factor > 1.0
                    and (now - self.last_oom_time) > 600
                ):
                    self.dynamic_overhead_factor = max(
                        1.0, self.dynamic_overhead_factor - 0.05
                    )
                    logger.info(
                        f"System stable. Adaptive overhead factor refined to {self.dynamic_overhead_factor:.2f}"
                    )

                est_mem = self.estimate_memory_gb(config, group_size=group_size)
                in_cooldown = (now - self.last_oom_time) < self.oom_cooldown
                target_threshold = self.default_threshold * (
                    0.8 if in_cooldown else 1.0
                )

                if self.device_count > 0:
                    per_device = self._snapshot_device_memory(target_threshold)
                    pending_by_device = self._pending_startup_by_device(now)
                    free_devices_sorted = sorted(
                        self.free_devices,
                        key=lambda d: per_device.get(d, {}).get("available", 0.0),
                        reverse=True,
                    )
                    candidate_devices = free_devices_sorted[:group_size]
                    if len(candidate_devices) < group_size:
                        await self.lock.wait()
                        continue

                    can_schedule = True
                    for d in candidate_devices:
                        adjusted_available = (
                            per_device[d]["available"] - pending_by_device.get(d, 0.0)
                        )
                        if adjusted_available < est_mem:
                            can_schedule = False
                            break

                    if can_schedule:
                        for d in candidate_devices:
                            self.free_devices.remove(d)

                        self.active_jobs[job_id] = {
                            "est_mem": est_mem,
                            "est_mem_per_device": est_mem,
                            "started_at": now,
                            "config": config,  # Keep config for debugging/recovery
                            "devices": candidate_devices,
                            "group_size": group_size,
                        }
                        free_pool_adjusted = sum(
                            max(
                                0.0,
                                per_device[d]["available"] - pending_by_device.get(d, 0.0),
                            )
                            for d in self.free_devices
                        )
                        logger.info(
                            f"Job {job_id} scheduled. Est/device: {est_mem:.2f}GB (Factor: {self.dynamic_overhead_factor:.2f}), "
                            f"group_size={group_size}, devices={candidate_devices}, free_pool_adjusted={free_pool_adjusted:.1f}GB"
                        )
                        return candidate_devices
                else:
                    # CPU mode or unknown? Fallback to sequential for safety
                    if not self.active_jobs:
                        self.active_jobs[job_id] = {
                            "est_mem": 0.0,
                            "started_at": now,
                            "devices": [],
                        }
                        return []

                await self.lock.wait()

    def report_oom(self):
        """Notification of runtime OOM to adjust future scheduling."""
        self.last_oom_time = asyncio.get_event_loop().time()
        # Increase safety margin more significantly if OOM occurs
        self.dynamic_overhead_factor = min(4.0, self.dynamic_overhead_factor + 0.2)
        logger.info(
            f"Adaptive Scaling: Runtime OOM detected. Increasing safety margin. Current factor: {self.dynamic_overhead_factor:.2f}"
        )

    async def release(self, job_id: str):
        async with self.lock:
            if job_id in self.active_jobs:
                devices = self.active_jobs[job_id].get("devices", [])
                for d in devices:
                    self.free_devices.add(d)
                del self.active_jobs[job_id]
                self.lock.notify_all()

    def get_status(self) -> Dict[str, Any]:
        now = asyncio.get_event_loop().time()
        in_cooldown = (now - self.last_oom_time) < self.oom_cooldown
        target_threshold = self.default_threshold * (0.8 if in_cooldown else 1.0)

        if self.device_count > 0:
            per_device = self._snapshot_device_memory(target_threshold)
            total_mem = sum(v["total"] for v in per_device.values())
            current_allocated = sum(v["allocated"] for v in per_device.values())
            available_capacity = sum(
                per_device[d]["available"] for d in self.free_devices if d in per_device
            )
            per_device_rows = [
                {
                    "device_id": d,
                    "total_gb": round(v["total"], 2),
                    "allocated_gb": round(v["allocated"], 2),
                    "available_gb": round(v["available"], 2),
                    "is_free": d in self.free_devices,
                }
                for d, v in sorted(per_device.items())
            ]
        else:
            mem_info = compat.get_memory_info()
            total_mem = mem_info["total"]
            current_allocated = mem_info["allocated"]
            available_capacity = max(0, total_mem * target_threshold - current_allocated)
            per_device_rows = []

        return {
            "active_tasks": len(self.active_jobs),
            "total_memory_gb": round(total_mem, 2),
            "allocated_memory_gb": round(current_allocated, 2),
            "available_capacity_gb": round(available_capacity, 2),
            "device_count": self.device_count,
            "free_devices": sorted(self.free_devices),
            "per_device_memory_gb": per_device_rows,
            "is_in_cooldown": in_cooldown,
            "cooldown_remaining": max(0, self.oom_cooldown - (now - self.last_oom_time))
            if in_cooldown
            else 0,
            "dynamic_overhead_factor": round(self.dynamic_overhead_factor, 2),
        }


resource_manager = ResourceManager()


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
    # Uvicorn handles SIGINT/SIGTERM natively. We rely on that to trigger the shutdown phase.
    # No custom signal handler needed here, as it conflicts with Uvicorn.

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


def _parse_group_sizes() -> list[int]:
    raw = os.getenv("TRAINING_GROUP_SIZES", "").strip()
    max_devices = max(1, compat.device_count())
    if raw:
        raw_sizes: list[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                raw_sizes.append(int(part))
            except ValueError:
                continue
    else:
        if compat.is_npu and max_devices >= 4:
            # Prioritize 2-way DDP to fill 4 cards with two concurrent models.
            raw_sizes = [2, 1, 4]
        else:
            raw_sizes = [1, 2, 4]

    sizes: list[int] = []
    seen = set()
    for s in raw_sizes:
        if s <= 0 or s > max_devices or s in seen:
            continue
        sizes.append(s)
        seen.add(s)

    if not sizes:
        return [1]

    return sizes


def _is_oom_error(error_msg: str) -> bool:
    msg = error_msg.lower()
    oom_keywords = [
        "out of memory",
        "acl api failed",
        "failed to allocate",
        "tried to allocate",
        "memory_allocation_failure",
        "npu out of memory",
    ]
    return any(k in msg for k in oom_keywords)


def _apply_stability_policy(config: Dict[str, Any]) -> Dict[str, Any]:
    mode = os.getenv("TRAINING_STABILITY_MODE", "proactive").lower()
    if mode == "off":
        return dict(config)

    cfg = dict(config)
    L = int(cfg.get("num_layers", 0))
    d = int(cfg.get("d_model", 0))
    lr = float(cfg.get("learning_rate", 0.0))

    risk = 0
    if L >= 16:
        risk += 1
    if L >= 20:
        risk += 1
    if d <= 192:
        risk += 1
    if d <= 160:
        risk += 1
    if lr >= 5e-4:
        risk += 1
    if lr >= 6e-4:
        risk += 1

    lr_scale = 1.0
    if risk >= 2:
        lr_scale = 0.5
    if risk >= 3:
        lr_scale = 0.25
    if risk >= 4:
        lr_scale = 0.2

    if lr_scale < 1.0:
        new_lr = max(1e-4, lr * lr_scale)
        cfg["learning_rate"] = new_lr

    if compat.is_npu and risk >= 2:
        cfg["amp_enabled_override"] = True
        cfg["amp_dtype_override"] = "bf16"

    # Stricter guard for high-risk configs
    if risk >= 3:
        cfg["loss_guard_override"] = 12.0

    if lr_scale < 1.0 or cfg.get("amp_dtype_override") or cfg.get("loss_guard_override"):
        logger.info(
            f"Stability policy applied: risk={risk}, lr={lr:.2e}->{cfg['learning_rate']:.2e}, "
            f"amp={cfg.get('amp_dtype_override','default')}, guard={cfg.get('loss_guard_override','default')}"
        )

    return cfg


async def guardian_task():
    """Background task to ensure system health and resource cleanup."""
    logger.info("Guardian task started.")
    while app_state["is_running"]:
        try:
            # Regularly clear cache to help with fragmentation
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

    # Dynamic resource management
    logger.info(f"Job {job_id} requesting resources...")

    async def run_training():
        # Double check inside the lock for existing results (prevents redundant work from near-simultaneous requests)
        existing_loss = db.get_existing_run(api_key, config)
        if existing_loss is not None:
            total_used = db.get_total_flops(api_key)
            logger.info(
                f"Job {job_id} found existing result in DB after acquiring resources. Skipping."
            )
            return {"loss": existing_loss, "total_flops_used": total_used}

        # Initialize Job Tracking
        run_id = db.initialize_run(api_key, config)
        logger.info(
            f"Job {job_id} (run_id: {run_id}) started training on {DEVICE_TYPE}."
        )
        db.update_run_status(run_id, "RUNNING")

        group_sizes = _parse_group_sizes()
        last_error: Exception | None = None

        for idx, group_size in enumerate(group_sizes):
            device_ids = await resource_manager.acquire(job_id, config, group_size)
            try:
                logger.info(
                    f"Job {job_id} attempting group_size={group_size} devices={device_ids}"
                )
                # Proactive stability policy (single attempt by default)
                cfg = _apply_stability_policy(config)
                enable_fallback = os.getenv("TRAINING_ENABLE_FALLBACK", "0") == "1"

                if not enable_fallback:
                    loss = await run_in_threadpool(trainer.train, cfg, device_ids)
                else:
                    # Optional fallback chain if explicitly enabled
                    attempt_configs = [
                        ("stable", cfg),
                        (
                            "lr_x0.5_bf16",
                            {
                                **cfg,
                                "learning_rate": max(1e-4, cfg["learning_rate"] * 0.5),
                                "amp_enabled_override": True,
                                "amp_dtype_override": "bf16",
                            },
                        ),
                    ]
                    loss = None
                    for tag, attempt_cfg in attempt_configs:
                        try:
                            logger.info(f"Job {job_id} stability_attempt={tag}")
                            loss = await run_in_threadpool(
                                trainer.train, attempt_cfg, device_ids
                            )
                            break
                        except Exception as inner_e:
                            if "loss_diverged" in str(inner_e).lower():
                                logger.warning(
                                    f"Job {job_id} divergence detected on attempt={tag}; retrying fallback."
                                )
                                continue
                            raise
                    if loss is None:
                        raise RuntimeError("LOSS_DIVERGED: all stability attempts failed")

                # Record success
                db.update_run_status(run_id, "SUCCESS", loss=loss)
                db.update_total_flops(api_key, float(train_flops))
                new_total_used = db.get_total_flops(api_key)
                logger.info(f"Job {job_id} completed successfully. Loss: {loss}")
                return {"loss": loss, "total_flops_used": new_total_used}

            except Exception as e:
                error_msg = str(e)
                last_error = e
                if "loss_diverged" in error_msg.lower():
                    logger.warning(
                        f"Job {job_id} failed due to loss divergence at group_size={group_size}. "
                        f"Trying larger group if available."
                    )
                    if idx < len(group_sizes) - 1:
                        continue
                if _is_oom_error(error_msg):
                    logger.warning(
                        f"Job {job_id} hit OOM at group_size={group_size}. "
                        f"Trying larger group if available."
                    )
                    resource_manager.report_oom()
                    if idx < len(group_sizes) - 1:
                        continue
                    db.update_run_status(
                        run_id, "FAILED", error_message="NPU out of memory"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="NPU out of memory, please retry later",
                        headers={"Retry-After": "30"},
                    )

                logger.error(f"Job {job_id} failed: {error_msg}")
                db.update_run_status(run_id, "FAILED", error_message=error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            finally:
                compat.empty_cache()
                await resource_manager.release(job_id)

        if last_error is not None:
            raise last_error
        return {"loss": float("nan"), "total_flops_used": db.get_total_flops(api_key)}

    # Use shield to ensure training completes even if client disconnects
    return await asyncio.shield(run_training())


@app.get("/status")
async def get_status(wait: bool = False):
    if wait:
        try:
            # Wait for a job to finish or a timeout (to prevent hanging forever)
            async with resource_manager.lock:
                # We wait on the condition which is notified in release()
                await asyncio.wait_for(resource_manager.lock.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            pass
    return resource_manager.get_status()


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
