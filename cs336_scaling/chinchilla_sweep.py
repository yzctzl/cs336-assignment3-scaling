# pyright: reportArgumentType=none
import concurrent.futures
import json
import logging
import os
import threading
import time
import weakref
from threading import Lock
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global shutdown event for graceful exit
shutdown_event = threading.Event()
# Track active sessions to force-close them on shutdown
active_sessions = weakref.WeakSet()

# Constants
API_URL = "http://localhost:8000/loss"
STATUS_URL = "http://localhost:8000/status"
PREVIOUS_RUNS_URL = "http://localhost:8000/previous_runs"
API_KEY = "chinchilla_method2_sweep_key"

# Global lock for results file access
results_lock = Lock()


def get_loss(config: Dict[str, Any]) -> float:
    """Robustly gets loss with infinite dynamic retry and background completion check."""

    def check_background_completion():
        """Checks if the task already completed in the background on the server."""
        try:
            resp = requests.get(
                PREVIOUS_RUNS_URL, params={"api_key": API_KEY}, timeout=30
            )
            if resp.status_code == 200:
                runs = resp.json().get("previous_runs", [])
                for run in runs:
                    # Match config (simplistic check based on key params)
                    if (
                        run.get("d_model") == config["d_model"]
                        and run.get("num_layers") == config["num_layers"]
                        and abs(run.get("learning_rate", 0) - config["learning_rate"])
                        < 1e-9
                        and run.get("train_flops") == config["train_flops"]
                    ):
                        if run.get("status") == "SUCCESS":
                            return run.get("loss")
        except Exception as e:
            logger.debug(f"Error checking background completion: {e}")
        return None

    while True:
        # 1. Attempt to execute the task
        try:
            if shutdown_event.is_set():
                return float("nan")

            with requests.Session() as session:
                active_sessions.add(session)
                try:
                    # Block indefinitely for the slot
                    response = session.get(
                        API_URL, params={**config, "api_key": API_KEY}, timeout=None
                    )
                except (
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                ):
                    if shutdown_event.is_set():
                        return float("nan")
                    raise

                if response.status_code == 200:
                    return response.json()["loss"]

                if response.status_code in [404, 403]:
                    logger.error(f"Fatal error {response.status_code}: {response.text}")
                    return float("nan")

                # If 500 or 503, we fall through to the wait logic
                if response.status_code == 503:
                    pass  # Expected "busy" state
                else:
                    logger.error(
                        f"Server error {response.status_code}. Waiting for signal..."
                    )

        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.Timeout,
        ) as e:
            logger.warning(f"Connection issue: {e}. Checking background...")
            loss = check_background_completion()
            if loss is not None:
                return loss
            # Fall through to wait logic

        except Exception as e:
            logger.error(f"Unexpected error: {e}. Retrying after signal...")

        # 2. Wait for signal (Barrier)
        # We failed to get a result (503, 500, or Network Error).
        # We block here until the server tells us a job has finished (capacity freed).
        try:
            if shutdown_event.is_set():
                return float("nan")

            with requests.Session() as wait_session:
                active_sessions.add(wait_session)
                # This blocks indefinitly until server notifies (or we kill it)
                wait_session.get(f"{STATUS_URL}?wait=true", timeout=None)
        except Exception:
            # If the wait itself fails (e.g. server down), we sleep briefly to avoid tight loop
            if shutdown_event.wait(5):
                return float("nan")


class SweepScheduler:
    """Capacity-aware reactive dispatcher for Scaling Law sweeps."""

    def __init__(self, csv_path: str, results_file: str):
        self.csv_path = csv_path
        self.results_file = results_file
        self.df_sweep = pd.read_csv(csv_path, dtype={"dataset": str})
        self.results = []
        self.completed_keys = set()
        self.load_results()

    def load_results(self):
        """Loads existing results and populates completion set."""
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, "r") as f:
                    self.results = json.load(f)
                    for r in self.results:
                        # Use a robust key for tracking
                        # Legacy results might not have 'dataset', assume '256' or handle gracefully
                        # But (N, D, LR) should be unique enough per file
                        key = (
                            float(r["N"]),
                            float(r["D"]),
                            float(r["LR"]),
                            r.get("dataset", "256"),  # Default for legacy
                        )
                        self.completed_keys.add(key)
                logger.info(
                    f"Loaded {len(self.results)} existing results from {self.results_file}"
                )
            except Exception as e:
                logger.error(f"Failed to load results: {e}")

    def save_result(self, result: Dict[str, Any]):
        """Atomically saves a new result."""
        with results_lock:
            self.results.append(result)
            temp_file = f"{self.results_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump(self.results, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, self.results_file)

    def estimate_job_memory(self, row: pd.Series) -> float:
        """Heuristic to estimate model memory before submittal (matches server logic)."""
        d_model = int(row["d_model"])
        num_layers = int(row["layers"])
        n_params = 12 * num_layers * (d_model**2)
        model_gb = (n_params * 12) / (1024**3)
        # Simplified activation overhead
        # Use .get for batch_size since it's missing in some CSVs
        batch_size = int(row.get("batch_size", 128))
        act_gb = (batch_size * 512 * d_model * num_layers * 16) / (1024**3)
        return model_gb + act_gb + 0.5

    def process_row(self, row: pd.Series):
        if shutdown_event.is_set():
            return

        budget = float(row["Budget"])
        lr = float(row["LR"])
        n_target = float(row["N"])
        n_params = 12 * int(row["layers"]) * (int(row["d_model"]) ** 2)
        d_tokens = budget // (6 * n_params)
        dataset = row["dataset"]

        # If legacy results default to "256" and current is "256", fine.
        key = (float(n_params), float(d_tokens), lr, str(dataset))
        if key in self.completed_keys:
            return

        bucket_batch_size = int(row.get("batch_size", 128))
        config = {
            "d_model": int(row["d_model"]),
            "num_layers": int(row["layers"]),
            "num_heads": int(row["heads"]),
            "batch_size": bucket_batch_size,
            "learning_rate": lr,
            "train_flops": int(budget),
            "dataset": dataset,
        }

        est_mem = self.estimate_job_memory(row)

        # Capacity Pre-check: Check shutdown_event in loop
        while not shutdown_event.is_set():
            try:
                status = requests.get(STATUS_URL, timeout=10).json()
                avail = status.get("available_capacity_gb", 0)
                if avail >= est_mem:
                    break

                logger.info(
                    f"Local pre-check: Insufficient capacity for N={n_params:.1e} "
                    f"(Need {est_mem:.1f}GB, Have {avail:.1f}GB). Waiting..."
                )

                # Check event during long wait simulation
                for _ in range(7):  # 7 * 10s = 70s wait equivalent
                    if shutdown_event.is_set():
                        return
                    try:
                        # Use short wait to be responsive
                        requests.get(f"{STATUS_URL}?wait=true", timeout=10)
                    except Exception:
                        pass

            except Exception:
                time.sleep(5)

        if shutdown_event.is_set():
            return

        logger.info(
            f"Dispatching N={n_params:.2e}, D={d_tokens:.0e}, Budget={budget:.1e}, LR={lr}"
        )
        try:
            loss = get_loss(config)
        except Exception as e:
            logger.error(f"Task failed: {e}")
            return

        if shutdown_event.is_set():
            return

        if not np.isnan(loss):
            result = {
                "N": n_params,
                "D": d_tokens,
                "C": budget,
                "LR": lr,
                "loss": loss,
                "dataset": dataset,
                "target_n": n_target,
            }
            self.save_result(result)
            self.completed_keys.add(key)
            logger.info(
                f"Completed: N={n_params:.2e}, D={d_tokens:.2e}, LR={lr:.2e} -> Loss={loss:.4f}"
            )

    def run(self, max_concurrent: int = 128):
        logger.info(f"Starting Sweep for {self.csv_path}...")

        # Manually manage executor to allow fast shutdown (avoid 'with' block's forced wait)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
        futures = []

        try:
            # Submit tasks
            for _, row in self.df_sweep.iterrows():
                if shutdown_event.is_set():
                    break
                futures.append(executor.submit(self.process_row, row))

            # Wait for completion or shutdown
            # We convert list to set for efficient removal
            pending_futures = set(futures)
            while pending_futures and not shutdown_event.is_set():
                # Wait for at least one future to complete, but timeout quickly to check shutdown_event
                done, _ = concurrent.futures.wait(
                    pending_futures,
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for f in done:
                    pending_futures.remove(f)
                    try:
                        f.result()
                    except Exception as e:
                        logger.error(f"Sweep task failed with error: {e}")

        except KeyboardInterrupt:
            logger.info("\nRun interrupted by user.")
            shutdown_event.set()

        finally:
            if shutdown_event.is_set():
                logger.info("Shutdown event set. Cancelling pending tasks...")
                # Best effort cancel for pending tasks
                for f in futures:
                    f.cancel()

                logger.info("Shutting down executor (wait=False)...")
                # Python 3.9+ supports cancel_futures=True
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)
            logger.info("Sweep run finished.")


def process_directory(directory: str, max_concurrent: int = 4):
    csv_files = [
        f
        for f in os.listdir(directory)
        if f.endswith("budget.csv") or f.endswith("supple.csv")
    ]
    csv_files.sort()

    if not csv_files:
        logger.error(f"No sweep files found in {directory}")
        return

    for csv_file in csv_files:
        if shutdown_event.is_set():
            break

        logger.info(f"=== Starting Sweep: {csv_file} ===")
        csv_path = os.path.join(directory, csv_file)
        results_file = os.path.join(directory, "results.json")
        scheduler = SweepScheduler(csv_path, results_file)
        scheduler.run(max_concurrent=max_concurrent)


if __name__ == "__main__":
    import argparse
    import signal

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory with CSV files")
    parser.add_argument("--concurrent", type=int, default=1)
    args = parser.parse_args()

    # Graceful Shutdown Handling
    def signal_handler(sig, frame):
        logger.info("\nSIGINT received. Setting shutdown event...")
        shutdown_event.set()

        # Force close all active sessions to unblock threads waiting on generic read()
        logger.info(
            f"Closing {len(active_sessions)} active sessions to unblock helper threads..."
        )
        for session in list(active_sessions):
            try:
                session.close()
            except Exception:
                pass

    signal.signal(signal.SIGINT, signal_handler)

    process_directory(args.directory, max_concurrent=args.concurrent)

    if shutdown_event.is_set():
        logger.info(
            "Shutdown event was set. Exiting via os._exit to bypass daemon thread waits."
        )
        logging.shutdown()
        os._exit(0)
