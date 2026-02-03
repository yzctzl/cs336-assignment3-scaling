# pyright: reportArgumentType=none
import concurrent.futures
import json
import logging
import os
import time
from threading import Lock
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests
from requests.exceptions import ConnectionError, Timeout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        try:
            # We use a very long timeout for the training request itself
            # but wrap it in logic that can recover from disconnections.
            response = requests.get(
                API_URL, params={**config, "api_key": API_KEY}, timeout=None
            )

            if response.status_code == 200:
                return response.json()["loss"]

            if response.status_code == 404:
                logger.error(f"Invalid configuration (404): {config}")
                return float("nan")

            if response.status_code == 403:
                detail = response.json().get("detail", "")
                logger.error(f"Forbidden (403): {detail}")
                return float("nan")

            if response.status_code == 503:
                # Server busy or OOM. Wait for a completion.
                logger.info(
                    "Server busy (503). Waiting for capacity via long polling..."
                )
                try:
                    requests.get(f"{STATUS_URL}?wait=true", timeout=70)
                except Exception:
                    time.sleep(10)
                continue

            logger.error(f"Server error {response.status_code}. Retrying...")
            time.sleep(10)

        except (Timeout, ConnectionError) as e:
            logger.warning(
                f"Connection issue: {e}. Checking if task finished in background..."
            )
            # If we timed out or lost connection, the server might still be working (asyncio.shield)
            loss = check_background_completion()
            if loss is not None:
                logger.info("Task found completed in background. Recovered loss.")
                return loss

            logger.info("Task not found in background. Re-submitting in 30s...")
            time.sleep(30)
        except Exception as e:
            logger.error(f"Unexpected error: {e}. Retrying...")
            time.sleep(10)


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

        # Capacity Pre-check: Don't even bother the server if we know it's full
        while True:
            try:
                status = requests.get(STATUS_URL, timeout=10).json()
                avail = status.get("available_capacity_gb", 0)
                if avail >= est_mem:
                    break

                logger.info(
                    f"Local pre-check: Insufficient capacity for N={n_params:.1e} "
                    f"(Need {est_mem:.1f}GB, Have {avail:.1f}GB). Waiting..."
                )
                requests.get(f"{STATUS_URL}?wait=true", timeout=70)
            except Exception:
                time.sleep(10)

        logger.info(
            f"Dispatching N={n_params:.2e}, D={d_tokens:.0e}, Budget={budget:.1e}, LR={lr}"
        )
        loss = get_loss(config)

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
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent
        ) as executor:
            futures = [
                executor.submit(self.process_row, row)
                for _, row in self.df_sweep.iterrows()
            ]
            done, _ = concurrent.futures.wait(futures)
            for f in done:
                try:
                    f.result()
                except Exception as e:
                    logger.error(f"Sweep task failed with error: {e}")


def process_directory(directory: str, max_concurrent: int = 32):
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
        logger.info(f"=== Starting Sweep: {csv_file} ===")
        csv_path = os.path.join(directory, csv_file)
        results_file = os.path.join(directory, csv_file.replace(".csv", ".json"))
        scheduler = SweepScheduler(csv_path, results_file)
        scheduler.run(max_concurrent=max_concurrent)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory with CSV files")
    parser.add_argument("--concurrent", type=int, default=32)
    args = parser.parse_args()

    # Graceful Shutdown Handling
    import signal
    import sys

    def signal_handler(sig, frame):
        logger.info("\nGraceful shutdown initiated. Cancelling pending tasks...")
        # Since we are in a thread pool, we can't easily kill running threads
        # but we can stop submitting new ones. The executor context manager handles wait.
        # But we force exit to be responsive.
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    process_directory(args.directory, max_concurrent=args.concurrent)
