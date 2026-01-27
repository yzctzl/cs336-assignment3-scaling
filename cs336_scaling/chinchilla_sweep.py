# pyright: reportArgumentType=none
import json
import logging
import os
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
API_URL = "http://localhost:8000/loss"
API_KEY = "chinchilla_method2_sweep_key"
TARGET_BUDGET = 1e19
SWEEP_CSV = "cs336_scaling/sweep/low_budget.csv"
RESULTS_FILE = "artifacts/chinchilla_sweep/low/results.json"
PLOTS_DIR = "artifacts/chinchilla_sweep/low"


def get_loss(config: Dict[str, Any], poll_interval: int = 30) -> float:
    """Robustly gets loss by polling, especially for long runs."""
    try:
        response = requests.get(API_URL, params=config, timeout=10)
        if response.status_code == 200:
            return response.json()["loss"]
    except Exception:
        pass

    while True:
        try:
            response = requests.get(API_URL, params=config, timeout=300)
            if response.status_code == 200:
                return response.json()["loss"]
            elif response.status_code == 500:
                detail = response.json().get("detail", "")
                if "out of memory" in detail.lower():
                    logger.warning(f"OOM for config: {config}")
                    return float("nan")
                logger.error(f"Server error: {detail}")
                return float("nan")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            logger.info(
                f"Training in progress for C={config['train_flops']:.1e}. Polling..."
            )
            time.sleep(poll_interval)
        except Exception as e:
            logger.error(f"Unexpected error in get_loss: {e}")
            return float("nan")


def chinchilla_model(X, E, A, B, alpha, beta):
    N, D = X
    return E + A / (N**alpha) + B / (D**beta)


def run_sweep():
    if not os.path.exists(SWEEP_CSV):
        logger.error(f"Sweep CSV not found at {SWEEP_CSV}")
        return []

    df_sweep = pd.read_csv(SWEEP_CSV)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            all_results = json.load(f)
        logger.info(f"Loaded {len(all_results)} previous data points.")
    else:
        all_results = []

    # Iterate through unique configs in CSV
    for _, row in df_sweep.iterrows():
        c = float(row["Budget"])
        n_target = float(row["Group_Center_N"])
        lr = float(row["LR"])

        config = {
            "d_model": int(row["d_model"]),
            "num_layers": int(row["layers"]),
            "num_heads": int(row["heads"]),
            "batch_size": 128
            if row["layers"] < 10
            else 64,  # Adaptive BS to prevent OOM
            "learning_rate": lr,
            "train_flops": int(c),
            "api_key": API_KEY,
        }

        # Calculate actual N params
        n_params = 12 * config["num_layers"] * (config["d_model"] ** 2)

        # Skip if already done
        exists = any(
            abs(r["C"] - c) / c < 0.01
            and abs(r["N"] - n_params) / n_params < 0.01
            and abs(r["LR"] - lr) / lr < 0.01
            for r in all_results
        )
        if exists:
            continue

        logger.info(f"Running N={n_params:.2e}, C={c:.2e}, LR={lr}")
        loss = get_loss(config)

        if not np.isnan(loss):
            d_tokens = c / (6 * n_params)
            all_results.append(
                {
                    "N": n_params,
                    "D": d_tokens,
                    "C": c,
                    "LR": lr,
                    "loss": loss,
                    "target_n": n_target,
                }
            )
            with open(RESULTS_FILE, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Saved: Loss={loss:.4f}")

    return all_results


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    results = run_sweep()
