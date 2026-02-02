# pyright: reportArgumentType=none
import json
import logging
import os
import time
from typing import Any, Dict

import click
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


def run_sweep_file(csv_path: str, results_file: str):
    df_sweep = pd.read_csv(csv_path, dtype={"dataset": str})

    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            all_results = json.load(f)
        logger.info(f"Loaded {len(all_results)} previous data points.")
    else:
        all_results = []

    # Iterate through unique configs in CSV
    for _, row in df_sweep.iterrows():
        c = float(row["Budget"])
        n_target = float(row["N"])
        lr = float(row["LR"])

        config = {
            "d_model": int(row["d_model"]),
            "num_layers": int(row["layers"]),
            "num_heads": int(row["heads"]),
            "batch_size": 128,
            "learning_rate": lr,
            "train_flops": int(c),
            "api_key": API_KEY,
            "dataset": row["dataset"],
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
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Saved: Loss={loss:.4f}")

    return all_results


@click.command()
@click.option(
    "--dir",
    "directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory containing budget/supple CSV files",
)
def main(directory: str):
    """
    Scans the given directory for all CSV files, runs the sweep for each,
    and updates a results.json in the same directory.
    """
    results_file = os.path.join(directory, "results.json")

    # 获取目录下所有的 CSV 文件
    csv_files = [
        f
        for f in os.listdir(directory)
        if f.endswith("budget.csv") or f.endswith("supple.csv")
    ]

    if not csv_files:
        logger.error(f"No CSV files found in {directory}")
        return

    logger.info(f"Found {len(csv_files)} CSV files in {directory}: {csv_files}")

    for csv_file in csv_files:
        csv_path = os.path.join(directory, csv_file)
        logger.info(f"--- Processing {csv_file} ---")
        run_sweep_file(csv_path, results_file)


if __name__ == "__main__":
    main()
