import argparse
import math
import os
from typing import Dict, List

import pandas as pd

VOCAB_SIZE = 32000
NUM_HEADS = 8


def solve_d_model_from_n_non_emb(n_non_emb: float, layers: int) -> int:
    d_raw = math.sqrt(max(n_non_emb, 0.0) / (12.0 * layers))
    d = int(round(d_raw / NUM_HEADS) * NUM_HEADS)
    return max(64, min(1024, d))


def compute_row(budget: float, layers: int, d_model: int, lr: float) -> Dict[str, float]:
    n_non_emb = int(12 * layers * (d_model**2))
    n_emb = int(VOCAB_SIZE * d_model)
    n_total = int(n_non_emb + n_emb)
    tokens = budget / (6 * n_total)
    dn_ratio = tokens / n_total
    return {
        "Budget": f"{budget:.0e}",
        "layers": layers,
        "d_model": d_model,
        "heads": NUM_HEADS,
        "LR": float(f"{lr:.6f}"),
        "N_non_emb": n_non_emb,
        "N_emb": n_emb,
        "N": n_total,
        "D_N_ratio": round(dn_ratio, 4),
        "Tokens": float(f"{tokens:.6e}"),
        "dataset": "sp6",
    }


def build_budget_df(
    budget: float,
    n_center: float,
    layers: List[int],
    n_mults: List[float],
    lrs: List[float],
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    seen = set()
    for layer in layers:
        for m in n_mults:
            target_n = n_center * m
            d_model = solve_d_model_from_n_non_emb(target_n, layer)
            for lr in lrs:
                key = (layer, d_model, round(lr, 9))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(compute_row(budget, layer, d_model, lr))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["N_non_emb", "LR"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="artifacts/chinchilla_sweep/sp6/stageB_plus",
        help="Output folder for StageB+ csv files",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # StageB+ anchors provided by user judgement.
    # 1e16: N_opt around 5.3e6
    # 6e15: N_opt around 2.0e6
    cfg = {
        1e16: {
            "n_center": 5.3e6,
            "layers": [8, 10],
            "n_mults": [0.88, 0.94, 1.00, 1.06, 1.12],
            "lrs": [3.2e-4, 3.6e-4, 4.0e-4],
        },
        6e15: {
            "n_center": 2.0e6,
            "layers": [6, 8],
            # denser low-N window around the observed trough region
            "n_mults": [0.75, 0.85, 0.95, 1.00, 1.05, 1.15, 1.30],
            # keep LR centered near 4e-4 with mild perturbations
            "lrs": [3.4e-4, 4.0e-4, 4.6e-4],
        },
    }

    total_flops = 0.0
    for budget, c in cfg.items():
        df = build_budget_df(
            budget=budget,
            n_center=c["n_center"],
            layers=c["layers"],
            n_mults=c["n_mults"],
            lrs=c["lrs"],
        )
        name = f"{budget:.0e}".replace("+", "") + "_budget.csv"
        path = os.path.join(args.out_dir, name)
        df.to_csv(path, index=False)
        total_flops += budget * len(df)
        print(
            f"Wrote {path}: runs={len(df)}, N_non_emb=[{int(df['N_non_emb'].min())}, {int(df['N_non_emb'].max())}]"
        )

    print(f"Estimated StageB+ FLOPs: {total_flops:.3e}")


if __name__ == "__main__":
    main()
