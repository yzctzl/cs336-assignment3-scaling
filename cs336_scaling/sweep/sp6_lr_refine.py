import json
import os
from typing import Iterable

import numpy as np
import pandas as pd

VOCAB_SIZE = 32000
NUM_HEADS = 8


def _solve_d_model(n_total: float, layers: int) -> int:
    a = 12 * layers
    b = VOCAB_SIZE
    c = -n_total
    d_raw = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    d_model = int(round(d_raw / NUM_HEADS) * NUM_HEADS)
    d_model = max(64, min(1024, d_model))
    return d_model


def _pick_nearby_archs(
    n_targets: list[float], layers_list: Iterable[int]
) -> list[tuple[int, int]]:
    picks = []
    for n_total in n_targets:
        best = None
        for L in layers_list:
            d_model = _solve_d_model(n_total, L)
            n_non_emb = 12 * L * (d_model**2)
            n_emb = VOCAB_SIZE * d_model
            n_total_actual = n_non_emb + n_emb
            err = abs(n_total_actual - n_total)
            if best is None or err < best[0]:
                best = (err, L, d_model)
        if best is not None:
            picks.append((best[1], best[2]))
    # dedupe while preserving order
    seen = set()
    unique = []
    for L, d in picks:
        if (L, d) not in seen:
            seen.add((L, d))
            unique.append((L, d))
    return unique


def generate_lr_refine(
    results_dir: str,
    out_csv: str,
    lr_list: list[float],
    layers_list: Iterable[int],
    n_multipliers: list[float],
):
    results = []
    for f in os.listdir(results_dir):
        if f.startswith("results_") and f.endswith(".json"):
            with open(os.path.join(results_dir, f)) as fh:
                results.extend(json.load(fh))

    byC = {}
    for r in results:
        byC.setdefault(r["C"], []).append(r)

    rows = []
    for C, rs in byC.items():
        # only use sp6
        rs = [r for r in rs if r.get("dataset") == "sp6"]
        if not rs:
            continue
        min_r = min(rs, key=lambda r: r["loss"])
        n0 = float(min_r["N"])
        targets = [n0 * m for m in n_multipliers]
        archs = _pick_nearby_archs(targets, layers_list)
        for L, d_model in archs:
            n_non_emb = 12 * L * (d_model**2)
            n_emb = VOCAB_SIZE * d_model
            n_total = n_non_emb + n_emb
            tokens = C / (6 * n_total)
            dn_ratio = tokens / n_total
            for lr in lr_list:
                rows.append(
                    {
                        "Budget": f"{C:.0e}",
                        "layers": L,
                        "d_model": d_model,
                        "heads": NUM_HEADS,
                        "LR": lr,
                        "N_non_emb": int(n_non_emb),
                        "N_emb": int(n_emb),
                        "N": int(n_total),
                        "D_N_ratio": round(dn_ratio, 2),
                        "Tokens": float(f"{tokens:.2e}"),
                        "dataset": "sp6",
                    }
                )

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["Budget", "layers", "d_model", "LR"]
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    RESULTS_DIR = "artifacts/chinchilla_sweep/sp6_full"
    OUT_CSV = "artifacts/chinchilla_sweep/sp6_full/sp6_lr_refine.csv"
    LR_LIST = [2e-4, 4e-4, 6e-4]
    LAYERS = [6, 8, 12, 16, 20, 24]
    N_MULT = [0.7, 0.85, 1.0, 1.2, 1.4]
    generate_lr_refine(RESULTS_DIR, OUT_CSV, LR_LIST, LAYERS, N_MULT)
