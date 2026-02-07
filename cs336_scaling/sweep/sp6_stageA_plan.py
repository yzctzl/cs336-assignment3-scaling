import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

VOCAB_SIZE = 32000
NUM_HEADS = 8

DEFAULT_BUDGETS = [1e15, 3e15, 6e15, 1e16, 3e16, 6e16]
DEFAULT_LAYERS = [6, 8, 10, 12, 14, 16]
DEFAULT_STAGE_A_POINTS = {
    1e15: 16,
    3e15: 16,
    6e15: 16,
    1e16: 12,
    3e16: 10,
    6e16: 8,
}

MIN_D_BY_LAYER = {
    6: 128,
    8: 160,
    10: 192,
    12: 224,
    14: 256,
    16: 288,
}

BASELINE_LR_BY_LAYER = {
    6: 4e-4,
    8: 4e-4,
    10: 4e-4,
    12: 3.5e-4,
    14: 3e-4,
    16: 3e-4,
}

def budget_label(budget: float) -> str:
    return f"{budget:.0e}".replace("+", "")


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_stage_a_points(raw: str) -> Dict[float, int]:
    points: Dict[float, int] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        left, right = chunk.split(":")
        points[float(left.strip())] = int(right.strip())
    return points


def get_active_layers(budget: float, all_layers: Iterable[int]) -> List[int]:
    selected = sorted(set(int(x) for x in all_layers))
    if budget <= 3e15:
        return [l for l in selected if l in [6, 8, 10, 12]]
    if budget <= 1e16:
        return [l for l in selected if l in [6, 8, 10, 12, 14]]
    return [l for l in selected if l in [8, 10, 12, 14, 16]]


def solve_d_model_from_n_total(n_total: float, layers: int) -> int:
    a = 12 * layers
    b = VOCAB_SIZE
    c = -n_total
    d_raw = (-b + math.sqrt(b**2 - 4 * a * c)) / (2 * a)
    d = int(round(d_raw / NUM_HEADS) * NUM_HEADS)
    d = max(64, min(1024, d))
    return d


def solve_d_model_from_n_non_emb(n_non_emb: float, layers: int) -> int:
    d_raw = math.sqrt(max(n_non_emb, 0.0) / (12 * layers))
    d = int(round(d_raw / NUM_HEADS) * NUM_HEADS)
    d = max(64, min(1024, d))
    return d


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


def select_log_spaced(candidates: List[Dict[str, float]], num_points: int) -> List[Dict[str, float]]:
    if len(candidates) <= num_points:
        return sorted(candidates, key=lambda r: r["N_non_emb"])

    candidates = sorted(candidates, key=lambda r: r["N_non_emb"])
    n_vals = np.array([float(r["N_non_emb"]) for r in candidates], dtype=np.float64)
    log_n = np.log10(n_vals)
    targets = np.linspace(log_n.min(), log_n.max(), num_points)

    remaining = set(range(len(candidates)))
    chosen: List[int] = []
    for t in targets:
        idx = min(remaining, key=lambda i: abs(log_n[i] - t))
        chosen.append(idx)
        remaining.remove(idx)

    selected = [candidates[i] for i in sorted(chosen)]
    return selected


def build_stage_a_for_budget(
    budget: float,
    num_points: int,
    layers: List[int],
    dn_min: float,
    dn_max: float,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    seen = set()
    for L in layers:
        min_d = MIN_D_BY_LAYER[L]
        lr = BASELINE_LR_BY_LAYER[L]
        for d in range(min_d, 1024 + 1, 8):
            n_non_emb = 12 * L * (d**2)
            n_total = n_non_emb + VOCAB_SIZE * d
            dn_ratio = budget / (6 * n_total * n_total)
            if dn_ratio < dn_min or dn_ratio > dn_max:
                continue
            key = (L, d, lr)
            if key in seen:
                continue
            seen.add(key)
            rows.append(compute_row(budget, L, d, lr))

    rows = sorted(rows, key=lambda r: r["N_non_emb"])
    selected = select_log_spaced(rows, num_points)
    df = pd.DataFrame(selected)
    if not df.empty:
        df = df.sort_values("N_non_emb").reset_index(drop=True)
    return df


def load_stage_a_results(out_dir: str, budget: float) -> List[Dict[str, float]]:
    name = f"results_sp6_stageA_{budget_label(budget)}_budget.json"
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    stable = [
        r
        for r in data
        if r.get("dataset") == "sp6"
        and isinstance(r.get("loss"), (int, float))
        and math.isfinite(float(r["loss"]))
        and float(r["loss"]) <= 20.0
    ]
    return stable


def recover_stage_a_arch(stage_a_df: pd.DataFrame, result: Dict[str, float]) -> Tuple[int, int, float]:
    # Prefer exact match on N_non_emb + LR.
    sub = stage_a_df[
        (stage_a_df["N_non_emb"].astype(float) == float(result["N"]))
        & (np.isclose(stage_a_df["LR"].astype(float), float(result["LR"]), atol=1e-12))
    ]
    if len(sub) == 0:
        # Fallback: nearest N_non_emb.
        idx = (stage_a_df["N_non_emb"].astype(float) - float(result["N"])).abs().idxmin()
        row = stage_a_df.loc[idx]
    else:
        row = sub.iloc[0]
    return int(row["layers"]), int(row["d_model"]), float(row["LR"])


def estimate_total_flops(stage_a: Dict[float, pd.DataFrame]) -> float:
    total = 0.0
    for budget, df in stage_a.items():
        total += budget * len(df)
    return total


def write_csvs(out_dir: str, stage_a: Dict[float, pd.DataFrame]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for budget, df in stage_a.items():
        if df.empty:
            continue
        name = f"sp6_stageA_{budget_label(budget)}_budget.csv"
        df.to_csv(os.path.join(out_dir, name), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="artifacts/chinchilla_sweep/sp6",
        help="Output directory for stage csv files",
    )
    parser.add_argument(
        "--target-flops",
        type=float,
        default=1.8e18,
        help="Total scan budget cap; script aborts if exceeded",
    )
    parser.add_argument(
        "--budgets",
        default="1e15,3e15,6e15,1e16,3e16,6e16",
        help="Comma-separated budget list",
    )
    parser.add_argument(
        "--layers",
        default="6,8,10,12,14,16",
        help="Comma-separated candidate layer list",
    )
    parser.add_argument(
        "--stage-a-points",
        default="1e15:16,3e15:16,6e15:16,1e16:12,3e16:10,6e16:8",
        help="Budget:count mapping for Stage A",
    )
    parser.add_argument(
        "--dn-min",
        type=float,
        default=0.2,
        help="Lower D/N target for Stage A candidate generation",
    )
    parser.add_argument(
        "--dn-max",
        type=float,
        default=220.0,
        help="Upper D/N target for Stage A candidate generation",
    )
    parser.add_argument(
        "--max-no-layer-expand-budget",
        type=float,
        default=1e16,
        help="Do not expand layers above Stage A anchor for budgets <= this value",
    )
    args = parser.parse_args()

    budgets = parse_float_list(args.budgets)
    layers = parse_int_list(args.layers)
    points = parse_stage_a_points(args.stage_a_points)

    stage_a: Dict[float, pd.DataFrame] = {}
    for budget in budgets:
        active_layers = get_active_layers(budget, layers)
        n_points = points.get(budget, DEFAULT_STAGE_A_POINTS.get(budget, 8))
        df = build_stage_a_for_budget(
            budget=budget,
            num_points=n_points,
            layers=active_layers,
            dn_min=args.dn_min,
            dn_max=args.dn_max,
        )
        if not df.empty:
            if not df["N_non_emb"].is_monotonic_increasing:
                raise RuntimeError(f"Stage A N_non_emb not monotonic for budget {budget}")
        stage_a[budget] = df

    total_est = estimate_total_flops(stage_a)
    if total_est > args.target_flops:
        raise SystemExit(
            f"Refusing to write files: estimated FLOPs {total_est:.3e} exceeds cap {args.target_flops:.3e}"
        )

    write_csvs(args.out_dir, stage_a)

    print("Stage A summary:")
    for budget in budgets:
        df = stage_a.get(budget, pd.DataFrame())
        print(f"  C={budget_label(budget)} runs={len(df)}")
    print(f"Estimated total FLOPs: {total_est:.3e}")


if __name__ == "__main__":
    main()
