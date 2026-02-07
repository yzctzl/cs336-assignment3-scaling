import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd

VOCAB_SIZE = 32000
NUM_HEADS = 8


@dataclass(frozen=True)
class ScanBlock:
    layers: Tuple[int, ...]
    n_mults: Tuple[float, ...]
    lrs: Tuple[float, ...]


@dataclass(frozen=True)
class BudgetPlan:
    budget: float
    n_center: float
    blocks: Tuple[ScanBlock, ...]


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


def iter_budget_rows(plan: BudgetPlan) -> Iterable[Dict[str, float]]:
    seen = set()
    for block in plan.blocks:
        for layer in block.layers:
            for mult in block.n_mults:
                d_model = solve_d_model_from_n_non_emb(plan.n_center * mult, layer)
                for lr in block.lrs:
                    key = (layer, d_model, round(lr, 9))
                    if key in seen:
                        continue
                    seen.add(key)
                    yield compute_row(plan.budget, layer, d_model, lr)


def build_budget_df(plan: BudgetPlan) -> pd.DataFrame:
    rows = list(iter_budget_rows(plan))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["N_non_emb", "LR"]).reset_index(drop=True)


def default_plans() -> List[BudgetPlan]:
    # Unified Stage-B strategy:
    # - absorb old B: keep high-budget (3e16/6e16) refinement
    # - absorb B+: keep 6e15/1e16 fixes around known troughs
    # - fix old B failure mode: avoid over-concentrating on tiny windows
    return [
        BudgetPlan(
            budget=6e15,
            n_center=2.0e6,
            blocks=(
                # Main local refinement around 6e15 trough.
                ScanBlock(
                    layers=(6, 8),
                    n_mults=(0.85, 1.00),
                    lrs=(3.4e-4, 4.0e-4),
                ),
                # Right-side check to avoid left-edge bias.
                ScanBlock(
                    layers=(6, 8),
                    n_mults=(1.15,),
                    lrs=(4.0e-4,),
                ),
            ),
        ),
        BudgetPlan(
            budget=1e16,
            n_center=5.3e6,
            blocks=(
                # Core region around observed 1e16 trough.
                ScanBlock(
                    layers=(8, 10),
                    n_mults=(0.82, 0.94, 1.06),
                    lrs=(3.2e-4, 3.8e-4),
                ),
                # Low-N rescue points from B+ experience.
                ScanBlock(
                    layers=(8, 10),
                    n_mults=(0.74,),
                    lrs=(3.6e-4,),
                ),
            ),
        ),
        BudgetPlan(
            budget=3e16,
            n_center=6.6e6,
            blocks=(
                # Broaden around both old-B minimum and monotonic-growth hypothesis.
                ScanBlock(
                    layers=(8, 10),
                    n_mults=(0.62, 0.88, 1.12),
                    lrs=(3.0e-4, 3.5e-4),
                ),
            ),
        ),
        BudgetPlan(
            budget=6e16,
            n_center=8.5e6,
            blocks=(
                # Intentionally wide to test "unexpected low-N optimum" vs expected scaling.
                ScanBlock(
                    layers=(8, 10),
                    n_mults=(0.55, 0.85, 1.30),
                    lrs=(2.8e-4, 3.3e-4),
                ),
            ),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="artifacts/chinchilla_sweep/sp6/stageB",
        help="Output folder for unified Stage-B csv files",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    total_flops = 0.0
    for plan in default_plans():
        df = build_budget_df(plan)
        budget_name = f"{plan.budget:.0e}".replace("+", "")
        path = os.path.join(args.out_dir, f"{budget_name}_budget.csv")
        df.to_csv(path, index=False)
        total_flops += plan.budget * len(df)
        n_min = int(df["N_non_emb"].min())
        n_max = int(df["N_non_emb"].max())
        print(f"Wrote {path}: runs={len(df)}, N_non_emb=[{n_min}, {n_max}]")

    print(f"Estimated unified Stage-B FLOPs: {total_flops:.3e}")


if __name__ == "__main__":
    main()
