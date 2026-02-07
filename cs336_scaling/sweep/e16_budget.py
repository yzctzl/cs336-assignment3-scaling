import os

import numpy as np
import pandas as pd

VOCAB_SIZE = 32000
NUM_HEADS = 8


def generate_fixed_layer_sweep(
    budget=6e15, L=6, num_points=18, lr=4e-4, dn_min=11, dn_max=180
):

    # Previous analysis showed 4e-4 caused instability (Loss~7.8) for some N.
    # We use a geometric spread:
    # 2e-4: Safe harbor (stability)
    # 4e-4: Previous baseline (performance)
    # 6e-4: Aggressive (for smallest N)
    learning_rates = [2e-4, 4e-4, 6e-4]

    # Range of D/N (Total) from 11 to 180
    n_total_min = np.sqrt(budget / (6 * dn_max))
    n_total_max = np.sqrt(budget / (6 * dn_min))

    # Generate target N_total in log space
    target_n_totals = np.logspace(
        np.log10(n_total_min), np.log10(n_total_max), num_points * 2
    )  # Generate more to account for rounding/clashing

    plan = []
    for t_n_total in target_n_totals:
        # Solve 12*L*d^2 + V*d - t_n_total = 0
        a = 12 * L
        b = VOCAB_SIZE
        c = -t_n_total
        d_raw = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        # Aligned d_model (must be divisible by num_heads)
        d_model = int(round(d_raw / NUM_HEADS) * NUM_HEADS)
        d_model = max(64, min(1024, d_model))

        # Calculate actuals
        n_non_emb = 12 * L * (d_model**2)
        n_emb = VOCAB_SIZE * d_model
        n_total = n_non_emb + n_emb

        tokens = budget / (6 * n_total)
        dn_ratio = tokens / n_total

        for lr in learning_rates:
            plan.append(
                {
                    "Budget": f"{budget:.0e}",
                    "layers": L,
                    "d_model": d_model,
                    "heads": NUM_HEADS,
                    # "batch_size": BATCH_SIZE,
                    "LR": lr,
                    "N_non_emb": int(n_non_emb),
                    "N_emb": int(n_total - n_non_emb),
                    "N": int(n_total),
                    "D_N_ratio": round(dn_ratio, 2),
                    "Tokens": float(f"{tokens:.2e}"),
                    "dataset": "sp6",
                }
            )

    df = pd.DataFrame(plan)
    # Drop duplicate (d_model, LR) pairs
    df = (
        df.drop_duplicates(subset=["d_model", "LR"])
        .sort_values(["d_model", "LR"])
        .reset_index(drop=True)
    )

    # If we have too many points, sample distinct architectures then explode LRs
    # Ideally we keep all LRs for sampled architecture.
    unique_d_models = df["d_model"].unique()
    if len(unique_d_models) > num_points:
        # Sample architectures
        indices = np.linspace(0, len(unique_d_models) - 1, num_points).astype(int)
        selected_d_models = unique_d_models[indices]
        df = df[df["d_model"].isin(selected_d_models)].reset_index(drop=True)  # pyright: ignore[reportArgumentType]

    return df


BUDGET = "6e16"

df_fixed = generate_fixed_layer_sweep(float(BUDGET), L=12, num_points=12)
os.makedirs(f"artifacts/chinchilla_sweep/{BUDGET}", exist_ok=True)
df_fixed.to_csv(f"artifacts/chinchilla_sweep/{BUDGET}/{BUDGET}_budget.csv", index=False)

print(
    df_fixed[["layers", "d_model", "N_non_emb", "N", "D_N_ratio"]].to_string(  # pyright: ignore[reportAttributeAccessIssue]
        index=False
    )
)
