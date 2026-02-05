import os

import numpy as np
import pandas as pd


def generate_fixed_layer_sweep(budget=6e15, L=6, num_points=18, lr = 5e-4):
    VOCAB_SIZE = 10000
    # BATCH_SIZE = 128
    LEARNING_RATE = lr
    NUM_HEADS = 8

    # Range of D/N (Total) from 11 to 180
    n_total_min = np.sqrt(budget / (6 * 180))
    n_total_max = np.sqrt(budget / (6 * 11))

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

        plan.append(
            {
                "Budget": f"{budget:.0e}",
                "layers": L,
                "d_model": d_model,
                "heads": NUM_HEADS,
                # "batch_size": BATCH_SIZE,
                "LR": LEARNING_RATE,
                "N_non_emb": int(n_non_emb),
                "N_emb": int(n_total - n_non_emb),
                "N": int(n_total),
                "D_N_ratio": round(dn_ratio, 2),
                "Tokens": float(f"{tokens:.2e}"),
                "dataset": "tss",
            }
        )

    df = pd.DataFrame(plan)
    # Drop duplicate d_models to keep exactly one point per distinct architecture
    df = (
        df.drop_duplicates(subset=["d_model"])
        .sort_values("d_model")
        .reset_index(drop=True)
    )

    # If we have too many points, sample them evenly to reach target
    if len(df) > num_points:
        indices = np.linspace(0, len(df) - 1, num_points).astype(int)
        df = df.iloc[indices].reset_index(drop=True)

    return df


BUDGET = "1e15"

df_fixed = generate_fixed_layer_sweep(float(BUDGET), L=6, num_points=18)
os.makedirs(f"artifacts/chinchilla_sweep/{BUDGET}", exist_ok=True)
df_fixed.to_csv(f"artifacts/chinchilla_sweep/{BUDGET}/{BUDGET}_budget.csv", index=False)

print(
    df_fixed[["layers", "d_model", "N_non_emb", "N", "D_N_ratio"]].to_string(
        index=False
    )
)
