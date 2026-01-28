import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ---------------- Configuration ----------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TARGET_BUDGET = 1e19
RESULT_DIRS = [
    "artifacts/chinchilla_sweep/low",
    "artifacts/chinchilla_sweep/mid",
    "artifacts/chinchilla_sweep/high",
]
OUTPUT_DIR = "artifacts/chinchilla_sweep/final_plots_definitive"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Tool Functions ----------------


def get_non_embedding_params(row):
    """Robustly calculate non-embedding parameters."""
    layers = row.get("num_layers")
    d_model = row.get("d_model")
    if layers is not None and d_model is not None and not pd.isna(layers):
        return 12 * layers * (d_model**2)
    if "N" in row and not pd.isna(row["N"]):
        return row["N"]
    return None


def load_data():
    """Load, clean, and extract Best LR for each (C, N)."""
    all_data = []
    for d in RESULT_DIRS:
        res_file = os.path.join(d, "results.json")
        if os.path.exists(res_file):
            with open(res_file, "r") as f:
                try:
                    all_data.extend(json.load(f))
                except:
                    pass

    if not all_data:
        raise ValueError("No data found!")
    df = pd.DataFrame(all_data)

    # Standardize columns
    if "C" not in df.columns:
        for col in ["train_flops", "Budget", "Cost"]:
            if col in df.columns:
                df["C"] = df[col]
                break
    if "LR" not in df.columns and "learning_rate" in df.columns:
        df["LR"] = df["learning_rate"]

    df["C"] = df["C"].astype(float)
    df["N"] = df.apply(get_non_embedding_params, axis=1)
    df = df.dropna(subset=["N", "C", "loss"])
    df["D"] = df["C"] / (6 * df["N"])

    # Filter: Keep only Best LR for each (C, N) pair
    # This eliminates vertical noise in the plots
    df_best = df.loc[df.groupby(["C", "N"])["loss"].idxmin()].reset_index(drop=True)
    return df_best.sort_values(["C", "N"])


# ---------------- Fitting Logic (Hybrid + Smooth) ----------------


def parabolic_fit_and_extract(df):
    """
    Fits a parabola for EACH Budget C.
    Returns:
      1. fit_curves: Smooth curve data for Plot 1 (visual only).
      2. optima: Valid optimal points for Plots 2 & 3 (Vertex if U-shape, Boundary if Monotonic).
    """
    unique_cs = sorted(df["C"].unique())
    optima = []
    fit_curves = []

    for c in unique_cs:
        sub = df[df["C"] == c].sort_values("N")
        if len(sub) < 3:
            continue

        log_n = np.log10(sub["N"].values)
        loss = sub["loss"].values

        try:
            # Fit Parabola: Loss = a*(logN)^2 + b*logN + c
            coeffs = np.polyfit(log_n, loss, 2)
            a, b, intercept = coeffs

            # 1. Generate Smooth Curve for Visualization (Plot 1)
            # Extrapolate slightly for better visuals
            x_smooth = np.linspace(log_n.min() - 0.1, log_n.max() + 0.1, 100)
            y_smooth = np.polyval(coeffs, x_smooth)
            fit_curves.append({"C": c, "N": 10**x_smooth, "loss": y_smooth})

            # 2. Extract Optimum (Hybrid Strategy)
            vertex_x = -b / (2 * a)
            is_convex = a > 0
            # Check if vertex is within valid range (with slight margin)
            margin = (log_n.max() - log_n.min()) * 0.2
            is_in_range = (log_n.min() - margin) <= vertex_x <= (log_n.max() + margin)

            if is_convex and is_in_range:
                # Perfect U-shape -> Use Vertex
                opt_n = 10**vertex_x
                opt_loss = np.polyval(coeffs, vertex_x)
                source_type = "vertex"
            else:
                # Monotonic or Vertex too far -> Use Empirical Minimum
                idx_min = np.argmin(loss)
                opt_n = sub.iloc[idx_min]["N"]
                opt_loss = sub.iloc[idx_min]["loss"]
                source_type = "boundary"

            # Find corresponding LR (from nearest actual data point)
            closest_idx = (sub["N"] - opt_n).abs().idxmin()
            opt_lr = sub.loc[closest_idx, "LR"]

            optima.append(
                {
                    "C": c,
                    "N": opt_n,
                    "D": c / (6 * opt_n),
                    "LR": opt_lr,
                    "loss": opt_loss,
                    "type": source_type,
                }
            )

        except Exception as e:
            logger.warning(f"Fit failed for {c}: {e}")

    return pd.DataFrame(optima), fit_curves


def power_law(x, k, a):
    return k * (x**a)


def fit_power_law(x, y):
    """Linear regression in log-log space"""
    slope, intercept = np.polyfit(np.log10(x), np.log10(y), 1)
    return 10**intercept, slope


# ---------------- Main Execution ----------------


def main():
    # 1. Load Data
    df = load_data()

    # 2. Extract Optima & Smooth Curves
    df_opt, fit_curves = parabolic_fit_and_extract(df)

    if df_opt.empty:
        logger.error("No optima found.")
        return

    # 3. Fit Scaling Laws (N vs C, D vs C)
    k_n, a_n = fit_power_law(df_opt["C"], df_opt["N"])
    k_d, a_d = fit_power_law(df_opt["C"], df_opt["D"])

    # 4. Predict for 1e19
    N_pred = power_law(TARGET_BUDGET, k_n, a_n)
    D_pred = power_law(TARGET_BUDGET, k_d, a_d)

    # 5. Fit LR Scaling (Using all data N > 1e5 for robustness)
    df_lr_fit = df[df["N"] > 1e5]
    if len(df_lr_fit) < 5:
        df_lr_fit = df
    k_lr, a_lr = fit_power_law(df_lr_fit["N"], df_lr_fit["LR"])
    LR_pred = power_law(N_pred, k_lr, a_lr)

    logger.info("=" * 40)
    logger.info("FINAL PREDICTION @ 1e19 FLOPs")
    logger.info(f"N_opt: {N_pred:.3e}")
    logger.info(f"D_opt: {D_pred:.3e}")
    logger.info(f"LR_opt: {LR_pred:.3e}")
    logger.info("=" * 40)

    # ---------------- Plotting ----------------
    plt.style.use("seaborn-v0_8-paper")
    unique_cs = sorted(df["C"].unique())
    color_map = {c: plt.cm.viridis(i / len(unique_cs)) for i, c in enumerate(unique_cs)}

    # --- Plot 1: Iso-FLOPs (Smooth Curves + Optima) ---
    fig1, ax1 = plt.subplots(figsize=(10, 7))

    # A. Draw raw data points (faint)
    for c in unique_cs:
        sub = df[df["C"] == c]
        ax1.scatter(sub["N"], sub["loss"], color=color_map[c], alpha=0.3, s=20)

    # B. Draw Smooth Fitted Curves (The "Original Script" look)
    for curve in fit_curves:
        c = curve["C"]
        ax1.plot(curve["N"], curve["loss"], color=color_map[c], linewidth=2, alpha=0.8)

    # C. Mark Optima (Distinguish Vertex vs Boundary)
    for _, row in df_opt.iterrows():
        c = row["C"]
        marker = (
            "*" if row["type"] == "vertex" else "D"
        )  # Star = Perfect, Diamond = Boundary
        # Make markers large and white-bordered for visibility
        ax1.scatter(
            row["N"],
            row["loss"],
            color="white",
            edgecolor=color_map[c],
            s=120,
            marker=marker,
            zorder=10,
            linewidth=1.5,
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("Non-Embedding Parameters (N)", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("1. Iso-FLOPs Profiles (Smooth Fits)", fontsize=14)

    # Custom Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="gray", lw=2, label="Fitted Curves"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="k",
            markersize=12,
            label="Vertex Optima (U-Shape)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="k",
            markersize=8,
            label="Boundary Optima",
        ),
    ]
    ax1.legend(handles=legend_elements, loc="upper left")
    plt.tight_layout()
    fig1.savefig(f"{OUTPUT_DIR}/1_iso_flops_smooth.png", dpi=300)

    # --- Plot 2: Scaling N ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(
        df_opt["C"],
        df_opt["N"],
        s=80,
        c=np.log10(df_opt["C"]),
        cmap="viridis",
        edgecolors="k",
        zorder=5,
    )

    c_line = np.logspace(13, 20, 100)
    ax2.loglog(
        c_line,
        power_law(c_line, k_n, a_n),
        "k--",
        lw=2,
        label=f"Fit: $N \\propto C^{{{a_n:.3f}}}$",
    )
    ax2.scatter(
        [TARGET_BUDGET],
        [N_pred],
        color="magenta",
        marker="*",
        s=300,
        zorder=10,
        label="1e19 Prediction",
    )

    ax2.set_xlabel("Compute (FLOPs)")
    ax2.set_ylabel("Optimal N")
    ax2.set_title("2. Optimal Model Scaling")
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.2)
    plt.tight_layout()
    fig2.savefig(f"{OUTPUT_DIR}/2_scaling_N.png", dpi=300)

    # --- Plot 3: Scaling D ---
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.scatter(
        df_opt["C"],
        df_opt["D"],
        s=80,
        c=np.log10(df_opt["C"]),
        cmap="viridis",
        edgecolors="k",
        zorder=5,
    )
    ax3.loglog(
        c_line,
        power_law(c_line, k_d, a_d),
        "k--",
        lw=2,
        label=f"Fit: $D \\propto C^{{{a_d:.3f}}}$",
    )
    ax3.scatter(
        [TARGET_BUDGET],
        [D_pred],
        color="magenta",
        marker="*",
        s=300,
        zorder=10,
        label="1e19 Prediction",
    )
    ax3.set_xlabel("Compute (FLOPs)")
    ax3.set_ylabel("Optimal D")
    ax3.set_title("3. Optimal Data Scaling")
    ax3.legend()
    ax3.grid(True, which="both", ls="--", alpha=0.2)
    plt.tight_layout()
    fig3.savefig(f"{OUTPUT_DIR}/3_scaling_D.png", dpi=300)

    # --- Plot 4: LR Scaling (Integrated) ---
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    # Plot all Best-LR points as background
    sc = ax4.scatter(
        df["N"],
        df["LR"],
        c=np.log10(df["C"]),
        cmap="plasma",
        s=40,
        edgecolors="k",
        alpha=0.6,
        label="Best LRs",
    )
    plt.colorbar(sc, label="Log10(Compute)")

    n_line = np.logspace(4, 11, 100)
    ax4.loglog(
        n_line,
        power_law(n_line, k_lr, a_lr),
        "k--",
        lw=2,
        label=f"Fit: $LR \\propto N^{{{a_lr:.3f}}}$",
    )
    ax4.scatter(
        [N_pred],
        [LR_pred],
        color="magenta",
        marker="*",
        s=300,
        zorder=10,
        label="1e19 Prediction",
    )

    ax4.set_xlabel("Non-Embedding Parameters (N)")
    ax4.set_ylabel("Optimal Learning Rate")
    ax4.set_title("4. Optimal Learning Rate Scaling")
    ax4.legend()
    ax4.grid(True, which="both", ls="--", alpha=0.2)
    plt.tight_layout()
    fig4.savefig(f"{OUTPUT_DIR}/4_scaling_LR.png", dpi=300)

    logger.info(f"Done! Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
