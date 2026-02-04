import os

import click
import matplotlib.pyplot as plt
import numpy as np

from cs336_scaling.plot.utils import (
    SliceParam,
    fit_isoflop_curve,
    get_min_stats_per_n,
    get_optimal_stats_per_n,
    load_data,
)


def plot_results(
    data, cleaned_n, cleaned_loss, poly_coeffs, n_opt, loss_opt, output_path, fit_slice
):
    a, b, c = poly_coeffs

    plt.figure(figsize=(10, 6))

    # 1. 绘制原始采样点 (背景)
    plt.scatter(
        [d["N"] for d in data],
        [d["loss"] for d in data],
        color="blue",
        alpha=0.2,
        s=10,
        label="Raw Data (all LRs)",
    )

    # 2. 区分拟合区间内和区间外的 LR 修正点
    # 获取索引
    all_indices = np.arange(len(cleaned_n))
    used_indices = all_indices[fit_slice]
    ignored_indices = np.array([i for i in all_indices if i not in used_indices])

    # 拟合用到的点 (High visibility)
    plt.scatter(
        np.array(cleaned_n)[used_indices],
        np.array(cleaned_loss)[used_indices],
        color="red",
        marker="o",
        s=40,
        label="Included Points (for Fit)",
    )

    # 被忽略的点 (Low visibility)
    if len(ignored_indices) > 0:
        plt.scatter(
            np.array(cleaned_n)[ignored_indices],
            np.array(cleaned_loss)[ignored_indices],
            color="gray",
            marker="x",
            alpha=0.5,
            s=30,
            label="Ignored Points",
        )

    # 3. 绘制拟合的 U 型曲线
    n_smooth = np.logspace(
        np.log10(min(cleaned_n) * 0.5), np.log10(max(cleaned_n) * 1.5), 200
    )
    y_smooth = a * (np.log10(n_smooth) ** 2) + b * np.log10(n_smooth) + c
    plt.plot(
        n_smooth, y_smooth, "purple", lw=2, label=f"Parabolic Fit (N_opt={n_opt:.2e})"
    )

    # 4. 标记顶点
    plt.axvline(n_opt, color="green", linestyle="--", alpha=0.5)
    plt.scatter(
        [n_opt],
        [loss_opt],
        color="gold",
        edgecolor="black",
        s=150,
        zorder=5,
        label="Vertex Optima",
    )

    plt.xscale("log")
    plt.xlabel("Non-Embedding Parameters (N)")
    plt.ylabel("Loss")
    plt.title("Iso-FLOPs Profile")

    # 5. 限制坐标轴区间
    plt.xlim(min(cleaned_n) * 0.8, max(cleaned_n) * 1.2)
    # y 轴通常取 loss 的范围，并留一点 padding
    y_min = min(min(cleaned_loss), loss_opt)
    y_max = max(cleaned_loss)
    plt.ylim(y_min * 0.9, y_max * 1.1)

    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.1)

    output_path = os.path.join(output_path, "iso_flops.pdf")
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", transparent=False, facecolor="white"
    )
    print(f"Plot saved to {output_path}")


def plot_points_only(cleaned_n, cleaned_loss, output_path, fit_slice):
    plt.figure(figsize=(10, 6))

    # Apply slice
    all_indices = np.arange(len(cleaned_n))
    used_indices = all_indices[fit_slice]

    n_to_plot = cleaned_n[used_indices]
    loss_to_plot = cleaned_loss[used_indices]

    # Plot lines and points
    # plt.plot(n_to_plot, loss_to_plot, "b-", alpha=0.3)  # Connect lines lightly
    plt.scatter(
        n_to_plot,
        loss_to_plot,
        color="purple",
        marker="o",
        s=50,
        label="Min Loss (Observed)",
    )

    plt.xscale("log")
    plt.xlabel("Non-Embedding Parameters (N)")
    plt.ylabel("Loss")
    plt.title("Iso-FLOPs Profile (Min Points Only)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.1)

    output_path = os.path.join(output_path, "iso_flops_points.pdf")
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", transparent=False, facecolor="white"
    )
    print(f"Point-only plot saved to {output_path}")


@click.command()
@click.option(
    "--result",
    type=click.Path(exists=False, dir_okay=True, writable=True),
    help="artifact path",
)
@click.option(
    "--fit-range",
    type=SliceParam(),
    default=":",
    help="result range to fit (e.g., '1:-3' or ':')",
)
@click.option(
    "--point",
    is_flag=True,
    help="Only plot minimum points without fitting",
)
@click.option(
    "--fit-lr",
    is_flag=True,
    default=False,
    help="Use quadratic fit to estimate optimal LR/Loss per N (default: False, use raw min)",
)
def main(result, fit_range, point, fit_lr):
    data = load_data(os.path.join(result, "results.json"))

    if point:
        # For point mode, we usually just want raw points,
        # but technically we could plot fitted points too if requested.
        # Following user request "simple point", we stick to raw min unless configured otherwise?
        # User said "Just draw the minimum point".
        # Let's assume point mode uses raw min mostly, but respecting lr_fit flag is cleaner.
        if fit_lr:
            cleaned_n, _, cleaned_loss = get_optimal_stats_per_n(data)
        else:
            cleaned_n, cleaned_loss = get_min_stats_per_n(data)
        plot_points_only(cleaned_n, cleaned_loss, result, fit_range)
        return

    # Data Preparation Strategy
    if fit_lr:
        # Use quadratic fit on LRs to find theoretical minimum
        cleaned_n, _, cleaned_loss = get_optimal_stats_per_n(data)
        print("Using [Fitted] LR minima.")
    else:
        # Use raw observed minimum
        cleaned_n, cleaned_loss = get_min_stats_per_n(data)
        print("Using [Raw] observed minima.")

    # Global Iso-FLOPs Fit
    a, b, c, n_opt, loss_opt = fit_isoflop_curve(cleaned_n, cleaned_loss, fit_range)

    # Print Fitting Result
    print("--- Fitting Result ---")
    print(f"Formula: Loss = {a:.4f} * (log10(N))^2 + ({b:.4f}) * log10(N) + ({c:.4f})")
    print(f"Optimal N (N_opt): {n_opt:.2e}")
    print(f"Projected Min Loss: {loss_opt:.4f}")

    # Plot
    plot_results(
        data, cleaned_n, cleaned_loss, (a, b, c), n_opt, loss_opt, result, fit_range
    )


if __name__ == "__main__":
    main()
