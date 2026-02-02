import os

import click
import matplotlib.pyplot as plt
import numpy as np

from cs336_scaling.plot.utils import SliceParam, get_optimal_stats_per_n, load_data


def fit_lr_scaling(ns, best_lrs, fit_slice):
    """
    Fit a power-law relationship: LR = k * N^alpha
    Equivalent to: log10(LR) = alpha * log10(N) + log10(k)
    """
    log_n = np.log10(np.array(ns)[fit_slice])
    log_lr = np.log10(np.array(best_lrs)[fit_slice])

    # Linear fit in log-log space
    alpha, beta = np.polyfit(log_n, log_lr, 1)
    k = 10**beta

    return alpha, k


def plot_results(data, ns, best_lrs, scaling_coeffs, output_path, fit_slice):
    alpha, k = scaling_coeffs

    plt.figure(figsize=(10, 6))

    # 绘制原始采样点 (背景)
    plt.scatter(
        [d["N"] for d in data],
        [d["LR"] for d in data],
        color="blue",
        alpha=0.2,
        s=10,
        label="Raw Data (All Samples)",
    )

    # 区分拟合区间内和区间外的 LR 修正点
    all_indices = np.arange(len(ns))
    used_indices = all_indices[fit_slice]
    ignored_indices = np.array([i for i in all_indices if i not in used_indices])

    # Included points
    plt.scatter(
        np.array(ns)[used_indices],
        np.array(best_lrs)[used_indices],
        color="red",
        marker="o",
        s=50,
        label="Included Best LRs",
    )

    # Ignored points
    if len(ignored_indices) > 0:
        plt.scatter(
            np.array(ns)[ignored_indices],
            np.array(best_lrs)[ignored_indices],
            color="gray",
            marker="x",
            alpha=0.5,
            s=40,
            label="Ignored Best LRs",
        )

    # 绘制拟合的 Scaling Law 曲线
    n_smooth = np.logspace(np.log10(min(ns) * 0.8), np.log10(max(ns) * 1.25), 200)
    lr_smooth = k * (n_smooth**alpha)
    plt.plot(
        n_smooth,
        lr_smooth,
        "purple",
        lw=2,
        label=f"Scaling Law: LR = {k:.2e} * N^({alpha:.4f})",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Non-Embedding Parameters (N)")
    plt.ylabel("Learning Rate (LR)")
    plt.title("LR vs N Scaling Law @ 1e14")

    # 限制坐标轴区间
    plt.xlim(min(ns) * 0.8, max(ns) * 1.25)
    plt.ylim(min(best_lrs) * 0.5, max(best_lrs) * 2.0)

    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.1)

    output_path = os.path.join(output_path, "lr_n_scaling.pdf")
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", transparent=False, facecolor="white"
    )
    print(f"Plot saved to {output_path}")


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
def main(result, fit_range):
    data = load_data(os.path.join(result, "results.json"))

    # 1. Get best LR for each N
    ns, best_lrs, _ = get_optimal_stats_per_n(data)

    # 2. Fit LR scaling law
    alpha, k = fit_lr_scaling(ns, best_lrs, fit_range)

    # 3. Print results
    print("--- LR Scaling 拟合结果 ---")
    print(f"拟合公式: LR = {k:.4e} * N^({alpha:.4f})")
    print(f"Log-Log 线性公式: log10(LR) = {alpha:.4f} * log10(N) + {np.log10(k):.4f}")

    # 4. Plot
    plot_results(data, ns, best_lrs, (alpha, k), result, fit_range)


if __name__ == "__main__":
    main()
