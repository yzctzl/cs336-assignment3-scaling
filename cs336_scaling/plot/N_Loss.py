import os

import click
import matplotlib.pyplot as plt
import numpy as np

from cs336_scaling.plot.utils import SliceParam, get_optimal_stats_per_n, load_data


def fit_isoflop_curve(cleaned_n, cleaned_loss, fit_slice):
    """
    Log-Log 空间二次拟合 (寻找 U 型底)
    我们拟合: Loss = a*(log10(N))^2 + b*log10(N) + c
    """
    # 选取中间部分进行拟合，对首尾不稳定的点做切片 (保持原逻辑)
    log_n = np.log10(cleaned_n[fit_slice])
    y = np.array(cleaned_loss[fit_slice])

    # 使用 numpy 的多项式拟合
    iso_poly = np.polyfit(log_n, y, 2)
    a, b, c = iso_poly

    # 计算 U 型底顶点 N_opt
    # 对数空间极值点: log_n_opt = -b / (2a)
    log_n_opt = -b / (2 * a)
    n_opt = 10**log_n_opt

    # 获取最优 Loss
    loss_opt = a * log_n_opt**2 + b * log_n_opt + c

    return a, b, c, n_opt, loss_opt


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
    plt.title("Iso-FLOPs Profile @ 1e14")

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

    # 局部 LR 修正
    cleaned_n, _, cleaned_loss = get_optimal_stats_per_n(data)

    # 全局 Iso-FLOPs 拟合
    a, b, c, n_opt, loss_opt = fit_isoflop_curve(cleaned_n, cleaned_loss, fit_range)

    # 打印拟合公式及结果
    print("--- 拟合结果 ---")
    print(
        f"完整拟合公式: Loss = {a:.4f} * (log10(N))^2 + ({b:.4f}) * log10(N) + ({c:.4f})"
    )
    print(f"最优参数量 N_opt: {n_opt:.2e}")
    print(f"对应最低 Loss 预估: {loss_opt:.4f}")

    # 绘图
    plot_results(
        data, cleaned_n, cleaned_loss, (a, b, c), n_opt, loss_opt, result, fit_range
    )


if __name__ == "__main__":
    main()
