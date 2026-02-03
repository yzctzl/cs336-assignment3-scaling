import os

import click
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from cs336_scaling.plot.utils import (
    SliceParam,
    fit_isoflop_curve,
    get_min_stats_per_n,
    get_optimal_stats_per_n,
    load_data,
)


def plot_multi_isoflops(budgets_data, output_path, fit_slice, fit_lr, do_fit):
    plt.figure(figsize=(10, 7))

    # Custom "Cool" Gradient: Sky Blue -> Blue -> Purple -> Dark Purple
    colors_list = ["skyblue", "blue", "purple", "indigo"]
    cmap = LinearSegmentedColormap.from_list("CustomCool", colors_list, N=100)

    # Generate colors from the custom map
    # Using a range from 0.2 to 1.0 to skip the very lightest if needed, or 0 to 1
    colors = cmap(np.linspace(0, 1, len(budgets_data)))

    for idx, (budget, data_entry) in enumerate(budgets_data.items()):
        cleaned_n = data_entry["n"]
        cleaned_loss = data_entry["loss"]

        color = colors[idx]
        label_prefix = f"C={budget:.0e}"

        # 1. Plot raw min points (Always shown)
        # Simplify label to just "C=..." inside the chart to save space
        label_text = label_prefix if not do_fit else f"{label_prefix} Observed"

        plt.scatter(
            cleaned_n,
            cleaned_loss,
            color=color,
            marker="o",
            s=50,
            alpha=0.8,
            label=label_text,
        )

        if do_fit and "coeffs" in data_entry:
            # Use provided coeffs
            a, b, c = data_entry["coeffs"]
            n_opt = data_entry["n_opt"]
            loss_opt = data_entry["loss_opt"]

            # 2. Plot fitted curve (IsoFLOPs parabola)
            n_smooth = np.logspace(
                np.log10(min(cleaned_n) * 0.5), np.log10(max(cleaned_n) * 1.5), 200
            )
            y_smooth = a * (np.log10(n_smooth) ** 2) + b * np.log10(n_smooth) + c

            plt.plot(
                n_smooth,
                y_smooth,
                color=color,
                linestyle="--",
                lw=1.5,
                label=f"{label_prefix} Fit",
            )

            # 3. Mark vertex
            plt.scatter(
                [n_opt],
                [loss_opt],
                color=color,
                edgecolor="white",
                s=120,
                marker="*",
                zorder=5,
                linewidth=1.0,
            )

    plt.xscale("log")
    plt.xlabel("Non-Embedding Parameters (N)", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Iso-FLOPs Profiles", fontsize=14, pad=15)

    # Legend inside the plot area
    plt.legend(loc="best", frameon=True, fontsize="small")
    plt.grid(True, which="both", ls="-", alpha=0.1)

    plt.tight_layout()

    filename = "multi_iso_c_flops.pdf"
    if fit_lr:
        filename = "multi_iso_c_flops_fitted_lr.pdf"

    final_path = os.path.join(output_path, filename)
    plt.savefig(
        final_path, dpi=300, bbox_inches="tight", transparent=False, facecolor="white"
    )
    print(f"Plot saved to {final_path}")


@click.command()
@click.argument("result_dirs", nargs=-1, type=click.Path(exists=True, file_okay=False))
@click.option(
    "--output",
    type=click.Path(exists=False, dir_okay=True, writable=True),
    default=".",
    help="Output directory for the plot",
)
@click.option(
    "--fit-range",
    type=SliceParam(),
    default=":",
    help="Result range to fit (e.g., '1:-3' or ':')",
)
@click.option(
    "--fit-lr",
    is_flag=True,
    default=False,
    help="Use quadratic fit to estimate optimal LR/Loss per N (default: False, use raw min)",
)
@click.option(
    "--fit",
    is_flag=True,
    default=False,
    help="Enable IsoFLOPs curve fitting and plotting",
)
def main(result_dirs, output, fit_range, fit_lr, fit):
    """
    Plot IsoFLOPs curves for multiple compute budgets.
    Pass multiple directory paths containing results.json.
    """
    if not result_dirs:
        print("Please provide at least one result directory.")
        return

    budgets_data = {}

    for res_dir in result_dirs:
        json_path = os.path.join(res_dir, "results.json")
        if not os.path.exists(json_path):
            print(f"Skipping {res_dir}: results.json not found.")
            continue

        try:
            budget_str = os.path.basename(res_dir.rstrip("/"))
            budget_val = float(budget_str)
        except ValueError:
            budget_val = 0.0  # Fallback

        print(f"Processing budget: {budget_val} from {res_dir}...")

        data = load_data(json_path)

        if fit_lr:
            cleaned_n, _, cleaned_loss = get_optimal_stats_per_n(data)
        else:
            cleaned_n, cleaned_loss = get_min_stats_per_n(data)

        budgets_data[budget_val] = {
            "n": cleaned_n,
            "loss": cleaned_loss,
        }

        if fit:
            # Fit IsoFLOPs curve only if requested
            a, b, c, n_opt, loss_opt = fit_isoflop_curve(
                cleaned_n, cleaned_loss, fit_range
            )
            budgets_data[budget_val].update(
                {"coeffs": (a, b, c), "n_opt": n_opt, "loss_opt": loss_opt}
            )
            print(f"  [Fit Result C={budget_val:.0e}]")
            print(f"  Formula: Loss = {a:.4f}*(logN)^2 + {b:.4f}*logN + {c:.4f}")
            print(f"  Optimal N: {n_opt:.2e}, Min Loss: {loss_opt:.4f}")

    if not budgets_data:
        print("No valid data found.")
        return

    try:
        sorted_keys = sorted(
            budgets_data.keys(),
            key=lambda x: float(x) if isinstance(x, (int, float, str)) else x,
        )
        sorted_budgets_data = {k: budgets_data[k] for k in sorted_keys}
    except ValueError:
        sorted_budgets_data = dict(sorted(budgets_data.items()))

    plot_multi_isoflops(sorted_budgets_data, output, fit_range, fit_lr, fit)


if __name__ == "__main__":
    main()
