import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def load_data(filepath):
    """Loads training run data from a JSON file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)


def get_optimal_points(data):
    """
    Finds the model size (N) that achieves the minimum loss for each compute budget (C).
    Returns arrays of compute budgets, optimal parameters, and optimal dataset size.
    """
    budgets = {}
    for run in data:
        C = run["compute_budget"]
        if C not in budgets or run["final_loss"] < budgets[C]["final_loss"]:
            budgets[C] = run

    sorted_budgets = sorted(budgets.keys())
    c_vals = np.array(sorted_budgets)
    n_vals = np.array([budgets[C]["parameters"] for C in sorted_budgets])
    # Compute dataset size D = C / 6N
    d_vals = c_vals / (6 * n_vals)

    return c_vals, n_vals, d_vals


def power_law(C, k, alpha):
    """Power law function: y = k * C^alpha"""
    return k * (C**alpha)


def fit_scaling_law(x_data, y_data):
    """Fits a power law using scipy.optimize.curve_fit."""
    # Using log-space fitting for better stability with large ranges
    # log(y) = log(k) + alpha * log(x)
    log_x = np.log10(x_data)
    log_y = np.log10(y_data)

    # Define linear function for log-space fitting
    def linear_model(lx, log_k, alpha):
        return log_k + alpha * lx

    popt, _ = curve_fit(linear_model, log_x, log_y)
    log_k, alpha = popt
    return 10**log_k, alpha


def plot_scaling_law(c_vals, y_vals, k, alpha, label, ylabel, title, filename, targets):
    """Generates a log-log plot for the scaling law."""
    plt.figure(figsize=(10, 6))
    plt.loglog(c_vals, y_vals, "o", label="Data points (min loss)")

    # Plot fitted line with extrapolation
    plot_extrap = np.logspace(np.log10(c_vals.min()), 24.2, 100)
    plt.loglog(
        plot_extrap,
        power_law(plot_extrap, k, alpha),
        "--",
        label=f"Fit: {label} = {k:.2e} * C^{alpha:.2f}",
    )

    # Mark targets
    colors = ["r", "g"]
    for i, C in enumerate(targets):
        plt.axvline(
            C,
            color=colors[i % len(colors)],
            linestyle=":",
            alpha=0.5,
            label=f"{C:.0e} FLOPs",
        )

    plt.xlabel("Compute Budget (C) [FLOPs]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(filename)
    print(f"Saved {filename}")


def main():
    data_path = "data/isoflops_curves.json"
    try:
        data = load_data(data_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    c_vals, n_opt, d_opt = get_optimal_points(data)

    # Fit scaling laws using scipy.optimize.curve_fit
    k_n, alpha_n = fit_scaling_law(c_vals, n_opt)
    k_d, alpha_d = fit_scaling_law(c_vals, d_opt)

    print("--- Scaling Laws (Fitted using scipy.optimize.curve_fit) ---")
    print(f"Model size:   N_opt = {k_n:.4e} * C^{alpha_n:.4f}")
    print(f"Dataset size: D_opt = {k_d:.4e} * C^{alpha_d:.4f}")

    # Extrapolate to target budgets
    targets = [1e23, 1e24]
    print("\n--- Extrapolated Optimal Sizes ---")
    for C in targets:
        n_pred = power_law(C, k_n, alpha_n)
        d_pred = power_law(C, k_d, alpha_d)
        print(f"Budget C = {C:.0e} FLOPs:")
        print(f"  Optimal parameters (N_opt): {n_pred:.2e}")
        print(f"  Optimal tokens (D_opt):     {d_pred:.2e}")

    # Plot results
    plot_scaling_law(
        c_vals,
        n_opt,
        k_n,
        alpha_n,
        "N_opt",
        "Optimal Model Size (N) [Parameters]",
        "Scaling Law for Model Size (IsoFLOPs Method)",
        "artifacts/chinchilla_isoflops/n_opt_scaling.png",
        targets,
    )
    plot_scaling_law(
        c_vals,
        d_opt,
        k_d,
        alpha_d,
        "D_opt",
        "Optimal Dataset Size (D) [Tokens]",
        "Scaling Law for Dataset Size (IsoFLOPs Method)",
        "artifacts/chinchilla_isoflops/d_opt_scaling.png",
        targets,
    )


if __name__ == "__main__":
    main()
