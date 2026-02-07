import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VOCAB_SIZE = 32000
NUM_HEADS = 8
BASELINE_LR_BY_LAYER = {
    6: 4e-4,
    8: 4e-4,
    10: 4e-4,
    12: 3.5e-4,
    14: 3e-4,
    16: 3e-4,
}


@dataclass
class FitResult:
    k: float
    a: float

    def predict(self, c: float) -> float:
        return self.k * (c**self.a)


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def budget_label(budget: float) -> str:
    return f"{budget:.0e}".replace("+", "")


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


def load_runs(results_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for file_name in sorted(os.listdir(results_dir)):
        if not file_name.startswith("results_") or not file_name.endswith(".json"):
            continue
        result_path = os.path.join(results_dir, file_name)
        base = file_name[len("results_") : -len(".json")]
        csv_path = os.path.join(results_dir, f"{base}.csv")
        csv_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

        with open(result_path, "r") as f:
            runs = json.load(f)

        for r in runs:
            if r.get("dataset") != "sp6":
                continue
            loss = float(r["loss"])
            c = float(r["C"])
            n_non_emb = float(r["N"])
            lr = float(r["LR"])

            layer = None
            d_model = None
            if not csv_df.empty:
                m = csv_df[
                    (csv_df["N_non_emb"].astype(float) == n_non_emb)
                    & (np.isclose(csv_df["LR"].astype(float), lr, atol=1e-12))
                ]
                if len(m) == 0:
                    idx = (csv_df["N_non_emb"].astype(float) - n_non_emb).abs().idxmin()
                    row = csv_df.loc[idx]
                else:
                    row = m.iloc[0]
                layer = int(row["layers"])
                d_model = int(row["d_model"])

            rows.append(
                {
                    "source_file": file_name,
                    "C": c,
                    "N_non_emb": n_non_emb,
                    "LR": lr,
                    "loss": loss,
                    "layers": layer,
                    "d_model": d_model,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["source_file", "C", "N_non_emb", "LR", "loss", "layers", "d_model"]
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["C", "N_non_emb"]).reset_index(drop=True)


def apply_filters(df: pd.DataFrame, loss_hard_cap: float, use_iqr: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    removed: List[pd.DataFrame] = []
    hard_removed = df[df["loss"] > loss_hard_cap]
    if len(hard_removed) > 0:
        removed.append(hard_removed.assign(filter_reason="hard_cap"))
    filtered = df[df["loss"] <= loss_hard_cap].copy()

    if use_iqr:
        kept_groups = []
        iqr_removed_groups = []
        for c, g in filtered.groupby("C"):
            if len(g) < 4:
                kept_groups.append(g)
                continue
            q1 = g["loss"].quantile(0.25)
            q3 = g["loss"].quantile(0.75)
            iqr = q3 - q1
            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr
            keep = g[(g["loss"] >= low) & (g["loss"] <= high)]
            drop = g[(g["loss"] < low) | (g["loss"] > high)]
            kept_groups.append(keep)
            if len(drop) > 0:
                iqr_removed_groups.append(drop.assign(filter_reason="iqr"))
        filtered = pd.concat(kept_groups, ignore_index=True) if kept_groups else filtered
        if iqr_removed_groups:
            removed.append(pd.concat(iqr_removed_groups, ignore_index=True))

    removed_df = pd.concat(removed, ignore_index=True) if removed else pd.DataFrame()
    return filtered.sort_values(["C", "N_non_emb"]).reset_index(drop=True), removed_df


def method2_empirical(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, g in df.groupby("C"):
        best = g.loc[g["loss"].idxmin()]
        rows.append(
            {
                "method": "empirical",
                "C": float(c),
                "N_opt": float(best["N_non_emb"]),
                "loss_opt": float(best["loss"]),
                "LR_opt": float(best["LR"]),
                "layers_opt": int(best["layers"]) if pd.notna(best["layers"]) else np.nan,
                "d_model_opt": int(best["d_model"]) if pd.notna(best["d_model"]) else np.nan,
                "source": "min_loss",
            }
        )
    out = pd.DataFrame(rows).sort_values("C").reset_index(drop=True)
    out["D_opt"] = out["C"] / (6 * out["N_opt"])
    return out


def method2_quadratic(df: pd.DataFrame, empirical_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    empirical_by_c = {float(r["C"]): r for _, r in empirical_df.iterrows()}
    for c, g in df.groupby("C"):
        g = g.sort_values("N_non_emb")
        if len(g) < 3:
            e = empirical_by_c[float(c)]
            rows.append(
                {
                    "method": "quadratic",
                    "C": float(c),
                    "N_opt": float(e["N_opt"]),
                    "loss_opt": float(e["loss_opt"]),
                    "LR_opt": float(e["LR_opt"]),
                    "layers_opt": e["layers_opt"],
                    "d_model_opt": e["d_model_opt"],
                    "source": "fallback_empirical_too_few_points",
                }
            )
            continue

        x = np.log10(g["N_non_emb"].astype(float).values)
        y = g["loss"].astype(float).values
        a, b, c0 = np.polyfit(x, y, 2)
        if a <= 0:
            e = empirical_by_c[float(c)]
            rows.append(
                {
                    "method": "quadratic",
                    "C": float(c),
                    "N_opt": float(e["N_opt"]),
                    "loss_opt": float(e["loss_opt"]),
                    "LR_opt": float(e["LR_opt"]),
                    "layers_opt": e["layers_opt"],
                    "d_model_opt": e["d_model_opt"],
                    "source": "fallback_empirical_non_convex",
                }
            )
            continue

        xv = -b / (2 * a)
        if xv < x.min() or xv > x.max():
            e = empirical_by_c[float(c)]
            rows.append(
                {
                    "method": "quadratic",
                    "C": float(c),
                    "N_opt": float(e["N_opt"]),
                    "loss_opt": float(e["loss_opt"]),
                    "LR_opt": float(e["LR_opt"]),
                    "layers_opt": e["layers_opt"],
                    "d_model_opt": e["d_model_opt"],
                    "source": "fallback_empirical_vertex_out_of_range",
                }
            )
            continue

        n_opt = float(10**xv)
        loss_opt = float(a * xv * xv + b * xv + c0)
        nearest_idx = (g["N_non_emb"] - n_opt).abs().idxmin()
        nearest = g.loc[nearest_idx]
        rows.append(
            {
                "method": "quadratic",
                "C": float(c),
                "N_opt": n_opt,
                "loss_opt": loss_opt,
                "LR_opt": float(nearest["LR"]),
                "layers_opt": int(nearest["layers"]) if pd.notna(nearest["layers"]) else np.nan,
                "d_model_opt": int(nearest["d_model"]) if pd.notna(nearest["d_model"]) else np.nan,
                "source": "quadratic_vertex",
            }
        )

    out = pd.DataFrame(rows).sort_values("C").reset_index(drop=True)
    out["D_opt"] = out["C"] / (6 * out["N_opt"])
    return out


def fit_power_law(c: np.ndarray, y: np.ndarray) -> FitResult:
    if len(c) == 0:
        raise ValueError("Cannot fit power law with zero points")
    if len(c) == 1:
        # Single-point fallback: use constant predictor.
        return FitResult(k=float(y[0]), a=0.0)
    slope, intercept = np.polyfit(np.log10(c), np.log10(y), 1)
    return FitResult(k=float(10**intercept), a=float(slope))


def build_candidate_architectures(pred_n: float) -> List[Dict[str, float]]:
    rows = []
    for l in [10, 12, 14, 16]:
        d = solve_d_model_from_n_non_emb(pred_n, l)
        n_non_emb_actual = int(12 * l * (d**2))
        rows.append(
            {
                "layers": l,
                "d_model": d,
                "N_non_emb_actual": n_non_emb_actual,
                "abs_error_to_pred": float(abs(n_non_emb_actual - pred_n)),
            }
        )
    return rows


def evaluate_fit_holdout(summary_df: pd.DataFrame) -> Dict[str, float]:
    budgets = sorted(float(x) for x in summary_df["C"].unique())
    folds = []
    for c_hold in budgets:
        train = summary_df[summary_df["C"] != c_hold]
        if len(train) < 2:
            continue
        fit = fit_power_law(train["C"].values, train["N_opt"].values)
        pred = float(fit.predict(c_hold))
        actual = float(summary_df[summary_df["C"] == c_hold].iloc[0]["N_opt"])
        rel_error = abs(pred - actual) / max(actual, 1.0)
        folds.append(
            {
                "holdout_budget": c_hold,
                "predicted_N_opt": pred,
                "actual_N_opt": actual,
                "abs_rel_error": float(rel_error),
            }
        )
    if not folds:
        return {
            "num_folds": 0,
            "mean_abs_rel_error": None,
            "median_abs_rel_error": None,
            "max_abs_rel_error": None,
            "folds": [],
        }

    errs = np.array([float(x["abs_rel_error"]) for x in folds], dtype=np.float64)
    return {
        "num_folds": int(len(folds)),
        "mean_abs_rel_error": float(errs.mean()),
        "median_abs_rel_error": float(np.median(errs)),
        "max_abs_rel_error": float(errs.max()),
        "folds": folds,
    }


def choose_better_method(v_emp: Dict[str, float], v_quad: Dict[str, float]) -> str:
    emp_med = v_emp.get("median_abs_rel_error")
    quad_med = v_quad.get("median_abs_rel_error")
    if emp_med is None and quad_med is None:
        return "empirical"
    if emp_med is None:
        return "quadratic"
    if quad_med is None:
        return "empirical"
    if emp_med <= quad_med:
        return "empirical"
    return "quadratic"


def build_large_scale_validation_plan(
    method: str,
    fit: FitResult,
    budgets: List[float],
    n_mults: List[float],
    lr_mults: List[float],
    low_budget_layers: List[int],
    high_budget_layers: List[int],
    max_no_layer_expand_budget: float,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for budget in budgets:
        layer_candidates = (
            low_budget_layers if budget <= max_no_layer_expand_budget else high_budget_layers
        )
        pred_center = float(fit.predict(budget))
        for n_mult in n_mults:
            target_n = pred_center * n_mult
            for layer in layer_candidates:
                d_model = solve_d_model_from_n_non_emb(target_n, layer)
                base_lr = BASELINE_LR_BY_LAYER.get(layer, 3e-4)
                for lr_mult in lr_mults:
                    lr = max(1e-4, min(1e-3, base_lr * lr_mult))
                    row = compute_row(budget, layer, d_model, lr)
                    row["method"] = method
                    row["pred_N_center"] = pred_center
                    row["n_mult"] = float(n_mult)
                    row["source"] = "large_scale_validation"
                    rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["method", "Budget", "layers", "d_model", "LR"])
    return df.sort_values(["method", "Budget", "N_non_emb", "LR"]).reset_index(drop=True)


def plot_isoflops_empirical(df: pd.DataFrame, empirical: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(10, 6))
    budgets = sorted(df["C"].unique())
    cmap = plt.cm.viridis(np.linspace(0, 1, len(budgets)))
    color = {c: cmap[i] for i, c in enumerate(budgets)}
    for c in budgets:
        g = df[df["C"] == c]
        plt.scatter(g["N_non_emb"], g["loss"], s=30, color=color[c], alpha=0.6)
        b = empirical[empirical["C"] == c].iloc[0]
        plt.scatter([b["N_opt"]], [b["loss_opt"]], marker="*", s=220, color=color[c], edgecolors="black")
    plt.xscale("log")
    plt.xlabel("N_non_emb")
    plt.ylabel("Loss")
    plt.title("Method2 Empirical IsoFLOPs Minima")
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_isoflops_quadratic(df: pd.DataFrame, quadratic: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(10, 6))
    budgets = sorted(df["C"].unique())
    cmap = plt.cm.plasma(np.linspace(0, 1, len(budgets)))
    color = {c: cmap[i] for i, c in enumerate(budgets)}
    for c in budgets:
        g = df[df["C"] == c].sort_values("N_non_emb")
        plt.scatter(g["N_non_emb"], g["loss"], s=28, color=color[c], alpha=0.55)
        if len(g) >= 3:
            x = np.log10(g["N_non_emb"].values)
            y = g["loss"].values
            a, b, c0 = np.polyfit(x, y, 2)
            x_line = np.linspace(x.min(), x.max(), 120)
            y_line = a * x_line * x_line + b * x_line + c0
            plt.plot(10**x_line, y_line, color=color[c], linewidth=1.5, alpha=0.9)
        q = quadratic[quadratic["C"] == c].iloc[0]
        plt.scatter([q["N_opt"]], [q["loss_opt"]], marker="D", s=90, color=color[c], edgecolors="black")
    plt.xscale("log")
    plt.xlabel("N_non_emb")
    plt.ylabel("Loss")
    plt.title("Method2 Quadratic IsoFLOPs Minima")
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_scaling(
    summary_emp: pd.DataFrame,
    summary_quad: pd.DataFrame,
    fit_emp: FitResult,
    fit_quad: FitResult,
    target_budget: float,
    out_n_path: str,
    out_d_path: str,
) -> None:
    c_grid = np.logspace(
        np.log10(min(summary_emp["C"].min(), summary_quad["C"].min())),
        np.log10(target_budget),
        120,
    )

    plt.figure(figsize=(9, 6))
    plt.loglog(summary_emp["C"], summary_emp["N_opt"], "o", label="Empirical minima")
    plt.loglog(summary_quad["C"], summary_quad["N_opt"], "s", label="Quadratic minima")
    plt.loglog(c_grid, fit_emp.predict(c_grid), "--", label=f"Empirical fit a={fit_emp.a:.3f}")
    plt.loglog(c_grid, fit_quad.predict(c_grid), "--", label=f"Quadratic fit a={fit_quad.a:.3f}")
    plt.scatter([target_budget], [fit_emp.predict(target_budget)], marker="*", s=160, label="Empirical @1e19")
    plt.scatter([target_budget], [fit_quad.predict(target_budget)], marker="*", s=160, label="Quadratic @1e19")
    plt.xlabel("Compute C")
    plt.ylabel("N_opt")
    plt.title("Method2 Scaling: N_opt(C)")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_n_path, dpi=220)
    plt.close()

    plt.figure(figsize=(9, 6))
    d_emp = summary_emp["C"] / (6 * summary_emp["N_opt"])
    d_quad = summary_quad["C"] / (6 * summary_quad["N_opt"])
    plt.loglog(summary_emp["C"], d_emp, "o", label="Empirical minima")
    plt.loglog(summary_quad["C"], d_quad, "s", label="Quadratic minima")
    plt.loglog(c_grid, c_grid / (6 * fit_emp.predict(c_grid)), "--", label="Empirical fit")
    plt.loglog(c_grid, c_grid / (6 * fit_quad.predict(c_grid)), "--", label="Quadratic fit")
    plt.scatter([target_budget], [target_budget / (6 * fit_emp.predict(target_budget))], marker="*", s=160, label="Empirical @1e19")
    plt.scatter([target_budget], [target_budget / (6 * fit_quad.predict(target_budget))], marker="*", s=160, label="Quadratic @1e19")
    plt.xlabel("Compute C")
    plt.ylabel("D_opt")
    plt.title("Method2 Scaling: D_opt(C)")
    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_d_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="artifacts/chinchilla_sweep/sp6_method2_nightly",
        help="Directory containing results_*.json and corresponding csv files",
    )
    parser.add_argument(
        "--target-budget",
        type=float,
        default=1e19,
        help="Target compute budget for extrapolation",
    )
    parser.add_argument(
        "--loss-hard-cap",
        type=float,
        default=20.0,
        help="Discard points with loss above this threshold",
    )
    parser.add_argument(
        "--no-iqr-filter",
        action="store_true",
        help="Disable per-budget IQR outlier filtering",
    )
    parser.add_argument(
        "--validation-budgets",
        default="1e17,3e17,1e18",
        help="Comma-separated larger budgets for validation-plan csv",
    )
    parser.add_argument(
        "--max-no-layer-expand-budget",
        type=float,
        default=1e16,
        help="For budgets <= this value, use low-budget layer set only",
    )
    parser.add_argument(
        "--low-budget-layers",
        default="6,8,10,12",
        help="Layer candidates for lower budgets",
    )
    parser.add_argument(
        "--high-budget-layers",
        default="12,14,16",
        help="Layer candidates for larger budgets",
    )
    parser.add_argument(
        "--validation-n-mult",
        default="0.85,1.0,1.15",
        help="Multipliers around predicted N_opt for large-scale validation plan",
    )
    parser.add_argument(
        "--validation-lr-mult",
        default="0.9,1.0",
        help="Multipliers around baseline LR for large-scale validation plan",
    )
    args = parser.parse_args()

    validation_budgets = parse_float_list(args.validation_budgets)
    low_budget_layers = parse_int_list(args.low_budget_layers)
    high_budget_layers = parse_int_list(args.high_budget_layers)
    validation_n_mult = parse_float_list(args.validation_n_mult)
    validation_lr_mult = parse_float_list(args.validation_lr_mult)

    os.makedirs(args.results_dir, exist_ok=True)

    raw = load_runs(args.results_dir)
    if raw.empty:
        print("No sp6 result files found; nothing to analyze.")
        return

    filtered, removed = apply_filters(
        raw, loss_hard_cap=args.loss_hard_cap, use_iqr=not args.no_iqr_filter
    )
    if filtered.empty:
        print("All points were filtered out; adjust thresholds.")
        return

    empirical = method2_empirical(filtered)
    quadratic = method2_quadratic(filtered, empirical)

    fit_emp = fit_power_law(empirical["C"].values, empirical["N_opt"].values)
    fit_quad = fit_power_law(quadratic["C"].values, quadratic["N_opt"].values)

    pred_emp_n = float(fit_emp.predict(args.target_budget))
    pred_quad_n = float(fit_quad.predict(args.target_budget))

    summary = pd.concat([empirical, quadratic], ignore_index=True).sort_values(
        ["method", "C"]
    )
    summary.to_csv(
        os.path.join(args.results_dir, "analysis", "summary.csv"),
        index=False,
    )

    holdout_emp = evaluate_fit_holdout(empirical)
    holdout_quad = evaluate_fit_holdout(quadratic)
    selected_method = choose_better_method(holdout_emp, holdout_quad)

    prediction = {
        "target_budget": args.target_budget,
        "filters": {
            "loss_hard_cap": args.loss_hard_cap,
            "iqr_enabled": not args.no_iqr_filter,
            "num_raw_points": int(len(raw)),
            "num_filtered_points": int(len(filtered)),
            "num_removed_points": int(len(removed)),
        },
        "empirical": {
            "fit": {"k": fit_emp.k, "a": fit_emp.a},
            "prediction": {
                "N_opt": pred_emp_n,
                "D_opt": args.target_budget / (6 * pred_emp_n),
            },
            "candidate_architectures": build_candidate_architectures(pred_emp_n),
        },
        "quadratic": {
            "fit": {"k": fit_quad.k, "a": fit_quad.a},
            "prediction": {
                "N_opt": pred_quad_n,
                "D_opt": args.target_budget / (6 * pred_quad_n),
            },
            "candidate_architectures": build_candidate_architectures(pred_quad_n),
        },
    }
    with open(os.path.join(args.results_dir, "analysis", "prediction.json"), "w") as f:
        json.dump(prediction, f, indent=2)

    validation = {
        "selected_method": selected_method,
        "max_no_layer_expand_budget": args.max_no_layer_expand_budget,
        "validation_budgets": validation_budgets,
        "low_budget_layers": low_budget_layers,
        "high_budget_layers": high_budget_layers,
        "validation_n_mult": validation_n_mult,
        "validation_lr_mult": validation_lr_mult,
        "empirical": holdout_emp,
        "quadratic": holdout_quad,
    }
    with open(os.path.join(args.results_dir, "analysis", "validation.json"), "w") as f:
        json.dump(validation, f, indent=2)

    large_emp = build_large_scale_validation_plan(
        method="empirical",
        fit=fit_emp,
        budgets=validation_budgets,
        n_mults=validation_n_mult,
        lr_mults=validation_lr_mult,
        low_budget_layers=low_budget_layers,
        high_budget_layers=high_budget_layers,
        max_no_layer_expand_budget=args.max_no_layer_expand_budget,
    )
    large_quad = build_large_scale_validation_plan(
        method="quadratic",
        fit=fit_quad,
        budgets=validation_budgets,
        n_mults=validation_n_mult,
        lr_mults=validation_lr_mult,
        low_budget_layers=low_budget_layers,
        high_budget_layers=high_budget_layers,
        max_no_layer_expand_budget=args.max_no_layer_expand_budget,
    )
    large_all = pd.concat([large_emp, large_quad], ignore_index=True).sort_values(
        ["method", "Budget", "N_non_emb", "LR"]
    )
    large_all.to_csv(
        os.path.join(args.results_dir, "analysis", "large_scale_validation_plan.csv"),
        index=False,
    )
    large_selected = large_all[large_all["method"] == selected_method].copy()
    large_selected.to_csv(
        os.path.join(args.results_dir, "analysis", "large_scale_validation_selected.csv"),
        index=False,
    )

    plot_isoflops_empirical(
        filtered,
        empirical,
        os.path.join(args.results_dir, "analysis", "isoflops_empirical.png"),
    )
    plot_isoflops_quadratic(
        filtered,
        quadratic,
        os.path.join(args.results_dir, "analysis", "isoflops_quadratic.png"),
    )
    plot_scaling(
        summary_emp=empirical,
        summary_quad=quadratic,
        fit_emp=fit_emp,
        fit_quad=fit_quad,
        target_budget=args.target_budget,
        out_n_path=os.path.join(args.results_dir, "analysis", "scaling_N.png"),
        out_d_path=os.path.join(args.results_dir, "analysis", "scaling_D.png"),
    )

    print("Wrote:")
    print(f"  {os.path.join(args.results_dir, 'analysis', 'summary.csv')}")
    print(f"  {os.path.join(args.results_dir, 'analysis', 'prediction.json')}")
    print(f"  {os.path.join(args.results_dir, 'analysis', 'validation.json')}")
    print(f"  {os.path.join(args.results_dir, 'analysis', 'large_scale_validation_plan.csv')}")
    print(
        f"  {os.path.join(args.results_dir, 'analysis', 'large_scale_validation_selected.csv')}"
    )
    print(f"  {os.path.join(args.results_dir, 'analysis', 'isoflops_empirical.png')}")
    print(f"  {os.path.join(args.results_dir, 'analysis', 'isoflops_quadratic.png')}")
    print(f"  {os.path.join(args.results_dir, 'analysis', 'scaling_N.png')}")
    print(f"  {os.path.join(args.results_dir, 'analysis', 'scaling_D.png')}")
    print(
        "Validation policy:"
        f" no layer expansion at budgets <= {budget_label(args.max_no_layer_expand_budget)}; "
        f"larger budgets use layers {','.join(str(x) for x in high_budget_layers)}."
    )


if __name__ == "__main__":
    main()
