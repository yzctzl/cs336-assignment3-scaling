import json

import click
import numpy as np

VOCAB_SIZES = {
    "tss": 10000,
    "256": 256,
    "owt": 32000,
    "sp6": 32000,
}


class SliceParam(click.ParamType):
    name = "slice"

    def convert(self, value, param, ctx):
        if isinstance(value, slice):
            return value
        try:
            parts = [int(p) if p.strip() else None for p in value.split(":")]
            return slice(*parts[:3])
        except ValueError:
            self.fail(f"{value!r} not a valid slice (e.g., '1:5' or '::2')", param, ctx)


def load_data(result_path):
    with open(result_path) as raw_json:
        return json.load(raw_json)


def calculate_effective_n(d):
    """
    Calculate effective parameter count including embeddings.
    Assumes tied embeddings (1x vocab_size * d_model) and L=12 for d_model estimation.
    """
    n_non_embed = d["N"]

    # Estimate d_model assuming L=12
    # N = 12 * L * d^2  => d = sqrt(N / 144)
    L = 12
    d_model = np.sqrt(n_non_embed / (12 * L))

    # Get vocab size (default to 256 for legacy compatibility)
    dataset = d.get("dataset", "256")
    vocab_size = VOCAB_SIZES.get(str(dataset), 256)

    # Add embedding parameters (vocab_size * d_model)
    return n_non_embed + (vocab_size * d_model)


def get_optimal_stats_per_n(data, use_embed=False):
    """
    对于每个 N，通过对 LR-Loss 响应做二次拟合，找到最优的 LR 和对应的 Loss 估算值。
    """
    unique_ns = sorted(list(set([d["N"] for d in data])))
    ns = []
    best_lrs = []
    best_losses = []

    for n in unique_ns:
        n_group = [d for d in data if d["N"] == n]
        log_lrs = np.log10([d["LR"] for d in n_group])

        if use_embed:
            # Calculate effective N for this group (using the first item as representative)
            current_n = calculate_effective_n(n_group[0])
        else:
            current_n = n

        losses = np.array([d["loss"] for d in n_group])

        # 对 LR 响应做二次拟合: Loss = a*(log10(LR))^2 + b*log10(LR) + c
        try:
            lr_poly = np.polyfit(log_lrs, losses, 2)
            if lr_poly[0] > 0:  # 开口向上，有最小值
                opt_log_lr = -lr_poly[1] / (2 * lr_poly[0])

                # 安全限制：避免外推过远
                opt_log_lr_clipped = np.clip(
                    opt_log_lr, np.min(log_lrs), np.max(log_lrs)
                )
                best_lr = 10**opt_log_lr_clipped

                # 估算最优 Loss
                best_loss = (
                    lr_poly[0] * opt_log_lr**2 + lr_poly[1] * opt_log_lr + lr_poly[2]
                )
                best_loss = min(best_loss, np.min(losses))
            else:
                best_lr = n_group[np.argmin(losses)]["LR"]
                best_loss = np.min(losses)
        except (RuntimeError, np.RankWarning):  # pyright: ignore[reportAttributeAccessIssue]
            best_lr = n_group[np.argmin(losses)]["LR"]
            best_loss = np.min(losses)

        ns.append(current_n)
        best_lrs.append(best_lr)
        best_losses.append(best_loss)

    return np.array(ns), np.array(best_lrs), np.array(best_losses)


def get_min_stats_per_n(data, use_embed=False):
    """
    Groups data by N and returns the actual minimum loss for each N (no LR fitting).
    """
    unique_ns = sorted(list(set([d["N"] for d in data])))
    ns = []
    min_losses = []

    for n in unique_ns:
        n_group = [d for d in data if d["N"] == n]
        losses = [d["loss"] for d in n_group]

        if use_embed:
            ns.append(calculate_effective_n(n_group[0]))
        else:
            ns.append(n)

        min_losses.append(min(losses))

    return np.array(ns), np.array(min_losses)


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
