import json

import click
import numpy as np


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


def get_optimal_stats_per_n(data):
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

        ns.append(n)
        best_lrs.append(best_lr)
        best_losses.append(best_loss)

    return np.array(ns), np.array(best_lrs), np.array(best_losses)
