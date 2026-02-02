import json
import os

import click
import numpy as np
import pandas as pd


def generate_supplemental_plan(
    results_path, output_path, C_budget=1e14, dataset="256", fixed_l=6
):
    # 加载现有结果
    with open(results_path, "r") as f:
        data = json.load(f)

    df_results = pd.DataFrame(data)
    unique_ns = sorted(df_results["N"].unique())

    supplemental_rows = []

    print(f"### 正在分析 {len(unique_ns)} 个 N 档位的 LR 覆盖情况... ###")

    for n in unique_ns:
        # 获取该 N 档位的所有实验点
        group: pd.DataFrame = df_results.loc[df_results["N"] == n]
        group = group.sort_values(by=["LR"])
        lrs = group["LR"].to_numpy()
        losses = group["loss"].to_numpy()

        if len(losses) < 2:
            continue

        min_idx = np.argmin(losses)

        # 策略 1：最优值在左边界 -> 探测更小的 LR
        if min_idx == 0:
            print(f"  [!] N={n:7d}: 最优值在左边界 ({lrs[0]:.2e})，补扫更小 LR")
            new_lrs = [lrs[0] * 0.25, lrs[0] * 0.5]

        # 策略 2：最优值在右边界 -> 探测更大的 LR
        elif min_idx == len(losses) - 1:
            print(f"  [!] N={n:7d}: 最优值在右边界 ({lrs[-1]:.2e})，补扫更大 LR")
            new_lrs = [lrs[-1] * 2.0, lrs[-1] * 4.0]

        # 策略 3：虽然是 V 型，但如果 V 型开口太宽或 Loss 差值极小（平原区），进行插值
        else:
            loss_range = np.max(losses) - np.min(losses)
            if loss_range < 0.05:  # 阈值可调，代表 Loss 几乎没动
                print(f"  [?] N={n:7d}: 处于 Loss 平原，进行对数插值")
                lr_best = lrs[min_idx]
                new_lrs = [
                    np.sqrt(lr_best * lrs[min_idx - 1]),
                    np.sqrt(lr_best * lrs[min_idx + 1]),
                ]
            else:
                continue  # 该点已找到完美的 V 型底，跳过

        # 生成兼容格式的行
        # 反推 d_model 以保持 architecture 一致
        d_model = int(np.sqrt(n / (12 * fixed_l)))
        tokens = C_budget / (6 * n)

        for lr in new_lrs:
            supplemental_rows.append(
                {
                    "Budget": f"{C_budget:.0e}",
                    "N": int(n),
                    "d_model": d_model,
                    "layers": fixed_l,
                    "heads": 4 if d_model < 128 else 8,
                    "LR": float(f"{lr:.2e}"),
                    "Tokens": float(f"{tokens:.2e}"),
                    "dataset": dataset,
                    "D_N_ratio": round(tokens / n, 2),
                }
            )

    if not supplemental_rows:
        print("🎉 所有档位似乎都已找到最优 V 型底，无需补充！")
        return None

    # 导出并去重
    df_supp = pd.DataFrame(supplemental_rows).drop_duplicates(subset=["N", "LR"])

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_supp.to_csv(output_path, index=False)
    print(f"\n✅ 增量计划已生成：{output_path} (共 {len(df_supp)} 个点)")
    return df_supp


@click.command()
@click.option(
    "--result",
    default="artifacts/chinchilla_sweep/1e14",
    type=click.Path(exists=True, dir_okay=True, readable=True),
    help="Path to results.json and output supplemental supple.csv",
)
@click.option("--budget", default=1e14, type=int, help="FLOPs budget")
@click.option("--dataset", default="256", help="Dataset identifier")
@click.option("--fixed-l", default=6, type=int, help="Fixed number of layers")
def main(result, budget, dataset, fixed_l):
    """
    分析现有实验结果，为没有找到 V 型底部的 N 档位生成增量补扫计划。
    """
    result_path = os.path.join(result, "results.json")
    output_path = os.path.join(result, f"{budget:.0e}_supple.csv".replace('+', ''))
    df_supp = generate_supplemental_plan(
        result_path, output_path, budget, dataset, fixed_l
    )
    if df_supp is not None:
        print(df_supp)


if __name__ == "__main__":
    main()
