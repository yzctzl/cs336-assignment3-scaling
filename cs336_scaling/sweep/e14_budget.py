import os

import numpy as np
import pandas as pd


def generate_scaling_sweep(budgets):
    FIXED_L = 6
    plan_rows = []

    for C_budget in budgets:
        # 1. 动态确定 N 的搜索范围
        # 基于 1e14 拟合出的 N_opt = 1.19e5
        # 算力增加，最优 N 按照 sqrt(C) 比例移动 (Chinchilla 经验)
        center_n = 1.19e5 * (C_budget / 1e14) ** 0.5

        # 3e14: 4/12, e14: 3/9
        d, s = (4, 12) if C_budget < 5e14 else (3, 9)

        # 覆盖中心点左右各 d 倍的范围，确保捕捉 U 型底
        target_ns = np.logspace(np.log10(center_n / d), np.log10(center_n * d), s)

        for t_n in target_ns:
            # 2. 架构对齐 (Non-Embedding Params: 12 * L * d^2)
            d_raw = np.sqrt(t_n / (12 * FIXED_L))
            d_model = int(max(16, round(d_raw / 8) * 8))
            n_actual = 12 * FIXED_L * (d_model**2)

            # 3. 计算 D (Tokens)
            tokens = C_budget / (6 * n_actual)
            dn_ratio = tokens / n_actual

            # 4. 使用你实测拟合的 LR 公式: LR = 18.2 * N^(-0.5993)
            lr_anchor = 18.2 * (n_actual**-0.5993)

            # 保持 0.7x, 1.0x, 1.4x 的窄窗口采样，防止 Loss 暴跳
            lrs = [lr_anchor * 0.7, lr_anchor, lr_anchor * 1.4]

            for lr in lrs:
                plan_rows.append(
                    {
                        "Budget": f"{C_budget:.0e}",  # 这里会生成 "3e+14"
                        "N": int(n_actual),
                        "d_model": d_model,
                        "layers": FIXED_L,
                        "heads": 4 if d_model < 128 else 8,
                        "LR": float(f"{lr:.2e}"),
                        "Tokens": float(f"{tokens:.2e}"),
                        "dataset": "256",
                        "D_N_ratio": round(dn_ratio, 2),
                    }
                )

    df = pd.DataFrame(plan_rows)
    # 去重
    df = df.drop_duplicates(subset=["Budget", "N", "LR"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    target_budgets = [3e14]  # , 6e14
    df_plan = generate_scaling_sweep(target_budgets)

    # 路径确保存在
    save_dir = "artifacts/chinchilla_sweep"
    os.makedirs(save_dir, exist_ok=True)

    for b in target_budgets:
        # 修正匹配逻辑：统一不带加号的格式
        b_label = f"{b:.0e}"  # "3e+14"
        b_file_name = b_label.replace("+", "")  # "3e14"

        subset = df_plan[df_plan["Budget"] == b_label]

        file_path = f"{save_dir}/{b_file_name}/{b_file_name}_budget.csv"
        subset.to_csv(file_path, index=False)

        print(f"\n### {b_file_name} 计划生成成功 (共 {len(subset)} 条) ###")
        print(f"文件位置: {file_path}")
        # 展示部分数据点，检查 D/N 覆盖情况
        preview = subset.iloc[1::6]  # 跨步展示不同模型的中间 LR
        print(preview[["N", "d_model", "LR", "D_N_ratio"]].to_string(index=False))
