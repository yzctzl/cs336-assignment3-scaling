import os

import numpy as np
import pandas as pd


def generate_scaling_sweep(budgets):
    FIXED_L = 6
    plan_rows = []

    for C_budget in budgets:
        # --- 核心修改：大幅右移搜索中心以应对 vocab=256 的特性 ---
        # 既然 1e14 在 1.19e5，而 6e14 在 6e5 还在跌
        # 我们对 N_opt 采用更激进的缩放估计 (接近 C^1.0 的 Kaplan 缩放)
        if C_budget < 4e14:  # 针对 3e14
            center_n = 6.0e5
            range_factor = 4.0  # 扫 1.5e5 到 2.4e6
        else:  # 针对 6e14
            center_n = 1.5e6
            range_factor = 4.0  # 扫 3.7e5 到 6.0e6

        num_models = 12  # 保持 12 个模型点以平衡分辨率和算力

        target_ns = np.logspace(
            np.log10(center_n / range_factor),
            np.log10(center_n * range_factor),
            num_models,
        )

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
    df = df.drop_duplicates(subset=["Budget", "N", "LR"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    target_budgets = [3e14, 6e14]
    df_plan = generate_scaling_sweep(target_budgets)

    # 路径确保存在
    save_dir = "artifacts/chinchilla_sweep"
    os.makedirs(save_dir, exist_ok=True)

    for b in target_budgets:
        b_label = f"{b:.0e}"  # 匹配 DataFrame 中的 "3e+14" 或 "6e+14"
        b_file_name = b_label.replace("+", "")

        subset = df_plan[df_plan["Budget"] == b_label]

        if subset.empty:
            print(f"⚠️ 无法匹配 Budget {b_label}，请检查格式")
            continue

        # 确保子目录存在
        os.makedirs(f"{save_dir}/{b_file_name}", exist_ok=True)
        file_path = f"{save_dir}/{b_file_name}/{b_file_name}_budget.csv"
        subset.to_csv(file_path, index=False)

        print(f"\n### {b_file_name} 计划生成成功 (共 {len(subset)} 条任务) ###")
        print(f"搜索区间: N 从 {subset['N'].min():.2e} 到 {subset['N'].max():.2e}")
        # 检查 D/N 覆盖情况
        preview = subset.iloc[1::3]  # 展示每个模型中间的 LR
        print(preview[["N", "d_model", "LR", "D_N_ratio"]].to_string(index=False))
