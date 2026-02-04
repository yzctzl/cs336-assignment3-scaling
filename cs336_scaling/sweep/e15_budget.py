import os
import numpy as np
import pandas as pd

def generate_stable_v10k_sweep(budgets):
    # 增加深度至 12 层以提升大模型训练稳定性
    FIXED_L = 12 
    plan_rows = []

    for C_budget in budgets:
        # 1. 动态预估 N_opt 中心点
        # 考虑到 V=10k 的 Embedding 成本，1e15 时 N_opt 约在 2.5M
        # 按照 C^0.5 规律平移中心
        center_n = 2.5e6 * (C_budget / 1e15) ** 0.5
        
        # 2. 窄范围高分辨率扫描 (仅覆盖中心点左右各 3 倍)
        # 采样 10 个点以确保每档实验量控制在 30 个左右
        target_ns = np.logspace(np.log10(center_n / 3), np.log10(center_n * 3), 10)

        for t_n in target_ns:
            # 3. 架构对齐 (Non-Embedding Params)
            d_raw = np.sqrt(t_n / (12 * FIXED_L))
            d_model = int(max(64, round(d_raw / 32) * 32)) # 步进设为 32 以对齐硬件优化
            n_actual = 12 * FIXED_L * (d_model**2)

            # 4. 算力约束下的 Token 数 D
            tokens = C_budget / (6 * n_actual)
            dn_ratio = tokens / n_actual

            # 5. 科学的 LR 推断：基于 N^-0.5 规律
            # 锚点设定：1M 参数时 LR 约 6e-3 (对应 V=10k 经验值)
            lr_anchor = 0.006 * (n_actual / 1e6) ** -0.5
            
            # 使用更密的探测步长 [0.8x, 1.0x, 1.25x]
            lrs = [lr_anchor * 0.8, lr_anchor, lr_anchor * 1.25]

            for lr in lrs:
                plan_rows.append({
                    "Budget": f"{C_budget:.0e}",
                    "N": int(n_actual),
                    "d_model": d_model,
                    "layers": FIXED_L,
                    "heads": 8 if d_model >= 128 else 4,
                    "LR": float(f"{lr:.2e}"),
                    "Tokens": float(f"{tokens:.2e}"),
                    "dataset": "tss",
                    "D_N_ratio": round(dn_ratio, 2),
                })

    df = pd.DataFrame(plan_rows)
    return df.drop_duplicates(subset=["Budget", "N", "LR"]).reset_index(drop=True)

if __name__ == "__main__":
    # 生成 3e15 和 6e15 的计划
    target_budgets = [3e15, 6e15]
    df_plan = generate_stable_v10k_sweep(target_budgets)

    save_dir = "artifacts/chinchilla_sweep/"
    os.makedirs(save_dir, exist_ok=True)

    for b in target_budgets:
        b_label = f"{b:.0e}"
        b_file_name = b_label.replace("+", "")
        subset = df_plan[df_plan["Budget"] == b_label]
        
        os.makedirs(f"{save_dir}/{b_file_name}", exist_ok=True)
        file_path = f"{save_dir}/{b_file_name}/{b_file_name}_budget.csv"
        subset.to_csv(file_path, index=False)

        print(f"\n### {b_file_name} 稳定版计划生成 (共 {len(subset)} 个任务) ###")
        print(f"搜索区间: N 从 {subset['N'].min():.2e} 到 {subset['N'].max():.2e}")
        # 预览
        preview = subset.drop_duplicates(subset="N")
        print(preview[["N", "d_model", "LR", "D_N_ratio"]].to_string(index=False))