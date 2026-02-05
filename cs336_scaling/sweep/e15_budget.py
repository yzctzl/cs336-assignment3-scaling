import os

import numpy as np
import pandas as pd


def generate_chinchilla_6e15_v10k_sweep():
    # 1. 设定基础参数
    C_budget = 6.0e15
    FIXED_L = 6
    FIXED_LR = 5e-4
    VOCAB_SIZE = 10000

    # 2. 根据 D/N 约束区间 [11, 180] 确定总参数量 N_total 的物理边界
    # 公式推导: C = 6 * N_total * D => D = C / (6 * N_total)
    # D/N = C / (6 * N_total^2) => N_total = sqrt(C / (6 * (D/N)))
    n_total_min = np.sqrt(C_budget / (6 * 180))  # 约 2.36M
    n_total_max = np.sqrt(C_budget / (6 * 11))  # 约 9.53M

    # 3. 在 log 空间生成 16 个目标总参数量点
    target_ns = np.logspace(np.log10(n_total_min), np.log10(n_total_max), 16)

    plan_rows = []

    for t_n in target_ns:
        # 4. 求解架构参数 d_model
        # N_total = N_logic + N_emb
        # N_total = (12 * L * d^2) + (V * d)
        # 这是一个一元二次方程: 144*d^2 + 10000*d - t_n = 0
        a = 12 * FIXED_L
        b = VOCAB_SIZE
        c = -t_n

        # 求根公式: d = (-b + sqrt(b^2 - 4ac)) / 2a
        d_raw = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

        # 将 d_model 对齐到 4 的倍数，防止 16 个点发生折叠合并
        d_model = int(max(64, round(d_raw / 4) * 4))

        # 5. 重新计算精确的参数量和 Token 数
        n_logic = 12 * FIXED_L * (d_model**2)
        n_emb = VOCAB_SIZE * d_model
        n_total_actual = n_logic + n_emb

        # 严格按照 6ND 计算 Token 数以保证算力对齐
        tokens = C_budget / (6 * n_total_actual)
        dn_ratio_actual = tokens / n_total_actual

        plan_rows.append(
            {
                "Budget": "6e15",
                "N_logic": int(n_logic),
                "N_emb": int(n_emb),
                "N": int(n_total_actual),
                "d_model": d_model,
                "layers": FIXED_L,
                "heads": 8,
                "LR": float(f"{FIXED_LR:.2e}"),
                "Tokens": float(f"{tokens:.2e}"),
                "dataset": "tss",
                "D_N_ratio": round(dn_ratio_actual, 2),
            }
        )

    df = pd.DataFrame(plan_rows)
    # 按照 N_total 排序以方便绘图
    df = (
        df.sort_values("N")
        .drop_duplicates(subset=["N"])
        .reset_index(drop=True)
    )
    return df


if __name__ == "__main__":
    df_plan = generate_chinchilla_6e15_v10k_sweep()

    # 保存结果
    save_path = "artifacts/chinchilla_sweep/6e15/v10k_aligned.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_plan.to_csv(save_path, index=False)

    print(f"实际生成点数: {len(df_plan)}")
    print(
        f"总参数量范围: {df_plan['N'].min():.2e} -> {df_plan['N'].max():.2e}"
    )
    print(
        f"D/N (全口径) 区间: {df_plan['D_N_ratio'].min()} -> {df_plan['D_N_ratio'].max()}"
    )
    print("-" * 60)
    # 预览关键列
    print(
        df_plan[["N", "d_model", "LR", "Tokens", "D_N_ratio"]].to_string(
            index=False
        )
    )
