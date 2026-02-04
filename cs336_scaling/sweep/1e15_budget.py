import os

import numpy as np
import pandas as pd


def gen_1e15_sweep():
    C_budget = 1.0e15
    FIXED_L = 8 

    # --- 核心修改 1：右移 N 序列以匹配 V=10k ---
    # 之前 N 太大导致 D 太小。我们要找 D/N 在 20-200 之间的区间
    # 1e15 预算下，N 的黄金搜索区应在 5e5 到 5e6 之间
    target_ns = np.logspace(np.log10(5e5), np.log10(5e6), 12)

    plan_rows = []
    for t_n in target_ns:
        d_raw = np.sqrt(t_n / (12 * FIXED_L))
        d_model = int(max(64, round(d_raw / 16) * 16)) # 提高 d_model 下限
        n_actual = 12 * FIXED_L * (d_model**2)

        tokens = C_budget / (6 * n_actual)
        dn_ratio = tokens / n_actual

        # --- 核心修改 2：使用更保守的子词 LR 锚点 ---
        # 针对 10k 词表，降低常数项以增强稳定性
        lr_anchor = 12.0 * (n_actual**-0.56)
        
        # 适当拉开 LR 间距以探测更广范围 [0.5, 1.0, 2.0]
        lrs = [lr_anchor * 0.5, lr_anchor, lr_anchor * 2.0]

        for lr in lrs:
            plan_rows.append({
                "Budget": "1e15",
                "N": int(n_actual),
                "d_model": d_model,
                "layers": FIXED_L,
                "heads": 8 if d_model >= 128 else 4,
                "LR": float(f"{lr:.2e}"),
                "Tokens": float(f"{tokens:.2e}"),
                "dataset": "tss",
                "D_N_ratio": round(dn_ratio, 2),
            })
    
    return pd.DataFrame(plan_rows).drop_duplicates(subset=["N", "LR"])


if __name__ == "__main__":
    df_plan = gen_1e15_sweep()

    # 路径确保存在
    save_dir = "artifacts/chinchilla_sweep/1e15"
    os.makedirs(save_dir, exist_ok=True)

    file_path = f"{save_dir}/1e15_budget.csv"
    df_plan.to_csv(file_path, index=False)

    print(f"### 1e15 (V=10k) 广撒网计划生成成功 (共 {len(df_plan)} 个任务) ###")
    print(f"N 范围: {df_plan['N'].min():.2e} -> {df_plan['N'].max():.2e}")
    print(f"D/N 范围: {df_plan['D_N_ratio'].min()} -> {df_plan['D_N_ratio'].max()}")

    # 预览
    preview = df_plan.iloc[1::4]  # 展示每个模型其中一个 LR
    print("\n预览数据点：")
    print(preview[["N", "d_model", "LR", "D_N_ratio"]].to_string(index=False))
