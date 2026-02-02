import numpy as np
import pandas as pd


def generate_perfect_1e14_sweep():
    # 设定总算力预算
    C_budget = 1.0e14

    # 1. 设定目标参数量 N 的序列 (对数均匀分布)
    # 从 3,000 到 3,000,000，采样 18 个点以获得极高分辨率
    target_ns = np.logspace(np.log10(3e3), np.log10(3e6), 18)

    # 固定层数 L，这是消除架构噪声的关键
    # 选 L=6 是因为在极小模型下它比 L=12 更容易训练，比 L=1/2 表达能力更稳
    FIXED_L = 6

    plan_rows = []

    for t_n in target_ns:
        # 2. 根据 N ≈ 12 * L * d^2 反推 d_model
        d_raw = np.sqrt(t_n / (12 * FIXED_L))

        # 强制 d_model 是 8 的倍数，且最小不低于 16
        d_model = int(max(16, round(d_raw / 8) * 8))

        # 3. 计算实际非 Embedding 参数量 N
        n_actual = 12 * FIXED_L * (d_model**2)

        # 4. 计算对应的训练 Token 数 D (根据 C = 6ND)
        tokens = C_budget / (6 * n_actual)

        # 5. 激进的 LR 搜索：每个点配 3 个 LR 锚点
        # 公式来自上次 commit：N 越小，LR 越高
        lr_anchor = 58.1 * n_actual ** -0.676
        lrs = [lr_anchor * 0.5, lr_anchor, lr_anchor * 2.0]

        for lr in lrs:
            plan_rows.append(
                {
                    "Budget": "1e14",
                    "N_actual": int(n_actual),
                    "d_model": d_model,
                    "layers": FIXED_L,
                    "heads": 4 if d_model < 128 else 8,
                    "LR": float(f"{lr:.2e}"),
                    "Tokens": float(f"{tokens:.2e}"),
                    "D_N_ratio": round(tokens / n_actual, 2),
                }
            )

    df = pd.DataFrame(plan_rows)
    # 去重：因为 round(d/8) 可能导致相邻的 target_n 指向同一个 d_model
    df = df.drop_duplicates(subset=["N_actual", "LR"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df_plan = generate_perfect_1e14_sweep()

    print("### 1e14 扫描计划预览 (每种模型仅展示中间 LR) ###")
    preview = df_plan.iloc[1::3].copy()  # 每 3 行取一行（取中间那个 LR）
    print(
        preview[["N_actual", "d_model", "layers", "LR", "D_N_ratio"]].to_string(
            index=False
        )
    )

    # 导出 CSV
    df_plan.to_csv("1e14_budget.csv", index=False)
