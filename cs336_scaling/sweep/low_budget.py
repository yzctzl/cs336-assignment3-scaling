import numpy as np
import pandas as pd


# --- 核心改进: 针对小模型的暴力广角 LR ---
def get_aggressive_lrs(n_center):
    """
    针对小模型实施极其激进的 LR 搜索。
    目标：必须看到 Loss 上升（U型右壁）。
    """
    # 策略：分段设置极宽的锚点
    if n_center < 5.0e5:  # < 500K (1e13 档位)
        # 激进尝试到 0.1
        # 采样点: [0.001, 0.003, 0.01, 0.03, 0.1]
        min_lr, max_lr = 1.0e-3, 1.0e-1
    elif n_center < 1.5e6:  # < 1.5M (3e13 - 1e14)
        # 稍微收敛
        # 采样点: [0.0005, ..., 0.05]
        min_lr, max_lr = 5.0e-4, 5.0e-2
    elif n_center < 5.0e6:  # < 5M (3e14 - 1e15)
        min_lr, max_lr = 2.0e-4, 2.0e-2
    else:  # > 5M (常规大模型)
        # 回归到之前的科学计算 (Kaplan scaling)
        lr_anchor = 0.0025 * (n_center / 1e6) ** -0.5
        min_lr = lr_anchor / 4.0
        max_lr = lr_anchor * 4.0
        # 物理限制
        max_lr = min(1.0, max_lr)

    # 物理下限
    min_lr = max(1e-6, min_lr)

    # 生成 5 个对数均匀点
    lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), 5)

    return [float(f"{lr:.2e}") for lr in lrs]


# 2. 模型库 (Besiroglu 修正版中心)
def get_optimal_n_besiroglu(compute_budget):
    A, alpha = 482.01, 0.3478
    B, beta = 2085.43, 0.3658
    G = compute_budget / 6.0
    term1 = (alpha * A) / (beta * B)
    term2 = G**beta
    n_opt = (term1 * term2) ** (1 / (alpha + beta))
    return n_opt


def get_model_registry():
    candidates = []
    # 1. 重点覆盖区：d 在 32 到 256 之间，步长设小以提高 U 型底分辨率
    for d in [32, 48, 64, 80, 96, 128, 160, 192, 224, 256]:
        for l in [2, 3, 4, 6, 8, 12]:  # noqa: E741
            n_params = 12 * l * (d**2)
            candidates.append(
                {
                    "n_params": n_params,
                    "d_model": d,
                    "num_layers": l,
                    "num_heads": 4 if d < 128 else 8,
                }
            )

    # 2. 扩展区：覆盖更高算力下的最优 N
    for d in [320, 384, 448, 512]:
        for l in [4, 8, 12, 16, 20]:  # noqa: E741
            n_params = 12 * l * (d**2)
            candidates.append(
                {"n_params": n_params, "d_model": d, "num_layers": l, "num_heads": 16}
            )

    df = pd.DataFrame(candidates).drop_duplicates(subset=["n_params"])
    return df.sort_values("n_params").reset_index(drop=True)


# 3. 扫描策略 (重点关注低算力区的 Range)
schedule = [
    # Low Compute: 之前的实验证明 1e13 的底在 100K 附近
    # 我们从 0.1 开始扫 (约 22K)，绝对能看到左侧 Loss 上升
    {"C": 1e13, "Range": [0.1, 3.0], "N_Models": 8, "LR_Points": 5},
    {"C": 3e13, "Range": [0.1, 3.0], "N_Models": 9, "LR_Points": 5},
    {"C": 1e14, "Range": [0.15, 3.5], "N_Models": 9, "LR_Points": 5},
    # Mid Compute: 理论最优 N 增加，Range 跟随平移
    {"C": 3e14, "Range": [0.2, 4.0], "N_Models": 9, "LR_Points": 5},
    {"C": 6e14, "Range": [0.25, 4.0], "N_Models": 8, "LR_Points": 5},
    # High Compute: 逐步减少模型数量，增加预测的可信度
    # {"C": 1e15, "Range": [0.4, 3.0],  "N_Models": 8, "LR_Points": 5},
    # {"C": 5e15, "Range": [0.5, 2.5],  "N_Models": 7, "LR_Points": 3},
    # {"C": 1e16, "Range": [0.6, 2.0],  "N_Models": 6, "LR_Points": 2},
]

# 4. 生成计划
df_models = get_model_registry()
plan_rows = []

print("Generating Aggressive Wide-Angle Plan...")

for stage in schedule:
    budget = stage["C"]
    center_n = get_optimal_n_besiroglu(budget)

    # 使用激进的 LR 生成器
    potential_lrs = get_aggressive_lrs(center_n)

    # 采样点逻辑
    if stage["LR_Points"] == 5:
        stage_lrs = potential_lrs
    elif stage["LR_Points"] == 3:
        stage_lrs = [potential_lrs[1], potential_lrs[2], potential_lrs[3]]
    elif stage["LR_Points"] == 2:
        stage_lrs = [potential_lrs[1], potential_lrs[3]]
    else:
        # if stage["LR_Points"] == 1:
        stage_lrs = [potential_lrs[2]]

    targets = np.logspace(
        np.log10(center_n * stage["Range"][0]),
        np.log10(center_n * stage["Range"][1]),
        stage["N_Models"],
    )

    selected_ids = []
    for t in targets:
        closest_idx = (df_models["n_params"] - t).abs().idxmin()
        selected_ids.append(closest_idx)
    selected_ids = sorted(list(set(selected_ids)))

    for idx in selected_ids:
        model = df_models.iloc[idx]
        for lr in stage_lrs:
            plan_rows.append(
                {
                    "Budget": f"{budget:.0e}",
                    "Besiroglu_N_Opt": f"{center_n:.2e}",
                    "Params": int(model["n_params"]),
                    "d_model": int(model["d_model"]),
                    "layers": int(model["num_layers"]),
                    "heads": int(model["num_heads"]),
                    "LR": lr,
                    "dataset": "256",
                    "Cost_Val": f"{budget:.0e}",
                }
            )

df_plan = pd.DataFrame(plan_rows)

print("\n--- Aggressive LR Check (Must see high values!) ---")
debug_view = (
    df_plan.groupby("Budget")
    .agg(
        {"Besiroglu_N_Opt": "first", "LR": lambda x: [f"{v:.1e}" for v in np.unique(x)]}
    )
    .reset_index()
)
debug_view["SortKey"] = debug_view["Budget"].astype(float)
print(debug_view.sort_values("SortKey").drop("SortKey", axis=1).to_string(index=False))

print("\n--- CSV Output ---")
print(df_plan.to_csv(index=False))
