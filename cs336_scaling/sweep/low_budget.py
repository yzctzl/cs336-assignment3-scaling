import numpy as np
import pandas as pd


# 1. 动态 LR 生成器
def get_scientific_lrs(n_center):
    """
    根据模型大小 N_center 动态计算 5 个 LR 搜索点。
    原理：模型越大，LR 上界越低。
    """
    # 始终锚定 API 的下界
    min_lr = 1.0e-4

    # 动态设定上界 (Heuristic based on Kaplan scaling)
    if n_center < 8.0e5:  # < 800K 参数 (对应 1e13 ~ 6e13)
        max_lr = 1.0e-3  # 允许最大 LR
    elif n_center < 1.8e6:  # ~ 1.6M 参数 (对应 1e14 ~ 3e14)
        max_lr = 8.0e-4  # 稍微降低风险
    else:  # > 2M 参数 (对应 6e14)
        max_lr = 6.0e-4  # 进一步降低

    # 生成 5 个对数均匀分布的点
    lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), 5)

    # 格式化保留有效数字，避免浮点误差
    return [float(f"{lr:.2e}") for lr in lrs]


# 2. 模型库生成 (与之前一致，加上筛选逻辑)
def get_model_registry():
    candidates = []
    for d in range(64, 1025, 32):
        for l in range(2, 25, 1):  # noqa: E741
            valid_heads = [h for h in [2, 4, 8, 16] if d % h == 0]
            if not valid_heads:
                continue
            h = max(valid_heads)
            n_params = 12 * l * (d**2)

            # 过滤掉形状怪异的模型 (AspectRatio)
            ratio = d / l
            if ratio < 10 or ratio > 200:
                continue

            candidates.append(
                {
                    "n_params": n_params,
                    "d_model": d,
                    "num_layers": l,
                    "num_heads": h,
                    "batch_size": 128,
                }
            )
    return pd.DataFrame(candidates).sort_values("n_params").reset_index(drop=True)


# 3. 扫描策略定义 (Low Compute Only)
low_compute_schedule = [
    {"C": 1e13, "Center": 2.9e5, "Range": [0.3, 4.0], "N_Models": 6},
    {"C": 3e13, "Center": 5.0e5, "Range": [0.3, 4.0], "N_Models": 7},
    {"C": 6e13, "Center": 7.1e5, "Range": [0.3, 4.0], "N_Models": 7},
    {"C": 1e14, "Center": 9.1e5, "Range": [0.33, 3.0], "N_Models": 7},
    {"C": 3e14, "Center": 1.6e6, "Range": [0.33, 3.0], "N_Models": 7},
    {"C": 6e14, "Center": 2.2e6, "Range": [0.33, 3.0], "N_Models": 7},
]

# 4. 生成详细执行表
df_models = get_model_registry()
plan_rows = []

for stage in low_compute_schedule:
    # A. 确定该算力下的 LR 搜索组
    stage_lrs = get_scientific_lrs(stage["Center"])

    # B. 确定模型参数点
    targets = np.logspace(
        np.log10(stage["Center"] * stage["Range"][0]),
        np.log10(stage["Center"] * stage["Range"][1]),
        stage["N_Models"],
    )

    selected_ids = []
    for t in targets:
        closest_idx = (df_models["n_params"] - t).abs().idxmin()
        selected_ids.append(closest_idx)
    selected_ids = sorted(list(set(selected_ids)))

    # C. 生成笛卡尔积 (Models x LRs)
    for idx in selected_ids:
        model = df_models.iloc[idx]
        for lr in stage_lrs:
            plan_rows.append(
                {
                    "Budget": f"{int(stage["C"]):.1e}",
                    "Group_Center_N": f"{stage['Center']:.1e}",
                    "Params": int(model["n_params"]),
                    "d_model": int(model["d_model"]),
                    "layers": int(model["num_layers"]),
                    "heads": int(model["num_heads"]),
                    "LR": lr,
                    "Cost": f"{int(stage["C"]):.1e}",
                }
            )

df_plan = pd.DataFrame(plan_rows)

# 5. 打印预览和统计
print(f"Total Low-Compute Runs: {len(df_plan)}")
# print(f"Total Cost: {df_plan['Cost'].sum():.2e} FLOPs")
# print(f"Cost % of 2e18: {(df_plan['Cost'].sum() / 2e18) * 100:.3f}%")
print("\n--- Sample of Dynamic LR Strategy ---")
# 打印不同组的 LR 看看是否变化
print(df_plan.groupby("Budget")["LR"].unique())

# 输出 CSV
print("\n--- Full CSV Output Below ---")
print(df_plan.to_csv(index=False))
