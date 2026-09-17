import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ====================== 在这里改你的文件名 ======================
FILE_PATH = "gnina_output1/all_poses.csv"
SAVE_DIR = "result_plots"  # 所有图片都会放进这里
# ===============================================================

# 自动创建文件夹
os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(FILE_PATH)

# 5 种方法
methods = [
    "default2017",
    "crossdock_default2018",
    "redock_default2018",
    "dense",
    "general_default2018"
]

# 自动构造列名
aff_cols = [f"{m}_Affinity" for m in methods]
pose_cols = [f"{m}_CNNpose" for m in methods]
cnnaff_cols = [f"{m}_CNNaffinity" for m in methods]
intra_cols = [f"{m}_Intramol" for m in methods]

# ===================== 绘图风格设置 =====================
# 设置字体为Arial（学术论文常用字体）
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# ==================================================================================
# 图1：Affinity 小提琴图
# ==================================================================================
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=df[aff_cols], palette=colors, linewidth=1.5, ax=ax)
ax.set_title("Binding Affinity Distribution", fontsize=20, weight='bold')
ax.set_ylabel("Affinity (kcal/mol)", fontsize=16, weight='bold')
ax.set_xlabel("Method", fontsize=16, weight='bold')
# 刻度向内、加粗边框，y轴刻度字体变大
ax.tick_params(axis='x', direction='out', width=1.5, length=6, labelsize=14)
ax.tick_params(axis='y', direction='in', width=1.5, length=6, labelsize=14)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
# 刻度标签不加粗
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
ax.set_ylim(-9, -4)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "affinity_violin.png"), dpi=300)
plt.close()

# ==================================================================================
# 图2：CNNpose 小提琴图
# ==================================================================================
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=df[pose_cols], palette=colors, linewidth=1.5, ax=ax)
ax.set_title("CNN Pose Confidence Distribution", fontsize=20, weight='bold')
ax.set_ylabel("CNNpose", fontsize=16, weight='bold')
ax.set_xlabel("Method", fontsize=16, weight='bold')
# 刻度向内、加粗边框，y轴刻度字体变大
ax.tick_params(axis='x', direction='out', width=1.5, length=6)
ax.tick_params(axis='y', direction='in', width=1.5, length=6)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
# 刻度标签不加粗
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "cnnpose_violin.png"), dpi=300)
plt.close()

# ==================================================================================
# 图3：CNNaffinity 小提琴图
# ==================================================================================
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=df[cnnaff_cols], palette=colors, linewidth=1.5, ax=ax)
ax.set_title("CNN Affinity Distribution", fontsize=20, weight='bold')
ax.set_ylabel("CNNaffinity", fontsize=16, weight='bold')
ax.set_xlabel("Method", fontsize=16, weight='bold')
# 刻度向内、加粗边框，y轴刻度字体变大
ax.tick_params(axis='x', direction='out', width=1.5, length=6)
ax.tick_params(axis='y', direction='in', width=1.5, length=6)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
# 刻度标签不加粗
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "cnn_affinity_violin.png"), dpi=300)
plt.close()

# ==================================================================================
# 图4：Intramol 箱线图
# ==================================================================================
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df[intra_cols], palette=colors, linewidth=1.5, ax=ax)
ax.set_title("Intramolecular Energy Distribution", fontsize=18, weight='bold')
ax.set_ylabel("Intramol (lower = more stable)", fontsize=14, weight='bold')
ax.set_xlabel("Method", fontsize=14, weight='bold')
# 刻度向内、加粗边框，y轴刻度字体变大
ax.tick_params(axis='x', direction='out', width=1.5, length=6, labelsize=14)
ax.tick_params(axis='y', direction='in', width=1.5, length=6, labelsize=14)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
# 刻度标签不加粗
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "intramol_boxplot.png"), dpi=300)
plt.close()

# ============================================================
# 热图 1：CNNpose
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(df[pose_cols].corr(), annot=True, cmap="RdYlGn", 
            vmin=0.5, vmax=1, fmt=".2f", linewidths=1.5, ax=ax)
ax.set_title("CNNpose Correlation",  fontsize=18,weight='bold')
# 刻度向内、加粗边框
ax.tick_params(axis='x', direction='in', width=1.5, length=6, labelsize=12)
ax.tick_params(axis='y', direction='in', width=1.5, length=6, labelsize=12)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
# 刻度标签不加粗
ax.set_xticklabels(ax.get_xticklabels())
ax.set_yticklabels(ax.get_yticklabels())
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "corr_cnnpose.png"), dpi=300)
plt.close()

# ============================================================
# 热图 2：Affinity
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(df[aff_cols].corr(), annot=True, cmap="RdYlGn", 
            vmin=0.5, vmax=1, fmt=".2f", linewidths=1.5, ax=ax)
ax.set_title("Affinity Correlation", fontsize=18, weight='bold')
# 刻度向内、加粗边框
ax.tick_params(axis='x', direction='in', width=1.5, length=6, labelsize=12)
ax.tick_params(axis='y', direction='in', width=1.5, length=6, labelsize=12)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
# 刻度标签不加粗
ax.set_xticklabels(ax.get_xticklabels())
ax.set_yticklabels(ax.get_yticklabels())
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "corr_affinity.png"), dpi=300)
plt.close()

# ============================================================
# 热图 3：CNNaffinity
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(df[cnnaff_cols].corr(), annot=True, cmap="RdYlGn", 
            vmin=0.5, vmax=1, fmt=".2f", linewidths=1.5, ax=ax)
ax.set_title("CNNaffinity Correlation",  fontsize=18,weight='bold')
# 刻度向内、加粗边框
ax.tick_params(axis='x', direction='in', width=1.5, length=6, labelsize=12)
ax.tick_params(axis='y', direction='in', width=1.5, length=6, labelsize=12)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
# 刻度标签不加粗
ax.set_xticklabels(ax.get_xticklabels())
ax.set_yticklabels(ax.get_yticklabels())
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "corr_cnnaff.png"), dpi=300)
plt.close()

# ============================================================
# 热图 4：Intramol
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(df[intra_cols].corr(), annot=True, cmap="RdYlGn", 
            vmin=0.5, vmax=1, fmt=".2f", linewidths=1.5, ax=ax)
ax.set_title("Intramol Correlation",  fontsize=18,weight='bold')
# 刻度向内、加粗边框
ax.tick_params(axis='x', direction='in', width=1.5, length=6, labelsize=12)
ax.tick_params(axis='y', direction='in', width=1.5, length=6, labelsize=12)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
# 刻度标签不加粗
ax.set_xticklabels(ax.get_xticklabels())
ax.set_yticklabels(ax.get_yticklabels())
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "corr_intramol.png"), dpi=300)
plt.close()

# ===================== 输出统计 =====================
stats = pd.DataFrame({
    "Method": methods,
    "Mean_Affinity": [round(df[c].mean(), 3) for c in aff_cols],
    "Mean_CNNpose": [round(df[c].mean(), 3) for c in pose_cols],
    "Mean_CNNAff": [round(df[c].mean(), 3) for c in cnnaff_cols],
    "Mean_Intramol": [round(df[c].mean(), 3) for c in intra_cols]
})

print("===== 方法统计结果 =====")
print(stats)
stats.to_csv("statistics_summary.csv", index=False, encoding="utf-8-sig")

print(f"\n✅ 全部图片已保存到：{SAVE_DIR}/")