import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import os

# ======================
# 🔧 自动创建输出文件夹
# ======================
out_dir = "gnina_results_analysis"
os.makedirs(out_dir, exist_ok=True)

# ======================
# 1. 读取数据
# ======================
df = pd.read_csv("gnina_output1/best_results_wide.csv")

# ======================
# 2. 定义所有GNINA模型列
# ======================
models = [
    "default2017",
    "crossdock_default2018",
    "redock_default2018",
    "dense",
    "general_default2018"
]

aff_cols   = [m + "_Affinity" for m in models]
pose_cols  = [m + "_CNNpose" for m in models]
cnn_cols   = [m + "_CNNaffinity" for m in models]

# ======================
# 3. 计算 Z-score
# ======================
def zscore(col):
    return (col - col.mean()) / col.std()

for col in aff_cols + pose_cols + cnn_cols:
    df[col + "_z"] = zscore(df[col])

# ======================
# 4. 综合评分
# ======================
pose_z_cols = [c + "_z" for c in pose_cols]
cnn_z_cols  = [c + "_z" for c in cnn_cols]

df["pose_mean_z"] = df[pose_z_cols].mean(axis=1)
df["cnn_mean_z"]  = df[cnn_z_cols].mean(axis=1)
df["Final_score"] = df["pose_mean_z"] + df["cnn_mean_z"]

# ======================
# 5. 总排名
# ======================
df_rank = df.sort_values("Final_score", ascending=False)

# ======================
# 6. 【关键】同分子去重：只保留每个分子得分最高的构象
# ======================
df_rank["Molecule"] = df_rank["Ligand"].str.replace(r"_\d+$", "", regex=True)
df_unique = df_rank.loc[df_rank.groupby("Molecule")["Final_score"].idxmax()]
df_unique = df_unique.sort_values("Final_score", ascending=False)

# ======================
# 7. 输出排名结果
# ======================
print("\n===== ✅ 去重后最终排名 Top10 =====")
print(df_unique[["Molecule", "Ligand", "Final_score"]].head(10))

# ======================
# 8. 每个模型单独Top10
# ======================
for m in models:
    col = m + "_CNNaffinity"
    top = df_unique.sort_values(col, ascending=False).head(10)
    print(f"\n----- {m} CNNaffinity Top10 -----")
    print(top[["Molecule", "Ligand", col]])

# ======================
# 9. 共识排名
# ======================
counter = Counter()
for m in models:
    col = m + "_CNNaffinity"
    top20 = df_unique.sort_values(col, ascending=False).head(20)
    for mol in top20["Molecule"]:
        counter[mol] += 1

consensus = pd.DataFrame({
    "Molecule": counter.keys(),
    "Consensus_count": counter.values()
}).sort_values("Consensus_count", ascending=False)

print("\n===== 🎯 共识排名 Top10 =====")
print(consensus.head(10))

# ======================
# 📊 全套论文统计图（全部保存到文件夹）
# ======================
# 设置字体为Arial（所有文本统一使用Arial）
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# 辅助函数：设置图形样式（刻度向内、边框加粗、标签加粗）
def set_axes_style(ax, title=None, xlabel=None, ylabel=None):
    """设置轴的样式：刻度向内、边框加粗、标题和标签加粗"""
    # 刻度向内
    ax.tick_params(axis='x', direction='in', width=1.5, length=6)
    ax.tick_params(axis='y', direction='in', width=1.5, length=6)
    # 边框加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # 刻度标签加粗
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    # 标题和轴标签加粗
    if title:
        ax.set_title(title, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')

# 1. 最终得分分布
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df_unique["Final_score"], bins=30, kde=True, color="#2E86AB", ax=ax)
set_axes_style(ax, title="Distribution of Final Score", xlabel="Final Score", ylabel="Count")
plt.tight_layout()
plt.savefig(f"{out_dir}/fig1_final_score_dist.png", dpi=300)
plt.close()

# 2. 共识次数柱状图
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x="Consensus_count", data=consensus, palette="viridis", ax=ax)
set_axes_style(ax, title="Consensus Count Distribution", xlabel="Consensus Count", ylabel="Number of Molecules")
plt.tight_layout()
plt.savefig(f"{out_dir}/fig2_consensus_count.png", dpi=300)
plt.close()

# 3. CNNaffinity 箱线图
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_unique[cnn_cols], palette="coolwarm", ax=ax)
set_axes_style(ax, title="CNNaffinity Across 5 Models", xlabel="Model", ylabel="CNNaffinity")
# 单独设置x轴刻度标签旋转
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.tight_layout()
plt.savefig(f"{out_dir}/fig3_cnn_affinity_boxplot.png", dpi=300)
plt.close()

# 4. 模型相关性热图
corr = df_unique[cnn_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="RdBu_r", vmin=0.5, vmax=1, fmt=".2f", ax=ax)
# 热图需要单独设置
ax.set_title("Model Correlation (CNNaffinity)", fontweight='bold')
# 边框加粗
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
# 刻度向内
ax.tick_params(axis='x', direction='in', width=1.5, length=6)
ax.tick_params(axis='y', direction='in', width=1.5, length=6)
# 刻度标签加粗
ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
plt.tight_layout()
plt.savefig(f"{out_dir}/fig4_model_correlation.png", dpi=300)
plt.close()

# 5. Top20 分子条形图
top20 = df_unique.head(20)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="Molecule", y="Final_score", data=top20, palette="magma", ax=ax)
set_axes_style(ax, title="Top 20 Molecules by Final Score", xlabel="Molecule", ylabel="Final Score")
# 单独设置x轴刻度标签旋转
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.savefig(f"{out_dir}/fig5_top20_molecules.png", dpi=300)
plt.close()

# ======================
# 📌 【你要的图】分子在哪些模型出现 + 出现次数
# ======================
def get_top_mols(df, col, n=20):
    return set(df.sort_values(col, ascending=False).head(n)["Molecule"])

top_sets = {}
for m in models:
    col = m + "_CNNaffinity"
    top_sets[m] = get_top_mols(df_unique, col, 20)

# 构建出现矩阵
presence = []
all_mols = sorted({mol for m in top_sets for mol in top_sets[m]})

for mol in all_mols:
    row = {"Molecule": mol}
    for m in models:
        row[m] = 1 if mol in top_sets[m] else 0
    row["Total"] = sum(row[m] for m in models)
    presence.append(row)

pres_df = pd.DataFrame(presence).sort_values("Total", ascending=False)

# 画图：分子 × 模型出现热图
fig, ax = plt.subplots(figsize=(12, 10))
plot_models = models
df_pivot = pres_df.pivot_table(index="Molecule", columns=plot_models, values="Total", fill_value=0)
df_pivot = pres_df.set_index("Molecule")[plot_models]

sns.heatmap(df_pivot, cmap=["white", "#c75336"], cbar=False, linewidths=0.5, linecolor="gray", ax=ax)
ax.set_title("Molecule Presence in Top20 of Each Model \n (1=present, 0=not present)", fontweight='bold', fontsize=16)
ax.set_xlabel("Model", fontweight='bold', fontsize=14)
ax.set_ylabel("Molecule", fontweight='bold', fontsize=14)
# 边框加粗
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
# 刻度向内
ax.tick_params(axis='x', direction='out', width=1.5, length=6)
ax.tick_params(axis='y', direction='out', width=1.5, length=6)
# 刻度标签加粗
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
plt.tight_layout()
plt.savefig(f"{out_dir}/fig6_molecule_presence_heatmap.png", dpi=300)
plt.close()

# 同时画：按出现次数排序的条形图
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x="Molecule", y="Total", data=pres_df.head(30), palette="Blues_r", ax=ax)
set_axes_style(ax, title="Top Molecules by Consensus Count (Total Appearances)", xlabel="Molecule", ylabel="Total Appearances")
# 单独设置x轴刻度标签旋转
ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
plt.tight_layout()
plt.savefig(f"{out_dir}/fig7_consensus_count_bar.png", dpi=300)
plt.close()

# ======================
# 💾 所有表格保存到新文件夹
# ======================
df_rank.to_csv(f"{out_dir}/final_ranking_all_poses.csv", index=False)
df_unique.to_csv(f"{out_dir}/final_ranking_unique.csv", index=False)
consensus.to_csv(f"{out_dir}/consensus_ranking.csv", index=False)

# ======================
# ✅ 完成提示
# ======================
print(f"\n🎉 全部分析完成！")
print(f"📁 所有结果已保存到文件夹：{out_dir}/")
print(f"✅ 7 张高清论文图")
print(f"✅ 3 个最终排名表格")