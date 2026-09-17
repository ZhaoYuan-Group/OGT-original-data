#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

# -------------------------- Global Plot Settings --------------------------
matplotlib.rcParams['axes.unicode_minus'] = False
rcParams['font.sans-serif'] = ['Arial', 'Arial Unicode MS']
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 11
rcParams['figure.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# 新增：全局设置刻度线方向、边框粗细
rcParams['xtick.direction'] = 'out'        # x轴刻度线向内
rcParams['ytick.direction'] = 'in'        # y轴刻度线向内
rcParams['axes.linewidth'] = 1.5          # 边框线加粗
rcParams['xtick.major.width'] = 1.5       # x轴主刻度线宽度
rcParams['ytick.major.width'] = 1.5       # y轴主刻度线宽度
rcParams['xtick.major.size'] = 3          # x轴主刻度线长度
rcParams['ytick.major.size'] = 3          # y轴主刻度线长度

# -------------------------- Your Color Palette --------------------------
COLOR_LIST = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
    '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
    '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    '#393b79', '#5254a3', '#6b4c9a', '#8c6d31'
]

# -------------------------- 1. Load Data --------------------------
df = pd.read_csv("result.csv", sep=",")
df.columns = df.columns.str.strip()

# -------------------------- Step 1: Calculate GninaScore --------------------------
df["GninaScore"] = df["CNN_affinity"] * df["CNN_pose"]

# -------------------------- Step 2: Convert GBSA to Positive --------------------------
df["GBSA_positive"] = -df["GBSA"]

# -------------------------- Step 3: Min-Max Normalization --------------------------
def min_max_norm(s):
    xmin = s.min()
    xmax = s.max()
    if xmax == xmin:
        return pd.Series([1.0]*len(s), index=s.index)
    return (s - xmin) / (xmax - xmin)

df["Norm_GBSA"] = min_max_norm(df["GBSA_positive"])
df["Norm_Gnina"] = min_max_norm(df["GninaScore"])

# -------------------------- Step 4: Final Score & Ranking --------------------------
df["FinalScore"] = (df["Norm_GBSA"] ** 0.7) * (df["Norm_Gnina"] ** 0.3)
df["Rank"] = df["FinalScore"].rank(method="min", ascending=False).astype(int)

df_sorted = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)

# -------------------------- Save Results --------------------------
df_sorted.to_csv("final_consensus_scores.csv", index=False, encoding="utf-8-sig")

# -------------------------- Plotting (PNG ONLY) --------------------------

# Figure 1: Final Score Bar Plot
plt.figure(figsize=(8, 6))
colors = [COLOR_LIST[i % len(COLOR_LIST)] for i in range(len(df_sorted))]
bars = plt.bar(df_sorted["Compound"], df_sorted["FinalScore"], color=colors, width=0.6)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.title('MM/GBSA + Gnina Consensus Score Ranking', fontweight='bold')
plt.xlabel('Compound ID', fontweight='bold')
plt.ylabel('Final Consensus Score', fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, df_sorted["FinalScore"].max() * 1.15)
plt.xticks(rotation=45, ha='center')

# 加粗边框
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)

plt.tight_layout()
plt.savefig('1_consensus_score_ranking.png', dpi=300)
plt.close()

# Figure 2: GBSA vs GninaScore Dual Axis Plot
fig, ax1 = plt.subplots(figsize=(8, 6))

# 加粗边框
for spine in ax1.spines.values():
    spine.set_linewidth(1.5)

# 柱状图：每个化合物独立颜色
ax1.bar(df_sorted["Compound"], df_sorted["GBSA_positive"], color=colors, alpha=0.7, width=0.5, label='GBSA Positive')
ax1.set_xlabel('Compound ID', fontweight='bold')
ax1.set_ylabel('GBSA Positive Value (-GBSA)', fontweight='bold')
ax1.tick_params(axis='y', direction='in', width=1.5, length=6)
ax1.tick_params(axis='x', direction='out',width=1.5, length=6)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.tick_params(axis='x', rotation=45)
ax2 = ax1.twinx()
# 加粗ax2边框
for spine in ax2.spines.values():
    spine.set_linewidth(1.5)

# 纯黑色折线+圆点
ax2.plot(df_sorted["Compound"], df_sorted["GninaScore"], 
         marker='o', linewidth=2, markersize=7,
         color='black', markerfacecolor='black', markeredgecolor='black',
         label='GninaScore')

ax2.set_ylabel('GninaScore (CNN_affinity × CNN_pose)', fontweight='bold')
ax2.tick_params(axis='y', direction='in', width=1.5, length=6)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', bbox_to_anchor=(0.98, 0.98))
plt.title('GBSA vs GninaScore Comparison', fontweight='bold')
plt.tight_layout()
plt.savefig('2_gbsa_gnina_comparison.png', dpi=300)
plt.close()

# Figure 3: Normalized Score Scatter Plot
plt.figure(figsize=(8, 6))

# 加粗边框
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)

scatter = plt.scatter(df_sorted["Norm_GBSA"], df_sorted["Norm_Gnina"], 
                      c=df_sorted["FinalScore"], cmap='viridis', s=150, alpha=0.8, edgecolors='k')

for i, row in df_sorted.iterrows():
    plt.text(row["Norm_GBSA"] + 0.02, row["Norm_Gnina"] + 0.02, row["Compound"], fontsize=9, fontweight='medium')

plt.title('Normalized GBSA vs GninaScore', fontweight='bold')
plt.xlabel('Normalized GBSA Score', fontweight='bold')
plt.ylabel('Normalized Gnina Score', fontweight='bold')
plt.grid(linestyle='--', alpha=0.7)
plt.xlim(-0.05, 1.15)
plt.ylim(-0.05, 1.05)
cbar = plt.colorbar(scatter, label='Final Consensus Score')
cbar.set_label('Final Consensus Score', fontweight='bold')
cbar.outline.set_visible(False)
plt.tight_layout()
plt.savefig('3_normalized_score_scatter.png', dpi=300)
plt.close()

# Figure 4: Horizontal Score Bar Plot
plt.figure(figsize=(8, 6))

# 加粗边框
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.5)

df_reversed = df_sorted.iloc[::-1]
colors_rev = [COLOR_LIST[i % len(COLOR_LIST)] for i in range(len(df_reversed))]
bars = plt.barh(df_reversed["Compound"], df_reversed["FinalScore"], color=colors_rev, height=0.6)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.02, bar.get_y() + bar.get_height()/2., f'{width:.3f}', ha='left', va='center', fontsize=9)

plt.title('Final Consensus Score Ranking (Horizontal)', fontweight='bold')
plt.xlabel('Final Consensus Score', fontweight='bold')
plt.ylabel('Compound ID', fontweight='bold')
plt.tick_params(axis='x', direction='in', width=1.5, length=6)
plt.tick_params(axis='y', direction='out', width=1.5, length=6)
plt.xlim(0, df_sorted["FinalScore"].max() * 1.15)
plt.tight_layout()
plt.savefig('4_horizontal_score_ranking.png', dpi=300)
plt.close()

# -------------------------- Console Output --------------------------
print("✅ Calculation completed successfully!")
print(df_sorted[["Compound", "GBSA", "GninaScore", "FinalScore", "Rank"]].to_string(index=False))