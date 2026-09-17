#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

# ============================================================
# 0. Output Folder
# ============================================================

OUTPUT_DIR = "consensus_sensitivity_results"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# -------------------------- Global Plot Settings --------------------------
matplotlib.rcParams['axes.unicode_minus'] = False
rcParams['font.sans-serif'] = ['Arial', 'Arial Unicode MS']
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# -------------------------- Color Palette --------------------------
COLOR_LIST = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
    '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
    '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
    '#393b79', '#5254a3', '#6b4c9a', '#8c6d31'
]

# ============================================================
# 1. Load Data
# ============================================================

df = pd.read_csv("result.csv", sep=",")
df.columns = df.columns.str.strip()

required_columns = [
    "Compound",
    "CNN_affinity",
    "CNN_pose",
    "GBSA"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df = df.dropna(
    subset=[
        "Compound",
        "CNN_affinity",
        "CNN_pose",
        "GBSA"
    ]
).copy()

# ============================================================
# 2. Calculate Gnina Score
# ============================================================

df["GninaScore"] = (
    df["CNN_affinity"] *
    df["CNN_pose"]
)

# ============================================================
# 3. Convert GBSA to Positive
# ============================================================

df["GBSA_positive"] = -df["GBSA"]

# ============================================================
# 4. Min-Max Normalization
# ============================================================

def min_max_norm(s):

    xmin = s.min()
    xmax = s.max()

    if xmax == xmin:
        return pd.Series(
            [1.0] * len(s),
            index=s.index
        )

    return (
        (s - xmin) /
        (xmax - xmin)
    )


df["Norm_GBSA"] = min_max_norm(
    df["GBSA_positive"]
)

df["Norm_Gnina"] = min_max_norm(
    df["GninaScore"]
)

# ============================================================
# 5. Sensitivity Analysis
# ============================================================

weight_schemes = {
    "0.5_0.5": (0.5, 0.5),
    "0.7_0.3": (0.7, 0.3),
    "0.3_0.7": (0.3, 0.7)
}

for name, (w_gbsa, w_gnina) in weight_schemes.items():

    score_col = f"Score_{name}"
    rank_col = f"Rank_{name}"

    df[score_col] = (
        (df["Norm_GBSA"] ** w_gbsa) *
        (df["Norm_Gnina"] ** w_gnina)
    )

    df[rank_col] = (
        df[score_col]
        .rank(
            method="min",
            ascending=False
        )
        .astype(int)
    )

# ============================================================
# 6. Main Analysis = 0.7 / 0.3
# ============================================================

df["FinalScore"] = df["Score_0.7_0.3"]
df["Rank"] = df["Rank_0.7_0.3"]

df_sorted = (
    df.sort_values(
        "FinalScore",
        ascending=False
    )
    .reset_index(drop=True)
)

# ============================================================
# 7. Save Main Results
# ============================================================

df_sorted.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_consensus_scores.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 8. Sensitivity Analysis Table
# ============================================================

sensitivity_cols = [
    "Compound",
    "Norm_GBSA",
    "Norm_Gnina",
    "Score_0.5_0.5",
    "Rank_0.5_0.5",
    "Score_0.7_0.3",
    "Rank_0.7_0.3",
    "Score_0.3_0.7",
    "Rank_0.3_0.7"
]

sensitivity_df = df[
    sensitivity_cols
].copy()

sensitivity_df = (
    sensitivity_df
    .sort_values(
        "Rank_0.7_0.3"
    )
    .reset_index(drop=True)
)

sensitivity_df[
    "DeltaRank_0.5_0.5"
] = (
    sensitivity_df["Rank_0.5_0.5"] -
    sensitivity_df["Rank_0.7_0.3"]
)

sensitivity_df[
    "DeltaRank_0.3_0.7"
] = (
    sensitivity_df["Rank_0.3_0.7"] -
    sensitivity_df["Rank_0.7_0.3"]
)

sensitivity_df[
    "MaxRankShift"
] = np.maximum(
    sensitivity_df[
        "DeltaRank_0.5_0.5"
    ].abs(),

    sensitivity_df[
        "DeltaRank_0.3_0.7"
    ].abs()
)

sensitivity_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "sensitivity_analysis_ranking.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 9. Spearman Rank Correlation
# ============================================================

rank_cols = [
    "Rank_0.5_0.5",
    "Rank_0.7_0.3",
    "Rank_0.3_0.7"
]

spearman_corr = (
    df[rank_cols]
    .corr(method="spearman")
)

spearman_corr.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "sensitivity_spearman_correlation.csv"
    ),
    encoding="utf-8-sig"
)

# ============================================================
# 10. Top-N Stability
# ============================================================

def get_top_n(df, rank_col, n):

    return set(
        df.loc[
            df[rank_col] <= n,
            "Compound"
        ]
    )


top_n_results = []

for n in [5, 10, 20]:

    if len(df) < n:
        continue

    top_05 = get_top_n(
        df,
        "Rank_0.5_0.5",
        n
    )

    top_07 = get_top_n(
        df,
        "Rank_0.7_0.3",
        n
    )

    top_03 = get_top_n(
        df,
        "Rank_0.3_0.7",
        n
    )

    common_all = (
        top_05 &
        top_07 &
        top_03
    )

    union_all = (
        top_05 |
        top_07 |
        top_03
    )

    jaccard = (
        len(common_all) /
        len(union_all)
        if len(union_all) > 0
        else np.nan
    )

    top_n_results.append(
        {
            "TopN": n,
            "Common_All_Three": len(common_all),
            "Common_Fraction": (
                len(common_all) / n
            ),
            "Jaccard_All": jaccard,
            "Common_Compounds": "; ".join(
                sorted(
                    map(
                        str,
                        common_all
                    )
                )
            )
        }
    )

top_n_df = pd.DataFrame(
    top_n_results
)

top_n_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "sensitivity_topN_stability.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 11. Figure 1
# Final Consensus Score Ranking
# ============================================================

plt.figure(
    figsize=(10, 6)
)

colors = [
    COLOR_LIST[
        i % len(COLOR_LIST)
    ]
    for i in range(
        len(df_sorted)
    )
]

bars = plt.bar(
    df_sorted["Compound"],
    df_sorted["FinalScore"],
    color=colors,
    width=0.6
)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() +
        bar.get_width() / 2,
        height + 0.02,
        f'{height:.3f}',
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.title(
    'MM/GBSA + Gnina Consensus Score Ranking',
    fontweight='bold'
)

plt.xlabel(
    'Compound ID',
    fontweight='medium'
)

plt.ylabel(
    'Final Consensus Score',
    fontweight='medium'
)

plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.7
)

max_score = df_sorted["FinalScore"].max()

plt.ylim(
    0,
    max_score * 1.15
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        '1_consensus_score_ranking.png'
    ),
    dpi=300
)

plt.close()

# ============================================================
# 12. Figure 2
# GBSA vs GninaScore Dual Axis Plot
# ============================================================

fig, ax1 = plt.subplots(
    figsize=(10, 6)
)

ax1.bar(
    df_sorted["Compound"],
    df_sorted["GBSA_positive"],
    color=colors,
    alpha=0.7,
    width=0.5,
    label='GBSA Positive'
)

ax1.set_xlabel(
    'Compound ID',
    fontweight='medium'
)

ax1.set_ylabel(
    'GBSA Positive Value (-GBSA)',
    fontweight='medium'
)

ax1.grid(
    axis='y',
    linestyle='--',
    alpha=0.5
)

ax2 = ax1.twinx()

ax2.plot(
    df_sorted["Compound"],
    df_sorted["GninaScore"],
    marker='o',
    linewidth=2,
    markersize=7,
    color='black',
    markerfacecolor='black',
    markeredgecolor='black',
    label='GninaScore'
)

ax2.set_ylabel(
    'GninaScore (CNN_affinity × CNN_pose)',
    fontweight='medium'
)

lines1, labels1 = (
    ax1.get_legend_handles_labels()
)

lines2, labels2 = (
    ax2.get_legend_handles_labels()
)

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc='upper left'
)

plt.title(
    'GBSA vs GninaScore Comparison',
    fontweight='bold'
)

ax1.tick_params(
    axis='x',
    rotation=45
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        '2_gbsa_gnina_comparison.png'
    ),
    dpi=300
)

plt.close()

# ============================================================
# 13. Figure 3
# Normalized Score Scatter Plot
# ============================================================

plt.figure(
    figsize=(8, 8)
)

scatter = plt.scatter(
    df_sorted["Norm_GBSA"],
    df_sorted["Norm_Gnina"],
    c=df_sorted["FinalScore"],
    cmap='viridis',
    s=150,
    alpha=0.8,
    edgecolors='k'
)

for _, row in df_sorted.iterrows():

    plt.text(
        row["Norm_GBSA"] + 0.02,
        row["Norm_Gnina"] + 0.02,
        str(row["Compound"]),
        fontsize=9,
        fontweight='medium'
    )

plt.title(
    'Normalized GBSA vs GninaScore',
    fontweight='bold'
)

plt.xlabel(
    'Normalized GBSA Score',
    fontweight='medium'
)

plt.ylabel(
    'Normalized Gnina Score',
    fontweight='medium'
)

plt.grid(
    linestyle='--',
    alpha=0.7
)

plt.xlim(
    -0.05,
    1.05
)

plt.ylim(
    -0.05,
    1.05
)

plt.colorbar(
    scatter,
    label='Final Consensus Score'
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        '3_normalized_score_scatter.png'
    ),
    dpi=300
)

plt.close()

# ============================================================
# 14. Figure 4
# Horizontal Score Bar Plot
# ============================================================

plt.figure(
    figsize=(10, 6)
)

df_reversed = (
    df_sorted.iloc[::-1]
)

colors_rev = [
    COLOR_LIST[
        i % len(COLOR_LIST)
    ]
    for i in range(
        len(df_reversed)
    )
]

bars = plt.barh(
    df_reversed["Compound"],
    df_reversed["FinalScore"],
    color=colors_rev,
    height=0.6
)

for bar in bars:

    width = bar.get_width()

    plt.text(
        width + 0.02,
        bar.get_y() +
        bar.get_height() / 2,
        f'{width:.3f}',
        ha='left',
        va='center',
        fontsize=9
    )

plt.title(
    'Final Consensus Score Ranking (Horizontal)',
    fontweight='bold'
)

plt.xlabel(
    'Final Consensus Score',
    fontweight='medium'
)

plt.ylabel(
    'Compound ID',
    fontweight='medium'
)

plt.grid(
    axis='x',
    linestyle='--',
    alpha=0.7
)

plt.xlim(
    0,
    max_score * 1.15
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        '4_horizontal_score_ranking.png'
    ),
    dpi=300
)

plt.close()

# ============================================================
# 15. Figure 5
# Ranking Sensitivity Plot
# ============================================================

plt.figure(
    figsize=(10, 7)
)

x_positions = [
    0,
    1,
    2
]

x_labels = [
    "0.5 / 0.5",
    "0.7 / 0.3",
    "0.3 / 0.7"
]

for _, row in sensitivity_df.iterrows():

    ranks = [
        row["Rank_0.5_0.5"],
        row["Rank_0.7_0.3"],
        row["Rank_0.3_0.7"]
    ]

    plt.plot(
        x_positions,
        ranks,
        marker="o",
        markersize=5,
        linewidth=1.2,
        alpha=0.65
    )

    plt.text(
        1.03,
        row["Rank_0.7_0.3"],
        str(row["Compound"]),
        fontsize=7,
        va="center"
    )

plt.xticks(
    x_positions,
    x_labels
)

plt.xlabel(
    "GBSA / Gnina Weight",
    fontweight="medium"
)

plt.ylabel(
    "Compound Rank",
    fontweight="medium"
)

plt.title(
    "Sensitivity Analysis of Consensus Ranking",
    fontweight="bold"
)

plt.gca().invert_yaxis()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "5_ranking_sensitivity.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 16. Figure 6
# Spearman Rank Correlation Heatmap
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 5)
)

corr_matrix = (
    spearman_corr.values
)

im = ax.imshow(
    corr_matrix,
    vmin=0,
    vmax=1,
    cmap="viridis"
)

labels = [
    "0.5 / 0.5",
    "0.7 / 0.3",
    "0.3 / 0.7"
]

ax.set_xticks(
    np.arange(
        len(labels)
    )
)

ax.set_yticks(
    np.arange(
        len(labels)
    )
)

ax.set_xticklabels(
    labels
)

ax.set_yticklabels(
    labels
)

for i in range(
    len(labels)
):
    for j in range(
        len(labels)
    ):

        ax.text(
            j,
            i,
            f"{corr_matrix[i, j]:.3f}",
            ha="center",
            va="center",
            fontsize=11
        )

ax.set_xlabel(
    "Weight Scheme",
    fontweight="bold",
    fontsize=12
)

ax.set_ylabel(
    "Weight Scheme",
    fontweight="bold",
    fontsize=12
)

ax.set_title(
    "Spearman Rank Correlation",
    fontweight="bold",
    fontsize=14
)

cbar = plt.colorbar(
    im,
    ax=ax
)

cbar.set_label(
    "Spearman ρ",
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "6_spearman_rank_correlation.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 17. Figure 7
# Maximum Rank Shift
# ============================================================

rank_shift_plot = (
    sensitivity_df
    .sort_values(
        "MaxRankShift",
        ascending=True
    )
)

plt.figure(
    figsize=(9, 7)
)

plt.barh(
    rank_shift_plot["Compound"],
    rank_shift_plot["MaxRankShift"]
)

plt.xlabel(
    "Maximum Absolute Rank Shift",
    fontweight="medium"
)

plt.ylabel(
    "Compound ID",
    fontweight="medium"
)

plt.title(
    "Maximum Rank Shift Across Weighting Schemes",
    fontweight="bold"
)

plt.grid(
    axis="x",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "7_maximum_rank_shift.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 18. Overall Stability Summary
# ============================================================

off_diagonal = []

for i in range(
    len(spearman_corr)
):
    for j in range(
        i + 1,
        len(spearman_corr)
    ):

        off_diagonal.append(
            spearman_corr.iloc[
                i,
                j
            ]
        )

mean_spearman = np.mean(
    off_diagonal
)

median_rank_shift = (
    sensitivity_df[
        "MaxRankShift"
    ].median()
)

max_rank_shift = (
    sensitivity_df[
        "MaxRankShift"
    ].max()
)

summary_df = pd.DataFrame(
    {
        "Metric": [
            "Mean pairwise Spearman rho",
            "Median maximum rank shift",
            "Largest observed rank shift"
        ],
        "Value": [
            mean_spearman,
            median_rank_shift,
            max_rank_shift
        ]
    }
)

summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "sensitivity_summary.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 19. Console Output
# ============================================================

print(
    "\n========================================"
)

print(
    "Calculation completed successfully!"
)

print(
    "========================================\n"
)

print(
    f"All results have been saved to:\n"
    f"{os.path.abspath(OUTPUT_DIR)}\n"
)

print(
    "Main consensus ranking "
    "(GBSA/Gnina = 0.7/0.3):\n"
)

print(
    df_sorted[
        [
            "Compound",
            "GBSA",
            "GninaScore",
            "FinalScore",
            "Rank"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\n========================================"
)

print(
    "Sensitivity Analysis"
)

print(
    "========================================\n"
)

print(
    sensitivity_df[
        [
            "Compound",
            "Rank_0.5_0.5",
            "Rank_0.7_0.3",
            "Rank_0.3_0.7",
            "DeltaRank_0.5_0.5",
            "DeltaRank_0.3_0.7",
            "MaxRankShift"
        ]
    ].to_string(
        index=False
    )
)

print(
    "\n========================================"
)

print(
    "Spearman Rank Correlation"
)

print(
    "========================================\n"
)

print(
    spearman_corr
)

print(
    "\n========================================"
)

print(
    "Top-N Stability"
)

print(
    "========================================\n"
)

if len(top_n_df) > 0:

    print(
        top_n_df.to_string(
            index=False
        )
    )

else:

    print(
        "Dataset is too small "
        "for Top-5/10/20 analysis."
    )

print(
    "\n========================================"
)

print(
    "Overall Stability Summary"
)

print(
    "========================================"
)

print(
    f"Mean pairwise Spearman rho: "
    f"{mean_spearman:.3f}"
)

print(
    f"Median maximum rank shift: "
    f"{median_rank_shift:.1f}"
)

print(
    f"Largest observed rank shift: "
    f"{max_rank_shift:.0f}"
)

print(
    "\nGenerated files:"
)

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):
    print(
        f" - {filename}"
    )