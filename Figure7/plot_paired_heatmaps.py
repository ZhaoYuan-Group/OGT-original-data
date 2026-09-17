from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / ".python_plot_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

MPL_CONFIG_DIR = ROOT / ".matplotlib"
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

from plot_descriptor_family_comparison import MODEL_FILES, build_source_data


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7


OUT = ROOT / "outputs" / "figure_styles"
SHORT_FAMILIES = [
    "Morgan / RDKit",
    "Atom-pair",
    "Topological torsion",
    "Auxiliary fingerprints",
    "Physicochemical",
]


def text_color(cmap, norm, value: float) -> str:
    r, g, b, _ = cmap(norm(value))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.56 else "#203047"


def add_vector_colorbar(fig, cmap, bounds: list[float]) -> None:
    """Add a vector-only colour scale, avoiding an embedded raster gradient."""
    cax = fig.add_axes(bounds)
    segments = 65
    for index in range(segments):
        lower = index * 65 / segments
        height = 65 / segments
        patch = plt.Rectangle((0, lower), 1, height,
                              facecolor=cmap((lower + height / 2) / 65),
                              edgecolor="none")
        cax.add_patch(patch)
    cax.set_xlim(0, 1)
    cax.set_ylim(0, 65)
    cax.set_xticks([])
    cax.set_yticks([0, 20, 40, 60])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.set_ylabel("Relative importance (%)", fontsize=6.5, labelpad=5)
    cax.tick_params(axis="y", labelsize=6.0, length=2)
    for spine in cax.spines.values():
        spine.set_visible(False)


def draw_pair(
    matrix: np.ndarray,
    model_names: list[str],
    title: str,
    stem: str,
    colors: list[str],
) -> None:
    cmap = LinearSegmentedColormap.from_list(stem, colors)
    norm = Normalize(vmin=0, vmax=65)

    # Portrait layout: descriptor families form the vertical axis and the two models form columns.
    display = matrix.T
    fig, ax = plt.subplots(figsize=(3.55, 5.10), facecolor="white")
    fig.subplots_adjust(left=0.34, right=0.80, top=0.76, bottom=0.10)
    fig.text(0.035, 0.975, title, ha="left", va="top", fontsize=10,
             fontweight="bold", color="#22324A")

    # pcolormesh writes each heatmap cell as vector geometry in the SVG,
    # rather than embedding a raster image.
    im = ax.pcolormesh(display, cmap=cmap, norm=norm, shading="flat")
    ax.set_xlim(0, display.shape[1])
    ax.set_ylim(display.shape[0], 0)
    ax.set_xticks(np.arange(len(model_names)) + 0.5, model_names,
                  fontsize=7.2, fontweight="bold")
    ax.set_yticks(np.arange(len(SHORT_FAMILIES)) + 0.5, SHORT_FAMILIES,
                  fontsize=6.8)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False,
                   labelbottom=False, pad=5, length=0)
    ax.tick_params(axis="y", length=0)

    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            value = display[i, j]
            ax.text(j + 0.5, i + 0.5, f"{value:.1f}", ha="center", va="center",
                    fontsize=7.0, color=text_color(cmap, norm, value),
                    fontweight="bold" if value >= 30 else "normal")

    for spine in ax.spines.values():
        spine.set_visible(False)

    add_vector_colorbar(fig, cmap, [0.85, 0.10, 0.024, 0.66])

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / f"{stem}.png", dpi=420, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def draw_pair_horizontal(
    matrix: np.ndarray,
    model_names: list[str],
    title: str,
    stem: str,
    colors: list[str],
) -> None:
    """Landscape counterpart: models form rows and descriptor families form columns."""
    cmap = LinearSegmentedColormap.from_list(stem, colors)
    norm = Normalize(vmin=0, vmax=65)

    fig, ax = plt.subplots(figsize=(7.15, 3.55), facecolor="white")
    fig.subplots_adjust(left=0.18, right=0.84, top=0.71, bottom=0.17)
    fig.text(0.035, 0.975, title, ha="left", va="top", fontsize=10,
             fontweight="bold", color="#22324A")

    # pcolormesh preserves every colored cell as editable vector geometry.
    im = ax.pcolormesh(matrix, cmap=cmap, norm=norm, shading="flat")
    ax.set_xlim(0, matrix.shape[1])
    ax.set_ylim(matrix.shape[0], 0)
    ax.set_xticks(np.arange(len(SHORT_FAMILIES)) + 0.5, SHORT_FAMILIES,
                  fontsize=6.8, fontweight="bold", rotation=18, ha="left",
                  rotation_mode="anchor")
    ax.set_yticks(np.arange(len(model_names)) + 0.5, model_names,
                  fontsize=7.2, fontweight="bold")
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False,
                   labelbottom=False, pad=9, length=0)
    ax.tick_params(axis="y", length=0, pad=6)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j + 0.5, i + 0.5, f"{value:.1f}", ha="center", va="center",
                    fontsize=7.0, color=text_color(cmap, norm, value),
                    fontweight="bold" if value >= 30 else "normal")

    for spine in ax.spines.values():
        spine.set_visible(False)

    add_vector_colorbar(fig, cmap, [0.865, 0.17, 0.018, 0.54])

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / f"{stem}.png", dpi=420, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    shares_by_model, _ = build_source_data(ROOT / "outputs")
    models = list(MODEL_FILES)
    data = np.vstack([shares_by_model[model] for model in models]) * 100

    draw_pair(
        data[[0, 1]],
        [models[0], "Transfer_MLP"],
        "Ridge and Transfer_MLP",
        "08_ridge_transfer_mlp_heatmap_vertical",
        ["#FFFAF5", "#FAD5B4", "#ED984E", "#A94B12"],
    )
    draw_pair_horizontal(
        data[[0, 1]],
        [models[0], "Transfer_MLP"],
        "Ridge and Transfer_MLP",
        "10_ridge_transfer_mlp_heatmap_horizontal",
        ["#FFFAF5", "#FAD5B4", "#ED984E", "#A94B12"],
    )
    draw_pair(
        data[[2, 3]],
        [models[2], models[3]],
        "LightGBM and SVM",
        "09_lightgbm_svm_heatmap_vertical",
        ["#F7FAFC", "#C8D9EA", "#6D9BC4", "#214E7A"],
    )
    print("Created two paired heatmaps")


if __name__ == "__main__":
    main()
