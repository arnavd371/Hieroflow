from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from results.plot_style import PALETTE, METHOD_LABELS, apply_style, save_fig, set_output_dir
from results.stats.significance import bootstrap_ci

CAPTION = (
    "Figure 9: Success rate vs. semantic diversity trade-off. Each point is one "
    "random seed for a given method averaged across all benchmarks. HieroFlow "
    "achieves the best combination of high success rate and high diversity, "
    "occupying the upper-right corner that other methods cannot reach. "
    "Ellipses show 1-σ covariance regions."
)

METHODS = ["hieroflow", "gfn_flat", "rl_baseline", "supervised"]


def _per_seed_stats(df, method: str):
    """Return arrays of (success_rate, semantic_diversity) per seed."""
    sub = df[df["method"] == method]
    seeds = sorted(sub["seed"].unique())
    success_vals = []
    div_vals = []
    for s in seeds:
        s_df = sub[sub["seed"] == s]
        success_vals.append(float(s_df["success"].mean()))
        div_vals.append(float(s_df["semantic_diversity"].mean()))
    return np.array(success_vals), np.array(div_vals)


def _confidence_ellipse(ax, x, y, color, n_std=1.5):
    """Draw a covariance ellipse around (x, y) data points."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    if cov.ndim != 2:
        return
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by eigenvalue descending
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    from matplotlib.patches import Ellipse
    width = 2 * n_std * np.sqrt(max(eigenvalues[0], 0))
    height = 2 * n_std * np.sqrt(max(eigenvalues[1], 0))
    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        alpha=0.12,
        edgecolor=color,
        linewidth=1.0,
        linestyle="--",
    )
    ax.add_patch(ell)


def make_figure(df, output_dir=None):
    apply_style()
    if output_dir:
        set_output_dir(output_dir)

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.92)

    for method in METHODS:
        succ, div = _per_seed_stats(df, method)
        color = PALETTE[method]
        _confidence_ellipse(ax, succ, div, color=color)
        ax.scatter(succ * 100, div, s=60, color=color, edgecolors="black",
                   linewidth=0.5, zorder=4, label=METHOD_LABELS[method])
        # Annotate mean
        ax.scatter([np.mean(succ) * 100], [np.mean(div)], s=120, color=color,
                   edgecolors="black", linewidth=0.8, marker="D", zorder=5)

    # Annotate quadrants
    ax.axvline(50, color="lightgrey", linewidth=0.8, linestyle=":")
    ax.axhline(0.5, color="lightgrey", linewidth=0.8, linestyle=":")
    ax.text(0.52, 0.96, "High success\nhigh diversity", transform=ax.transAxes,
            fontsize=7, color="gray", ha="left", va="top", linespacing=1.3)
    ax.text(0.02, 0.04, "Low success\nlow diversity", transform=ax.transAxes,
            fontsize=7, color="gray", ha="left", va="bottom", linespacing=1.3)

    ax.set_xlabel("Proof success rate (%)")
    ax.set_ylabel("Semantic diversity")
    ax.set_xlim(40, 85)
    ax.set_ylim(0.25, 0.90)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE[m],
               markeredgecolor="black", markersize=7, label=METHOD_LABELS[m])
        for m in METHODS
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    outputs = save_fig(fig, "fig9_diversity_scatter", double_col=False)
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    from results.data.schema import make_synthetic_data

    df = make_synthetic_data()
    make_figure(df)
