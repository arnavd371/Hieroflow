from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from results.plot_style import PALETTE, METHOD_LABELS, apply_style, save_fig, set_output_dir
from results.stats.significance import bootstrap_ci

CAPTION = (
    "Figure 8: Per-benchmark success-rate heatmap. Each cell shows the mean proof "
    "success rate (%) averaged over 5 seeds. HieroFlow leads on every benchmark–metric "
    "combination; the relative advantage is largest on the harder ProofNet split."
)

METHODS = ["hieroflow", "gfn_flat", "rl_baseline", "supervised"]
BENCHMARKS = ["leandojo_mathlib", "minif2f", "proofnet"]
BENCHMARK_LABELS = {
    "leandojo_mathlib": "LeanDojo\nMathlib",
    "minif2f": "MiniF2F",
    "proofnet": "ProofNet",
}


def make_figure(df, output_dir=None):
    apply_style()
    if output_dir:
        set_output_dir(output_dir)

    # Build the success-rate matrix: rows=methods, cols=benchmarks
    mat = np.zeros((len(METHODS), len(BENCHMARKS)))
    ci_lo = np.zeros_like(mat)
    ci_hi = np.zeros_like(mat)

    benchmarks_present = [b for b in BENCHMARKS if b in set(df["benchmark"])]

    for i, method in enumerate(METHODS):
        for j, bench in enumerate(benchmarks_present):
            vals = (
                df[(df["method"] == method) & (df["benchmark"] == bench)]
                .groupby("seed")["success"]
                .mean()
                .to_numpy(dtype=float)
            )
            m = float(np.mean(vals)) if vals.size else 0.0
            lo, hi = bootstrap_ci(vals)
            mat[i, j] = m * 100
            ci_lo[i, j] = lo * 100
            ci_hi[i, j] = hi * 100

    fig, axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]})
    fig.subplots_adjust(left=0.22, right=0.97, bottom=0.15, top=0.90, wspace=0.08)

    # Left panel: heatmap
    ax = axes[0]
    vmin, vmax = 35, 75
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.Blues

    im = ax.imshow(mat[:, :len(benchmarks_present)], cmap=cmap, norm=norm,
                   aspect="auto")

    # Annotate cells with value ± half-CI
    for i in range(len(METHODS)):
        for j in range(len(benchmarks_present)):
            val = mat[i, j]
            half_ci = (ci_hi[i, j] - ci_lo[i, j]) / 2
            text = f"{val:.1f}\n±{half_ci:.1f}"
            # Choose text color based on background brightness
            cell_norm = norm(val)
            text_color = "white" if cell_norm > 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5,
                    color=text_color, linespacing=1.3)

    ax.set_xticks(range(len(benchmarks_present)))
    ax.set_xticklabels([BENCHMARK_LABELS[b] for b in benchmarks_present], fontsize=8)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=8)
    ax.set_title("Success rate (%)", fontsize=8, pad=4)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # Right panel: average across benchmarks (bar chart)
    ax2 = axes[1]
    avg = mat[:, :len(benchmarks_present)].mean(axis=1)
    colors = [PALETTE[m] for m in METHODS]
    bars = ax2.barh(range(len(METHODS)), avg, color=colors, edgecolor="black",
                    linewidth=0.5)
    for bar, val in zip(bars, avg):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", ha="left", fontsize=7.5)
    ax2.set_yticks([])
    ax2.set_xlabel("Avg. success (%)", fontsize=8)
    ax2.set_xlim(0, max(avg) + 10)
    ax2.invert_yaxis()
    ax2.set_title("Average", fontsize=8, pad=4)

    outputs = save_fig(fig, "fig8_per_benchmark", double_col=True)
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    from results.data.schema import make_synthetic_data

    df = make_synthetic_data()
    make_figure(df)
