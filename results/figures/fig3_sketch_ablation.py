from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from results.plot_style import PALETTE, apply_style, save_fig, set_output_dir
from results.stats.significance import bootstrap_ci

CAPTION = (
    "Figure 3: Ablation over maximum sketch depth. Both proof success and semantic "
    "diversity improve as sketch depth increases from 0, plateauing at depth 3. "
    "Depth 0 recovers the flat GFlowNet baseline."
)


def _depth_values(df, metric: str):
    out = {}
    for depth in range(6):
        if depth == 0:
            subset = df[df["method"] == "gfn_flat"]
        else:
            subset = df[(df["method"] == "hieroflow") & (df["sketch_depth"] == depth)]
        vals = subset.groupby("seed")[metric].mean().to_numpy(dtype=float)
        mean = float(np.mean(vals)) if vals.size else 0.0
        lo, hi = bootstrap_ci(vals)
        out[depth] = (mean, lo, hi)
    return out


def make_figure(df, output_dir=None):
    apply_style()
    if output_dir:
        set_output_dir(output_dir)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    depths = np.arange(6)
    succ = _depth_values(df, "success")
    sem = _depth_values(df, "semantic_diversity")

    succ_mean = np.array([succ[d][0] for d in depths])
    succ_lo = np.array([succ[d][1] for d in depths])
    succ_hi = np.array([succ[d][2] for d in depths])

    sem_mean = np.array([sem[d][0] for d in depths])
    sem_lo = np.array([sem[d][1] for d in depths])
    sem_hi = np.array([sem[d][2] for d in depths])

    ax1.plot(depths, succ_mean * 100, color=PALETTE["hieroflow"], marker="o")
    ax1.fill_between(depths, succ_lo * 100, succ_hi * 100, color=PALETTE["hieroflow"], alpha=0.15)

    ax2.plot(depths, sem_mean, color=PALETTE["rl_baseline"], marker="s")
    ax2.fill_between(depths, sem_lo, sem_hi, color=PALETTE["rl_baseline"], alpha=0.15)

    ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.text(0.05, 0.96, "Flat GFlowNet", transform=ax1.transAxes, fontsize=8, color="gray", va="top")

    best_idx = int(np.argmax(succ_mean))
    ax1.plot(depths[best_idx], succ_mean[best_idx] * 100, marker="*", markersize=10, color=PALETTE["hieroflow"])

    flat_success = succ_mean[0] * 100
    gain = succ_mean[best_idx] * 100 - flat_success
    ax1.text(
        0.55,
        0.28,
        f"Optimal depth\\n+{gain:.1f}pp vs flat",
        transform=ax1.transAxes,
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "lightgrey"},
    )

    ax1.set_xticks(depths)
    ax1.set_xticklabels(["0\\n(flat)", "1", "2", "3", "4", "5"])
    ax1.set_xlabel("Sketch depth limit")
    ax1.set_ylabel("Proof success rate (%)", color=PALETTE["hieroflow"])
    ax2.set_ylabel("Semantic diversity", color=PALETTE["rl_baseline"])

    outputs = save_fig(fig, "fig3_sketch_ablation", double_col=False)
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    from results.data.schema import make_synthetic_data

    df = make_synthetic_data()
    make_figure(df)
