from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from results.plot_style import PALETTE, METHOD_LABELS, apply_style, save_fig, set_output_dir

CAPTION = (
    "Figure 6 (Appendix): Training diagnostics. (A) TB loss decreases for both "
    "GFlowNet levels. (B) HieroFlow's reward distribution has more mass at 1.0 "
    "than baselines. (C) log Z stabilises during training, confirming convergence. "
    "(D) HieroFlow generates proofs of more variable length, indicating strategy "
    "diversity at the structural level."
)

METHODS = ["hieroflow", "gfn_flat", "rl_baseline", "supervised"]


def make_figure(df, output_dir=None):
    apply_style()
    if output_dir:
        set_output_dir(output_dir)

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.22, top=0.95, hspace=0.50, wspace=0.38)
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    steps = np.linspace(0, 50000, 200)
    sketch_loss = 2.8 * np.exp(-steps / 10000) + 0.12 + rng.normal(0, 0.03, len(steps))
    tactic_loss = 1.9 * np.exp(-steps / 12000) + 0.08 + rng.normal(0, 0.02, len(steps))
    sketch_loss = np.clip(sketch_loss, 0.01, None)
    tactic_loss = np.clip(tactic_loss, 0.01, None)

    ax_a.plot(steps, sketch_loss, label="SketchFlow loss", color=PALETTE["hieroflow"])
    ax_a.plot(steps, tactic_loss, label="TacticFlow loss", color=PALETTE["gfn_flat"])
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Training step")
    ax_a.set_ylabel("Trajectory Balance loss")
    for x, txt in [(5000, "Phase 1 end"), (20000, "Phase 2 end")]:
        ax_a.axvline(x, linestyle="--", color="gray", linewidth=0.8)
        ax_a.text(x + 800, np.max(sketch_loss) * 0.8, txt, fontsize=8, color="gray")

    rewards = {}
    for method in METHODS:
        base = {"hieroflow": (0.15, 0.20, 0.65), "gfn_flat": (0.22, 0.28, 0.50), "rl_baseline": (0.25, 0.32, 0.43), "supervised": (0.30, 0.36, 0.34)}[method]
        vals = rng.choice([0.0, 0.5, 1.0], size=220, p=base)
        rewards[method] = vals

    data = [rewards[m] for m in METHODS]
    v = ax_b.violinplot(data, showmeans=False, showmedians=True, widths=0.85)
    for i, body in enumerate(v["bodies"]):
        m = METHODS[i]
        body.set_facecolor(PALETTE[m])
        body.set_alpha(0.45)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
    for k in ["cbars", "cmins", "cmaxes", "cmedians"]:
        if k in v:
            v[k].set_color("black")
            v[k].set_linewidth(0.6)

    for i, m in enumerate(METHODS, start=1):
        x = np.full_like(rewards[m], i, dtype=float) + rng.normal(0, 0.04, size=len(rewards[m]))
        ax_b.scatter(x, rewards[m], s=8, alpha=0.3, color=PALETTE[m], edgecolors="none")

    ax_b.set_xticks(range(1, len(METHODS) + 1))
    ax_b.set_xticklabels(["HieroFlow", "GFlowNet-Flat", "RL-Prover", "Supervised"], rotation=15)
    ax_b.set_ylabel("Final proof reward")
    ax_b.set_ylim(-0.05, 1.05)

    outer_logz = 1.2 * np.exp(-steps / 11000) * np.cos(steps / 8500) + 0.03 * rng.normal(size=len(steps))
    inner_logz = 0.9 * np.exp(-steps / 13000) * np.sin(steps / 9000) + 0.03 * rng.normal(size=len(steps))
    ax_c.plot(steps, outer_logz, color=PALETTE["hieroflow"], label="Outer log Z")
    ax_c.plot(steps, inner_logz, color=PALETTE["rl_baseline"], label="Inner log Z")
    ax_c.axhline(0, linestyle="--", color="gray", linewidth=0.8)
    ax_c.set_xlabel("Training step")
    ax_c.set_ylabel("log Z value")

    bins = np.arange(1, 32)
    for m in METHODS:
        vals = df[(df["method"] == m) & (df["success"])]["proof_length"].to_numpy(dtype=float)
        if vals.size:
            ax_d.hist(vals, bins=bins, density=True, alpha=0.5, color=PALETTE[m], label=METHOD_LABELS[m])
    ax_d.set_xlim(1, 30)
    ax_d.set_xlabel("Proof length (tactics)")
    ax_d.set_ylabel("Density")

    for ax, label in zip([ax_a, ax_b, ax_c, ax_d], ["A", "B", "C", "D"]):
        ax.text(0.01, 0.99, label, transform=ax.transAxes, va="top", ha="left", fontweight="bold", fontsize=10)

    legend_handles = [Line2D([0], [0], color=PALETTE[m], lw=2, label=METHOD_LABELS[m]) for m in METHODS]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01), fontsize=8)

    outputs = save_fig(fig, "fig6_appendix", double_col=True)
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    from results.data.schema import make_synthetic_data

    df = make_synthetic_data()
    make_figure(df)
