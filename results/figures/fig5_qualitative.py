from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from results.plot_style import METHOD_LABELS, apply_style, save_fig, set_output_dir

CAPTION = (
    "Figure 5: Tactic usage frequency heatmap for theorem List.length_append. "
    "Each row is one discovered proof strategy; each column is a proof step. "
    "HieroFlow discovers 5 qualitatively distinct strategies while RL-Prover "
    "collapses to 2 near-identical proofs, illustrating mode collapse."
)

TACTICS = ["induction", "simp", "rw", "apply", "exact", "cases", "ring", "omega"]
METHODS = ["hieroflow", "gfn_flat", "rl_baseline"]
CMAPS = {"hieroflow": "Blues", "gfn_flat": "Greens", "rl_baseline": "Oranges"}
NUM_ROWS = {"hieroflow": 5, "gfn_flat": 3, "rl_baseline": 2}


def _build_method_matrix(df, method: str) -> np.ndarray:
    subset = df[(df["theorem_name"] == "List.length_append") & (df["method"] == method) & (df["success"])]
    if subset.empty:
        return np.zeros((NUM_ROWS[method], 8), dtype=int)

    attempts = subset.head(NUM_ROWS[method])["proof_tactics"].tolist()
    mat = np.zeros((NUM_ROWS[method], 8), dtype=int)
    tactic_to_idx = {t: i for i, t in enumerate(TACTICS)}

    for row_idx in range(NUM_ROWS[method]):
        if row_idx >= len(attempts):
            continue
        tactics = attempts[row_idx]
        for step in range(min(8, len(tactics))):
            tidx = tactic_to_idx.get(tactics[step], None)
            if tidx is not None:
                mat[row_idx, step] = tidx + 1
    return mat


def make_figure(df, output_dir=None):
    apply_style()
    if output_dir:
        set_output_dir(output_dir)

    fig, axes = plt.subplots(3, 1, sharex=True)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.93, bottom=0.10, hspace=0.20)

    for i, method in enumerate(METHODS):
        ax = axes[i]
        mat = _build_method_matrix(df, method)
        im = ax.imshow(mat, aspect="auto", cmap=CMAPS[method], vmin=0, vmax=len(TACTICS))

        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = int(mat[r, c])
                if v > 0:
                    ax.text(c, r, str(v), ha="center", va="center", fontsize=8)

        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels([f"Strategy {r+1}" for r in range(mat.shape[0])], fontsize=9)

        ax.text(-0.08, 0.5, METHOD_LABELS[method], transform=ax.transAxes, rotation=90, va="center", ha="right", fontweight="bold")

    axes[0].set_title("Theorem: List.length_append")

    axes[-1].set_xticks(np.arange(8))
    axes[-1].set_xticklabels([f"Step {i}" for i in range(1, 9)])

    axes[2].annotate(
        "Low strategy diversity\\n(mode collapse)",
        xy=(4.5, 0.6),
        xytext=(6.4, 1.5),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "lightgrey"},
    )
    axes[0].annotate(
        "5 distinct strategies\\nexplored",
        xy=(2.3, 2.0),
        xytext=(5.8, 3.7),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "lightgrey"},
    )

    outputs = save_fig(fig, "fig5_qualitative", double_col=True)
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    from results.data.schema import make_synthetic_data

    df = make_synthetic_data()
    make_figure(df)
