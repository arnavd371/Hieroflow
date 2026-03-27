from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from results.plot_style import PALETTE, METHOD_LABELS, apply_style, save_fig, set_output_dir
from results.stats.significance import bootstrap_ci

CAPTION = (
    "Figure 2: Diversity of generated proofs. HieroFlow produces proofs that are "
    "more diverse at tactic, subgoal, and semantic levels than all baselines, "
    "confirming that hierarchical sketch-level planning improves strategy coverage."
)

ORDER = ["hieroflow", "gfn_flat", "rl_baseline", "supervised"]
METRICS = [
    ("tactic_diversity", "Tactic diversity", "Tactic entropy (bits)", (0, 3)),
    ("unique_subgoal_rate", "Subgoal diversity", "Unique subgoal rate", (0, 1)),
    ("semantic_diversity", "Semantic diversity", "Semantic diversity", (0, 1)),
]


def _values(df, method: str, metric: str):
    per_seed = df[df["method"] == method].groupby("seed")[metric].mean().to_numpy(dtype=float)
    m = float(np.mean(per_seed)) if per_seed.size else 0.0
    lo, hi = bootstrap_ci(per_seed)
    return m, (max(0.0, m - lo), max(0.0, hi - m))


def make_figure(df, output_dir=None):
    apply_style()
    if output_dir:
        set_output_dir(output_dir)

    fig, axes = plt.subplots(1, 3, sharey=True)

    y = np.arange(len(ORDER), dtype=float)
    y_rev = y[::-1]

    rl_reference = {}
    for i, (metric, panel_title, xlabel, xlim) in enumerate(METRICS):
        ax = axes[i]
        means, errs = [], []
        for method in ORDER:
            mean, (err_lo, err_hi) = _values(df, method, metric)
            means.append(mean)
            errs.append((err_lo, err_hi))
            if method == "rl_baseline":
                rl_reference[metric] = mean

        errs_arr = np.array(errs).T
        ax.barh(
            y_rev,
            means,
            xerr=errs_arr,
            color=[PALETTE[m] for m in ORDER],
            edgecolor="black",
            linewidth=0.5,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "capsize": 2},
        )
        ax.set_xlim(*xlim)
        ax.set_title(panel_title, fontsize=9)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_yticks(y_rev)
            ax.set_yticklabels([METHOD_LABELS[m] for m in ORDER])
        else:
            ax.set_yticks(y_rev)
            ax.set_yticklabels([])

        if metric in {"unique_subgoal_rate", "semantic_diversity"}:
            ax.axvline(rl_reference[metric], linestyle="--", color=PALETTE["rl_baseline"], linewidth=0.8, alpha=0.8)

    legend_items = [Patch(facecolor=PALETTE[m], edgecolor="black", label=METHOD_LABELS[m]) for m in ORDER]
    fig.legend(handles=legend_items, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))

    outputs = save_fig(fig, "fig2_diversity", double_col=False)
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    from results.data.schema import make_synthetic_data

    df = make_synthetic_data()
    make_figure(df)
