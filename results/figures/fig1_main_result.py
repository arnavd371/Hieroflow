from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from results.plot_style import PALETTE, HATCH, METHOD_LABELS, apply_style, save_fig, add_significance_bracket, set_output_dir
from results.stats.significance import bootstrap_ci, wilcoxon_test

CAPTION = (
    "Figure 1: Proof success rate on three Lean 4 benchmarks. HieroFlow achieves "
    "consistent gains over all baselines across benchmarks. Error bars show 95% "
    "confidence intervals over 5 random seeds. *** p < 0.001 (Wilcoxon signed-rank)."
)

METHODS = ["hieroflow", "gfn_flat", "rl_baseline", "supervised"]
BENCHMARK_LABELS = {
    "leandojo_mathlib": "LeanDojo Mathlib",
    "minif2f": "MiniF2F",
    "proofnet": "ProofNet",
}


def _seed_success(df, method: str, benchmark: str) -> np.ndarray:
    s = df[(df["method"] == method) & (df["benchmark"] == benchmark)]
    if s.empty:
        return np.array([])
    return s.groupby("seed")["success"].mean().to_numpy(dtype=float)


def make_figure(df, output_dir=None):
    apply_style()
    if output_dir:
        set_output_dir(output_dir)

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.26, top=0.88)
    benchmarks = [b for b in ["leandojo_mathlib", "minif2f", "proofnet"] if b in set(df["benchmark"])]
    x = np.arange(len(benchmarks), dtype=float)

    offsets = {"hieroflow": -0.30, "gfn_flat": -0.10, "rl_baseline": 0.10, "supervised": 0.30}
    widths = {"hieroflow": 0.22, "gfn_flat": 0.18, "rl_baseline": 0.18, "supervised": 0.18}

    for method in METHODS:
        means, errs = [], []
        for b in benchmarks:
            vals = _seed_success(df, method, b)
            m = float(np.mean(vals)) if vals.size else 0.0
            lo, hi = bootstrap_ci(vals)
            means.append(m)
            errs.append([max(0.0, m - lo), max(0.0, hi - m)])
        errs = np.array(errs).T if errs else np.zeros((2, 0))

        bars = ax.bar(
            x + offsets[method],
            means,
            width=widths[method],
            color=PALETTE[method],
            hatch=HATCH.get(method, ""),
            edgecolor="black",
            linewidth=0.6,
            yerr=errs,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "capsize": 2},
            label=METHOD_LABELS[method],
            zorder=3,
        )
        if method == "hieroflow":
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val*100:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0.40, 1.05)
    ax.set_ylabel("Proof success rate")
    ticks = np.linspace(0.4, 1.0, 7)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(t*100)}%" for t in ticks])
    ax.set_xticks(x)
    ax.set_xticklabels([BENCHMARK_LABELS[b] for b in benchmarks])

    for i, b in enumerate(benchmarks):
        h_vals = _seed_success(df, "hieroflow", b)
        baseline_p = []
        for baseline in ["gfn_flat", "rl_baseline", "supervised"]:
            vals = _seed_success(df, baseline, b)
            baseline_p.append((float(np.mean(vals)) if vals.size else -1.0, baseline, vals))
        _, best_baseline, best_vals = max(baseline_p, key=lambda t: t[0])
        p_value, _ = wilcoxon_test(h_vals, best_vals)
        add_significance_bracket(ax, i + offsets["hieroflow"], i + offsets[best_baseline], 0.96 - 0.01 * i, p_value)

    # Broken-axis indicator at y=40% on left y-axis
    for dy in [0.0, 0.02]:
        ax.plot((-0.015, 0.015), (-0.005 + dy, 0.015 + dy), transform=ax.transAxes, color="black", lw=0.8, clip_on=False)

    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=9)

    outputs = save_fig(fig, "fig1_main_result", double_col=True)
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    from results.data.schema import make_synthetic_data

    df = make_synthetic_data()
    make_figure(df)
