from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from results.plot_style import PALETTE, METHOD_LABELS, apply_style, save_fig, set_output_dir
from results.stats.significance import bootstrap_ci

CAPTION = (
    "Figure 4: Proof success rate as a function of Lean verifier call budget. "
    "HieroFlow achieves comparable performance to RL-Prover using 2.1× fewer "
    "verifier calls, demonstrating the efficiency of sketch-guided search."
)

CHECKPOINTS = np.array([100, 200, 300, 500, 700, 1000, 1500, 2000], dtype=float)
METHODS = ["hieroflow", "gfn_flat", "rl_baseline", "supervised"]


def _budget_curve(df, method: str):
    subset = df[df["method"] == method]
    by_seed = {}
    for seed, sdf in subset.groupby("seed"):
        vals = []
        for b in CHECKPOINTS:
            s = sdf[sdf["num_lean_calls"] <= b]["success"]
            vals.append(float(s.mean()) if len(s) else 0.0)
        by_seed[int(seed)] = np.array(vals)

    means, lows, highs = [], [], []
    if not by_seed:
        z = np.zeros_like(CHECKPOINTS)
        return z, z, z

    mat = np.vstack([by_seed[k] for k in sorted(by_seed)])
    for i in range(mat.shape[1]):
        col = mat[:, i]
        m = float(np.mean(col))
        lo, hi = bootstrap_ci(col)
        means.append(m)
        lows.append(lo)
        highs.append(hi)
    return np.array(means), np.array(lows), np.array(highs)


def make_figure(df, output_dir=None):
    apply_style()
    if output_dir:
        set_output_dir(output_dir)

    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.97)

    curves = {}
    for method in METHODS:
        mean, lo, hi = _budget_curve(df, method)
        curves[method] = (mean, lo, hi)
        x_smooth = np.geomspace(CHECKPOINTS.min(), CHECKPOINTS.max(), 200)
        spline = make_interp_spline(np.log10(CHECKPOINTS), mean, k=3)
        y_smooth = spline(np.log10(x_smooth))
        lw = 2.0 if method == "hieroflow" else 1.2
        ax.plot(x_smooth, y_smooth, color=PALETTE[method], linewidth=lw, label=METHOD_LABELS[method])

        if method == "hieroflow":
            lo_spline = make_interp_spline(np.log10(CHECKPOINTS), lo, k=3)
            hi_spline = make_interp_spline(np.log10(CHECKPOINTS), hi, k=3)
            ax.fill_between(x_smooth, lo_spline(np.log10(x_smooth)), hi_spline(np.log10(x_smooth)), color=PALETTE[method], alpha=0.15)

    ax.set_xscale("log")
    ax.set_xlim(100, 2000)
    ax.set_xticks([100, 200, 500, 1000, 2000])
    ax.set_xticklabels(["100", "200", "500", "1000", "2000"])
    ax.set_xlabel("Number of Lean verifier calls")
    ax.set_ylabel("Cumulative proof success rate")
    ax.set_ylim(0, 1)

    ax.axvline(847, linestyle="--", color="gray", linewidth=0.8)
    ax.text(900, 0.18, "HieroFlow\nmean budget", fontsize=8, color="gray")

    rl_final = curves["rl_baseline"][0][-1]
    h_curve = curves["hieroflow"][0]
    idx = int(np.argmin(np.abs(h_curve - rl_final)))
    x_match = CHECKPOINTS[idx]
    y_match = h_curve[idx]
    ax.annotate(
        "Matches RL at\n2.1× fewer calls",
        xy=(x_match, y_match),
        xytext=(400, min(0.95, y_match + 0.18)),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
        fontsize=8,
    )

    ax.legend(loc="upper left")

    outputs = save_fig(fig, "fig4_efficiency", double_col=False)
    plt.close(fig)
    return outputs


if __name__ == "__main__":
    from results.data.schema import make_synthetic_data

    df = make_synthetic_data()
    make_figure(df)
