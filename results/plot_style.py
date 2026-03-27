from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

PALETTE = {
    "hieroflow": "#0077BB",
    "rl_baseline": "#EE7733",
    "gfn_flat": "#009988",
    "supervised": "#CC3311",
    "random": "#BBBBBB",
}

HATCH = {"hieroflow": "", "rl_baseline": "///", "gfn_flat": "...", "supervised": "xxx"}

METHOD_LABELS = {
    "hieroflow": "HieroFlow (ours)",
    "rl_baseline": "RL-Prover (ABEL)",
    "gfn_flat": "GFlowNet-Flat",
    "supervised": "Supervised (ReProver)",
    "random": "Random",
}

_OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "output"


def set_output_dir(path: str | Path) -> None:
    global _OUTPUT_DIR
    _OUTPUT_DIR = Path(path)


def apply_style() -> None:
    matplotlib.rcParams.update(
        {
            "figure.dpi": 300,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.figsize": (3.5, 2.625),
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "lines.linewidth": 1.5,
        }
    )


def save_fig(fig: plt.Figure, filename: str, double_col: bool = False) -> list[Path]:
    width = 7.0 if double_col else 3.5
    height = width / 1.618
    fig.set_size_inches(width, height)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for ext in ("pdf", "png"):
        out = _OUTPUT_DIR / f"{filename}.{ext}"
        fig.savefig(out)
        outputs.append(out)
    return outputs


def add_significance_bracket(ax: plt.Axes, x1: float, x2: float, y: float, p_value: float) -> None:
    if p_value < 0.001:
        label = "***"
    elif p_value < 0.01:
        label = "**"
    elif p_value < 0.05:
        label = "*"
    else:
        label = "ns"

    h = 0.012
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=0.8, clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h + 0.002, label, ha="center", va="bottom", fontsize=9)
