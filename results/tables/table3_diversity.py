from __future__ import annotations

import numpy as np
import pandas as pd

from results.plot_style import METHOD_LABELS
from results.stats.significance import bootstrap_ci

CAPTION = "Full diversity metrics across methods."

ORDER = ["hieroflow", "gfn_flat", "rl_baseline", "supervised"]
METRICS = [
    ("tactic_diversity", "Tactic entropy"),
    ("unique_subgoal_rate", "Unique subgoal rate"),
    ("semantic_diversity", "Semantic diversity"),
]


def generate_table3(df: pd.DataFrame) -> str:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Method & Tactic entropy & Unique subgoal rate & Semantic diversity \\\\",
        "\\midrule",
    ]
    for method in ORDER:
        vals = []
        for metric, _ in METRICS:
            arr = df[df["method"] == method].groupby("seed")[metric].mean().to_numpy(dtype=float)
            mean = float(np.mean(arr)) if arr.size else 0.0
            lo, hi = bootstrap_ci(arr)
            vals.append(f"{mean:.3f}\\scriptsize{{±{((hi-lo)/2):.3f}}}")
        lines.append(f"{METHOD_LABELS[method]} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    table = "\n".join(lines)
    print(table)
    return table


if __name__ == "__main__":
    from results.data.schema import make_synthetic_data

    generate_table3(make_synthetic_data())
