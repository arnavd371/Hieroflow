from __future__ import annotations

import numpy as np
import pandas as pd

from results.stats.significance import bootstrap_ci

CAPTION = "Sketch-depth ablation on synthetic benchmark aggregates."


def generate_table2(df: pd.DataFrame) -> str:
    nl = r"\\"
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        f"Depth & Success (\\%) & Semantic diversity {nl}",
        "\\midrule",
    ]
    for depth in range(6):
        if depth == 0:
            subset = df[df["method"] == "gfn_flat"]
        else:
            subset = df[(df["method"] == "hieroflow") & (df["sketch_depth"] == depth)]
        success = subset.groupby("seed")["success"].mean().to_numpy(dtype=float)
        sem = subset.groupby("seed")["semantic_diversity"].mean().to_numpy(dtype=float)
        s_m = float(np.mean(success)) if success.size else 0.0
        d_m = float(np.mean(sem)) if sem.size else 0.0
        s_lo, s_hi = bootstrap_ci(success)
        d_lo, d_hi = bootstrap_ci(sem)
        lines.append(
            f"{depth} & {s_m*100:.1f}\\scriptsize{{±{((s_hi-s_lo)/2)*100:.1f}}} & {d_m:.3f}\\scriptsize{{±{((d_hi-d_lo)/2):.3f}}} {nl}"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    table = "\n".join(lines)
    print(table)
    return table


if __name__ == "__main__":
    from results.data.schema import make_synthetic_data

    generate_table2(make_synthetic_data())
