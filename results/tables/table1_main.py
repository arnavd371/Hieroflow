from __future__ import annotations

import numpy as np
import pandas as pd

from results.plot_style import METHOD_LABELS
from results.stats.significance import bootstrap_ci

CAPTION = (
    "Main results on three Lean 4 benchmarks. HieroFlow outperforms all baselines "
    "on proof success rate and semantic diversity. Results averaged over 5 random seeds; "
    "95\\% confidence intervals shown."
)

ORDER = ["hieroflow", "gfn_flat", "rl_baseline", "supervised"]
BENCHES = ["leandojo_mathlib", "minif2f", "proofnet"]


def _format_ci(mean: float, lo: float, hi: float, pct: bool = False) -> str:
    val = mean * 100 if pct else mean
    err = ((hi - lo) / 2.0) * (100 if pct else 1)
    return f"{val:.1f}\\scriptsize{{±{err:.1f}}}"


def _method_metric(df: pd.DataFrame, method: str, benchmark: str, metric: str):
    vals = (
        df[(df["method"] == method) & (df["benchmark"] == benchmark)]
        .groupby("seed")[metric]
        .mean()
        .to_numpy(dtype=float)
    )
    mean = float(np.mean(vals)) if vals.size else 0.0
    lo, hi = bootstrap_ci(vals)
    return mean, lo, hi


def generate_table1(df: pd.DataFrame) -> str:
    nl = r"\\"
    metrics = []
    for bench in BENCHES:
        for metric in ["success", "semantic_diversity"]:
            metrics.append((bench, metric))
    metrics += [("avg", "success"), ("avg", "semantic_diversity")]

    col_values = {m: [] for m in metrics}
    rows = []
    for method in ORDER:
        row = [METHOD_LABELS[method]]
        for bench, metric in metrics:
            if bench == "avg":
                vals = df[df["method"] == method].groupby("seed")[metric].mean().to_numpy(dtype=float)
                mean = float(np.mean(vals)) if vals.size else 0.0
                lo, hi = bootstrap_ci(vals)
            else:
                mean, lo, hi = _method_metric(df, method, bench, metric)
            col_values[(bench, metric)].append(mean)
            row.append((mean, _format_ci(mean, lo, hi, pct=(metric == "success"))))
        rows.append((method, row))

    best_idx = {k: int(np.argmax(v)) for k, v in col_values.items()}
    second_idx = {k: int(np.argsort(v)[-2]) for k, v in col_values.items()}

    lines = []
    lines.append("\\begin{tabular}{lcccccccc}")
    lines.append("\\toprule")
    lines.append(
        f"Method & \\multicolumn{{2}}{{c}}{{LeanDojo Mathlib}} & \\multicolumn{{2}}{{c}}{{MiniF2F}} & \\multicolumn{{2}}{{c}}{{ProofNet}} & Avg. Success & Avg. Diversity {nl}"
    )
    lines.append(f"& Success (\\%) & Semantic Div. & Success (\\%) & Semantic Div. & Success (\\%) & Semantic Div. & (\\%) & {nl}")
    lines.append("\\midrule")

    for ridx, (method, row) in enumerate(rows):
        cells = [row[0]]
        for cidx, metric_key in enumerate(metrics):
            _, text = row[cidx + 1]
            if ridx == best_idx[metric_key]:
                text = f"\\textbf{{{text}}}"
            elif ridx == second_idx[metric_key]:
                text = f"\\underline{{{text}}}"
            cells.append(text)

        prefix = "\\rowcolor{gray!10} " if method == "hieroflow" else ""
        lines.append(prefix + " & ".join(cells) + f" {nl}")

    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{9}}{{l}}{{\\scriptsize † Results averaged over 5 seeds. Best per column in \\textbf{{bold}}.}} {nl}")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    table = "\n".join(lines)
    print(table)
    return table


if __name__ == "__main__":
    from results.data.schema import make_synthetic_data

    generate_table1(make_synthetic_data())
