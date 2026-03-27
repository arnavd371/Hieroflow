from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, wilcoxon

from results.stats.effect_size import cohens_d as cohens_d_impl, relative_improvement


METRICS = {
    "success": {"agg": "mean", "is_percentage_point": True, "name": "success"},
    "semantic_diversity": {"agg": "mean", "is_percentage_point": False, "name": "semantic diversity"},
}


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 500,
    confidence: float = 0.95,
) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return (0.0, 0.0)
    if vals.size == 1:
        return (float(vals[0]), float(vals[0]))

    ci = bootstrap(
        (vals,),
        statistic,
        confidence_level=confidence,
        n_resamples=n_resamples,
        method="percentile",
        random_state=42,
    )
    return float(ci.confidence_interval.low), float(ci.confidence_interval.high)


def _sig_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def wilcoxon_test(method_a_scores: np.ndarray, method_b_scores: np.ndarray) -> tuple[float, str]:
    a = np.asarray(method_a_scores, dtype=float)
    b = np.asarray(method_b_scores, dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return 1.0, "ns"
    a = a[:n]
    b = b[:n]

    if np.allclose(a, b):
        return 1.0, "ns"
    try:
        result = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", mode="auto")
        p = float(result.pvalue)
    except ValueError:
        p = 1.0
    return p, _sig_label(p)


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    return cohens_d_impl(group_a, group_b)


def _effect_size_bucket(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "small"
    if ad < 0.5:
        return "medium"
    if ad < 0.8:
        return "large"
    return "very large"


def _seed_scores(df: pd.DataFrame, method: str, benchmark: str, metric: str) -> np.ndarray:
    subset = df[(df["method"] == method) & (df["benchmark"] == benchmark)]
    if subset.empty:
        return np.array([])
    return (
        subset.groupby("seed", as_index=False)[metric]
        .mean(numeric_only=True)
        .sort_values("seed")[metric]
        .to_numpy(dtype=float)
    )


def run_all_significance_tests(df: pd.DataFrame) -> pd.DataFrame:
    baselines = ["gfn_flat", "rl_baseline", "supervised"]
    rows = []

    for benchmark in sorted(df["benchmark"].unique()):
        for metric in METRICS:
            h_scores = _seed_scores(df, "hieroflow", benchmark, metric)
            for baseline in baselines:
                b_scores = _seed_scores(df, baseline, benchmark, metric)
                p_value, sig = wilcoxon_test(h_scores, b_scores)
                d = cohens_d(h_scores, b_scores)
                rows.append(
                    {
                        "benchmark": benchmark,
                        "metric": metric,
                        "baseline": baseline,
                        "p_value": p_value,
                        "significance": sig,
                        "effect_size_cohens_d": d,
                    }
                )

    out = pd.DataFrame(rows)
    print("\nSignificance summary (HieroFlow vs baselines)")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return out


def _main_report_line(df: pd.DataFrame, benchmark: str, baseline: str) -> str:
    h_success = _seed_scores(df, "hieroflow", benchmark, "success")
    b_success = _seed_scores(df, baseline, benchmark, "success")
    p_value, _ = wilcoxon_test(h_success, b_success)
    d = cohens_d(h_success, b_success)
    baseline_mean = float(np.mean(b_success)) if b_success.size else 0.0
    proposed_mean = float(np.mean(h_success)) if h_success.size else 0.0
    imp = relative_improvement(baseline_mean, proposed_mean, is_percentage_point=True)

    p_text = "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"
    return (
        f"HieroFlow vs {baseline} on {benchmark}: {imp} success "
        f"({p_text}, d={d:.2f} [{_effect_size_bucket(d)}])"
    )


if __name__ == "__main__":
    from results.data.schema import make_synthetic_data

    synthetic_df = make_synthetic_data()
    run_all_significance_tests(synthetic_df)
    print("\nPaper-style statements")
    for b in sorted(synthetic_df["benchmark"].unique()):
        print(_main_report_line(synthetic_df, b, "rl_baseline"))
