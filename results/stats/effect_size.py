from __future__ import annotations

import numpy as np


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0

    var_a = np.var(a, ddof=1) if a.size > 1 else 0.0
    var_b = np.var(b, ddof=1) if b.size > 1 else 0.0
    pooled_denom = (a.size - 1) + (b.size - 1)
    if pooled_denom <= 0:
        return 0.0
    pooled_sd = np.sqrt(((a.size - 1) * var_a + (b.size - 1) * var_b) / pooled_denom)
    if pooled_sd == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def relative_improvement(baseline: float, proposed: float, is_percentage_point: bool = False) -> str:
    if is_percentage_point:
        delta_pp = (proposed - baseline) * 100.0
        return f"{delta_pp:+.1f}pp"
    if baseline == 0:
        return "inf%"
    rel = (proposed - baseline) / baseline * 100.0
    return f"{rel:+.1f}%"
