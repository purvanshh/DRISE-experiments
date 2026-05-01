"""Statistical tests for experiment comparisons."""

from __future__ import annotations

import math


def mcnemar_test(correct_a: list[bool], correct_b: list[bool]) -> dict[str, float | int]:
    """Compute McNemar's test with a lightweight fallback implementation."""

    if len(correct_a) != len(correct_b):
        raise ValueError("McNemar inputs must be aligned and equal in length.")

    a_only = sum(1 for left, right in zip(correct_a, correct_b) if left and not right)
    b_only = sum(1 for left, right in zip(correct_a, correct_b) if not left and right)
    statistic = 0.0
    if a_only + b_only > 0:
        statistic = ((abs(a_only - b_only) - 1.0) ** 2) / (a_only + b_only)

    try:
        from statsmodels.stats.contingency_tables import mcnemar as statsmodels_mcnemar
    except ImportError:
        p_value = math.erfc(math.sqrt(max(statistic, 0.0) / 2.0))
    else:  # pragma: no cover - depends on optional dependency
        result = statsmodels_mcnemar([[0, a_only], [b_only, 0]], exact=False, correction=True)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)

    return {
        "a_only": a_only,
        "b_only": b_only,
        "statistic": round(statistic, 6),
        "p_value": round(float(p_value), 6),
    }
