"""Confidence intervals (Geron, Ch. 2: ``scipy.stats.t.interval``)."""

from __future__ import annotations

import numpy as np
from scipy import stats as st


def mean_ci(values, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and the ``confidence`` interval of the mean from a t distribution.

    Returns ``(mean, low, high)``. With one value the interval is degenerate (mean, mean, mean).
    """
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return m, m, m
    low, high = st.t.interval(confidence, len(v) - 1, loc=m, scale=st.sem(v))
    return m, float(low), float(high)


def paired_diff_ci(a, b, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean of ``b - a`` over paired runs (same seeds), with its confidence interval."""
    return mean_ci(np.asarray(b, dtype=float) - np.asarray(a, dtype=float), confidence)
