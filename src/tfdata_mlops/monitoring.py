"""Drift checks on the prediction log (pure function, unit-tested).

Four things can go wrong after deployment, and each is visible in the requests alone: categories
the model never saw (the unknown bucket of ``StringLookup`` silently takes them), values outside
the training range, missing values (replaced by the median), and a shift of the whole input or
prediction distribution. Shifts are measured in training standard deviations, so they do not
depend on the units of a column.
"""

from __future__ import annotations

import math
from collections import Counter

from .data import CATEGORICAL, NUMERIC
from .inference import Profile


def _pct(count: int, total: int) -> float:
    return 100.0 * count / total


def summarize_predictions(rows: list[dict], cfg: dict, profile: Profile) -> dict:
    """``rows``: dicts with ``features`` (the raw request), ``prediction_usd``,
    ``unknown_category``, ``n_missing`` and ``n_out_of_range``. ``cfg``: the ``monitor`` section."""
    n = len(rows)
    report: dict = {"n_predictions": n, "alerts": []}
    if n < int(cfg["min_predictions"]):
        report["status"] = "insufficient-data"
        return report

    stats = profile.stats
    shifts = {}
    for name, mean, std in zip(NUMERIC, stats.means, stats.stds, strict=True):
        values = [
            float(r["features"][name])
            for r in rows
            if r["features"].get(name) is not None and not math.isnan(float(r["features"][name]))
        ]
        if values and std > 0:
            shifts[name] = (sum(values) / len(values) - mean) / std
    shown = Counter(str(r["features"].get(CATEGORICAL, "")) for r in rows)
    tvd = 0.5 * sum(
        abs(shown.get(c, 0) / n - profile.category_share.get(c, 0.0))
        for c in set(shown) | set(profile.category_share)
    )
    prediction_shift = (
        sum(float(r["prediction_usd"]) for r in rows) / n - profile.target_mean
    ) / profile.target_std
    report.update(
        {
            "unknown_category_rate_pct": _pct(sum(bool(r["unknown_category"]) for r in rows), n),
            "out_of_range_rate_pct": _pct(sum(r["n_out_of_range"] > 0 for r in rows), n),
            "missing_rate_pct": _pct(sum(r["n_missing"] > 0 for r in rows), n),
            "feature_shift_std": {k: round(v, 3) for k, v in shifts.items()},
            "largest_feature_shift": max(shifts, key=lambda k: abs(shifts[k])) if shifts else None,
            "category_tvd": tvd,
            "prediction_shift_std": prediction_shift,
            "mean_prediction_usd": sum(float(r["prediction_usd"]) for r in rows) / n,
        }
    )
    alerts = report["alerts"]
    if report["unknown_category_rate_pct"] > float(cfg["max_unknown_category_pct"]):
        alerts.append(
            f"{report['unknown_category_rate_pct']:.1f}% of requests have an unseen "
            f"{CATEGORICAL} (limit {cfg['max_unknown_category_pct']}%)"
        )
    if report["out_of_range_rate_pct"] > float(cfg["max_out_of_range_pct"]):
        alerts.append(
            f"{report['out_of_range_rate_pct']:.1f}% of requests have a value outside the "
            f"training range (limit {cfg['max_out_of_range_pct']}%)"
        )
    if report["missing_rate_pct"] > float(cfg["max_missing_rate_pct"]):
        alerts.append(
            f"{report['missing_rate_pct']:.1f}% of requests have a missing value "
            f"(limit {cfg['max_missing_rate_pct']}%)"
        )
    limit = float(cfg["max_feature_shift_std"])
    for name, shift in sorted(shifts.items(), key=lambda kv: -abs(kv[1])):
        if abs(shift) > limit:
            alerts.append(f"mean of {name} moved by {shift:+.2f} training standard deviations")
    if tvd > float(cfg["max_category_tvd"]):
        alerts.append(f"{CATEGORICAL} mix differs from training (total variation {tvd:.2f})")
    if abs(prediction_shift) > float(cfg["max_prediction_shift_std"]):
        alerts.append(
            f"mean prediction moved by {prediction_shift:+.2f} training standard deviations"
        )
    report["status"] = "alert" if alerts else "ok"
    return report
