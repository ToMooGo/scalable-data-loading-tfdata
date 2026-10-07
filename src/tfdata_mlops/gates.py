"""Deployment quality gates and the champion-vs-challenger rule (pure functions, unit-tested)."""

from __future__ import annotations

import numpy as np

from .baselines import linear_baseline
from .data import CATEGORICAL, NUMERIC, TARGET
from .inference import Profile, predict_rows
from .modeling import predict_dataset, regression_metrics
from .prepare import Prepared
from .tfrecord import tfrecord_reader_dataset


def evaluate_model(
    model, profile: Profile, prep: Prepared, compression: str | None = "GZIP"
) -> dict:
    """Metrics of a model on the held-out test rows, computed through both input paths.

    * request path: raw rows -> ``predict_rows`` (what the API does);
    * pipeline path: TFRecord shards -> tf.data -> ``model`` (what training did).

    Their difference is the training/serving skew; it must be zero up to floating-point noise.
    """
    test = prep.split.test
    rows = test.to_dict("records")
    served, _ = predict_rows(model, rows, profile)
    metrics = regression_metrics(test[TARGET].to_numpy(), served)

    ordered = tfrecord_reader_dataset(
        prep.tfrecords(compression)["test"],
        prep.stats,
        compression=compression,
        n_readers=1,
        shuffle_buffer_size=0,
        shuffle_files=False,
        batch_size=256,
        prefetch=1,
    )
    pipeline, targets = predict_dataset(model, ordered)
    assert np.allclose(targets, test[TARGET].to_numpy(), atol=1e-2)  # same rows, same order

    probe = dict(rows[0])
    unseen = predict_rows(model, [{**probe, CATEGORICAL: "SUBURB"}], profile)[0][0]
    gap = predict_rows(model, [{**probe, "total_bedrooms": None}], profile)[0][0]
    median = prep.stats.medians[NUMERIC.index("total_bedrooms")]
    filled = predict_rows(model, [{**probe, "total_bedrooms": median}], profile)[0][0]
    baseline = linear_baseline(prep)
    return {
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "test_r2": metrics["r2"],
        "baseline_linear_test_rmse": baseline["rmse"],
        "skew_max_abs_usd": float(np.abs(pipeline - served).max()),
        "unknown_category_ok": float(np.isfinite(unseen)),
        "missing_value_ok": float(np.isfinite(gap) and abs(gap - filled) < 1e-3),
    }


def _check(name: str, value: float, limit: float, passed: bool) -> dict:
    return {"check": name, "value": float(value), "limit": float(limit), "passed": bool(passed)}


def evaluate_gates(
    candidate: dict,
    champion: dict | None,
    gates: dict,
    tolerance_pct: float = 1.0,
    training_test_rmse: float | None = None,
) -> list[dict]:
    """One record per check: ``{"check", "value", "limit", "passed"}``.

    A candidate is deployable only if every check passes. ``training_test_rmse`` is the value the
    training run logged for the same model; the registered artifact must reproduce it.
    """
    m = candidate
    gain = 100 * (1 - m["test_rmse"] / m["baseline_linear_test_rmse"])
    checks = [
        _check(
            "test RMSE (USD)",
            m["test_rmse"],
            gates["max_test_rmse_usd"],
            m["test_rmse"] <= gates["max_test_rmse_usd"],
        ),
        _check(
            "test R-squared",
            m["test_r2"],
            gates["min_test_r2"],
            m["test_r2"] >= gates["min_test_r2"],
        ),
        _check(
            "RMSE gain over the linear baseline (%)",
            gain,
            gates["min_gain_over_linear_pct"],
            gain >= gates["min_gain_over_linear_pct"],
        ),
        _check(
            "training/serving skew, max |difference| (USD)",
            m["skew_max_abs_usd"],
            gates["max_skew_usd"],
            m["skew_max_abs_usd"] <= gates["max_skew_usd"],
        ),
        _check(
            "unseen category gives a finite answer",
            m["unknown_category_ok"],
            1,
            m["unknown_category_ok"] == 1,
        ),
        _check(
            "missing number is replaced by the median",
            m["missing_value_ok"],
            1,
            m["missing_value_ok"] == 1,
        ),
    ]
    if training_test_rmse is not None:
        diff = abs(m["test_rmse"] - training_test_rmse)
        checks.append(
            _check(
                "registered model reproduces the training run's RMSE, |difference| (USD)",
                diff,
                gates["max_reproduction_diff_usd"],
                diff <= gates["max_reproduction_diff_usd"],
            )
        )
    if champion:
        limit = champion["test_rmse"] * (1 + tolerance_pct / 100)
        checks.append(
            _check(
                "test RMSE vs champion (USD, may be at most tolerance worse)",
                m["test_rmse"],
                limit,
                m["test_rmse"] <= limit,
            )
        )
    return checks


def all_passed(checks: list[dict]) -> bool:
    return all(c["passed"] for c in checks)
