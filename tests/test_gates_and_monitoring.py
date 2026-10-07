import pytest

from tfdata_mlops.gates import all_passed, evaluate_gates
from tfdata_mlops.inference import Profile
from tfdata_mlops.monitoring import summarize_predictions

GATES = {
    "max_test_rmse_usd": 60000,
    "min_test_r2": 0.75,
    "min_gain_over_linear_pct": 10,
    "max_skew_usd": 0.05,
    "max_reproduction_diff_usd": 1.0,
}
GOOD = {
    "test_rmse": 52000.0,
    "test_r2": 0.79,
    "baseline_linear_test_rmse": 68000.0,
    "skew_max_abs_usd": 0.0,
    "unknown_category_ok": 1.0,
    "missing_value_ok": 1.0,
}


def failing(checks):
    return {c["check"] for c in checks if not c["passed"]}


def test_a_good_candidate_passes_every_gate():
    checks = evaluate_gates(GOOD, None, GATES, training_test_rmse=52000.0)
    assert all_passed(checks) and len(checks) == 7


@pytest.mark.parametrize(
    "change, check",
    [
        ({"test_rmse": 65000.0}, "test RMSE (USD)"),
        ({"test_r2": 0.5}, "test R-squared"),
        (
            {"test_rmse": 59000.0, "baseline_linear_test_rmse": 60000.0},
            "RMSE gain over the linear baseline (%)",
        ),
        ({"skew_max_abs_usd": 1.0}, "training/serving skew, max |difference| (USD)"),
        ({"unknown_category_ok": 0.0}, "unseen category gives a finite answer"),
        ({"missing_value_ok": 0.0}, "missing number is replaced by the median"),
    ],
)
def test_each_gate_can_fail_on_its_own(change, check):
    checks = evaluate_gates({**GOOD, **change}, None, GATES)
    assert check in failing(checks) and not all_passed(checks)


def test_the_registered_model_must_reproduce_the_training_rmse():
    checks = evaluate_gates(GOOD, None, GATES, training_test_rmse=51999.5)
    assert not failing(checks)
    assert any(
        "reproduces" in c
        for c in failing(evaluate_gates(GOOD, None, GATES, training_test_rmse=50000.0))
    )


def test_champion_challenger_tolerance():
    champion = {**GOOD, "test_rmse": 51500.0}
    assert not failing(evaluate_gates(GOOD, champion, GATES, 1.0))  # 1% worse is allowed
    worse = evaluate_gates({**GOOD, "test_rmse": 53000.0}, champion, GATES, 1.0)
    assert any("champion" in c for c in failing(worse))


# ------------------------------------------------------------------ monitoring
@pytest.fixture(scope="module")
def profile(prep):
    from tfdata_mlops.inference import build_profile

    return build_profile(prep.split.train, prep.stats, "all features")


CFG = {
    "min_predictions": 20,
    "max_unknown_category_pct": 1.0,
    "max_out_of_range_pct": 5.0,
    "max_missing_rate_pct": 5.0,
    "max_feature_shift_std": 0.5,
    "max_category_tvd": 0.2,
    "max_prediction_shift_std": 0.5,
}


def rows_from(df, profile, n=200, **override):
    out = []
    for rec in df.head(n).to_dict("records"):
        feats = {k: rec[k] for k in rec if k != "median_house_value"}
        feats.update(override)
        out.append(
            {
                "features": feats,
                "prediction_usd": profile.target_mean,
                "unknown_category": False,
                "n_missing": 0,
                "n_out_of_range": 0,
            }
        )
    return out


def test_too_few_requests_give_no_verdict(prep, profile):
    report = summarize_predictions(rows_from(prep.split.test, profile, n=5), CFG, profile)
    assert report["status"] == "insufficient-data" and report["alerts"] == []


def test_traffic_like_the_training_data_is_ok(prep, profile):
    rows = rows_from(prep.split.train, profile, n=len(prep.split.train))
    report = summarize_predictions(rows, CFG, profile)
    assert report["status"] == "ok", report["alerts"]
    assert max(abs(v) for v in report["feature_shift_std"].values()) < 0.1
    assert report["category_tvd"] < 0.01


def test_each_kind_of_drift_raises_its_own_alert(prep, profile):
    base = rows_from(prep.split.train, profile, n=300)

    def alerts(rows):
        return summarize_predictions(rows, CFG, profile)["alerts"]

    unseen = [{**r, "unknown_category": True} for r in base[:30]] + base[30:]
    assert any("unseen" in a for a in alerts(unseen))
    outside = [{**r, "n_out_of_range": 1} for r in base[:30]] + base[30:]
    assert any("outside the training range" in a for a in alerts(outside))
    gaps = [{**r, "n_missing": 1} for r in base[:30]] + base[30:]
    assert any("missing value" in a for a in alerts(gaps))
    pricey = [{**r, "prediction_usd": profile.target_mean + 2 * profile.target_std} for r in base]
    assert any("mean prediction moved" in a for a in alerts(pricey))
    rich = rows_from(prep.split.train, profile, n=300)
    for r in rich:
        r["features"]["median_income"] = r["features"]["median_income"] + 4 * profile.stats.stds[7]
    found = summarize_predictions(rich, CFG, profile)
    assert found["largest_feature_shift"] == "median_income"
    assert any("median_income" in a for a in found["alerts"])
    moved = [{**r, "features": {**r["features"], "ocean_proximity": "INLAND"}} for r in base]
    assert any("mix differs" in a for a in alerts(moved))


def test_missing_values_in_requests_do_not_break_the_shift_calculation(prep, profile):
    rows = rows_from(prep.split.train, profile, n=100, total_bedrooms=None)
    report = summarize_predictions(rows, CFG, profile)
    assert "total_bedrooms" not in report["feature_shift_std"]
    assert isinstance(profile, Profile)
