"""Serving-side handling of requests: the request path must agree with the training path."""

import numpy as np
import pytest

from tfdata_mlops import features as F
from tfdata_mlops.data import CATEGORICAL, NUMERIC
from tfdata_mlops.inference import (
    Profile,
    build_profile,
    grid_cell,
    predict_rows,
    rows_to_features,
)
from tfdata_mlops.modeling import df_to_features, predict_dollars


@pytest.fixture(scope="module")
def model(prep):
    import keras

    keras.utils.set_random_seed(0)
    return F.build_model(F.FEATURE_SETS["all features"], prep.stats)


@pytest.fixture(scope="module")
def profile(prep):
    return build_profile(prep.split.train, prep.stats, "all features")


def test_profile_round_trips_through_json(profile):
    import json

    again = Profile.from_dict(json.loads(json.dumps(profile.to_dict())))
    assert again == profile
    assert sum(profile.cell_counts) == profile.n_train and len(profile.cell_counts) == 400
    assert sum(profile.category_share.values()) == pytest.approx(1.0)
    assert 0 < profile.missing_pct < 10
    for r in profile.ranges.values():
        assert r["min"] <= r["p01"] <= r["p50"] <= r["p99"] <= r["max"]


def test_request_path_equals_training_path(prep, model, profile):
    """Same rows through ``predict_rows`` and through the DataFrame helper used in training."""
    test = prep.split.test
    served, flags = predict_rows(model, test.to_dict("records"), profile)
    reference = predict_dollars(model, df_to_features(test, prep.stats))
    assert np.abs(served - reference).max() < 1e-3
    assert len(flags) == len(test)


def test_missing_numbers_are_filled_with_the_training_median(prep, profile):
    row = prep.split.train.iloc[0].to_dict()
    row["total_bedrooms"] = None
    feats, (flag,) = rows_to_features([row], profile)
    median = prep.stats.medians[NUMERIC.index("total_bedrooms")]
    assert float(feats["total_bedrooms"].numpy()[0, 0]) == pytest.approx(median)
    assert flag.missing == ["total_bedrooms"]
    nan_row = {**row, "total_bedrooms": float("nan")}
    assert rows_to_features([nan_row], profile)[1][0].missing == ["total_bedrooms"]


def test_flags_for_unseen_categories_and_out_of_range_values(prep, profile):
    row = prep.split.train.iloc[1].to_dict()
    clean = rows_to_features([row], profile)[1][0]
    assert not clean.missing and not clean.unknown_category and not clean.out_of_range
    bad = {
        **row,
        CATEGORICAL: "SUBURB",
        "median_income": profile.ranges["median_income"]["max"] + 1,
    }
    flag = rows_to_features([bad], profile)[1][0]
    assert flag.unknown_category and flag.out_of_range == ["median_income"]


def test_the_model_gives_finite_answers_for_damaged_rows(prep, model, profile):
    row = prep.split.train.iloc[2].to_dict()
    damaged = [
        {**row, CATEGORICAL: "SUBURB"},
        {**row, "total_bedrooms": None},
        {**row, "median_income": 1000.0},
    ]
    prices, _ = predict_rows(model, damaged, profile)
    assert np.isfinite(prices).all()


def test_batch_and_single_requests_agree(prep, model, profile):
    rows = prep.split.test.head(12).to_dict("records")
    batch, _ = predict_rows(model, rows, profile)
    single = np.array([predict_rows(model, [r], profile)[0][0] for r in rows])
    # float32: another batch size may sum in another order (kernels differ between CPUs), which moves
    # a prediction by a few units in the last place, about 1e-7 of its value each; 1e-5 is far above that
    # and far below any real difference.
    np.testing.assert_allclose(batch, single, rtol=1e-5, atol=0.05)


def test_grid_cell_matches_the_layers(prep):
    import keras

    rng = np.random.default_rng(1)
    lat, lon = rng.uniform(31, 43, 300), rng.uniform(-126, -113, 300)
    lat_layer = keras.layers.Discretization(bin_boundaries=F.LAT_BOUNDARIES, output_mode="int")
    lon_layer = keras.layers.Discretization(bin_boundaries=F.LON_BOUNDARIES, output_mode="int")
    r = lat_layer(lat.astype("float32")[:, None]).numpy().ravel()
    c = lon_layer(lon.astype("float32")[:, None]).numpy().ravel()
    assert [grid_cell(a, b) for a, b in zip(lat, lon, strict=True)] == list(zip(r, c, strict=True))
    assert grid_cell(0, -200) == (0, 0) and grid_cell(90, 0) == (19, 19)


def test_non_finite_inputs_are_treated_as_missing(prep, profile):
    row = prep.split.train.iloc[0].to_dict()
    feats, (flag,) = rows_to_features([{**row, "population": float("nan")}], profile)
    assert flag.missing == ["population"] and np.isfinite(feats["population"].numpy()).all()
