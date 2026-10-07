"""Part C: the preprocessing layers against NumPy, and what the book's claims reduce to."""

import keras
import numpy as np
import pytest
from keras import layers

from tfdata_mlops import features as F
from tfdata_mlops import part_c_features as C
from tfdata_mlops.data import CATEGORICAL
from tfdata_mlops.modeling import df_to_features

EXPECTED_WIDTH = {
    "numeric only": 8,
    "+ ocean one-hot": 8 + 6,
    "+ ocean embedding": 8 + 2,
    "+ income buckets": 8 + 6 + 5,
    "+ age x ocean": 8 + 6 + 4,
    "+ grid one-hot (400)": 8 + 6 + 400,
    "+ grid embedding (400)": 8 + 6 + 8,
    "+ hashed grid (1000)": 8 + 6 + 8,
    "all features": 8 + 6 + 5 + 4 + 8,
}


def test_widths_of_all_feature_sets(prep):
    assert set(F.FEATURE_SETS) == set(EXPECTED_WIDTH)
    for name, fs in F.FEATURE_SETS.items():
        assert F.feature_width(fs, prep.stats) == EXPECTED_WIDTH[name]


def test_layers_without_learned_weights_equal_numpy(prep):
    table = C.layer_check(prep)
    assert set(table.feature_set) == {
        "numeric only",
        "+ ocean one-hot",
        "+ income buckets",
        "+ grid one-hot (400)",
    }
    assert table.max_abs_diff.max() < 1e-5


def test_the_book_numbers_are_in_the_layers():
    assert F.INCOME_BOUNDARIES == [1.5, 3.0, 4.5, 6.0]  # five buckets
    assert (
        F.GRID * F.GRID == 400 and F.LOCATION_HASH_BUCKETS == 1000 and F.AGE_X_OCEAN_BUCKETS == 100
    )
    assert len(F.LAT_BOUNDARIES) == len(F.LON_BOUNDARIES) == F.GRID - 1


def test_an_unseen_category_lands_in_the_unknown_bucket(prep, housing):
    inputs, features, blocks = F.build_preprocessing(F.FEATURE_SETS["+ ocean one-hot"], prep.stats)
    model = keras.Model(inputs, features)
    row = housing.head(3).copy()
    row[CATEGORICAL] = ["SUBURB", "INLAND", "<1H OCEAN"]
    out = model(df_to_features(row, prep.stats), training=False).numpy()[:, -6:]
    assert out[0].tolist() == [1, 0, 0, 0, 0, 0]  # index 0 = unknown
    assert out[1].argmax() == prep.stats.vocab.index("INLAND") + 1
    assert out.sum(axis=1).tolist() == [1, 1, 1]


def test_grid_cells_and_hash_buckets(prep):
    buckets = C.hash_buckets_of_cells()
    assert buckets.shape == (400,) and buckets.min() >= 0 and buckets.max() < 1000
    assert np.array_equal(buckets, C.hash_buckets_of_cells())  # deterministic
    # 400 cells into 1000 buckets: collisions are certain, and about as many as chance predicts
    expected = 1000 * (1 - (1 - 1 / 1000) ** 400)
    assert abs(len(set(buckets)) - expected) < 20
    table, summary = C.hash_collisions(prep)
    assert table.train_rows.sum() == len(prep.split.train)
    assert summary["cells_with_training_rows"] == int((table.train_rows > 0).sum())
    assert summary["distinct_buckets_occupied_cells"] <= summary["cells_with_training_rows"]
    cells = C.location_cells(prep.split.train)
    assert cells.min() >= 0 and cells.max() < 400


def test_discretization_layers_agree_with_numpy_digitize():
    rng = np.random.default_rng(0)
    lat = rng.uniform(30, 44, (500, 1)).astype("float32")
    layer = layers.Discretization(bin_boundaries=F.LAT_BOUNDARIES, output_mode="int")
    assert np.array_equal(layer(lat).numpy().ravel(), np.digitize(lat.ravel(), F.LAT_BOUNDARIES))


def test_an_embedding_is_a_one_hot_vector_times_a_matrix():
    """Why the 400-cell grid embedding is a Dense layer without bias (see features.py)."""
    n, dim = 400, 8
    emb = layers.Embedding(n, dim)
    cells = np.array([[3], [399], [17]])
    out = emb(cells).numpy()[:, 0, :]
    dense = layers.Dense(dim, use_bias=False)
    dense.build((None, n))
    dense.set_weights([emb.get_weights()[0]])
    one_hot = np.eye(n, dtype="float32")[cells.ravel()]
    assert np.allclose(dense(one_hot).numpy(), out, atol=1e-6)


def test_feature_set_validation():
    with pytest.raises(ValueError):
        F.FeatureSet(ocean="dense")
    with pytest.raises(ValueError):
        F.FeatureSet(location="grid")


def test_reference_features_reject_learned_blocks(prep):
    with pytest.raises(ValueError):
        F.reference_features(prep.split.valid, prep.stats, F.FEATURE_SETS["all features"])


def test_model_returns_one_price_per_row_and_is_deterministic(prep):
    model = F.build_model(F.FEATURE_SETS["all features"], prep.stats)
    x = df_to_features(prep.split.test.head(7), prep.stats)
    a, b = model(x, training=False).numpy(), model(x, training=False).numpy()
    assert a.shape == (7, 1) and np.array_equal(a, b)


def test_part_c_trains_each_head_and_selects_on_validation(prep):
    cfg = {
        "feature_sets": ["+ ocean one-hot", "+ grid one-hot (400)"],
        "n_seeds": 2,
        "batch_size": 64,
        "compression": "GZIP",
        "select_head": "mlp",
        "heads": {
            "linear": {"hidden": [], "learning_rate": 0.01, "epochs": 3, "patience": 2},
            "mlp": {"hidden": [8], "learning_rate": 0.01, "epochs": 3, "patience": 2},
        },
    }
    res = C.run_part_c(prep, cfg, seed=0)
    runs, summary = res.tables["feature_runs"], res.tables["feature_summary"]
    assert len(runs) == 2 * 2 * 2 and set(runs["head"]) == {"linear", "mlp"}
    assert np.isfinite(runs.valid_rmse).all() and (runs.test_rmse > 0).all()
    assert len(summary) == 4 and "valid_diff_vs_baseline" in summary
    chosen = summary[summary["head"] == "mlp"]
    assert res.extra["best_feature_set"] == chosen.loc[chosen.valid_rmse.idxmin(), "feature_set"]
    assert res.metrics["layer_check_max_abs_diff"] < 1e-5
