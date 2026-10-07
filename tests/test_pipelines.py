"""The tf.data pipelines (CSV and TFRecord) against plain pandas / NumPy computations."""

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from tfdata_mlops import benchmark as bm
from tfdata_mlops import part_a_csv, part_b_tfrecord
from tfdata_mlops.csv_pipeline import csv_reader_dataset
from tfdata_mlops.data import CATEGORICAL, NUMERIC, TARGET, TARGET_SCALE
from tfdata_mlops.tfrecord import (
    count_records,
    feature_description,
    make_example,
    serialized_example_size,
    tfrecord_reader_dataset,
)

ORDERED = dict(n_readers=1, shuffle_buffer_size=0, shuffle_files=False, prefetch=0)


def collect_dict(ds):
    feats = {k: [] for k in (*NUMERIC, CATEGORICAL)}
    ys = []
    for f, y in ds:
        for k in feats:
            feats[k].append(f[k].numpy().reshape(-1))
        ys.append(y.numpy().reshape(-1))
    return {k: np.concatenate(v) for k, v in feats.items()}, np.concatenate(ys)


def test_csv_pipeline_in_order_equals_pandas(prep):
    out = part_a_csv.pipeline_vs_pandas(prep, batch_size=16)
    assert out["rows"] == len(prep.split.train)
    assert out["max_abs_diff_inputs"] < 1e-4 and out["max_abs_diff_targets"] < 1e-5
    assert out["rows_with_missing_filled"] > 0  # the synthetic data has missing bedrooms


@pytest.mark.parametrize("reader", ["csv", "tfrecord", "tfrecord_gz"])
def test_shuffled_interleaved_pass_reads_every_row_exactly_once(prep, reader):
    kw = dict(batch_size=32, n_readers=3, shuffle_buffer_size=200, seed=1, mode="dict")
    if reader == "csv":
        ds = csv_reader_dataset(prep.csv["train"], prep.stats, n_parse_threads=2, **kw)
    else:
        kw.pop("mode")
        gz = reader == "tfrecord_gz"
        ds = tfrecord_reader_dataset(
            prep.tfrecords("GZIP" if gz else None)["train"],
            prep.stats,
            compression="GZIP" if gz else None,
            **kw,
        )
    feats, y = collect_dict(ds)
    want = np.sort(prep.split.train[TARGET].to_numpy() / TARGET_SCALE)
    assert np.allclose(np.sort(y), want, atol=1e-5)
    assert len(y) == len(prep.split.train)


def test_endless_dataset_runs_past_one_epoch(prep):
    ds = csv_reader_dataset(prep.csv["train"], prep.stats, repeat=None, batch_size=64, mode="dict")
    n = sum(int(y.shape[0]) for _, y in ds.take(3 * len(prep.split.train) // 64))
    assert n > 2 * len(prep.split.train)


def test_shuffling_changes_the_order_but_a_seed_makes_it_reproducible(prep):
    def run(seed):
        ds = csv_reader_dataset(
            prep.csv["train"], prep.stats, mode="dict", seed=seed, n_parse_threads=None
        )
        return collect_dict(ds)[1]

    a, b, c = run(1), run(1), run(2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    ordered = collect_dict(
        csv_reader_dataset(prep.csv["train"], prep.stats, mode="dict", **ORDERED)
    )[1]
    assert not np.array_equal(a, ordered)


def test_strict_and_default_regimes_return_the_same_data(prep):
    def run(strict):
        ds = csv_reader_dataset(
            prep.csv["train"], prep.stats, mode="dict", **ORDERED, n_parse_threads=None
        )
        return collect_dict(bm.regime(ds, strict))[1]

    assert np.array_equal(run(True), run(False))


def test_tfrecord_matches_csv_and_gzip_matches_plain(prep):
    out = part_b_tfrecord.csv_vs_tfrecord(prep)
    assert out["rows"] == len(prep.split.train)
    assert out["max_abs_diff_numeric"] == 0 and out["max_abs_diff_target"] == 0
    assert out["category_equal"] and out["gzip_identical_to_plain"]


def test_parsing_one_record_or_a_batch_gives_the_same_numbers(prep):
    kw = dict(batch_size=50, **ORDERED)
    one = collect_dict(
        tfrecord_reader_dataset(prep.tfrecord["train"], prep.stats, parse="single", **kw)
    )
    many = collect_dict(
        tfrecord_reader_dataset(prep.tfrecord["train"], prep.stats, parse="batch", **kw)
    )
    assert np.array_equal(one[1], many[1])
    assert all(np.array_equal(one[0][k], many[0][k]) for k in one[0])
    with pytest.raises(ValueError):
        tfrecord_reader_dataset(prep.tfrecord["train"], prep.stats, parse="nope")


def test_a_missing_feature_is_parsed_as_the_training_median(prep):
    out = part_b_tfrecord.missing_feature_default(prep)
    assert out["equal"]
    row = prep.split.train.iloc[0].to_dict()
    row["total_bedrooms"] = float("nan")
    example = make_example(row)
    assert "total_bedrooms" not in example.features.feature
    # the target has no default: a record without it is an error, not a silent zero
    broken = {k: v for k, v in example.features.feature.items() if k != TARGET}
    serialized = tf.train.Example(features=tf.train.Features(feature=broken)).SerializeToString()
    with pytest.raises(tf.errors.InvalidArgumentError):
        tf.io.parse_single_example(serialized, feature_description(prep.stats))


def test_the_wire_size_formula_is_exact(prep):
    out = part_b_tfrecord.wire_size_check(prep)
    assert out["formula_matches_all"] and out["file_bytes"] == out["predicted_file_bytes"]
    for row in prep.split.train.head(20).to_dict("records"):
        assert serialized_example_size(row) == len(make_example(row).SerializeToString())


def assert_no_silent_damage(table) -> None:
    """A flipped byte always fails the CRC. A truncated file fails too, except when a GZIP stream
    happens to be cut where a compressed block and a record end together: then it ends cleanly and
    fewer records come back. (The bytes of a shard differ between processes, because protobuf
    writes the features of an Example in no fixed order, so that case shows up now and then.)"""
    intact = table[table.case == "intact"]
    assert not intact.error_detected.any()
    n_intact = {r.file: int(r.result.split()[1]) for r in intact.itertuples()}
    for r in table[table.case != "intact"].itertuples():
        if "flipped" in r.case or not r.error_detected:
            assert r.error_detected or int(r.result.split()[1]) < n_intact[r.file], r
            assert r.error_detected or r.file == "GZIP", r
        if r.error_detected:
            assert r.result == "DataLossError", r


def test_every_kind_of_damage_is_detected(prep):
    table = part_b_tfrecord.corruption_checks(prep)
    assert set(table.case) == {"intact", "one byte flipped in the middle", "truncated at 60%"}
    assert_no_silent_damage(table)
    assert table[table.case == "one byte flipped in the middle"].error_detected.all()


def test_record_counts_and_verification(prep):
    n = len(prep.split.test)
    assert count_records(prep.tfrecord["test"]) == n
    assert count_records(prep.tfrecord_gz["test"], "GZIP") == n


def test_gzip_shards_are_smaller_and_uncompressed_tfrecord_is_larger_than_csv(prep):
    import os

    size = lambda paths: sum(os.path.getsize(p) for p in paths)  # noqa: E731
    csv, rec, gz = (size(p["train"]) for p in (prep.csv, prep.tfrecord, prep.tfrecord_gz))
    assert gz < csv < rec  # a TFRecord stores names and framing in every record


def test_shuffle_buffer_on_sorted_data(prep, tmp_path):
    rows = bm.shuffle_quality(
        prep.split.train,
        prep.stats,
        tmp_path,
        n_shards=6,
        buffer_sizes=[10, 100],
        n_readers=3,
        seed=0,
    )
    by = {r["label"]: r for r in rows}
    assert by["no shuffling"]["spearman"] > 0.99 and by["no shuffling"]["batch_diversity"] < 0.2
    assert by["buffer 10"]["spearman"] > by["buffer 100"]["spearman"] > 0.5
    assert by["interleave 3 files"]["batch_diversity"] > by["no shuffling"]["batch_diversity"] + 0.3
    # shuffling only the order of 6 files moves whole blocks: the order changes, but each batch
    # still comes from one block of the sorted target, so it is far less diverse than a batch
    # drawn from three files at once
    only_files = by["random file order only"]
    assert only_files["spearman"] < 0.99
    assert only_files["batch_diversity"] < 0.4
    assert only_files["batch_diversity"] < by["interleave 3 files"]["batch_diversity"]


def test_part_a_runs_and_reports_consistent_numbers(prep):
    res = part_a_csv.run_part_a(
        prep,
        {
            "repeats": 1,
            "step_ms_values": [0, 2],
            "prefetch_step_ms_values": [0, 2],
            "buffer_sizes": [10, 100],
            "batch_size": 32,
        },
        seed=0,
    )
    m = res.metrics
    assert m["pipeline_max_abs_diff_inputs"] < 1e-4 and m["numpy_examples_per_s"] > 0
    assert set(res.tables) >= {
        "csv_ablation",
        "prefetch_sweep",
        "shuffle_quality",
        "numpy_baseline",
    }
    sweep = res.tables["prefetch_sweep"]
    assert (sweep.predicted_ms_per_batch > 0).all()
    ab = res.tables["csv_ablation"]
    assert set(ab.regime) == {"strict", "default"} and ab.variant.nunique() == 6


def test_part_b_runs_on_small_data(prep):
    res = part_b_tfrecord.run_part_b(
        prep, {"repeats": 1, "scale_copies": [1, 2], "batch_size": 32}, seed=0
    )
    m = res.metrics
    assert m["wire_formula_matches_all"] == 1
    assert_no_silent_damage(res.tables["corruption"])
    assert m["csv_tfrecord_max_abs_diff"] == 0 and m["missing_feature_default_ok"] == 1
    assert m["tfrecord_gz_over_csv"] < 1 < m["tfrecord_over_csv"]
    scale = res.tables["scale_test"]
    assert list(scale.rows) == [len(prep.split.train) * c for c in (1, 2)]


def test_benchmark_formulas():
    assert bm.predicted_ms_per_batch(2.0, 5.0, prefetch=False) == 7.0
    assert bm.predicted_ms_per_batch(2.0, 5.0, prefetch=True) == 5.0
    assert bm.predicted_ms_per_batch(6.0, 5.0, prefetch=True) == 6.0
    assert bm.predicted_spearman(0, 1000) == 1.0 / np.sqrt(1.0)
    assert bm.predicted_spearman(1000, 1000) == 0.0
    assert 0.9 < bm.predicted_spearman(100, 10_000) < 1.0
    sorted_targets = np.arange(1000.0)
    assert bm.order_correlation(sorted_targets) == pytest.approx(1.0)
    assert abs(bm.order_correlation(np.random.default_rng(0).permutation(sorted_targets))) < 0.1
    flat = [np.full(10, 3.0), np.full(10, 5.0)]
    assert bm.batch_diversity(flat) < 0.01  # every batch is constant although the whole set varies
    rng = np.random.default_rng(1)
    mixed = [rng.normal(size=1000) for _ in range(5)]
    assert bm.batch_diversity(mixed) == pytest.approx(1.0, abs=0.05)


def test_confidence_intervals():
    from tfdata_mlops.stats import mean_ci, paired_diff_ci

    m, lo, hi = mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m == 3.0 and lo < 3.0 < hi and hi - m == pytest.approx(m - lo)
    assert mean_ci([2.0]) == (2.0, 2.0, 2.0)
    d, lo, hi = paired_diff_ci([10, 10, 10, 10], [9, 8, 9, 8])
    assert d == -1.5 and hi < 0  # a consistent drop is "significant"
    assert pd.notna(lo)


def test_measurement_processes_find_the_package_without_an_install(tmp_path, monkeypatch):
    """The scale test runs ``python -m tfdata_mlops.memprobe`` in child processes."""
    import subprocess
    import sys
    from pathlib import Path

    monkeypatch.delenv("PYTHONPATH", raising=False)
    out = subprocess.run(
        [sys.executable, "-c", "import tfdata_mlops, sys; print(tfdata_mlops.__file__)"],
        env=bm.child_env(),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert Path(out).resolve() == (Path(bm.__file__).parent / "__init__.py").resolve()
