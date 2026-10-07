"""Part B: the TFRecord format (Geron, Ch. 13, "The TFRecord Format").

Sizes, the wire-format formula, equivalence with the CSV pipeline, corruption detection, read speed
and the memory used while streaming.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from . import benchmark as bm
from .csv_pipeline import csv_reader_dataset
from .data import CATEGORICAL, NUMERIC
from .prepare import Prepared
from .results import PartResult
from .tfrecord import (
    RECORD_OVERHEAD_BYTES,
    count_records,
    make_example,
    serialized_example_size,
    tfrecord_reader_dataset,
)


def _ordered(builder, paths, stats, **kw):
    """One ordered pass: single reader, no shuffle, so two pipelines can be compared row by row."""
    return builder(
        paths,
        stats,
        n_readers=1,
        shuffle_buffer_size=0,
        shuffle_files=False,
        prefetch=0,
        batch_size=256,
        **kw,
    )


def csv_vs_tfrecord(prep: Prepared) -> dict:
    """Both pipelines must hand the model exactly the same numbers."""
    csv = _ordered(
        csv_reader_dataset, prep.csv["train"], prep.stats, mode="dict", n_parse_threads=None
    )
    rec = _ordered(tfrecord_reader_dataset, prep.tfrecord["train"], prep.stats, parse="batch")
    rec_gz = _ordered(
        tfrecord_reader_dataset,
        prep.tfrecord_gz["train"],
        prep.stats,
        parse="batch",
        compression="GZIP",
    )

    def collect(ds):
        feats = {k: [] for k in (*NUMERIC, CATEGORICAL)}
        ys = []
        for f, y in ds:
            for k in feats:
                feats[k].append(f[k].numpy().reshape(-1))
            ys.append(y.numpy().reshape(-1))
        return {k: np.concatenate(v) for k, v in feats.items()}, np.concatenate(ys)

    (fa, ya), (fb, yb), (fc, yc) = collect(csv), collect(rec), collect(rec_gz)
    numeric_diff = max(float(np.abs(fa[k] - fb[k]).max()) for k in NUMERIC)
    return {
        "rows": int(len(ya)),
        "max_abs_diff_numeric": numeric_diff,
        "max_abs_diff_target": float(np.abs(ya - yb).max()),
        "category_equal": bool((fa[CATEGORICAL] == fb[CATEGORICAL]).all()),
        "gzip_identical_to_plain": bool(
            all((fb[k] == fc[k]).all() for k in fb) and (yb == yc).all()
        ),
    }


def wire_size_check(prep: Prepared) -> dict:
    """The size of every serialised Example equals the protobuf wire-format formula, and so does
    the size of the files (record overhead: 8 + 4 + 4 bytes per record)."""
    rows = prep.split.train.to_dict("records")
    sizes = np.array([len(make_example(r).SerializeToString()) for r in rows])
    predicted = np.array([serialized_example_size(r) for r in rows])
    # per shard: sum over its records + 16 bytes of framing each
    file_bytes = sum(os.path.getsize(p) for p in prep.tfrecord["train"])
    return {
        "records": len(rows),
        "formula_matches_all": bool((sizes == predicted).all()),
        "mean_example_bytes": float(sizes.mean()),
        "min_example_bytes": int(sizes.min()),
        "max_example_bytes": int(sizes.max()),
        "file_bytes": int(file_bytes),
        "predicted_file_bytes": int(predicted.sum() + RECORD_OVERHEAD_BYTES * len(rows)),
    }


def _flip_byte(src: Path, dst: Path, offset: int) -> None:
    data = bytearray(src.read_bytes())
    data[offset] ^= 0xFF
    dst.write_bytes(bytes(data))


def corruption_checks(prep: Prepared) -> pd.DataFrame:
    """Damage one byte of a shard (or cut it short) and see what TensorFlow does."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        for label, compression, source in (
            ("uncompressed", None, prep.tfrecord["train"][0]),
            ("GZIP", "GZIP", prep.tfrecord_gz["train"][0]),
        ):
            src = Path(source)
            size = src.stat().st_size
            cases = {
                "intact": None,
                "one byte flipped in the middle": ("flip", size // 2),
                "truncated at 60%": ("cut", int(size * 0.6)),
            }
            for case, how in cases.items():
                dst = tmp / f"{label}_{len(rows)}{src.suffix}"
                if how is None:
                    shutil.copy(src, dst)
                elif how[0] == "flip":
                    _flip_byte(src, dst, how[1])
                else:
                    dst.write_bytes(src.read_bytes()[: how[1]])
                try:
                    n = count_records([dst], compression)  # checks every CRC
                    outcome, detected = f"read {n} records", False
                except tf.errors.OpError as exc:
                    outcome, detected = type(exc).__name__, True
                rows.append(
                    {"file": label, "case": case, "result": outcome, "error_detected": detected}
                )
    return pd.DataFrame(rows)


def missing_feature_default(prep: Prepared) -> dict:
    """An Example without ``total_bedrooms`` parses to the training median (book: default_value)."""
    row = prep.split.train.iloc[0].to_dict()
    row["total_bedrooms"] = float("nan")
    serialized = make_example(row).SerializeToString()
    assert "total_bedrooms" not in make_example(row).features.feature
    from .tfrecord import feature_description

    parsed = tf.io.parse_single_example(serialized, feature_description(prep.stats))
    median = prep.stats.medians[NUMERIC.index("total_bedrooms")]
    return {
        "parsed_value": float(parsed["total_bedrooms"].numpy()[0]),
        "training_median": float(median),
        "equal": bool(np.isclose(parsed["total_bedrooms"].numpy()[0], median)),
    }


def run_part_b(prep: Prepared, cfg: dict, seed: int) -> PartResult:
    batch = int(cfg.get("batch_size", 32))
    repeats = int(cfg.get("repeats", 3))
    copies_list = [int(c) for c in cfg.get("scale_copies", [1, 10, 100])]
    root = prep.root

    gz_csv = bm.gzip_csv_shards(prep.csv["train"], root / "csv_gz")
    formats = {
        "csv": {"paths": prep.csv["train"]},
        "csv.gz": {"paths": [str(p) for p in gz_csv]},
        "tfrecord": {"paths": prep.tfrecord["train"]},
        "tfrecord.gz": {"paths": prep.tfrecord_gz["train"]},
    }
    readers = bm.default_format_readers()
    spec = {
        "csv": ("csv", "csv"),
        "csv.gz": ("csv.gz", "csv.gz"),
        "tfrecord, parse one record at a time": (
            "tfrecord",
            "tfrecord, parse one record at a time",
        ),
        "tfrecord, parse a batch at a time": ("tfrecord", "tfrecord, parse a batch at a time"),
        "tfrecord.gz, parse a batch at a time": (
            "tfrecord.gz",
            "tfrecord.gz, parse a batch at a time",
        ),
    }
    plan = {
        name: {"paths": formats[files]["paths"], "reader": readers[reader]}
        for name, (files, reader) in spec.items()
    }
    speed = pd.DataFrame(
        bm.run_format_benchmark(plan, prep.stats, batch_size=batch, repeats=repeats, seed=seed)
    )
    csv_bytes = int(speed.loc[speed.format == "csv", "bytes"].iloc[0])
    speed["size_vs_csv"] = speed["bytes"] / csv_bytes
    speed = speed.drop(columns=["seconds_all"])

    equal = csv_vs_tfrecord(prep)
    wire = wire_size_check(prep)
    corrupt = corruption_checks(prep)
    missing = missing_feature_default(prep)
    scale = pd.DataFrame(
        bm.scale_test(
            prep.csv["train"],
            prep.tfrecord_gz["train"],
            root / "stats.json",
            copies_list,
            root / "scaled",
            batch,
        )
    )

    by_format = speed.set_index("format")
    metrics = {
        "csv_bytes": csv_bytes,
        "csv_gz_bytes": int(by_format.loc["csv.gz", "bytes"]),
        "tfrecord_bytes": int(by_format.loc["tfrecord, parse a batch at a time", "bytes"]),
        "tfrecord_gz_bytes": int(by_format.loc["tfrecord.gz, parse a batch at a time", "bytes"]),
        "tfrecord_over_csv": float(
            by_format.loc["tfrecord, parse a batch at a time", "size_vs_csv"]
        ),
        "tfrecord_gz_over_csv": float(
            by_format.loc["tfrecord.gz, parse a batch at a time", "size_vs_csv"]
        ),
        "csv_examples_per_s": float(by_format.loc["csv", "examples_per_second"]),
        "tfrecord_single_examples_per_s": float(
            by_format.loc["tfrecord, parse one record at a time", "examples_per_second"]
        ),
        "tfrecord_batch_examples_per_s": float(
            by_format.loc["tfrecord, parse a batch at a time", "examples_per_second"]
        ),
        "tfrecord_gz_batch_examples_per_s": float(
            by_format.loc["tfrecord.gz, parse a batch at a time", "examples_per_second"]
        ),
        "csv_tfrecord_max_abs_diff": max(
            equal["max_abs_diff_numeric"], equal["max_abs_diff_target"]
        ),
        "csv_tfrecord_category_equal": float(equal["category_equal"]),
        "wire_formula_matches_all": float(wire["formula_matches_all"]),
        "wire_file_bytes_exact": float(wire["file_bytes"] == wire["predicted_file_bytes"]),
        "corruption_all_detected": float(
            corrupt.loc[corrupt.case != "intact", "error_detected"].all()
            and not corrupt.loc[corrupt.case == "intact", "error_detected"].any()
        ),
        "missing_feature_default_ok": float(missing["equal"]),
        "scale_rows_max": int(scale.rows.iloc[-1]),
        "scale_stream_growth_mb_max": float(scale.stream_growth_mb.iloc[-1]),
        "scale_pandas_growth_mb_max": float(scale.pandas_growth_mb.iloc[-1]),
        "scale_stream_examples_per_s_max": float(scale.stream_examples_per_s.iloc[-1]),
    }
    return PartResult(
        "B-tfrecord",
        params={"batch_size": batch, "repeats": repeats, "scale_copies": copies_list},
        metrics=metrics,
        tables={
            "format_benchmark": speed,
            "corruption": corrupt,
            "scale_test": scale,
            "csv_tfrecord_equivalence": pd.DataFrame([equal]),
            "wire_size": pd.DataFrame([wire]),
        },
        extra={"missing": missing},
    )
