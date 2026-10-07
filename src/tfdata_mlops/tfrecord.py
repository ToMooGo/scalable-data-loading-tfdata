"""Part B: the TFRecord format and the Example protocol buffer (Geron, Ch. 13, "The TFRecord Format").

A TFRecord file is a sequence of binary records. Each record is stored as

    uint64 length | uint32 CRC of the length | data bytes | uint32 CRC of the data

so a corrupted record is detected when it is read. The data is usually a serialised ``Example``
protocol buffer: a dictionary of named features, each a list of byte strings, floats or int64s.

A feature that is missing from an ``Example`` is parsed to the ``default_value`` of its
``FixedLenFeature``. Missing values (``total_bedrooms`` has 207) are therefore simply left out and
come back as the training median, the same value the CSV pipeline uses for an empty field.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .data import CATEGORICAL, NUMERIC, TARGET, TARGET_SCALE, Stats

RECORD_OVERHEAD_BYTES = 8 + 4 + 4  # length, CRC of the length, CRC of the data


def _float_feature(value: float) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_example(row: Mapping) -> tf.train.Example:
    """One housing row as an ``Example``. NaN numbers are left out."""
    feature = {}
    for name in NUMERIC:
        value = row[name]
        if value is not None and not np.isnan(value):
            feature[name] = _float_feature(float(value))
    feature[CATEGORICAL] = _bytes_feature(str(row[CATEGORICAL]).encode("utf-8"))
    feature[TARGET] = _float_feature(float(row[TARGET]))
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _suffix(compression: str | None) -> str:
    return ".tfrecord.gz" if compression == "GZIP" else ".tfrecord"


def write_tfrecord_shards(
    df: pd.DataFrame,
    directory: str | Path,
    prefix: str,
    n_shards: int,
    compression: str | None = None,
) -> list[Path]:
    """Write ``df`` as ``n_shards`` TFRecord files of (almost) equal length.

    ``compression``: ``None`` or ``"GZIP"``. Rows keep the order of ``df``, which is already
    shuffled by the split (the book suggests shuffling while converting).
    """
    if compression not in (None, "GZIP"):
        raise ValueError("compression must be None or 'GZIP'")
    if not 1 <= n_shards <= len(df):
        raise ValueError(f"n_shards must be between 1 and {len(df)}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    options = tf.io.TFRecordOptions(compression_type=compression) if compression else None
    paths = []
    records = df.to_dict("records")
    for i, part in enumerate(np.array_split(np.arange(len(df)), n_shards)):
        path = directory / f"{prefix}_{i:02d}{_suffix(compression)}"
        with tf.io.TFRecordWriter(str(path), options) as writer:
            for j in part:
                writer.write(make_example(records[j]).SerializeToString())
        paths.append(path)
    return paths


def feature_description(stats: Stats) -> dict:
    """How to read an ``Example`` back: shape, type and default value of every feature."""
    description = {
        name: tf.io.FixedLenFeature([1], tf.float32, default_value=[median])
        for name, median in zip(NUMERIC, stats.medians, strict=True)
    }
    description[CATEGORICAL] = tf.io.FixedLenFeature([1], tf.string, default_value=[""])
    description[TARGET] = tf.io.FixedLenFeature([1], tf.float32)  # required
    return description


def _split_target(parsed: dict):
    target = parsed.pop(TARGET) / TARGET_SCALE
    return parsed, target


def parse_single(stats: Stats):
    """``parse(serialized_example)`` for one record at a time (``tf.io.parse_single_example``)."""
    description = feature_description(stats)

    def parse(serialized):
        return _split_target(tf.io.parse_single_example(serialized, description))

    return parse


def parse_batch(stats: Stats):
    """``parse(serialized_examples)`` for a whole batch (``tf.io.parse_example``)."""
    description = feature_description(stats)

    def parse(serialized):
        return _split_target(tf.io.parse_example(serialized, description))

    return parse


def tfrecord_reader_dataset(
    filepaths: Sequence[str],
    stats: Stats,
    *,
    compression: str | None = None,
    parse: str = "batch",
    repeat: int | None = 1,
    n_readers: int = 5,
    n_read_threads: int | None = None,
    shuffle_buffer_size: int = 10_000,
    n_parse_threads: int | None = None,
    batch_size: int = 32,
    prefetch: int | None = 1,
    shuffle_files: bool = True,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Read, shuffle, parse and batch TFRecord files; yields ``(features, y)`` like the CSV reader.

    ``parse="batch"`` parses a whole batch at once with ``tf.io.parse_example`` (what the book
    does); ``parse="single"`` parses record by record with ``tf.io.parse_single_example``.
    """
    if parse not in ("batch", "single"):
        raise ValueError("parse must be 'batch' or 'single'")
    dataset = tf.data.Dataset.list_files(
        list(map(str, filepaths)), shuffle=shuffle_files, seed=seed
    )
    dataset = dataset.repeat(repeat)
    dataset = dataset.interleave(
        lambda path: tf.data.TFRecordDataset(path, compression_type=compression),
        cycle_length=n_readers,
        num_parallel_calls=n_read_threads,
    )
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    if parse == "batch":
        dataset = dataset.batch(batch_size).map(
            parse_batch(stats), num_parallel_calls=n_parse_threads
        )
    else:
        dataset = dataset.map(parse_single(stats), num_parallel_calls=n_parse_threads).batch(
            batch_size
        )
    return dataset.prefetch(prefetch) if prefetch else dataset


def count_records(paths: Sequence[str | Path], compression: str | None = None) -> int:
    """Read every record and let TensorFlow check the CRCs. Returns the number of records.

    Raises ``tf.errors.DataLossError`` (or another ``tf.errors.OpError`` for a damaged GZIP
    stream) if a record is corrupted.
    """
    return int(
        tf.data.TFRecordDataset([str(p) for p in paths], compression_type=compression)
        .reduce(0, lambda n, _: n + 1)
        .numpy()
    )


# --------------------------------------------------------------------------------------------
# The size of a serialised Example, derived from the protobuf definition (see the report)
# --------------------------------------------------------------------------------------------
def _varint_len(n: int) -> int:
    length = 1
    while n >= 0x80:
        n >>= 7
        length += 1
    return length


def _field(payload_len: int) -> int:
    """Bytes of a length-delimited field: 1 tag byte (field numbers below 16), the length
    varint and the payload."""
    return 1 + _varint_len(payload_len) + payload_len


def _map_entry(key: str, value_payload_len: int) -> int:
    key_bytes = len(key.encode("utf-8"))
    entry_payload = _field(key_bytes) + _field(value_payload_len)  # key = 1, value = 2
    return _field(entry_payload)  # entry of ``map<string, Feature> feature = 1``


def serialized_example_size(row: Mapping) -> int:
    """Predicted ``len(example.SerializeToString())`` from the wire format, without building it."""
    entries = 0
    float_feature = _field(_field(4))  # Feature{float_list: FloatList{value: packed 1 float}}
    for name in NUMERIC:
        value = row[name]
        if value is not None and not np.isnan(value):
            entries += _map_entry(name, float_feature)
    entries += _map_entry(TARGET, float_feature)
    text = str(row[CATEGORICAL]).encode("utf-8")
    entries += _map_entry(CATEGORICAL, _field(_field(len(text))))  # Feature{bytes_list: ...}
    return _field(entries)  # Example{features = 1}


def example_to_dict(example: tf.train.Example) -> dict:
    """Plain Python view of an ``Example`` (for tests and notebooks)."""
    out = {}
    for name, feature in example.features.feature.items():
        kind = feature.WhichOneof("kind")
        out[name] = list(getattr(feature, kind).value)
    return out
