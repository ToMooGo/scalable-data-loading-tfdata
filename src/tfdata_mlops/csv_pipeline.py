"""Part A: the Data API on sharded CSV files (Geron, Ch. 13, "The Data API").

``csv_reader_dataset`` is the book's helper function, with its knobs exposed so each step can be
switched on or off and measured: list the files, interleave lines from several files at once,
shuffle, parse and standardise, batch, prefetch.

Two output shapes:

* ``mode="vector"``: ``(x, y)`` with ``x`` the eight numeric inputs, standardised inside the
  pipeline, exactly as the book's ``preprocess()`` does.
* ``mode="dict"``: ``(features, y)`` with raw, named features (including ``ocean_proximity``).
  The standardising and encoding then happen inside the model (see ``features.py``).
"""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf

from .data import CATEGORICAL, N_INPUTS, NUMERIC, TARGET_SCALE, Stats


def record_defaults(stats: Stats) -> list:
    """What ``tf.io.decode_csv`` uses for an empty field: the training median for a number, ""
    for the category. The target has no default (an empty tensor), so a missing target raises."""
    return [
        *(tf.constant(m, dtype=tf.float32) for m in stats.medians),
        tf.constant("", dtype=tf.string),
        tf.constant([], dtype=tf.float32),
    ]


def make_preprocess(stats: Stats, mode: str = "vector"):
    """Return ``preprocess(line)``: parse one CSV line (and standardise it in vector mode)."""
    if mode not in ("vector", "dict"):
        raise ValueError("mode must be 'vector' or 'dict'")
    defaults = record_defaults(stats)
    mean = tf.constant(stats.means, dtype=tf.float32)
    std = tf.constant(stats.stds, dtype=tf.float32)

    def preprocess_vector(line):
        fields = tf.io.decode_csv(line, record_defaults=defaults)
        x = tf.stack(fields[:N_INPUTS])  # the 8 numeric fields as one 1D tensor
        y = tf.stack(fields[-1:])
        return (x - mean) / std, y / TARGET_SCALE

    def preprocess_dict(line):
        fields = tf.io.decode_csv(line, record_defaults=defaults)
        features = {name: tf.reshape(fields[i], [1]) for i, name in enumerate(NUMERIC)}
        features[CATEGORICAL] = tf.reshape(fields[N_INPUTS], [1])
        return features, tf.reshape(fields[-1], [1]) / TARGET_SCALE

    return preprocess_vector if mode == "vector" else preprocess_dict


def csv_reader_dataset(
    filepaths: Sequence[str],
    stats: Stats,
    *,
    mode: str = "vector",
    repeat: int | None = 1,
    n_readers: int = 5,
    n_read_threads: int | None = None,
    shuffle_buffer_size: int = 10_000,
    n_parse_threads: int | None = 5,
    batch_size: int = 32,
    prefetch: int | None = 1,
    shuffle_files: bool = True,
    seed: int | None = None,
) -> tf.data.Dataset:
    """Load, shuffle, parse and batch the lines of many CSV files.

    The steps keep the order of the book's ``csv_reader_dataset``: ``list_files(...)``,
    ``.repeat(repeat)``, ``interleave`` (skipping each header line), ``shuffle``, ``map``
    (``preprocess``), ``batch``, ``prefetch(1)``. Shuffling after the interleave mixes lines from
    several files before they are parsed.

    ``repeat``: passes over the files (``None`` = forever, as the book uses for training).
    ``n_readers``: files read at the same time, their lines interleaved (``cycle_length``).
    ``n_read_threads`` / ``n_parse_threads``: threads for reading and parsing (``None`` = serial).
    ``shuffle_buffer_size``: 0 disables the shuffling buffer.
    ``prefetch``: batches prepared ahead of the training step (0 or ``None`` disables).
    """
    dataset = tf.data.Dataset.list_files(
        list(map(str, filepaths)), shuffle=shuffle_files, seed=seed
    )
    dataset = dataset.repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),  # skip the header row
        cycle_length=n_readers,
        num_parallel_calls=n_read_threads,
    )
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size, seed=seed)
    dataset = dataset.map(make_preprocess(stats, mode), num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(prefetch) if prefetch else dataset
