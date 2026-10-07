"""Measuring input pipelines (Parts A and B): throughput, shuffle quality, formats and memory.

Every timing is the median of several full passes over the data after one warm-up pass, so the
operating system's file cache is warm for every variant. ``step_ms`` simulates an accelerator that
needs that many milliseconds per batch, which is when prefetching matters (book, Figure 13-3).
"""

from __future__ import annotations

import gzip
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats as sps

from .csv_pipeline import csv_reader_dataset
from .data import CSV_COLUMNS, TARGET, Stats, write_csv_shards
from .tfrecord import tfrecord_reader_dataset

AUTOTUNE = tf.data.AUTOTUNE


def strict_options() -> tf.data.Options:
    """Switch off tf.data's automatic rewrites (parallel ``map``, injected ``prefetch`` ...).

    Recent TensorFlow versions apply some of the book's advice by themselves. With these options
    only the transformations written in the pipeline take effect, which is what the book's
    Figure 13-3 describes and what lets each step be measured on its own.
    """
    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = False
    return options


def regime(dataset: tf.data.Dataset, strict: bool) -> tf.data.Dataset:
    return dataset.with_options(strict_options()) if strict else dataset


def consume(dataset: tf.data.Dataset, step_ms: float = 0.0) -> tuple[int, int, float]:
    """Iterate once over ``dataset``. Returns ``(examples, batches, seconds)``."""
    sleep = step_ms / 1000.0
    examples = batches = 0
    start = time.perf_counter()
    for batch in dataset:
        first = batch[1] if isinstance(batch, tuple) else batch
        examples += int(first.shape[0])
        batches += 1
        if sleep:
            time.sleep(sleep)  # stand-in for a training step that does not use the CPU
    return examples, batches, time.perf_counter() - start


def throughput(
    make_dataset: Callable[[], tf.data.Dataset], *, repeats: int = 3, step_ms: float = 0.0
) -> dict:
    """Examples per second of one pass (median of ``repeats`` after a warm-up pass)."""
    consume(make_dataset(), 0.0)  # warm-up: file cache, graph tracing
    runs = [consume(make_dataset(), step_ms) for _ in range(repeats)]
    seconds = [r[2] for r in runs]
    examples = runs[0][0]
    median = statistics.median(seconds)
    return {
        "examples": examples,
        "batches": runs[0][1],
        "seconds_median": median,
        "seconds_all": seconds,
        "examples_per_second": examples / median,
        "step_ms": step_ms,
    }


# --------------------------------------------------------------------------------------------
# Part A: one switch per step of the book's pipeline
# --------------------------------------------------------------------------------------------
def csv_variants(n_readers: int = 5, shuffle_buffer: int = 10_000) -> list[tuple[str, dict]]:
    """Build the pipeline step by step; the last-but-one row is the book's function as printed."""
    serial = dict(
        n_readers=1, n_read_threads=None, shuffle_buffer_size=0, n_parse_threads=None, prefetch=0
    )
    return [
        ("1. one file at a time, no shuffle, serial parse", dict(serial)),
        (f"2. + interleave {n_readers} files", dict(serial, n_readers=n_readers)),
        (
            "3. + shuffle buffer",
            dict(serial, n_readers=n_readers, shuffle_buffer_size=shuffle_buffer),
        ),
        (
            "4. + 5 parse threads",
            dict(
                serial, n_readers=n_readers, shuffle_buffer_size=shuffle_buffer, n_parse_threads=5
            ),
        ),
        (
            "5. + prefetch(1)  [the book's function]",
            dict(
                serial,
                n_readers=n_readers,
                shuffle_buffer_size=shuffle_buffer,
                n_parse_threads=5,
                prefetch=1,
            ),
        ),
        (
            "6. + AUTOTUNE threads and prefetch",
            dict(
                n_readers=n_readers,
                n_read_threads=AUTOTUNE,
                shuffle_buffer_size=shuffle_buffer,
                n_parse_threads=AUTOTUNE,
                prefetch=AUTOTUNE,
            ),
        ),
    ]


def run_csv_ablation(
    paths: Sequence[str],
    stats: Stats,
    *,
    batch_size: int,
    step_ms_values: Sequence[float],
    repeats: int,
    seed: int,
    strict: bool,
) -> list[dict]:
    """Throughput of each pipeline variant; ``strict`` turns tf.data's own rewrites off."""
    rows = []
    for name, kwargs in csv_variants():
        for step_ms in step_ms_values:
            res = throughput(
                lambda kwargs=kwargs: regime(
                    csv_reader_dataset(
                        paths, stats, mode="dict", batch_size=batch_size, seed=seed, **kwargs
                    ),
                    strict,
                ),
                repeats=repeats,
                step_ms=step_ms,
            )
            rows.append({"variant": name, "regime": "strict" if strict else "default", **res})
    return rows


def prefetch_sweep(
    paths: Sequence[str],
    stats: Stats,
    *,
    batch_size: int,
    step_ms_values: Sequence[float],
    repeats: int,
    seed: int,
) -> list[dict]:
    """Seconds per batch with and without ``prefetch(1)`` as the training step gets slower.

    Run in the strict regime, with parallel parsing, so the only difference is the prefetch.
    Without prefetch a batch costs ``t_input + t_step``; with it, ``max(t_input, t_step)``.
    """
    base = dict(n_readers=5, n_read_threads=None, shuffle_buffer_size=10_000, n_parse_threads=5)
    rows = []
    for prefetch in (0, 1):
        for step_ms in step_ms_values:
            res = throughput(
                lambda prefetch=prefetch: regime(
                    csv_reader_dataset(
                        paths,
                        stats,
                        mode="dict",
                        batch_size=batch_size,
                        seed=seed,
                        prefetch=prefetch,
                        **base,
                    ),
                    True,
                ),
                repeats=repeats,
                step_ms=step_ms,
            )
            rows.append(
                {
                    "prefetch": prefetch,
                    "step_ms": step_ms,
                    "ms_per_batch": 1000 * res["seconds_median"] / res["batches"],
                    **res,
                }
            )
    return rows


def predicted_ms_per_batch(t_input_ms: float, step_ms: float, prefetch: bool) -> float:
    """Book Figure 13-3: serial ``t_input + t_step``; overlapped ``max(t_input, t_step)``."""
    return max(t_input_ms, step_ms) if prefetch else t_input_ms + step_ms


def numpy_in_memory_baseline(
    paths: Sequence[str], stats: Stats, *, batch_size: int, step_ms: float, repeats: int, seed: int
) -> dict:
    """The usual alternative: read every CSV with pandas, standardise in NumPy, slice batches.

    It is fast on small data but holds all of it in RAM, which is what the Data API avoids.
    """
    rng = np.random.default_rng(seed)
    times = []
    for _ in range(repeats + 1):
        start = time.perf_counter()
        frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        num = frame[CSV_COLUMNS[:8]].fillna(dict(zip(CSV_COLUMNS[:8], stats.medians, strict=True)))
        x = ((num - np.asarray(stats.means)) / np.asarray(stats.stds)).to_numpy(np.float32)
        order = rng.permutation(len(x))
        n = 0
        for i in range(0, len(x), batch_size):
            batch = x[order[i : i + batch_size]]
            n += len(batch)
            if step_ms:
                time.sleep(step_ms / 1000.0)
        times.append(time.perf_counter() - start)
    median = statistics.median(times[1:])  # drop the warm-up pass
    return {
        "examples": n,
        "seconds_median": median,
        "examples_per_second": n / median,
        "step_ms": step_ms,
        "resident_bytes": int(x.nbytes + frame.memory_usage(deep=True).sum()),
    }


# --------------------------------------------------------------------------------------------
# Part A: how well does a shuffle buffer mix data that arrives in sorted order?
# --------------------------------------------------------------------------------------------
def first_epoch_batches(dataset: tf.data.Dataset) -> list[np.ndarray]:
    """The targets of every batch of one pass, in the order the pipeline emits them."""
    return [y.numpy().ravel() for _, y in dataset]


def batch_diversity(batches: Sequence[np.ndarray]) -> float:
    """Average spread of the targets inside a batch, relative to the spread of the whole data.

    Gradient descent works best on batches that look like random draws from the data (book,
    "Shuffling the Data"): 1 means each batch is as varied as the whole data set, 0 means a batch
    holds near-identical targets, as when sorted data is read in order.
    """
    full = [b for b in batches if len(b) == len(batches[0])]
    overall = float(np.std(np.concatenate(batches)))
    return float(np.mean([b.std() for b in full]) / overall)


def order_correlation(targets_in_output_order: np.ndarray) -> float:
    """Spearman correlation between output position and target.

    The shards hold the data sorted by target, so the target tells where an example sat in the
    source. 1 means the order is untouched, 0 means it is fully mixed.
    """
    pos = np.arange(len(targets_in_output_order))
    return float(sps.spearmanr(pos, targets_in_output_order).statistic)


def shuffle_quality(
    train_df: pd.DataFrame,
    stats: Stats,
    directory: str | Path,
    *,
    n_shards: int,
    buffer_sizes: Sequence[int],
    n_readers: int,
    seed: int,
    batch_size: int = 64,
) -> list[dict]:
    """Write the training set sorted by target, then read it back with different shufflers."""
    sorted_df = train_df.sort_values(TARGET, kind="stable").reset_index(drop=True)
    paths = [str(p) for p in write_csv_shards(sorted_df, directory, "sorted", n_shards)]
    base = dict(
        mode="dict",
        batch_size=batch_size,
        n_parse_threads=None,
        prefetch=0,
        seed=seed,
        n_read_threads=None,
    )
    rows = []

    def measure(label: str, **kwargs) -> None:
        ds = csv_reader_dataset(paths, stats, **{**base, **kwargs})
        batches = first_epoch_batches(ds)
        rows.append(
            {
                "label": label,
                "n_readers": kwargs.get("n_readers", 1),
                "buffer": kwargs.get("shuffle_buffer_size", 0),
                "shuffle_files": kwargs.get("shuffle_files", False),
                "spearman": order_correlation(np.concatenate(batches)),
                "batch_diversity": batch_diversity(batches),
            }
        )

    measure("no shuffling", n_readers=1, shuffle_buffer_size=0, shuffle_files=False)
    measure("random file order only", n_readers=1, shuffle_buffer_size=0, shuffle_files=True)
    measure(
        f"interleave {n_readers} files",
        n_readers=n_readers,
        shuffle_buffer_size=0,
        shuffle_files=True,
    )
    for b in buffer_sizes:
        measure(f"buffer {b}", n_readers=1, shuffle_buffer_size=b, shuffle_files=False)
    for b in buffer_sizes:
        measure(
            f"interleave {n_readers} + buffer {b}",
            n_readers=n_readers,
            shuffle_buffer_size=b,
            shuffle_files=True,
        )
    return rows


def predicted_spearman(buffer_size: int, n: int) -> float:
    """Spearman correlation after a shuffle buffer of ``buffer_size`` on sorted data of ``n`` rows.

    Derived in the report: output position = input position + a geometric waiting time with mean
    ``B`` and variance about ``B^2``, so the correlation is ``1 / sqrt(1 + 12 (B / n)^2)``.
    It holds for ``B`` much smaller than ``n``; a buffer as large as the data mixes it fully.
    """
    if buffer_size >= n:
        return 0.0
    return float(1.0 / np.sqrt(1.0 + 12.0 * (buffer_size / n) ** 2))


# --------------------------------------------------------------------------------------------
# Part B: file formats
# --------------------------------------------------------------------------------------------
def gzip_csv_shards(paths: Sequence[str | Path], directory: str | Path) -> list[Path]:
    """Compress each CSV shard with gzip (a fair baseline for GZIP-compressed TFRecord)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = []
    for p in map(Path, paths):
        dst = directory / (p.name + ".gz")
        with p.open("rb") as src, gzip.open(dst, "wb", compresslevel=6) as fh:
            fh.write(src.read())
        out.append(dst)
    return out


def csv_gz_reader_dataset(paths: Sequence[str], stats: Stats, *, batch_size: int, seed: int):
    """Like the book's CSV reader, reading gzip-compressed shards with ``TextLineDataset``."""
    from .csv_pipeline import make_preprocess

    ds = tf.data.Dataset.list_files(list(map(str, paths)), shuffle=True, seed=seed)
    ds = ds.interleave(
        lambda p: tf.data.TextLineDataset(p, compression_type="GZIP").skip(1),
        cycle_length=5,
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.shuffle(10_000, seed=seed).map(
        make_preprocess(stats, "dict"), num_parallel_calls=AUTOTUNE
    )
    return ds.batch(batch_size).prefetch(AUTOTUNE)


def dir_bytes(paths: Sequence[str | Path]) -> int:
    return int(sum(os.path.getsize(p) for p in paths))


def run_format_benchmark(
    formats: dict[str, dict], stats: Stats, *, batch_size: int, repeats: int, seed: int
) -> list[dict]:
    """``formats``: name -> ``{"paths": [...], "reader": callable(paths, stats, batch_size, seed)}``."""
    rows = []
    for name, spec in formats.items():
        res = throughput(
            lambda spec=spec: spec["reader"](spec["paths"], stats, batch_size, seed),
            repeats=repeats,
        )
        rows.append({"format": name, "bytes": dir_bytes(spec["paths"]), **res})
    return rows


def default_format_readers() -> dict[str, Callable]:
    return {
        "csv": lambda p, st, bs, seed: csv_reader_dataset(
            p,
            st,
            mode="dict",
            batch_size=bs,
            seed=seed,
            n_read_threads=AUTOTUNE,
            n_parse_threads=AUTOTUNE,
            prefetch=AUTOTUNE,
        ),
        "csv.gz": lambda p, st, bs, seed: csv_gz_reader_dataset(p, st, batch_size=bs, seed=seed),
        "tfrecord, parse one record at a time": lambda p, st, bs, seed: tfrecord_reader_dataset(
            p,
            st,
            parse="single",
            batch_size=bs,
            seed=seed,
            n_read_threads=AUTOTUNE,
            n_parse_threads=AUTOTUNE,
            prefetch=AUTOTUNE,
        ),
        "tfrecord, parse a batch at a time": lambda p, st, bs, seed: tfrecord_reader_dataset(
            p,
            st,
            parse="batch",
            batch_size=bs,
            seed=seed,
            n_read_threads=AUTOTUNE,
            n_parse_threads=AUTOTUNE,
            prefetch=AUTOTUNE,
        ),
        "tfrecord.gz, parse a batch at a time": lambda p, st, bs, seed: tfrecord_reader_dataset(
            p,
            st,
            compression="GZIP",
            parse="batch",
            batch_size=bs,
            seed=seed,
            n_read_threads=AUTOTUNE,
            n_parse_threads=AUTOTUNE,
            prefetch=AUTOTUNE,
        ),
    }


# --------------------------------------------------------------------------------------------
# Scale: the same data repeated many times (a test of memory and speed, not of accuracy)
# --------------------------------------------------------------------------------------------
def write_scaled_shards(
    csv_paths: Sequence[str], tfrecord_gz_paths: Sequence[str], copies: int, directory: str | Path
) -> dict[str, list[str]]:
    """Repeat every shard's rows ``copies`` times, keeping the number of shards.

    The result is the same kind of data, ``copies`` times larger. It is only used to measure how
    memory and speed behave as the data grows; no model is ever trained on it.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = {"csv": [], "tfrecord_gz": []}
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    for i, (csv, rec) in enumerate(zip(csv_paths, tfrecord_gz_paths, strict=True)):
        header, _, body = Path(csv).read_bytes().partition(b"\n")
        dst = directory / f"scaled_{copies}x_{i:02d}.csv"
        dst.write_bytes(header + b"\n" + body * copies)
        out["csv"].append(str(dst))
        records = list(
            tf.data.TFRecordDataset(str(rec), compression_type="GZIP").as_numpy_iterator()
        )
        dst = directory / f"scaled_{copies}x_{i:02d}.tfrecord.gz"
        with tf.io.TFRecordWriter(str(dst), options) as writer:
            for _ in range(copies):
                for record in records:
                    writer.write(record)
        out["tfrecord_gz"].append(str(dst))
    return out


def child_env() -> dict:
    """Environment of a measurement process: it imports this very package, installed or not."""
    package_root = str(Path(__file__).resolve().parents[1])  # the folder that holds tfdata_mlops
    paths = [package_root, *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
    return {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "PYTHONPATH": os.pathsep.join(paths)}


def _probe(mode: str, paths: Sequence[str], stats_path: str | Path, batch_size: int) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "tfdata_mlops.memprobe",
        "--mode",
        mode,
        "--batch-size",
        str(batch_size),
        "--stats",
        str(stats_path),
        *(["--compression", "GZIP"] if mode == "stream" else []),
        *map(str, paths),
    ]
    res = subprocess.run(
        cmd, capture_output=True, text=True, env=child_env(), check=True, timeout=1800
    )
    return json.loads(res.stdout.strip().splitlines()[-1])


def scale_test(
    csv_paths: Sequence[str],
    tfrecord_gz_paths: Sequence[str],
    stats_path: str | Path,
    copies_list: Sequence[int],
    directory: str | Path,
    batch_size: int,
) -> list[dict]:
    """Stream the scaled TFRecord shards with tf.data and, for comparison, load the scaled CSV
    shards into pandas. Each measurement runs in a fresh process."""
    rows = []
    for copies in copies_list:
        scaled = write_scaled_shards(csv_paths, tfrecord_gz_paths, copies, directory)
        stream = _probe("stream", scaled["tfrecord_gz"], stats_path, batch_size)
        frame = _probe("pandas", scaled["csv"], stats_path, batch_size)
        rows.append(
            {
                "copies": copies,
                "rows": stream["examples"],
                "csv_mb": dir_bytes(scaled["csv"]) / 1e6,
                "tfrecord_gz_mb": dir_bytes(scaled["tfrecord_gz"]) / 1e6,
                "stream_growth_mb": stream["growth_mb"],
                "stream_examples_per_s": stream["examples_per_second"],
                "pandas_growth_mb": frame["growth_mb"],
                "pandas_seconds": frame["seconds"],
            }
        )
        for p in (*scaled["csv"], *scaled["tfrecord_gz"]):
            os.remove(p)  # large and easy to regenerate
    return rows
