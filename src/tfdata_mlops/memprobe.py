"""Read a set of shards in a fresh process and report how much memory the process grew by.

Run as ``python -m tfdata_mlops.memprobe --mode stream|pandas --stats stats.json shard1 shard2 ...``.

``stream`` reads TFRecord shards with the tf.data pipeline (one pass, 10,000-example shuffle
buffer). ``pandas`` is the alternative the Data API replaces: read every CSV into a DataFrame and
standardise it in NumPy. A fresh process per measurement keeps the numbers independent.
"""

from __future__ import annotations

import argparse
import json
import time

import psutil

from .data import CSV_COLUMNS, NUMERIC, Stats


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def probe_stream(paths, stats: Stats, compression, batch_size: int) -> dict:
    import tensorflow as tf

    from .tfrecord import tfrecord_reader_dataset

    # one warm-up batch so the runtime's own thread pools and buffers are already allocated
    warm = tfrecord_reader_dataset(
        paths[:1],
        stats,
        compression=compression,
        batch_size=batch_size,
        shuffle_buffer_size=0,
        n_readers=1,
        prefetch=0,
    )
    next(iter(warm))
    before = rss_mb()
    peak = before
    dataset = tfrecord_reader_dataset(
        paths,
        stats,
        compression=compression,
        batch_size=batch_size,
        n_read_threads=tf.data.AUTOTUNE,
        n_parse_threads=tf.data.AUTOTUNE,
        prefetch=tf.data.AUTOTUNE,
        seed=0,
    )
    start = time.perf_counter()
    n = 0
    for batches, (_, y) in enumerate(dataset, start=1):
        n += int(y.shape[0])
        if batches % 200 == 0:
            peak = max(peak, rss_mb())
    seconds = time.perf_counter() - start
    peak = max(peak, rss_mb())
    return {"examples": n, "seconds": seconds, "rss_before_mb": before, "rss_peak_mb": peak}


def probe_pandas(paths, stats: Stats) -> dict:
    import numpy as np
    import pandas as pd

    pd.DataFrame({"a": [1.0]}).to_numpy()  # import cost out of the way
    before = rss_mb()
    start = time.perf_counter()
    frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    num = frame[NUMERIC].fillna(dict(zip(NUMERIC, stats.medians, strict=True)))
    x = ((num - np.asarray(stats.means)) / np.asarray(stats.stds)).to_numpy(np.float32)
    seconds = time.perf_counter() - start
    peak = rss_mb()  # DataFrame, standardised copy and NumPy array are all still alive here
    assert list(frame.columns) == CSV_COLUMNS
    return {
        "examples": int(len(x)),
        "seconds": seconds,
        "rss_before_mb": before,
        "rss_peak_mb": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--mode", choices=("stream", "pandas"), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--compression", default=None)
    args = parser.parse_args()
    with open(args.stats, encoding="utf-8") as fh:
        stats = Stats.from_dict(json.load(fh))
    if args.mode == "stream":
        out = probe_stream(args.paths, stats, args.compression, args.batch_size)
    else:
        out = probe_pandas(args.paths, stats)
    out["mode"] = args.mode
    out["growth_mb"] = out["rss_peak_mb"] - out["rss_before_mb"]
    out["examples_per_second"] = out["examples"] / out["seconds"]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
