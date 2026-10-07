"""Part D: TensorFlow Datasets and image TFRecords (Geron, Ch. 13, "The TensorFlow Datasets Project").

``tfds.load("mnist")`` as in the book, MNIST written as TFRecord in three encodings, and the same
small classifier trained from both sources.
"""

from __future__ import annotations

import time

import pandas as pd

from . import benchmark as bm
from . import mnist_tfds as M
from .results import PartResult


def run_part_d(cfg: dict, seed: int, work_dir) -> PartResult:
    data_dir = cfg.get("data_dir", "data/tfds")
    n_train = cfg.get("n_train")
    epochs = int(cfg.get("epochs", 5))
    n_seeds = int(cfg.get("n_seeds", 3))
    n_shards = int(cfg.get("n_shards", 10))
    repeats = int(cfg.get("repeats", 3))
    batch = 32

    splits, source = M.load_tfds_mnist(data_dir, batch_size=batch, source=cfg.get("source"))
    n_batches = {k: int(sum(1 for _ in v)) for k, v in splits.items()}
    x_tr, y_tr, x_te, y_te = M.mnist_arrays(data_dir, source=cfg.get("source"))
    if n_train:
        x_tr, y_tr = x_tr[: int(n_train)], y_tr[: int(n_train)]

    rows = []
    for encoding in M.ENCODINGS:
        for compression in (None, "GZIP"):
            start = time.perf_counter()
            paths = M.write_mnist_tfrecords(
                x_tr,
                y_tr,
                work_dir,
                "train",
                encoding=encoding,
                n_shards=n_shards,
                compression=compression,
            )
            write_seconds = time.perf_counter() - start
            size = bm.dir_bytes(paths)
            variants = [("one record at a time, serial", dict(parallel=False, prefetch=False))]
            variants.append(("one record at a time, parallel", dict(parallel=True, prefetch=True)))
            if encoding == "raw":
                variants.append(
                    (
                        "a batch at a time, serial",
                        dict(parallel=False, prefetch=False, parse="batch"),
                    )
                )
                variants.append(
                    (
                        "a batch at a time, parallel",
                        dict(parallel=True, prefetch=True, parse="batch"),
                    )
                )
            for label, kw in variants:
                res = bm.throughput(
                    lambda kw=kw, paths=paths, enc=encoding, comp=compression: (
                        M.mnist_tfrecord_dataset(
                            paths, encoding=enc, compression=comp, batch_size=batch, **kw
                        )
                    ),
                    repeats=repeats,
                )
                rows.append(
                    {
                        "encoding": encoding,
                        "compression": compression or "none",
                        "reading": label,
                        "bytes": size,
                        "bytes_per_image": size / len(x_tr),
                        "write_seconds": write_seconds,
                        "examples_per_second": res["examples_per_second"],
                    }
                )
    formats = pd.DataFrame(rows)

    # all three encodings give back the original pixels
    lossless = {}
    probe = slice(0, 256)
    for encoding in M.ENCODINGS:
        paths = M.write_mnist_tfrecords(
            x_tr[probe], y_tr[probe], work_dir, "probe", encoding=encoding, n_shards=1
        )
        xb, yb = next(
            iter(
                M.mnist_tfrecord_dataset(
                    paths, encoding=encoding, batch_size=256, parallel=False, prefetch=False
                )
            )
        )
        lossless[encoding] = bool(
            (xb.numpy() == x_tr[probe]).all() and (yb.numpy() == y_tr[probe]).all()
        )

    # the same classifier, trained from tfds.load and from the TFRecord files
    raw_paths = M.write_mnist_tfrecords(
        x_tr, y_tr, work_dir, "fit", encoding="raw", n_shards=n_shards, compression="GZIP"
    )
    steps = len(x_tr) // batch
    test_ds = splits["test"]
    runs = []
    for k in range(n_seeds):
        s = seed + k
        for source_name in ("tfds.load", "TFRecord (raw, GZIP)"):
            model = M.build_classifier(s)
            if source_name == "tfds.load":
                train_ds = splits["train"].repeat().prefetch(1)  # exactly the book's call
            else:
                train_ds = M.mnist_tfrecord_dataset(
                    raw_paths,
                    encoding="raw",
                    compression="GZIP",
                    repeat=None,
                    shuffle_buffer_size=10_000,
                    parse="batch",
                    seed=s,
                )
            start = time.perf_counter()
            hist = model.fit(
                train_ds, epochs=epochs, steps_per_epoch=steps, validation_data=test_ds, verbose=0
            )
            seconds = time.perf_counter() - start
            runs.append(
                {
                    "input": source_name,
                    "seed": s,
                    "test_accuracy": float(hist.history["val_accuracy"][-1]),
                    "train_seconds": seconds,
                }
            )
    fits = pd.DataFrame(runs)
    by = fits.groupby("input").test_accuracy.mean()
    raw_row = formats[
        (formats.encoding == "raw")
        & (formats.compression == "none")
        & (formats.reading == "one record at a time, serial")
    ].iloc[0]
    metrics = {
        "tfds_source_is_mirror": float(source == "mirror"),
        "train_batches": n_batches["train"],
        "test_batches": n_batches["test"],
        "n_train_images": int(len(x_tr)),
        "raw_bytes_per_image": float(raw_row.bytes_per_image),
        "all_encodings_lossless": float(all(lossless.values())),
        "tfds_load_test_accuracy": float(by["tfds.load"]),
        "tfrecord_test_accuracy": float(by["TFRecord (raw, GZIP)"]),
    }
    spec = splits["train"].element_spec
    return PartResult(
        "D-tfds-mnist",
        params={
            "source": source,
            "epochs": epochs,
            "n_seeds": n_seeds,
            "n_shards": n_shards,
            "element_spec": str((spec[0].shape.as_list(), spec[0].dtype.name)),
        },
        metrics=metrics,
        tables={
            "mnist_formats": formats,
            "mnist_fits": fits,
            "mnist_lossless": pd.DataFrame([lossless]),
        },
        extra={"source": source},
    )
