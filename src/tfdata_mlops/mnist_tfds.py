"""Part D: TensorFlow Datasets and TFRecord images (Geron, Ch. 13, "The TensorFlow Datasets Project").

The book loads MNIST with one call::

    dataset = tfds.load(name="mnist", batch_size=32, as_supervised=True)

``load_tfds_mnist`` makes exactly that call. TFDS downloads the four official MNIST files from
Google Cloud Storage. If that host cannot be reached (firewalls, proxies, some CI runners),
the same four files are fetched from a public mirror, checked against the official MD5 checksums,
and handed to TFDS's own MNIST builder through a local web server. TFDS then prepares the data as
usual, so everything after the download is the real ``tfds.load``.

The second half stores the same images as TFRecord files in three encodings (raw bytes, PNG and a
serialised tensor: the options mentioned in the book) so their size and decoding speed can be
compared.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import http.server
import logging
import os
import socketserver
import threading
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import tensorflow as tf

log = logging.getLogger(__name__)

# Official MNIST files and their MD5 checksums (the same ones used by the book's data sources).
MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
MIRRORS = ("https://raw.githubusercontent.com/fgnt/mnist/master/",)
ENCODINGS = ("raw", "png", "tensor")


def _md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - the official checksums are MD5; this is integrity, not security
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_mirror_files(directory: str | Path) -> Path:
    """Download the four MNIST files from a mirror and verify their MD5 checksums."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, md5 in MNIST_FILES.items():
        dst = directory / name
        if dst.exists() and _md5(dst) == md5:
            continue
        last: Exception | None = None
        for base in MIRRORS:
            try:
                log.info("Downloading %s from %s", name, base)
                tmp = dst.with_suffix(dst.suffix + ".part")
                with urllib.request.urlopen(base + name, timeout=60) as resp, tmp.open("wb") as fh:
                    while chunk := resp.read(1 << 20):
                        fh.write(chunk)
                os.replace(tmp, dst)
                break
            except OSError as exc:
                last = exc
        else:
            raise RuntimeError(f"Could not download {name}: {last}")
        if _md5(dst) != md5:
            dst.unlink()
            raise RuntimeError(f"Checksum mismatch for {name}: it is not the official MNIST file")
    return directory


@contextlib.contextmanager
def _serve_directory(directory: Path) -> Iterator[str]:
    """Serve ``directory`` on 127.0.0.1 for the duration of the ``with`` block."""

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args) -> None:  # keep the logs clean
            pass

    server = socketserver.TCPServer(
        ("127.0.0.1", 0), functools.partial(Quiet, directory=str(directory))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


def _tfds_load(data_dir: str, batch_size: int) -> dict:
    import tensorflow_datasets as tfds

    return tfds.load(name="mnist", data_dir=data_dir, batch_size=batch_size, as_supervised=True)


def _is_prepared(data_dir: str) -> bool:
    """True when TFDS has already prepared MNIST in ``data_dir`` (no download is needed)."""
    return any(Path(data_dir, "mnist").glob("*/dataset_info.json"))


def _tfds_host_reachable(timeout: float = 5.0) -> bool:
    """Quick check of TFDS's download host, so a blocked network fails in seconds, not minutes."""
    from tensorflow_datasets.image_classification import mnist as tfds_mnist

    request = urllib.request.Request(
        tfds_mnist.MNIST.URL + "t10k-labels-idx1-ubyte.gz", method="HEAD"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return True
    except OSError:
        return False


def load_tfds_mnist(
    data_dir: str | Path = "data/tfds", batch_size: int = 32, source: str | None = None
) -> tuple[dict, str]:
    """``tfds.load(name="mnist", batch_size=..., as_supervised=True)``; returns ``(splits, source)``.

    ``source``: ``auto`` (TFDS's own download, then the verified mirror if that fails),
    ``tfds`` or ``mirror``. Defaults to the ``MNIST_SOURCE`` environment variable, else ``auto``.
    ``source`` in the result says which route was used.
    """
    source = (source or os.getenv("MNIST_SOURCE", "auto")).lower()
    data_dir = str(data_dir)
    if source == "tfds" or (
        source == "auto" and (_is_prepared(data_dir) or _tfds_host_reachable())
    ):
        try:
            # the mirror directory only exists if the mirror once supplied this data directory
            used_mirror = (Path(data_dir) / "mnist_mirror").exists()
            return _tfds_load(data_dir, batch_size), "mirror" if used_mirror else "tfds"
        except Exception as exc:  # network, proxy, rate limit ...
            if source == "tfds":
                raise
            log.warning("TFDS could not download MNIST (%s); using the verified mirror", exc)
    from tensorflow_datasets.image_classification import mnist as tfds_mnist

    files = ensure_mirror_files(Path(data_dir) / "mnist_mirror")
    original = tfds_mnist.MNIST.URL
    with _serve_directory(files) as url:
        no_proxy = os.environ.get("NO_PROXY", "")
        os.environ["NO_PROXY"] = f"127.0.0.1,localhost,{no_proxy}"
        tfds_mnist.MNIST.URL = url
        try:
            return _tfds_load(data_dir, batch_size), "mirror"
        finally:
            tfds_mnist.MNIST.URL = original
            os.environ["NO_PROXY"] = no_proxy


def mnist_arrays(data_dir: str | Path = "data/tfds", source: str | None = None):
    """The whole of MNIST as uint8 arrays: ``(x_train, y_train, x_test, y_test)``, images 28x28x1."""
    splits, _ = load_tfds_mnist(data_dir, batch_size=-1, source=source)  # batch_size=-1: one batch
    out = []
    for name in ("train", "test"):
        x, y = splits[name]
        out += [np.asarray(x), np.asarray(y)]
    return tuple(out)


# --------------------------------------------------------------------------------------------
# MNIST as TFRecord files
# --------------------------------------------------------------------------------------------
def _image_bytes(image: np.ndarray, encoding: str) -> bytes:
    if encoding == "raw":
        return image.astype(np.uint8).tobytes()
    if encoding == "png":
        return tf.io.encode_png(image.astype(np.uint8)).numpy()
    if encoding == "tensor":
        return tf.io.serialize_tensor(tf.constant(image.astype(np.uint8))).numpy()
    raise ValueError(f"encoding must be one of {ENCODINGS}")


def write_mnist_tfrecords(
    images: np.ndarray,
    labels: np.ndarray,
    directory: str | Path,
    prefix: str,
    *,
    encoding: str,
    n_shards: int,
    compression: str | None = None,
) -> list[Path]:
    """Write ``(image, label)`` pairs as ``Example`` records: ``image`` (bytes), ``label`` (int64)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    options = tf.io.TFRecordOptions(compression_type=compression) if compression else None
    suffix = ".tfrecord.gz" if compression == "GZIP" else ".tfrecord"
    paths = []
    for i, part in enumerate(np.array_split(np.arange(len(images)), n_shards)):
        path = directory / f"{prefix}_{encoding}_{i:02d}{suffix}"
        with tf.io.TFRecordWriter(str(path), options) as writer:
            for j in part:
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "image": tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[_image_bytes(images[j], encoding)]
                                )
                            ),
                            "label": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[int(labels[j])])
                            ),
                        }
                    )
                )
                writer.write(example.SerializeToString())
        paths.append(path)
    return paths


_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}


def _decoder(encoding: str):
    if encoding == "raw":
        return lambda b: tf.reshape(tf.io.decode_raw(b, tf.uint8), [28, 28, 1])
    if encoding == "png":
        return lambda b: tf.io.decode_png(b, channels=1)
    if encoding == "tensor":
        return lambda b: tf.reshape(tf.io.parse_tensor(b, tf.uint8), [28, 28, 1])
    raise ValueError(f"encoding must be one of {ENCODINGS}")


def mnist_tfrecord_dataset(
    paths: Sequence[str | Path],
    *,
    encoding: str,
    compression: str | None = None,
    batch_size: int = 32,
    parallel: bool = True,
    shuffle_buffer_size: int = 0,
    prefetch: bool = True,
    repeat: int | None = 1,
    seed: int | None = None,
    parse: str = "single",
) -> tf.data.Dataset:
    """``(image, label)`` batches from MNIST TFRecord files.

    ``parallel`` decodes with ``num_parallel_calls=AUTOTUNE`` and reads files in parallel;
    ``prefetch`` ends the pipeline with ``prefetch(AUTOTUNE)``. ``parse="batch"`` batches first and
    parses a whole batch per call (``parse_example``); it works for the ``raw`` encoding, because
    ``decode_raw`` accepts a vector of strings while ``decode_png`` and ``parse_tensor`` do not.
    """
    if parse not in ("single", "batch"):
        raise ValueError("parse must be 'single' or 'batch'")
    if parse == "batch" and encoding != "raw":
        raise ValueError("batch-wise parsing is only available for the 'raw' encoding")
    par = tf.data.AUTOTUNE if parallel else None

    ds = tf.data.TFRecordDataset(
        [str(p) for p in paths], compression_type=compression, num_parallel_reads=par
    )
    ds = ds.repeat(repeat)
    if shuffle_buffer_size:
        ds = ds.shuffle(shuffle_buffer_size, seed=seed)
    if parse == "batch":

        def parse_batch(serialized):
            parsed = tf.io.parse_example(serialized, _DESCRIPTION)
            images = tf.io.decode_raw(parsed["image"], tf.uint8)
            return tf.reshape(images, [-1, 28, 28, 1]), parsed["label"]

        ds = ds.batch(batch_size).map(parse_batch, num_parallel_calls=par)
    else:
        decode = _decoder(encoding)

        def parse_one(serialized):
            parsed = tf.io.parse_single_example(serialized, _DESCRIPTION)
            return decode(parsed["image"]), parsed["label"]

        ds = ds.map(parse_one, num_parallel_calls=par).batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE) if prefetch else ds


def build_classifier(seed: int = 0):
    """A small classifier of this project: flatten, one hidden layer of 128 units, Adam.

    The book elides the layers of its model (``Sequential([...])``) and uses the optimizer ``sgd``;
    this model is a choice of the project, used only to compare the two input routes.
    """
    import keras

    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            keras.layers.Input((28, 28, 1)),
            keras.layers.Rescaling(1.0 / 255),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model
