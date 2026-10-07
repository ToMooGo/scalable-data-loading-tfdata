"""MNIST helpers, offline: the checksum-verified mirror download (with tiny synthetic IDX files and a
fake network), the route ``load_tfds_mnist`` takes, and the three TFRecord encodings."""

import gzip
import hashlib
import io
import struct
import urllib.request

import numpy as np
import pytest

from tfdata_mlops import mnist_tfds as M


def idx_images(images: np.ndarray) -> bytes:
    """IDX3 file (the official MNIST format): magic 0x803, count, rows, cols, then the pixels."""
    n, rows, cols = images.shape
    return gzip.compress(struct.pack(">IIII", 0x803, n, rows, cols) + images.tobytes())


def idx_labels(labels: np.ndarray) -> bytes:
    return gzip.compress(struct.pack(">II", 0x801, len(labels)) + labels.tobytes())


@pytest.fixture()
def mirror(monkeypatch):
    """Four small IDX files with their MD5 checksums, served by a fake ``urlopen``."""
    rng = np.random.default_rng(0)
    files = {
        "train-images-idx3-ubyte.gz": idx_images(rng.integers(0, 256, (6, 28, 28), np.uint8)),
        "train-labels-idx1-ubyte.gz": idx_labels(rng.integers(0, 10, 6, np.uint8)),
        "t10k-images-idx3-ubyte.gz": idx_images(rng.integers(0, 256, (4, 28, 28), np.uint8)),
        "t10k-labels-idx1-ubyte.gz": idx_labels(rng.integers(0, 10, 4, np.uint8)),
    }
    monkeypatch.setattr(M, "MNIST_FILES", {k: hashlib.md5(v).hexdigest() for k, v in files.items()})
    monkeypatch.setattr(M, "MIRRORS", ("https://down.invalid/", "https://mirror.invalid/"))
    served = dict(files)  # what the mirror answers (a test may tamper with it)
    calls = []

    def urlopen(url, timeout=None):
        calls.append(url)
        if url.startswith("https://down.invalid/"):
            raise OSError("mirror down")
        return io.BytesIO(served[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return {"files": files, "served": served, "calls": calls}


def test_mirror_files_are_downloaded_once_and_checked(mirror, tmp_path):
    out = M.ensure_mirror_files(tmp_path / "m")
    for name, content in mirror["files"].items():
        assert (out / name).read_bytes() == content
    # the first mirror fails, the second one delivers every file
    assert len(mirror["calls"]) == 8 and not list(out.glob("*.part"))
    M.ensure_mirror_files(tmp_path / "m")
    assert len(mirror["calls"]) == 8  # verified files in the cache are not downloaded again


def test_a_file_with_the_wrong_checksum_is_refused_and_removed(mirror, tmp_path):
    name = "t10k-labels-idx1-ubyte.gz"
    mirror["served"][name] = mirror["files"][name] + b"tampered"
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        M.ensure_mirror_files(tmp_path)
    assert not (tmp_path / name).exists()
    (tmp_path / name).write_bytes(b"tampered cache")  # a bad cached file is not trusted either
    mirror["served"][name] = mirror["files"][name]
    M.ensure_mirror_files(tmp_path)
    assert (tmp_path / name).read_bytes() == mirror["files"][name]


def test_no_reachable_mirror_is_an_error(mirror, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "MIRRORS", ("https://down.invalid/",))
    with pytest.raises(RuntimeError, match="Could not download"):
        M.ensure_mirror_files(tmp_path)


def test_the_mirror_route_hands_the_files_to_tfds_over_a_local_server(
    mirror, tmp_path, monkeypatch
):
    """TFDS's MNIST builder gets the verified files from 127.0.0.1; its URL is restored after."""
    import os

    from tensorflow_datasets.image_classification import mnist as tfds_mnist

    monkeypatch.setenv("NO_PROXY", os.environ.get("NO_PROXY", ""))  # the route edits it
    original = tfds_mnist.MNIST.URL
    seen = {}

    def fake_tfds_load(data_dir, batch_size):
        url = tfds_mnist.MNIST.URL
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        name = "t10k-labels-idx1-ubyte.gz"
        with opener.open(url + name, timeout=10) as resp:
            seen["bytes"] = resp.read()
        seen["url"], seen["batch_size"] = url, batch_size
        return {"train": "ds"}

    monkeypatch.setattr(M, "_tfds_load", fake_tfds_load)
    splits, source = M.load_tfds_mnist(tmp_path, batch_size=32, source="mirror")
    assert source == "mirror" and splits == {"train": "ds"} and seen["batch_size"] == 32
    assert seen["url"].startswith("http://127.0.0.1:")
    assert seen["bytes"] == mirror["files"]["t10k-labels-idx1-ubyte.gz"]
    restored = tfds_mnist.MNIST.URL
    assert restored == original

    # auto: TFDS's host unreachable -> mirror; tfds: a failure is raised, not hidden
    monkeypatch.setattr(M, "_tfds_host_reachable", lambda timeout=5.0: False)
    assert M.load_tfds_mnist(tmp_path / "auto", source="auto")[1] == "mirror"

    def failing(data_dir, batch_size):
        raise ConnectionError("blocked")

    monkeypatch.setattr(M, "_tfds_load", failing)
    with pytest.raises(ConnectionError):
        M.load_tfds_mnist(tmp_path / "t", source="tfds")


@pytest.mark.parametrize("compression", [None, "GZIP"])
@pytest.mark.parametrize("encoding", M.ENCODINGS)
def test_every_encoding_gives_back_the_original_pixels(tmp_path, encoding, compression):
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, (10, 28, 28, 1), np.uint8)
    labels = rng.integers(0, 10, 10)
    paths = M.write_mnist_tfrecords(
        images, labels, tmp_path, "t", encoding=encoding, n_shards=3, compression=compression
    )
    assert len(paths) == 3
    assert all(p.name.endswith(".tfrecord.gz" if compression else ".tfrecord") for p in paths)
    ds = M.mnist_tfrecord_dataset(
        paths, encoding=encoding, compression=compression, batch_size=4, parallel=False
    )
    x = np.concatenate([b[0].numpy() for b in ds])
    y = np.concatenate([b[1].numpy() for b in ds])
    assert x.dtype == np.uint8 and x.shape == images.shape
    assert np.array_equal(x, images) and np.array_equal(y, labels)
    if encoding == "raw":  # one parse_example call per batch gives the same result
        ds = M.mnist_tfrecord_dataset(
            paths,
            encoding="raw",
            compression=compression,
            batch_size=4,
            parse="batch",
            parallel=False,
        )
        assert np.array_equal(np.concatenate([b[0].numpy() for b in ds]), images)


def test_unknown_encodings_and_parse_modes_are_refused(tmp_path):
    with pytest.raises(ValueError, match="encoding"):
        M.write_mnist_tfrecords(
            np.zeros((1, 28, 28, 1), np.uint8), [0], tmp_path, "t", encoding="jpeg", n_shards=1
        )
    with pytest.raises(ValueError, match="raw"):
        M.mnist_tfrecord_dataset([], encoding="png", parse="batch")
    with pytest.raises(ValueError, match="parse"):
        M.mnist_tfrecord_dataset([], encoding="raw", parse="row")
