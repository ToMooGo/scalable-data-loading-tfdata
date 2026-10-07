"""Step 0 of every run: split the data, learn the statistics and write the shards.

One call produces everything the parts need: CSV shards (the book's starting point), TFRecord shards
and GZIP-compressed TFRecord shards. Train, validation and test rows are never mixed in a shard.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .data import (
    HousingSplit,
    Stats,
    compute_stats,
    ensure_housing_csv,
    load_housing,
    split_housing,
    write_csv_shards,
)
from .tfrecord import write_tfrecord_shards

SPLITS = ("train", "valid", "test")


@dataclass
class Prepared:
    split: HousingSplit
    stats: Stats
    root: Path
    csv: dict[str, list[str]]
    tfrecord: dict[str, list[str]]
    tfrecord_gz: dict[str, list[str]]
    data_sha256: str

    def tfrecords(self, compression: str | None) -> dict[str, list[str]]:
        return self.tfrecord_gz if compression == "GZIP" else self.tfrecord

    @property
    def shard_counts(self) -> dict[str, int]:
        return {name: len(self.csv[name]) for name in SPLITS}

    def table(self, name: str):
        return getattr(self.split, name)


def prepare_frame(
    df,
    root: str | Path,
    seed: int,
    *,
    test_size: float = 0.2,
    valid_fraction: float = 0.25,
    shards: dict | None = None,
    data_sha256: str = "",
) -> Prepared:
    """Split ``df`` 60/20/20 (stratified on income), fit the statistics on the training rows only
    and write all shards below ``root``."""
    split = split_housing(df, seed, test_size, valid_fraction)
    stats = compute_stats(split.train)
    root = Path(root)
    shards = shards or {"train": 20, "valid": 10, "test": 10}
    csv, tfr, tfr_gz = {}, {}, {}
    for name in SPLITS:
        table, n = getattr(split, name), int(shards[name])
        csv[name] = [str(p) for p in write_csv_shards(table, root / "csv", name, n)]
        tfr[name] = [str(p) for p in write_tfrecord_shards(table, root / "tfrecord", name, n)]
        tfr_gz[name] = [
            str(p) for p in write_tfrecord_shards(table, root / "tfrecord_gz", name, n, "GZIP")
        ]
    (root / "stats.json").write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    return Prepared(split, stats, root, csv, tfr, tfr_gz, data_sha256)


def prepare_data(data_cfg: dict, seed: int) -> Prepared:
    """Load ``housing.csv`` (downloaded once, SHA-256 checked) and prepare it as ``prepare_frame``."""
    csv_path = ensure_housing_csv(data_cfg.get("cache_dir", "data"))
    return prepare_frame(
        load_housing(data_cfg.get("cache_dir", "data")),
        data_cfg.get("work_dir", "data/work"),
        seed,
        test_size=float(data_cfg.get("test_size", 0.2)),
        valid_fraction=float(data_cfg.get("valid_fraction", 0.25)),
        shards=data_cfg.get("shards"),
        data_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    )
