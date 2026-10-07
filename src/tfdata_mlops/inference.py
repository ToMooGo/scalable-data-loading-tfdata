"""Serving-side helpers: turn raw request rows into model inputs, and describe the training data.

The model already contains the standardising and encoding layers, so a request only needs the raw
columns. What it still needs is the handling of values the pipeline handles during training: a
missing number is replaced by the training median (the ``default_value`` of the TFRecord reader and
the ``record_defaults`` of the CSV reader), and a category that was never seen falls into the
unknown bucket of the ``StringLookup`` layer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import tensorflow as tf

from . import features as F
from .data import CATEGORICAL, NUMERIC, TARGET, TARGET_SCALE, Stats

QUANTILES = (0.01, 0.5, 0.99)


@dataclass(frozen=True)
class Profile:
    """What the model was trained on: the statistics of the pipeline plus the observed ranges."""

    stats: Stats
    ranges: dict[str, dict[str, float]]  # per numeric column: min, p01, p50, p99, max
    target_mean: float  # dollars
    target_std: float
    feature_set: str
    n_train: int
    category_share: dict[str, float] = field(default_factory=dict)
    missing_pct: float = 0.0  # share of training rows with at least one missing number
    cell_counts: list[int] = field(default_factory=list)  # training rows per grid cell (400)

    def to_dict(self) -> dict:
        return {
            "stats": self.stats.to_dict(),
            "ranges": self.ranges,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "feature_set": self.feature_set,
            "n_train": self.n_train,
            "category_share": self.category_share,
            "missing_pct": self.missing_pct,
            "cell_counts": self.cell_counts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Profile:
        return cls(
            Stats.from_dict(d["stats"]),
            d["ranges"],
            float(d["target_mean"]),
            float(d["target_std"]),
            d["feature_set"],
            int(d["n_train"]),
            d.get("category_share", {}),
            float(d.get("missing_pct", 0.0)),
            [int(c) for c in d.get("cell_counts", [])],
        )


def build_profile(train: pd.DataFrame, stats: Stats, feature_set: str) -> Profile:
    ranges = {}
    for name in NUMERIC:
        col = train[name].dropna()
        q = col.quantile(list(QUANTILES))
        ranges[name] = {
            "min": float(col.min()),
            "p01": float(q.iloc[0]),
            "p50": float(q.iloc[1]),
            "p99": float(q.iloc[2]),
            "max": float(col.max()),
        }
    shares = train[CATEGORICAL].value_counts(normalize=True)
    cells = np.digitize(train["latitude"], F.LAT_BOUNDARIES) * F.GRID + np.digitize(
        train["longitude"], F.LON_BOUNDARIES
    )
    return Profile(
        stats=stats,
        ranges=ranges,
        target_mean=float(train[TARGET].mean()),
        target_std=float(train[TARGET].std(ddof=0)),
        feature_set=feature_set,
        n_train=len(train),
        category_share={str(k): float(v) for k, v in shares.items()},
        missing_pct=float(100 * train[NUMERIC].isna().any(axis=1).mean()),
        cell_counts=[int(c) for c in np.bincount(cells, minlength=F.GRID * F.GRID)],
    )


@dataclass
class RowFlags:
    missing: list[str]
    unknown_category: bool
    out_of_range: list[str]


def grid_cell(latitude: float, longitude: float) -> tuple[int, int]:
    """Row and column of a location on the 20 x 20 grid (the cells of ``features.py``)."""
    return (
        int(np.digitize(latitude, F.LAT_BOUNDARIES)),
        int(np.digitize(longitude, F.LON_BOUNDARIES)),
    )


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def rows_to_features(
    rows: Sequence[Mapping], profile: Profile
) -> tuple[dict[str, tf.Tensor], list[RowFlags]]:
    """Raw request rows -> the model's input dict, plus data-quality flags for each row."""
    stats = profile.stats
    columns = {name: [] for name in NUMERIC}
    categories: list[str] = []
    flags: list[RowFlags] = []
    for row in rows:
        missing, outside = [], []
        for name, median in zip(NUMERIC, stats.medians, strict=True):
            value = row.get(name)
            if _is_missing(value):
                missing.append(name)
                value = median
            else:
                bounds = profile.ranges[name]
                if value < bounds["min"] or value > bounds["max"]:
                    outside.append(name)
            columns[name].append(float(value))
        category = str(row.get(CATEGORICAL, ""))
        categories.append(category)
        flags.append(RowFlags(missing, category not in stats.vocab, outside))
    features = {
        name: tf.constant(np.asarray(values, dtype=np.float32)[:, None])
        for name, values in columns.items()
    }
    features[CATEGORICAL] = tf.constant(np.asarray(categories, dtype=object)[:, None])
    return features, flags


def predict_rows(
    model, rows: Sequence[Mapping], profile: Profile
) -> tuple[np.ndarray, list[RowFlags]]:
    """Dollars for each raw row (the model predicts hundreds of thousands of dollars)."""
    features, flags = rows_to_features(rows, profile)
    out = model(features, training=False).numpy().ravel() * TARGET_SCALE
    return out, flags
