"""Part C: preprocessing inside the model (Geron, Ch. 13, "The Features API").

Compares feature sets (one-hot, buckets, a crossed grid of locations, embeddings) with several
seeds, checks the layers against plain NumPy, and counts the hash collisions of the book's
1,000-bucket location cross.
"""

from __future__ import annotations

import itertools

import keras
import numpy as np
import pandas as pd
from keras import layers

from . import features as F
from .data import CATEGORICAL
from .modeling import df_to_features, train_and_evaluate
from .prepare import Prepared
from .results import PartResult
from .stats import mean_ci, paired_diff_ci

BASELINE = "+ ocean one-hot"


def layer_check(prep: Prepared) -> pd.DataFrame:
    """Layers without learned weights must equal a NumPy computation of the same features."""
    rows = []
    valid = prep.split.valid
    inputs = df_to_features(valid, prep.stats)
    for name, fs in F.FEATURE_SETS.items():
        try:
            reference = F.reference_features(valid, prep.stats, fs)
        except ValueError:
            continue  # contains an embedding, which has no closed form
        raw_inputs, features, blocks = F.build_preprocessing(fs, prep.stats)
        out = keras.Model(raw_inputs, features)(inputs, training=False).numpy()
        rows.append(
            {
                "feature_set": name,
                "width": int(out.shape[1]),
                "max_abs_diff": float(np.abs(out - reference).max()),
                "blocks": ", ".join(f"{b} ({w})" for b, w in blocks),
            }
        )
    return pd.DataFrame(rows)


def location_cells(df: pd.DataFrame) -> np.ndarray:
    """Cell number 0-399 of every row on the 20 x 20 grid (latitude row * 20 + longitude column)."""
    lat = np.digitize(df["latitude"].to_numpy(), F.LAT_BOUNDARIES)
    lon = np.digitize(df["longitude"].to_numpy(), F.LON_BOUNDARIES)
    return lat * F.GRID + lon


def hash_buckets_of_cells(num_bins: int = F.LOCATION_HASH_BUCKETS) -> np.ndarray:
    """The bucket that ``HashedCrossing`` assigns to each of the 400 cells."""
    lat, lon = np.array(list(itertools.product(range(F.GRID), range(F.GRID)))).T
    crossing = layers.HashedCrossing(num_bins=num_bins)
    return crossing([lat.reshape(-1, 1), lon.reshape(-1, 1)]).numpy().ravel()


def hash_collisions(prep: Prepared) -> tuple[pd.DataFrame, dict]:
    """How many of the 400 cells (and how many training rows) share a bucket after hashing?"""
    n_bins = F.LOCATION_HASH_BUCKETS
    n_cells = F.GRID * F.GRID
    bucket_of_cell = hash_buckets_of_cells(n_bins)
    cells = location_cells(prep.split.train)
    counts = np.bincount(cells, minlength=n_cells)
    occupied = np.flatnonzero(counts)
    occupied_buckets = bucket_of_cell[occupied]
    bucket_cells = pd.Series(occupied_buckets).value_counts()
    colliding = set(bucket_cells[bucket_cells > 1].index)
    rows_in_collision = int(sum(counts[c] for c in occupied if bucket_of_cell[c] in colliding))
    expected_distinct = n_bins * (1 - (1 - 1 / n_bins) ** n_cells)
    summary = {
        "cells": n_cells,
        "buckets": n_bins,
        "cells_with_training_rows": int(len(occupied)),
        "distinct_buckets_all_cells": int(len(set(bucket_of_cell))),
        "expected_distinct_buckets_all_cells": float(expected_distinct),
        "distinct_buckets_occupied_cells": int(len(set(occupied_buckets))),
        "occupied_cells_sharing_a_bucket": int(
            sum(bucket_cells[bucket_cells > 1]) if len(colliding) else 0
        ),
        "training_rows_in_shared_buckets": rows_in_collision,
        "training_rows_in_shared_buckets_pct": 100 * rows_in_collision / len(cells),
    }
    table = pd.DataFrame(
        {
            "cell": np.arange(n_cells),
            "row": np.arange(n_cells) // F.GRID,
            "col": np.arange(n_cells) % F.GRID,
            "train_rows": counts,
            "bucket": bucket_of_cell,
        }
    )
    return table, summary


DEFAULT_HEADS = {
    "linear": {"hidden": [], "learning_rate": 0.005, "epochs": 150, "patience": 15},
    "mlp": {"hidden": [64, 64], "learning_rate": 0.002, "epochs": 100, "patience": 10},
}


def _summarise(runs: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Mean error with a 95% interval per feature set, and the paired difference from the baseline."""
    rows = []
    base = runs[runs.feature_set == BASELINE].sort_values("seed")
    for name in names:
        r = runs[runs.feature_set == name].sort_values("seed")
        v_mean, v_lo, v_hi = mean_ci(r.valid_rmse)
        t_mean, t_lo, t_hi = mean_ci(r.test_rmse)
        row = {
            "feature_set": name,
            "features": int(r.features.iloc[0]),
            "params": int(r.params.iloc[0]),
            "valid_rmse": v_mean,
            "valid_rmse_lo": v_lo,
            "valid_rmse_hi": v_hi,
            "test_rmse": t_mean,
            "test_rmse_lo": t_lo,
            "test_rmse_hi": t_hi,
            "test_r2": float(r.test_r2.mean()),
            "epochs_mean": float(r.epochs_run.mean()),
        }
        if len(base) == len(r) and name != BASELINE:
            d, lo, hi = paired_diff_ci(base.valid_rmse, r.valid_rmse)
            row.update(valid_diff_vs_baseline=d, valid_diff_lo=lo, valid_diff_hi=hi)
            row["significant_vs_baseline"] = bool(lo > 0 or hi < 0)
        rows.append(row)
    return pd.DataFrame(rows)


def run_part_c(prep: Prepared, cfg: dict, seed: int) -> PartResult:
    names = cfg.get("feature_sets") or list(F.FEATURE_SETS)
    n_seeds = int(cfg.get("n_seeds", 5))
    compression = cfg.get("compression", "GZIP")
    heads = cfg.get("heads") or DEFAULT_HEADS
    select_head = cfg.get("select_head", "mlp")
    batch_size = int(cfg.get("batch_size", 128))
    paths = prep.tfrecords(compression)
    n_train, n_valid = len(prep.split.train), len(prep.split.valid)

    rows = []
    for head, spec in heads.items():
        kw = dict(
            compression=compression,
            batch_size=batch_size,
            hidden=tuple(int(h) for h in spec["hidden"]),
            learning_rate=float(spec["learning_rate"]),
            epochs=int(spec["epochs"]),
            patience=int(spec["patience"]),
        )
        for name in names:
            fs = F.FEATURE_SETS[name]
            width = F.feature_width(fs, prep.stats)
            for k in range(n_seeds):
                res = train_and_evaluate(
                    fs, prep.stats, paths, n_train, n_valid, seed=seed + k, **kw
                )
                fit = res["fit"]
                rows.append(
                    {
                        "head": head,
                        "feature_set": name,
                        "seed": seed + k,
                        "features": width,
                        "params": res["n_params"],
                        "epochs_run": fit.epochs_run,
                        "best_epoch": fit.best_epoch,
                        "train_seconds": fit.train_seconds,
                        **{f"valid_{m}": v for m, v in res["valid"].items()},
                        **{f"test_{m}": v for m, v in res["test"].items()},
                    }
                )
    runs = pd.DataFrame(rows)
    summary = pd.concat(
        [_summarise(runs[runs["head"] == h], names).assign(head=h) for h in heads],
        ignore_index=True,
    )
    chosen = summary[summary["head"] == select_head]
    best = chosen.loc[chosen.valid_rmse.idxmin()]  # chosen on validation data only

    cell_table, collision = hash_collisions(prep)
    check = layer_check(prep)
    metrics = {
        "n_feature_sets": len(names),
        "n_seeds": n_seeds,
        "best_feature_set_valid_rmse": float(best.valid_rmse),
        "best_feature_set_test_rmse": float(best.test_rmse),
        "layer_check_max_abs_diff": float(check.max_abs_diff.max()),
        **{f"hash_{k}": float(v) for k, v in collision.items()},
    }
    for head in heads:
        part = summary[summary["head"] == head].set_index("feature_set")
        if BASELINE in part.index:
            metrics[f"{head}_baseline_valid_rmse"] = float(part.loc[BASELINE, "valid_rmse"])
        metrics[f"{head}_best_valid_rmse"] = float(part.valid_rmse.min())
        metrics[f"{head}_n_significant_gains"] = int(
            (part.get("valid_diff_hi", pd.Series(dtype=float)) < 0).sum()
        )
    return PartResult(
        "C-features",
        params={
            "feature_sets": names,
            "n_seeds": n_seeds,
            "heads": heads,
            "select_head": select_head,
            "best_feature_set": str(best.feature_set),
            "batch_size": batch_size,
            "compression": compression,
        },
        metrics=metrics,
        tables={
            "feature_runs": runs,
            "feature_summary": summary,
            "layer_check": check,
            "location_cells": cell_table,
            "hash_collisions": pd.DataFrame([collision]),
        },
        extra={"best_feature_set": str(best.feature_set), "categorical": CATEGORICAL},
    )
