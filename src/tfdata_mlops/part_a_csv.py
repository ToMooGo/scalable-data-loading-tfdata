"""Part A: the Data API on sharded CSV files (Geron, Ch. 13, "The Data API").

Measures each step of the book's input pipeline on its own and checks that the pipeline returns
exactly what pandas would.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import benchmark as bm
from .csv_pipeline import csv_reader_dataset
from .data import N_INPUTS, NUMERIC
from .prepare import Prepared
from .results import PartResult


def pipeline_vs_pandas(prep: Prepared, batch_size: int = 32) -> dict:
    """Read the training shards in order and compare with pandas + NumPy standardisation."""
    ds = csv_reader_dataset(
        prep.csv["train"],
        prep.stats,
        mode="vector",
        n_readers=1,
        shuffle_buffer_size=0,
        n_parse_threads=None,
        prefetch=0,
        shuffle_files=False,
        batch_size=batch_size,
    )
    x = np.concatenate([xb.numpy() for xb, _ in ds])
    y = np.concatenate([yb.numpy().ravel() for _, yb in ds])
    train = prep.split.train
    filled = train[NUMERIC].fillna(dict(zip(NUMERIC, prep.stats.medians, strict=True)))
    ref = ((filled - np.asarray(prep.stats.means)) / np.asarray(prep.stats.stds)).to_numpy()
    y_ref = train["median_house_value"].to_numpy() / 100_000.0
    assert x.shape == (len(train), N_INPUTS)
    return {
        "rows": int(len(x)),
        "max_abs_diff_inputs": float(np.abs(x - ref).max()),
        "max_abs_diff_targets": float(np.abs(y - y_ref).max()),
        "rows_with_missing_filled": int(train[NUMERIC].isna().any(axis=1).sum()),
    }


def _drop_lists(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    return df.drop(columns=[c for c in ("seconds_all",) if c in df.columns])


def run_part_a(prep: Prepared, cfg: dict, seed: int) -> PartResult:
    batch = int(cfg.get("batch_size", 32))
    repeats = int(cfg.get("repeats", 3))
    steps = [float(s) for s in cfg.get("step_ms_values", [0.0, 4.0])]
    sweep_steps = [float(s) for s in cfg.get("prefetch_step_ms_values", [0.0, 1.0, 2.0, 4.0, 8.0])]
    buffers = [int(b) for b in cfg.get("buffer_sizes", [100, 1000, 10_000])]
    paths = prep.csv["train"]
    n_train = len(prep.split.train)

    check = pipeline_vs_pandas(prep, batch)

    strict = bm.run_csv_ablation(
        paths,
        prep.stats,
        batch_size=batch,
        step_ms_values=steps,
        repeats=repeats,
        seed=seed,
        strict=True,
    )
    default = bm.run_csv_ablation(
        paths,
        prep.stats,
        batch_size=batch,
        step_ms_values=steps,
        repeats=repeats,
        seed=seed,
        strict=False,
    )
    ablation = _drop_lists(strict + default)

    sweep = _drop_lists(
        bm.prefetch_sweep(
            paths,
            prep.stats,
            batch_size=batch,
            step_ms_values=sweep_steps,
            repeats=repeats,
            seed=seed,
        )
    )
    # the model of Figure 13-3: serial = t_in + t_step, overlapped = max(t_in, t_step)
    t_in = float(sweep[(sweep.prefetch == 0) & (sweep.step_ms == 0)].ms_per_batch.iloc[0])
    sweep["predicted_ms_per_batch"] = [
        bm.predicted_ms_per_batch(t_in, s, bool(p))
        for s, p in zip(sweep.step_ms, sweep.prefetch, strict=True)
    ]

    numpy_rows = [
        bm.numpy_in_memory_baseline(
            paths, prep.stats, batch_size=batch, step_ms=s, repeats=repeats, seed=seed
        )
        for s in steps
    ]
    numpy_table = pd.DataFrame(numpy_rows)

    shuffle = pd.DataFrame(
        bm.shuffle_quality(
            prep.split.train,
            prep.stats,
            prep.root / "csv_sorted",
            n_shards=len(paths),
            buffer_sizes=buffers,
            n_readers=5,
            seed=seed,
        )
    )
    shuffle["predicted_spearman"] = [
        # the formula assumes a buffer much smaller than the data (here: at most a quarter)
        bm.predicted_spearman(int(b), n_train) if (r == 1 and 0 < b <= n_train / 4) else np.nan
        for b, r in zip(shuffle.buffer, shuffle.n_readers, strict=True)
    ]

    def eps(df, regime, variant_prefix, step=0.0):
        row = df[
            (df.regime == regime)
            & (df.variant.str.startswith(variant_prefix))
            & (df.step_ms == step)
        ]
        return float(row.examples_per_second.iloc[0])

    first, book = "1.", "5."
    metrics = {
        "n_train_rows": n_train,
        "pipeline_max_abs_diff_inputs": check["max_abs_diff_inputs"],
        "pipeline_max_abs_diff_targets": check["max_abs_diff_targets"],
        "strict_serial_examples_per_s": eps(ablation, "strict", first),
        "strict_book_examples_per_s": eps(ablation, "strict", book),
        "default_serial_examples_per_s": eps(ablation, "default", first),
        "default_book_examples_per_s": eps(ablation, "default", book),
        "numpy_examples_per_s": float(numpy_table.examples_per_second.iloc[0]),
        "numpy_resident_mb": float(numpy_table.resident_bytes.iloc[0]) / 1e6,
    }
    slow = max(steps)
    if slow > 0:
        strict_slow = ablation[(ablation.regime == "strict") & (ablation.step_ms == slow)]
        no_prefetch = float(
            strict_slow[strict_slow.variant.str.startswith("4.")].examples_per_second.iloc[0]
        )
        with_prefetch = float(
            strict_slow[strict_slow.variant.str.startswith("5.")].examples_per_second.iloc[0]
        )
        metrics["prefetch_speedup_at_slowest_step"] = with_prefetch / no_prefetch
        metrics["slowest_step_ms"] = slow
    buf = shuffle[(shuffle.n_readers == 1) & (shuffle.buffer > 0)]
    metrics["shuffle_spearman_no_shuffle"] = float(shuffle.iloc[0].spearman)
    metrics["shuffle_spearman_buffer_max"] = float(buf.iloc[-1].spearman)
    metrics["shuffle_buffer_max"] = int(buf.iloc[-1].buffer)
    return PartResult(
        "A-csv-pipeline",
        params={
            "batch_size": batch,
            "repeats": repeats,
            "step_ms_values": steps,
            "buffer_sizes": buffers,
            "n_csv_shards": len(paths),
            "n_readers": 5,
        },
        metrics=metrics,
        tables={
            "csv_ablation": ablation,
            "prefetch_sweep": sweep,
            "numpy_baseline": numpy_table,
            "shuffle_quality": shuffle,
            "csv_equivalence": pd.DataFrame([check]),
        },
        extra={"t_input_ms": t_in},
    )
