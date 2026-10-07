"""Part E: the model that gets deployed, and the proof that serving matches training.

The model takes raw columns. Its preprocessing layers are part of the saved file, so a request is
handled by the very same standardising, bucketing and encoding that the training pipeline fed it
with (the book's argument for putting preprocessing in the model, or in TF Transform).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from . import features as F
from .baselines import mean_baseline
from .data import CATEGORICAL, TARGET
from .gates import evaluate_model
from .inference import build_profile, predict_rows
from .modeling import train_and_evaluate
from .prepare import Prepared
from .results import PartResult


def error_by_group(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    frame = test.assign(pred=pred, err=pred - test[TARGET].to_numpy())
    income = pd.cut(
        frame["median_income"],
        [0, 1.5, 3, 4.5, 6, np.inf],
        labels=["<1.5", "1.5-3", "3-4.5", "4.5-6", ">6"],
    )
    out = []
    for kind, key in (("ocean_proximity", frame[CATEGORICAL]), ("median_income", income)):
        for group, part in frame.groupby(key, observed=True):
            out.append(
                {
                    "group_by": kind,
                    "group": str(group),
                    "rows": len(part),
                    "rmse": float(np.sqrt((part.err**2).mean())),
                    "mean_error": float(part.err.mean()),
                }
            )
    return pd.DataFrame(out)


def run_part_e(
    prep: Prepared, cfg: dict, seed: int, best_feature_set: str | None = None
) -> PartResult:
    name = cfg.get("feature_set", "auto")
    if name == "auto":
        name = best_feature_set or F.DEFAULT_FEATURE_SET
    fs = F.FEATURE_SETS[name]
    compression = cfg.get("compression", "GZIP")
    hidden = tuple(int(h) for h in cfg.get("hidden", [64, 64]))
    res = train_and_evaluate(
        fs,
        prep.stats,
        prep.tfrecords(compression),
        len(prep.split.train),
        len(prep.split.valid),
        compression=compression,
        seed=seed,
        batch_size=int(cfg.get("batch_size", 128)),
        hidden=hidden,
        learning_rate=float(cfg.get("learning_rate", 2e-3)),
        epochs=int(cfg.get("epochs", 100)),
        patience=int(cfg.get("patience", 10)),
    )
    model, fit = res["model"], res["fit"]
    profile = build_profile(prep.split.train, prep.stats, name)
    test = prep.split.test
    rows = test.to_dict("records")
    served, flags = predict_rows(model, rows, profile)

    evaluation = evaluate_model(model, profile, prep, compression)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.keras"
        model.save(path)
        reloaded = keras.saving.load_model(path, compile=False, safe_mode=True)
    roundtrip = float(np.abs(predict_rows(reloaded, rows, profile)[0] - served).max())

    rng = np.random.default_rng(seed)
    n_samples = min(int(cfg.get("n_samples", 200)), len(rows))
    samples = [
        {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rows[i].items()}
        for i in rng.choice(len(rows), size=n_samples, replace=False)
    ]
    metrics = {
        **evaluation,
        "valid_rmse": res["valid"]["rmse"],
        "valid_r2": res["valid"]["r2"],
        "baseline_mean_test_rmse": mean_baseline(prep)["rmse"],
        "roundtrip_max_abs_usd": roundtrip,
        "epochs_run": fit.epochs_run,
        "best_epoch": fit.best_epoch,
        "train_seconds": fit.train_seconds,
        "n_params": res["n_params"],
        "rows_with_missing_value": int(sum(bool(f.missing) for f in flags)),
        "rows_outside_training_range": int(sum(bool(f.out_of_range) for f in flags)),
    }
    predictions = pd.DataFrame(
        {
            "median_house_value": test[TARGET].to_numpy(),
            "prediction": served,
            CATEGORICAL: test[CATEGORICAL],
        }
    )
    return PartResult(
        "E-deployed-model",
        params={
            "feature_set": name,
            "hidden": list(hidden),
            "compression": compression,
            "seed": seed,
        },
        metrics=metrics,
        tables={
            "error_by_group": error_by_group(test, served),
            "training_history": pd.DataFrame(fit.history).assign(
                epoch=lambda d: np.arange(1, len(d) + 1)
            ),
            "test_predictions": predictions,
        },
        extra={"model": model, "profile": profile, "samples": samples, "feature_set": name},
    )
