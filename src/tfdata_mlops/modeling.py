"""Training and evaluating the housing regressor on tf.data pipelines (Geron, Ch. 13, "Using the
Dataset With tf.keras").

The book passes the datasets straight to ``fit()`` and ``evaluate()`` and gives the number of steps
per epoch, because the training set repeats forever. That is what ``fit_model`` does.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from .data import CATEGORICAL, NUMERIC, TARGET_SCALE, Stats
from .features import FeatureSet, build_model
from .tfrecord import tfrecord_reader_dataset


def set_seed(seed: int) -> None:
    keras.utils.set_random_seed(seed)  # Python, NumPy and TensorFlow


def df_to_features(df: pd.DataFrame, stats: Stats) -> dict[str, tf.Tensor]:
    """Raw rows as the model's input dict (each value shaped ``(n, 1)``).

    Missing numbers are replaced by the training medians, which is what both pipelines do.
    """
    filled = df[NUMERIC].fillna(dict(zip(NUMERIC, stats.medians, strict=True)))
    out = {name: tf.constant(filled[name].to_numpy(np.float32)[:, None]) for name in NUMERIC}
    out[CATEGORICAL] = tf.constant(df[CATEGORICAL].astype(str).to_numpy()[:, None])
    return out


def predict_dollars(model: keras.Model, features: dict, batch_size: int = 2048) -> np.ndarray:
    """Median house values in dollars for a dict of raw features (any number of rows)."""
    n = int(next(iter(features.values())).shape[0])
    outputs = []
    for start in range(0, n, batch_size):
        batch = {k: v[start : start + batch_size] for k, v in features.items()}
        outputs.append(model(batch, training=False).numpy().ravel())
    return np.concatenate(outputs) * TARGET_SCALE


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": 1.0 - ss_res / ss_tot,
    }


@dataclass
class FitResult:
    model: keras.Model
    history: dict
    epochs_run: int
    best_epoch: int
    train_seconds: float


def fit_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    valid_ds: tf.data.Dataset,
    *,
    steps_per_epoch: int,
    validation_steps: int,
    epochs: int = 100,
    patience: int = 10,
    verbose: int = 0,
) -> FitResult:
    """``fit()`` on a dataset that repeats forever, with early stopping on the validation loss.

    As in the book, the number of steps is given for the training set (it never ends) and for
    the validation set (so Keras knows where one pass over it stops).
    """
    start = time.perf_counter()
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=valid_ds,
        validation_steps=validation_steps,
        shuffle=False,  # shuffling happens in the dataset
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            )
        ],
        verbose=verbose,
    )
    seconds = time.perf_counter() - start
    val = history.history["val_loss"]
    return FitResult(
        model=model,
        history={k: [float(v) for v in vs] for k, vs in history.history.items()},
        epochs_run=len(val),
        best_epoch=int(np.argmin(val)) + 1,
        train_seconds=seconds,
    )


def tfrecord_datasets(
    paths: dict[str, Sequence[str]],
    stats: Stats,
    *,
    compression: str | None,
    batch_size: int,
    seed: int | None,
    n_readers: int = 5,
    shuffle_buffer_size: int = 10_000,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Training (repeats forever, shuffled), validation and test (one ordered pass) datasets."""
    train = tfrecord_reader_dataset(
        paths["train"],
        stats,
        compression=compression,
        repeat=None,
        batch_size=batch_size,
        n_readers=n_readers,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        n_read_threads=tf.data.AUTOTUNE,
        n_parse_threads=tf.data.AUTOTUNE,
        prefetch=tf.data.AUTOTUNE,
    )
    ordered = dict(
        compression=compression,
        batch_size=batch_size,
        n_readers=1,
        shuffle_buffer_size=0,
        shuffle_files=False,
        prefetch=tf.data.AUTOTUNE,
    )
    valid = tfrecord_reader_dataset(paths["valid"], stats, **ordered)
    test = tfrecord_reader_dataset(paths["test"], stats, **ordered)
    return train, valid, test


def predict_dataset(model: keras.Model, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Predictions and targets (both in dollars) for an ordered dataset."""
    preds, targets = [], []
    for features, y in dataset:
        preds.append(model(features, training=False).numpy().ravel())
        targets.append(y.numpy().ravel())
    return np.concatenate(preds) * TARGET_SCALE, np.concatenate(targets) * TARGET_SCALE


def train_and_evaluate(
    fs: FeatureSet,
    stats: Stats,
    paths: dict[str, Sequence[str]],
    n_train: int,
    n_valid: int,
    *,
    compression: str | None,
    seed: int,
    batch_size: int = 128,
    hidden: tuple[int, ...] = (64, 64),
    learning_rate: float = 2e-3,
    epochs: int = 100,
    patience: int = 10,
) -> dict:
    """Train one model from TFRecord shards and report validation and test errors in dollars."""
    set_seed(seed)
    model = build_model(fs, stats, hidden, learning_rate)
    train, valid, test = tfrecord_datasets(
        paths, stats, compression=compression, batch_size=batch_size, seed=seed
    )
    fit = fit_model(
        model,
        train,
        valid,
        steps_per_epoch=n_train // batch_size,
        validation_steps=-(-n_valid // batch_size),  # ceil: every validation row is used
        epochs=epochs,
        patience=patience,
    )
    out = {"fit": fit}
    for name, ds in (("valid", valid), ("test", test)):
        pred, true = predict_dataset(model, ds)
        out[name] = regression_metrics(true, pred)
        out[f"{name}_pred"] = pred
    out["n_params"] = int(model.count_params())
    out["model"] = model
    return out
