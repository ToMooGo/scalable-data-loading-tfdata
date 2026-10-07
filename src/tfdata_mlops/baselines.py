"""The yardsticks a neural network must beat (Geron, Ch. 2: the linear model on prepared data)."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, NUMERIC, TARGET
from .modeling import regression_metrics
from .prepare import Prepared


def linear_baseline(prep: Prepared) -> dict:
    """Chapter 2's recipe: median imputation, standardised numbers, one-hot category, linear model."""
    columns = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [CATEGORICAL]),
        ]
    )
    model = make_pipeline(columns, LinearRegression())
    train, test = prep.split.train, prep.split.test
    model.fit(train[NUMERIC + [CATEGORICAL]], train[TARGET])
    return regression_metrics(test[TARGET].to_numpy(), model.predict(test[NUMERIC + [CATEGORICAL]]))


def mean_baseline(prep: Prepared) -> dict:
    """Always predict the training mean."""
    test = prep.split.test
    return regression_metrics(
        test[TARGET].to_numpy(), np.full(len(test), prep.split.train[TARGET].mean())
    )
