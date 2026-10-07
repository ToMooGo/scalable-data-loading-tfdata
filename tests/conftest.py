"""Shared fixtures. Unit tests run offline on synthetic housing data with the same columns as the
book's ``housing.csv``; only the integration test downloads real data."""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# Prefect keeps its state under PREFECT_HOME (without a server: a temporary SQLite database). The
# tests use a throw-away folder and never reach a Prefect server configured in the shell. This must
# happen before Prefect is imported.
_PREFECT_HOME = tempfile.mkdtemp(prefix="prefect-home-")
atexit.register(shutil.rmtree, _PREFECT_HOME, ignore_errors=True)
os.environ["PREFECT_HOME"] = _PREFECT_HOME
os.environ.pop("PREFECT_API_URL", None)
os.environ["PREFECT_SERVER_ANALYTICS_ENABLED"] = "false"
os.environ["PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW"] = "ignore"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src", ROOT / "services" / "api"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tfdata_mlops.data import CSV_COLUMNS, OCEAN_VOCAB  # noqa: E402
from tfdata_mlops.prepare import prepare_frame  # noqa: E402


def synthetic_housing(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Rows shaped like the book's data: California coordinates, a skewed income, a few missing
    ``total_bedrooms`` and a price that depends on income, latitude and the category."""
    rng = np.random.default_rng(seed)
    income = np.clip(rng.lognormal(1.2, 0.5, n), 0.5, 15)
    lat = rng.uniform(32.6, 41.9, n)
    lon = rng.uniform(-124.3, -114.4, n)
    rooms = rng.lognormal(7.5, 0.6, n).round()
    bedrooms = (rooms * rng.uniform(0.15, 0.25, n)).round().astype(float)
    bedrooms[rng.random(n) < 0.02] = np.nan
    category = rng.choice(OCEAN_VOCAB, n, p=[0.45, 0.3, 0.001, 0.12, 0.129])
    price = 40_000 * income + 3_000 * (lat - 32) + 20_000 * (category == "NEAR BAY")
    price = np.clip(price + rng.normal(0, 25_000, n), 15_000, 500_001)
    df = pd.DataFrame(
        {
            "longitude": lon.round(2),
            "latitude": lat.round(2),
            "housing_median_age": rng.integers(1, 52, n).astype(float),
            "total_rooms": rooms,
            "total_bedrooms": bedrooms,
            "population": rng.lognormal(6.5, 0.6, n).round(),
            "households": rng.lognormal(5.8, 0.6, n).round(),
            "median_income": income.round(4),
            "ocean_proximity": category,
            "median_house_value": price.round(),
        }
    )
    return df[CSV_COLUMNS]


@pytest.fixture(scope="session")
def housing() -> pd.DataFrame:
    return synthetic_housing()


@pytest.fixture(scope="session")
def prep(housing, tmp_path_factory):
    root = tmp_path_factory.mktemp("work")
    return prepare_frame(housing, root, 7, shards={"train": 6, "valid": 3, "test": 3})


@pytest.fixture(autouse=True)
def _isolated_services(tmp_path, monkeypatch):
    """No test may touch a developer's MLflow store or prediction database, whatever the shell
    exports, or leave files in the repository (MLflow writes artifacts to ``./mlruns``)."""
    import mlflow

    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path}/isolated_mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/isolated_app.db")
    monkeypatch.delenv("API_URL", raising=False)
    mlflow.set_tracking_uri(uri)


@pytest.fixture(scope="session")
def prefect_server():
    """A throw-away Prefect API (SQLite in a temporary folder) for tests that run real flows."""
    from prefect.testing.utilities import prefect_test_harness

    with prefect_test_harness(server_startup_timeout=120):
        yield


@pytest.fixture()
def config_file(tmp_path):
    """``config_file(cfg)`` writes ``cfg`` as YAML and returns its path (the flows take a path)."""

    def write(cfg: dict, name: str = "config.yaml") -> str:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return str(path)

    return write
