"""Operational behaviour: registry round trip, safe loading, secrets, drift filtering, lineage.
(The model store and the deploy flow have their own modules.)"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import pytest
from mlflow.exceptions import MlflowException

from tfdata_mlops import tracking

ROOT = Path(__file__).resolve().parents[1]


def test_redact_hides_passwords_in_urls():
    cfg = {
        "database": {"url": "postgresql+psycopg://mlops:s3cret@postgres:5432/app"},
        "mlflow": {"tracking_uri": "http://mlflow:5000"},
        "list": ["sqlite:///app.db", "postgresql://u:p@h/db"],
    }
    out = tracking.redact(cfg)
    assert out["database"]["url"] == "postgresql+psycopg://mlops:***@postgres:5432/app"
    assert out["mlflow"]["tracking_uri"] == "http://mlflow:5000"
    assert out["list"] == ["sqlite:///app.db", "postgresql://u:***@h/db"]
    assert "s3cret" not in str(out)


def test_importing_run_flow_changes_nothing(tmp_path):
    import importlib

    import run_flow

    cwd = os.getcwd()
    importlib.reload(run_flow)
    assert os.getcwd() == cwd  # the chdir happens in main(), not on import
    assert Path(cwd).resolve() == tmp_path.resolve()


def test_prefect_stores_the_config_path_but_never_the_database_password(
    prefect_server, monkeypatch, config_file
):
    """A real flow run through ``run_flow.py``: the flow sees the database URL, Prefect does not."""
    from prefect.client.orchestration import get_client

    import flows.monitor_flow as mon
    import run_flow

    secret = "s3cret-pw"
    monkeypatch.setenv("DATABASE_URL", f"postgresql+psycopg://mlops:{secret}@db:5432/app")
    path = config_file({"extends": str(ROOT / "configs" / "monitor_flow_config.yaml")})
    seen = []
    monkeypatch.setattr(mon, "fetch_predictions", lambda url, *args: seen.append(url) or [])
    monkeypatch.setattr(mon.tracking, "configure", lambda *a, **k: None)

    out = run_flow.main(["--config", path])
    assert out["status"] == "insufficient-data" and secret in seen[0]  # the flow got the URL
    with get_client(sync_client=True) as client:
        stored = [run.parameters for run in client.read_flow_runs()]
    mine = [p for p in stored if path in json.dumps(p)]
    assert mine and all(set(p) == {"config_path"} for p in mine)
    assert secret not in json.dumps(stored)


def test_exit_code_reflects_gates_and_deployment():
    from run_flow import exit_code

    assert exit_code({"gates_passed": True, "deployed": True}) == 0
    assert exit_code({"gates_passed": False, "deployed": False}) == 2
    assert exit_code({"passed": False}) == 2
    assert exit_code({"gates_passed": True, "deployed": False}) == 3
    assert exit_code({"status": "ok"}) == 0  # monitor flow


def test_only_keras_artifacts_are_loaded(tmp_path):
    def artifact(flavors, with_file=True):
        root = tmp_path / str(len(list(tmp_path.iterdir())))
        (root / "data").mkdir(parents=True)
        import yaml

        (root / "MLmodel").write_text(yaml.safe_dump({"flavors": flavors}))
        if with_file:
            (root / "data" / "model.keras").write_bytes(b"x")
        return root

    ok = artifact({"keras": {"data": "data", "keras_backend": "tensorflow"}})
    assert tracking.check_artifact(ok, "ref").name == "model.keras"
    with pytest.raises(ValueError, match="Keras"):
        tracking.check_artifact(artifact({"sklearn": {"pickled_model": "model.pkl"}}), "ref")
    with pytest.raises(ValueError, match="Keras"):
        tracking.check_artifact(artifact({"keras": {"keras_backend": "jax"}}), "ref")
    with pytest.raises(ValueError, match="not found"):
        tracking.check_artifact(artifact({"keras": {"keras_backend": "tensorflow"}}, False), "ref")


def test_model_files_with_lambda_layers_are_refused(tmp_path):
    """safe_mode=True: a model that carries arbitrary code is not loaded."""
    import keras

    inp = keras.Input((2,))
    out = keras.layers.Lambda(lambda x: x * 2)(inp)
    path = tmp_path / "evil.keras"
    keras.Model(inp, out).save(path)
    with pytest.raises(ValueError, match="safe"):
        keras.saving.load_model(path, compile=False, safe_mode=True)


def test_registry_round_trip_with_profile_and_alias(prep, tmp_path):
    """Log -> register -> alias -> load: the loaded model predicts exactly what the original does."""
    import keras
    import mlflow

    from tfdata_mlops import features as F
    from tfdata_mlops.inference import build_profile, predict_rows

    tracking.configure(f"sqlite:///{tmp_path}/mlflow.db", "test")
    keras.utils.set_random_seed(0)
    model = F.build_model(F.FEATURE_SETS["all features"], prep.stats)
    profile = build_profile(prep.split.train, prep.stats, "all features")
    samples = [{"longitude": -120.0}]
    with mlflow.start_run():
        version = tracking.log_model(model, profile, samples)
    assert version == "1" and tracking.latest_version() == "1"
    assert tracking.get_alias_version(tracking.MODEL_NAME, "champion") is None  # no alias yet
    tracking.set_alias(tracking.MODEL_NAME, version)
    assert str(tracking.get_alias_version().version) == "1"
    tracking.set_version_tags(tracking.MODEL_NAME, version, {"data_sha256": "abc"})

    loaded = tracking.load_model(tracking.MODEL_NAME, "champion")
    assert loaded.version == "1" and loaded.profile == profile and loaded.samples == samples
    rows = prep.split.test.head(25).to_dict("records")
    assert np.array_equal(
        predict_rows(model, rows, profile)[0], predict_rows(loaded.model, rows, loaded.profile)[0]
    )
    assert tracking.load_model(tracking.MODEL_NAME, "1").run_id == loaded.run_id


def test_drift_check_ignores_demo_and_ci_traffic(tmp_path):
    from flows.monitor_flow import fetch_predictions
    from tfdata_mlops.db import Prediction, init_db, make_engine

    url = f"sqlite:///{tmp_path}/app.db"
    Session = init_db(make_engine(url))
    with Session() as s:
        for src in ["ui", "ui", "sample-unseen", "sample-clean", "ci", "api"]:
            s.add(
                Prediction(
                    source=src,
                    features={"median_income": 3.0, "ocean_proximity": "INLAND"},
                    prediction_usd=200000.0,
                    unknown_category=src == "sample-unseen",
                    n_missing=0,
                    n_out_of_range=0,
                    model_version="1",
                    latency_ms=2.0,
                )
            )
        s.commit()
    rows = fetch_predictions.fn(url, 24, ("sample-", "ci"))
    assert sorted(r["source"] for r in rows) == ["api", "ui", "ui"]
    assert set(rows[0]) >= {"features", "prediction_usd", "unknown_category", "n_missing"}
    assert len(fetch_predictions.fn(url, 24, ())) == 6
    # "_" and "%" in a prefix are literal characters, not SQL LIKE wildcards
    assert len(fetch_predictions.fn(url, 24, ("u_",))) == 6
    assert len(fetch_predictions.fn(url, 24, ("%",))) == 6


def test_monitor_flow_end_to_end_with_a_fake_profile(prep, tmp_path, monkeypatch, config_file):
    """Rows in the database -> verdict, including an alert for a flood of unseen area types."""
    import flows.monitor_flow as mon
    from tfdata_mlops.db import Prediction, init_db, make_engine
    from tfdata_mlops.inference import build_profile

    profile = build_profile(prep.split.train, prep.stats, "all features")
    url = f"sqlite:///{tmp_path}/app.db"
    Session = init_db(make_engine(url))
    with Session() as s:
        for i, rec in enumerate(prep.split.test.head(60).to_dict("records")):
            rec = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}
            rec.pop("median_house_value")
            s.add(
                Prediction(
                    source="api",
                    features=rec,
                    prediction_usd=profile.target_mean,
                    unknown_category=i % 2 == 0,
                    n_missing=0,
                    n_out_of_range=0,
                    model_version="1",
                    latency_ms=1.0,
                )
            )
        s.commit()
    monkeypatch.setattr(mon, "load_profile", lambda alias: profile)
    monkeypatch.setattr(mon, "fetch_predictions", mon.fetch_predictions.fn)  # no Prefect server
    monkeypatch.setattr(mon.tracking, "configure", lambda *a, **k: None)
    monkeypatch.setattr(
        mon.mlflow, "start_run", lambda *a, **k: __import__("contextlib").nullcontext()
    )
    monkeypatch.setattr(mon.mlflow, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(mon.tracking, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(mon.tracking, "log_json", lambda *a, **k: None)
    monkeypatch.setattr(mon, "get_run_logger", lambda: logging.getLogger("test"))
    cfg = {
        "database": {"url": url},
        "mlflow": {"tracking_uri": None, "experiment": "x"},
        "deploy": {"alias": "champion"},
        "monitor": {
            "window_hours": 24,
            "min_predictions": 20,
            "max_unknown_category_pct": 1.0,
            "max_out_of_range_pct": 5.0,
            "max_missing_rate_pct": 5.0,
            "max_feature_shift_std": 5.0,
            "max_category_tvd": 1.0,
            "max_prediction_shift_std": 5.0,
            "exclude_source_prefixes": [],
        },
    }
    report = mon.monitor_flow.fn(config_file(cfg))
    assert report["status"] == "alert" and report["unknown_category_rate_pct"] == 50.0
    assert any("unseen" in a for a in report["alerts"])


def test_an_interrupted_run_resumes_at_the_first_unfinished_part(prep, tmp_path):
    """A finished part is saved; with resume on it is reused, and a changed config recomputes it."""
    import mlflow

    from flows.train_flow import _run_part
    from tfdata_mlops.results import PartResult

    tracking.configure(f"sqlite:///{tmp_path}/resume.db", "resume-test")
    cfg = {"seed": 1, "resume": "true", "part_a": {"repeats": 1}}
    calls = []

    def compute():
        calls.append(1)
        return PartResult(name="A", params={"x": 1}, metrics={"m": 2.0})

    with mlflow.start_run() as parent:
        pid = parent.info.run_id
        first = _run_part("A", "A-test", prep, cfg, pid, compute)
        again = _run_part("A", "A-test", prep, cfg, pid, compute)  # resumed, not recomputed
        assert len(calls) == 1 and again.metrics == first.metrics == {"m": 2.0}
        cfg_changed = {**cfg, "part_a": {"repeats": 2}}
        _run_part("A", "A-test", prep, cfg_changed, pid, compute)  # different config -> recompute
        assert len(calls) == 2
        _run_part("A", "A-test", prep, {**cfg, "resume": "false"}, pid, compute)  # resume off
        assert len(calls) == 3


def test_latest_version_raises_real_registry_errors(monkeypatch):
    """Only "no such model" means "no version"; an unreachable registry is an error."""

    from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST

    class Client:
        def __init__(self, code):
            self.code = code

        def search_model_versions(self, query):
            raise MlflowException("boom", error_code=self.code)

    monkeypatch.setattr(tracking, "MlflowClient", lambda: Client(RESOURCE_DOES_NOT_EXIST))
    assert tracking.latest_version() is None
    monkeypatch.setattr(tracking, "MlflowClient", lambda: Client(INTERNAL_ERROR))
    with pytest.raises(MlflowException):
        tracking.latest_version()


def test_git_commit_falls_back_to_the_environment(monkeypatch):
    """In the Docker image there is no git: the GIT_COMMIT variable is recorded instead."""
    import subprocess

    import flows.train_flow as train

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(train.subprocess, "run", no_git)
    monkeypatch.setenv("GIT_COMMIT", "abc1234")
    assert train.git_commit() == "abc1234"
    monkeypatch.setenv("GIT_COMMIT", "  ")
    assert train.git_commit() == "unknown"
    monkeypatch.setattr(
        train.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="def5678-dirty\n"),
    )
    assert train.git_commit() == "def5678-dirty"  # git itself wins when it works
