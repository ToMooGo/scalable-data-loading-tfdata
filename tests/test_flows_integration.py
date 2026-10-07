"""End-to-end: train -> evaluate -> deploy with Prefect and a local MLflow (SQLite) on the quick
config. Downloads housing.csv and MNIST once (cached in data/). Run with ``pytest -m integration``."""

import pytest

pytestmark = pytest.mark.integration


def test_full_flow_quick(tmp_path, monkeypatch, prefect_server, config_file):
    from pathlib import Path

    import mlflow

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)  # data/ (the download cache) resolves from the repository root
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    from flows.full_flow import full_flow
    from tfdata_mlops.config import load_config

    # the flows take a config *path*: a small file on top of the quick config
    path = config_file(
        {
            "extends": str(root / "configs" / "quick_flow_config.yaml"),
            "data": {"work_dir": str(tmp_path / "work")},
            "reports": {
                "dir": str(tmp_path / "reports"),
                "figures_dir": str(tmp_path / "reports/figures"),
                "tables_dir": str(tmp_path / "reports/tables"),
            },
            "part_c": {"feature_sets": ["+ ocean one-hot", "all features"], "n_seeds": 2},
            "part_d": {"n_train": 3000, "n_shards": 3},
            "deploy": {"api_url": "http://127.0.0.1:9", "require_api": False, "reload_retries": 1},
        }
    )
    # artifacts in tmp_path, not ./mlruns in the repository
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(
        load_config(path)["mlflow"]["experiment"], artifact_location=(tmp_path / "art").as_uri()
    )
    out = full_flow(path)
    assert out["gates_passed"] and out["deployed"]
    assert (tmp_path / "reports" / "RESULTS.md").exists()
    assert len(list((tmp_path / "reports" / "figures").glob("*.png"))) >= 1

    from tfdata_mlops import tracking

    loaded = tracking.load_model(tracking.MODEL_NAME, "champion")
    assert loaded.version == out["model_version"] and loaded.profile.n_train > 10_000
