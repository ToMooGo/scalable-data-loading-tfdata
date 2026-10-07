"""Thin helpers around MLflow tracking and the Model Registry (Keras model + its input profile)."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import mlflow.keras  # noqa: E402
import yaml  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from .inference import Profile  # noqa: E402

log = logging.getLogger(__name__)

MODEL_NAME = "housing-price-model"
CHAMPION = "champion"
PROFILE_ARTIFACT = "profile/profile.json"
SAMPLES_ARTIFACT = "profile/samples.json"


def configure(tracking_uri: str | None, experiment: str) -> str:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    return mlflow.set_experiment(experiment).experiment_id


def _is_number(v) -> bool:
    return (
        isinstance(v, (int, float, np.integer, np.floating))
        and not isinstance(v, bool)
        and math.isfinite(float(v))
    )


def log_metrics(metrics: dict, prefix: str = "") -> None:
    mlflow.log_metrics({f"{prefix}{k}": float(v) for k, v in metrics.items() if _is_number(v)})


def log_params(params: dict, prefix: str = "") -> None:
    mlflow.log_params({f"{prefix}{k}": str(v)[:500] for k, v in params.items()})


def log_table(df: pd.DataFrame, artifact_file: str) -> None:
    mlflow.log_text(df.to_csv(index=False), artifact_file)


def log_json(obj, artifact_file: str) -> None:
    mlflow.log_text(json.dumps(obj, indent=2, default=float), artifact_file)


def log_model(model, profile: Profile, samples: list[dict], name: str = MODEL_NAME) -> str:
    """Log the Keras model (preprocessing layers included) and register it; return the version.

    The profile (training statistics and ranges) and a few example rows are stored in the same
    run, so the service can describe what the model was trained on and flag inputs outside it.
    """
    info = mlflow.keras.log_model(model, name="model", registered_model_name=name)
    mlflow.log_dict(profile.to_dict(), PROFILE_ARTIFACT)
    mlflow.log_dict(samples, SAMPLES_ARTIFACT)
    return str(info.registered_model_version)


def set_alias(name: str, version: str, alias: str = CHAMPION) -> None:
    MlflowClient().set_registered_model_alias(name, alias, str(version))


# what MLflow answers when the model or the alias does not exist yet (anything else, such as a
# connection error, is a real failure and is raised)
_MISSING = {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}


def get_alias_version(name: str = MODEL_NAME, alias: str = CHAMPION):
    try:
        return MlflowClient().get_model_version_by_alias(name, alias)
    except MlflowException as exc:
        if exc.error_code in _MISSING:
            return None
        raise


def set_version_tags(name: str, version: str, tags: dict) -> None:
    client = MlflowClient()
    for k, v in tags.items():
        client.set_model_version_tag(name, str(version), k, str(v))


_SECRET = re.compile(r"(://[^:/@\s]+:)[^@\s]+@")


def redact(obj):
    """Copy of a config with passwords in connection URLs replaced by ``***``."""
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return _SECRET.sub(r"\1***@", obj)
    return obj


def delete_alias(name: str, alias: str = CHAMPION) -> None:
    """Remove an alias (a rollback of the very first deployment: there is no older champion)."""
    try:
        MlflowClient().delete_registered_model_alias(name, alias)
    except MlflowException as exc:
        if exc.error_code not in _MISSING:
            raise


def latest_version(name: str = MODEL_NAME) -> str | None:
    try:
        versions = MlflowClient().search_model_versions(f"name='{name}'")
    except MlflowException as exc:
        if exc.error_code in _MISSING:  # no such model yet
            return None
        raise
    return str(max(int(v.version) for v in versions)) if versions else None


def run_metrics(run_id: str) -> dict:
    return dict(MlflowClient().get_run(run_id).data.metrics)


def check_artifact(root: Path, ref: str) -> Path:
    """Return the ``.keras`` file of a downloaded model, refusing anything else.

    Only the Keras flavor is accepted, and the file is loaded with ``safe_mode=True`` (no arbitrary
    code from ``Lambda`` layers). The artifact comes from the registry, so it is checked, not trusted.
    """
    mlmodel = yaml.safe_load((root / "MLmodel").read_text(encoding="utf-8"))
    flavors = mlmodel.get("flavors", {})
    if "keras" not in flavors or flavors["keras"].get("keras_backend") != "tensorflow":
        raise ValueError(f"{ref}: expected a Keras model with the TensorFlow backend")
    path = root / flavors["keras"].get("data", "data") / "model.keras"
    if not path.is_file():
        raise ValueError(f"{ref}: model.keras not found in the artifact")
    return path


@dataclass
class LoadedModel:
    model: object
    profile: Profile
    samples: list[dict]
    version: str
    run_id: str


def load_model(name: str = MODEL_NAME, version_or_alias: str = CHAMPION) -> LoadedModel:
    """Load a registered version (a number) or whatever an alias points at, with its profile."""
    import keras

    client = MlflowClient()
    if str(version_or_alias).isdigit():
        mv = client.get_model_version(name, str(version_or_alias))
    else:
        mv = client.get_model_version_by_alias(name, version_or_alias)
    ref = f"models:/{name}/{mv.version}"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(mlflow.artifacts.download_artifacts(artifact_uri=ref, dst_path=tmp))
        model = keras.saving.load_model(check_artifact(root, ref), compile=False, safe_mode=True)
        profile_dir = Path(tmp) / "run"
        profile_path = download_run_artifact(mv.run_id, PROFILE_ARTIFACT, profile_dir)
        samples_path = download_run_artifact(mv.run_id, SAMPLES_ARTIFACT, profile_dir)
        profile = Profile.from_dict(json.loads(profile_path.read_text(encoding="utf-8")))
        samples = json.loads(samples_path.read_text(encoding="utf-8"))
    return LoadedModel(model, profile, samples, str(mv.version), str(mv.run_id))


def download_run_artifact(run_id: str, path: str, dst: str | Path) -> Path:
    return Path(
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path, dst_path=str(dst))
    )
