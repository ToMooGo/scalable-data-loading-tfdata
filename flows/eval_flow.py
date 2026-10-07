"""Evaluation flow: re-evaluate the *registered* candidate on the held-out test set, apply the
quality gates and a champion-vs-challenger check. Only a passing candidate is deployed."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NONE

from tfdata_mlops import tracking
from tfdata_mlops.config import load_config
from tfdata_mlops.gates import all_passed, evaluate_gates, evaluate_model
from tfdata_mlops.prepare import prepare_data

ROOT = Path(__file__).resolve().parents[1]


@task(name="load-model", cache_policy=NONE)
def load_registered(version: str) -> tracking.LoadedModel:
    return tracking.load_model(tracking.MODEL_NAME, version)


@task(name="evaluate-model", cache_policy=NONE)
def evaluate(loaded: tracking.LoadedModel, config: dict) -> dict:
    # the split and the shards are rebuilt from the same data and seed: the test rows are the
    # ones the training run held out
    prep = prepare_data(config["data"], int(config["seed"]))
    return evaluate_model(
        loaded.model, loaded.profile, prep, config["part_c"].get("compression", "GZIP")
    )


@task(name="quality-gates", cache_policy=NONE)
def check_gates(candidate, champion, cfg, training_rmse) -> list[dict]:
    return evaluate_gates(
        candidate,
        champion,
        cfg["eval"]["gates"],
        float(cfg["eval"].get("champion_tolerance_pct", 1.0)),
        training_rmse,
    )


@flow(name="eval-flow", log_prints=True)
def eval_flow(config_path: str, model_version: str | None = None) -> dict:
    """``config_path``, not the config: Prefect stores flow parameters in plain text."""
    logger = get_run_logger()
    config = load_config(config_path)
    tracking.configure(config["mlflow"]["tracking_uri"], config["mlflow"]["experiment"])
    version = model_version or tracking.latest_version()
    if version is None:
        raise RuntimeError("No registered model found - run the train flow first.")
    loaded = load_registered(version)
    candidate = evaluate(loaded, config)
    training_rmse = tracking.run_metrics(loaded.run_id).get("test_rmse")

    champion = None
    mv = tracking.get_alias_version(tracking.MODEL_NAME, config["deploy"]["alias"])
    if mv is not None and str(mv.version) != str(version):
        champion = evaluate(load_registered(str(mv.version)), config)

    checks = check_gates(candidate, champion, config, training_rmse)
    passed = all_passed(checks)
    with mlflow.start_run(run_name="evaluate", tags={"flow": "eval", "passed": str(passed)}):
        mlflow.log_param("candidate_version", version)
        tracking.log_metrics(candidate, prefix="candidate_")
        if champion:
            tracking.log_metrics(champion, prefix="champion_")
        tracking.log_json(checks, "quality_gates.json")
        mlflow.log_metric("gates_passed", float(passed))
    report = ROOT / config.get("reports", {}).get("dir", "reports") / "quality_gates.json"
    try:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {"model_version": str(version), "passed": passed, "checks": checks}, indent=2
            ),
            encoding="utf-8",
        )
    except OSError as exc:  # a read-only reports folder must not stop an evaluation
        logger.warning("Could not write %s: %s", report, exc)
    for c in checks:
        logger.info(
            "%-4s %-72s value=%.4f limit=%.4f",
            "PASS" if c["passed"] else "FAIL",
            c["check"],
            c["value"],
            c["limit"],
        )
    return {"passed": passed, "checks": checks, "model_version": str(version), "metrics": candidate}


def start(config_path: str) -> dict:
    return eval_flow(config_path)
