"""Monitoring flow: read recent predictions from the database and check for input drift.

It compares the requests of the last days with what the deployed model was trained on: unseen
categories, values outside the training range, missing values and shifts of the means. Run once
with ``python run_flow.py --config configs/monitor_flow_config.yaml`` or on a schedule with
``python -m flows.serve_monitor`` (the ``monitor`` profile in docker-compose.yml).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import mlflow
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NONE
from sqlalchemy import select

from tfdata_mlops import tracking
from tfdata_mlops.config import load_config
from tfdata_mlops.db import Prediction, init_db, make_engine
from tfdata_mlops.inference import Profile
from tfdata_mlops.monitoring import summarize_predictions


@task(name="fetch-predictions", cache_policy=NONE)
def fetch_predictions(url: str, window_hours: float, exclude_prefixes=()) -> list[dict]:
    """Real traffic of the window. Requests whose ``source`` starts with one of
    ``exclude_prefixes`` (the UI's demo rows, CI smoke tests) are left out, so demos of unusual
    inputs do not raise drift alerts."""
    engine = make_engine(url)
    init_db(engine)
    since = datetime.now(UTC) - timedelta(hours=window_hours)
    query = select(
        Prediction.features,
        Prediction.prediction_usd,
        Prediction.unknown_category,
        Prediction.n_missing,
        Prediction.n_out_of_range,
        Prediction.source,
    ).where(Prediction.created_at >= since)
    for prefix in exclude_prefixes:  # autoescape: "_" and "%" in a prefix are literal characters
        query = query.where(~Prediction.source.startswith(prefix, autoescape=True))
    with engine.connect() as conn:
        out = [dict(r._mapping) for r in conn.execute(query)]
    engine.dispose()
    return out


@task(name="load-profile", cache_policy=NONE)
def load_profile(alias: str) -> Profile:
    return tracking.load_model(tracking.MODEL_NAME, alias).profile


@flow(name="monitor-flow", log_prints=True)
def monitor_flow(config_path: str) -> dict:
    """``config_path``, not the config: the resolved config holds the database URL with its
    password, and Prefect stores flow parameters in plain text."""
    logger = get_run_logger()
    config = load_config(config_path)
    mcfg = config["monitor"]
    tracking.configure(config["mlflow"]["tracking_uri"], config["mlflow"]["experiment"])
    rows = fetch_predictions(
        config["database"]["url"],
        float(mcfg["window_hours"]),
        tuple(mcfg.get("exclude_source_prefixes", [])),
    )
    if len(rows) < int(mcfg["min_predictions"]):
        logger.info("Only %d predictions in the window - not enough for a verdict.", len(rows))
        return {"n_predictions": len(rows), "alerts": [], "status": "insufficient-data"}
    report = summarize_predictions(rows, mcfg, load_profile(config["deploy"]["alias"]))

    tracking.configure(
        config["mlflow"]["tracking_uri"], config["mlflow"]["experiment"] + "-monitoring"
    )
    with mlflow.start_run(run_name=f"monitor-{datetime.now(UTC):%Y%m%d-%H%M}"):
        tracking.log_metrics(
            {
                k: v
                for k, v in report.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
        )
        tracking.log_metrics(report["feature_shift_std"], prefix="shift_")
        tracking.log_json(report, "monitor_report.json")
    for a in report["alerts"]:
        logger.warning("ALERT: %s", a)
    logger.info("Monitoring status: %s (%d predictions)", report["status"], report["n_predictions"])
    return report


@flow(name="scheduled-monitor")
def scheduled_monitor(config_path: str = "configs/monitor_flow_config.yaml") -> dict:
    """Entry point for the schedule: only the config path is stored in Prefect (no secrets)."""
    return monitor_flow(config_path)


def start(config_path: str) -> dict:
    return monitor_flow(config_path)
