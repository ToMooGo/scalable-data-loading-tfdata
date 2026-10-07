"""Full flow: train -> evaluate -> deploy, from one config."""

from __future__ import annotations

from prefect import flow, get_run_logger

from flows.deploy_flow import deploy_flow
from flows.eval_flow import eval_flow
from flows.train_flow import train_flow


@flow(name="full-flow", log_prints=True)
def full_flow(config_path: str) -> dict:
    """Every flow gets the config *path* and loads the file itself: Prefect stores flow and subflow
    parameters in plain text, and the resolved config contains the database URL."""
    trained = train_flow(config_path)
    evaluated = eval_flow(config_path, trained["model_version"])
    deployed = deploy_flow(config_path, trained["model_version"], evaluated["passed"])
    summary = {
        "parent_run_id": trained["parent_run_id"],
        "model_version": trained["model_version"],
        "gates_passed": evaluated["passed"],
        "deployed": deployed["deployed"],
        "api_reloaded": deployed.get("reloaded", False),
        "test_metrics": evaluated["metrics"],
    }
    get_run_logger().info("Full flow finished: %s", summary)
    return summary


def start(config_path: str) -> dict:
    return full_flow(config_path)
