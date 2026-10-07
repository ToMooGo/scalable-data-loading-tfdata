"""Deploy flow: promote the evaluated model to the ``champion`` alias and hot-reload the API.

Nothing is promoted without a passing quality-gate result: ``full_flow`` hands over the result of
its own evaluation, and ``python run_flow.py --flow deploy`` runs the evaluation first.

After the promotion the API is asked to reload; it must then serve the new version and answer a
canary prediction on a held-out row. If the API answers but cannot do that (an error status, a
reply that is not JSON, another version, a failed canary), the previous champion is restored and
the API is asked to load it again. If the API cannot be reached at all, the promotion stays (the
API loads the new champion on its next registry poll or start), unless ``deploy.require_api`` is
true: then the deployment is rolled back as well and the flow fails.
"""

from __future__ import annotations

import logging
import math
import os
import time

import httpx
import mlflow
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NONE
from prefect.exceptions import MissingContextError

from flows.eval_flow import eval_flow
from tfdata_mlops import tracking
from tfdata_mlops.config import load_config

CANARY_SOURCE = "deploy-canary"  # excluded from drift monitoring (configs/monitor_flow_config.yaml)


class ReloadFailed(RuntimeError):
    """The API answered, but does not serve the expected version correctly."""


def _logger():
    try:
        return get_run_logger()
    except MissingContextError:  # called outside a flow run (unit tests)
        return logging.getLogger(__name__)


def _http_client(api_url: str) -> httpx.Client:
    token = os.getenv("ADMIN_TOKEN")
    headers = {"X-Admin-Token": token} if token else {}
    return httpx.Client(base_url=api_url, timeout=60, headers=headers)


def _json(response: httpx.Response, what: str) -> dict:
    if response.status_code != 200:
        raise ReloadFailed(f"{what} answered HTTP {response.status_code}")
    try:
        body = response.json()
    except ValueError as exc:  # an HTML error page from a proxy, a truncated reply ...
        raise ReloadFailed(f"{what} did not answer JSON") from exc
    if not isinstance(body, dict):
        raise ReloadFailed(f"{what} answered a {type(body).__name__}, not a JSON object")
    return body


def _check_served(client: httpx.Client, expected: str) -> None:
    """POST /reload, then check the version on /health and a canary prediction."""
    _json(client.post("/reload"), "POST /reload")
    served = str(_json(client.get("/health"), "GET /health").get("model_version"))
    if served != expected:
        raise ReloadFailed(f"API serves version {served}, expected {expected}")
    sample = client.get("/samples", params={"kind": "clean", "seed": 0})
    if sample.status_code == 404:  # a model registered without example rows: nothing to send
        return
    row = _json(sample, "GET /samples").get("row")
    if not isinstance(row, dict):
        raise ReloadFailed("GET /samples returned no row")
    answer = _json(
        client.post("/predict", json={**row, "source": CANARY_SOURCE}), "canary POST /predict"
    )
    price = answer.get("price_usd")
    if (
        str(answer.get("model_version")) != expected
        or not isinstance(price, int | float)
        or not math.isfinite(price)
    ):
        raise ReloadFailed(f"canary prediction is not usable: {answer}")


@task(name="promote-to-champion", cache_policy=NONE)
def promote(version: str, alias: str) -> None:
    tracking.set_alias(tracking.MODEL_NAME, version, alias)
    _logger().info("%s v%s -> @%s", tracking.MODEL_NAME, version, alias)


@task(name="reload-api", cache_policy=NONE)
def reload_api(api_url: str, expected: str, retries: int) -> dict:
    """Ask the API to load the champion and check that it serves ``expected``.

    Returns ``reloaded`` and ``reachable``; ``reachable`` is False only if no attempt got an answer.
    """
    logger = _logger()
    last_error, reachable = None, False
    for attempt in range(1, retries + 1):
        try:
            with _http_client(api_url) as client:
                _check_served(client, expected)
            logger.info("API at %s now serves version %s", api_url, expected)
            return {"reloaded": True, "reachable": True, "served": expected}
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            last_error = f"API not reachable: {exc!r}"
        except (ReloadFailed, httpx.HTTPError) as exc:  # it answered, or broke off mid-request
            reachable, last_error = True, str(exc) or repr(exc)
        logger.warning("Reload attempt %d/%d failed: %s", attempt, retries, last_error)
        if attempt < retries:
            time.sleep(min(2 * attempt, 10))
    return {"reloaded": False, "reachable": reachable, "error": last_error}


def _rollback(previous, alias: str, api_url: str, retries: int) -> dict:
    """Restore the registry as it was before the promotion and ask the API to serve that again."""
    logger = _logger()
    if previous is None:  # the first deployment: there is no older champion to go back to
        logger.error("Rolling back: removing @%s (there was no champion before)", alias)
        tracking.delete_alias(tracking.MODEL_NAME, alias)
        return {"rolled_back_to": None}
    version = str(previous.version)
    logger.error("Rolling back: restoring the previous champion v%s", version)
    promote(version, alias)
    try:
        restored = reload_api(api_url, version, retries)
    except Exception as exc:  # the registry is restored; the API catches up on its next poll
        restored = {"reloaded": False, "error": repr(exc)}
    if not restored["reloaded"]:
        logger.error(
            "The API did not confirm v%s (%s); it loads it on its next registry poll",
            version,
            restored["error"],
        )
    return {"rolled_back_to": version, "api_restored": restored["reloaded"]}


@flow(name="deploy-flow", log_prints=True)
def deploy_flow(config_path: str, model_version: str, passed: bool) -> dict:
    """Promote ``model_version`` if ``passed`` is the quality-gate result ``True``.

    ``passed`` has no default on purpose: a version is never promoted without a gate result.
    """
    logger = _logger()
    config = load_config(config_path)
    tracking.configure(config["mlflow"]["tracking_uri"], config["mlflow"]["experiment"])
    dep = config["deploy"]
    version = str(model_version)
    if passed is not True:
        logger.error("Quality gates failed - keeping the current champion. Nothing deployed.")
        return {"deployed": False, "model_version": version}
    required = str(dep.get("require_api", False)).lower() == "true"
    retries = int(dep.get("reload_retries", 10))
    retries = retries if required else min(retries, 2)  # don't wait long for an optional API
    alias, api_url = dep["alias"], dep["api_url"]

    previous = tracking.get_alias_version(tracking.MODEL_NAME, alias)
    promote(version, alias)
    try:
        result = reload_api(api_url, version, retries)
    except Exception:  # anything unexpected while the new champion is half deployed
        _rollback(previous, alias, api_url, retries)
        raise
    if not result["reloaded"] and (result["reachable"] or required):
        restored = _rollback(previous, alias, api_url, retries)
        message = (
            f"The API did not serve v{version} ({result['error']}); "
            f"champion restored to {restored['rolled_back_to'] or 'none'}"
        )
        if required:
            raise RuntimeError(message)
        logger.error(message)
        return {"deployed": False, "model_version": version, **result, **restored}
    if not result["reloaded"]:
        logger.warning(
            "API not reachable - the model is promoted; the API loads it on its next registry "
            "poll or start."
        )
    with mlflow.start_run(run_name="deploy", tags={"flow": "deploy"}):
        mlflow.log_param("model_version", version)
        mlflow.log_metric("api_reloaded", float(result["reloaded"]))
    return {"deployed": True, "model_version": version, **result}


@flow(name="evaluate-and-deploy", log_prints=True)
def evaluate_and_deploy(config_path: str, model_version: str | None = None) -> dict:
    """``run_flow.py --flow deploy``: evaluate the version (default: the newest) against the
    quality gates, then deploy it only if it passed."""
    evaluated = eval_flow(config_path, model_version)
    deployed = deploy_flow(config_path, evaluated["model_version"], evaluated["passed"])
    return {"gates_passed": evaluated["passed"], **deployed}


def start(config_path: str) -> dict:
    return evaluate_and_deploy(config_path)
