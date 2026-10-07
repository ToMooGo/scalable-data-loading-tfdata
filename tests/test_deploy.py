"""Deploy flow: promotion only with a gate result, reload checks over real HTTP calls (a fake API
behind ``httpx.MockTransport``), and rollback of both the registry and the API."""

import contextlib
import inspect
import json
from types import SimpleNamespace

import httpx
import pytest

import flows.deploy_flow as deploy


class FakeRegistry:
    def __init__(self, champion):
        self.champion = champion
        self.deleted = False

    def get_alias_version(self, name, alias="champion"):
        return None if self.champion is None else SimpleNamespace(version=self.champion)

    def set_alias(self, name, version, alias="champion"):
        self.champion = str(version)

    def delete_alias(self, name, alias="champion"):
        self.champion, self.deleted = None, True


class FakeAPI:
    """The model service over HTTP: POST /reload loads whatever ``@champion`` points at."""

    def __init__(self, registry, served):
        self.registry, self.served = registry, served
        self.broken = set()  # versions whose model cannot be loaded (503 on /reload)
        self.stuck = False  # /reload answers 200 but keeps the old model
        self.health_html = False  # a proxy answers /health with an HTML page
        self.bad_canary = set()  # versions whose predictions are unusable
        self.unreachable = False
        self.explode = False  # a bug: the handler raises something unexpected
        self.requests = []
        self.reloads = []  # what @champion pointed at when each POST /reload arrived

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        if self.explode:
            raise KeyError("unexpected")
        body = json.loads(request.content) if request.content else None
        self.requests.append((request.method, request.url.path, body))
        route = (request.method, request.url.path)
        if route == ("POST", "/reload"):
            target = self.registry.champion
            self.reloads.append(target)
            if target in self.broken:
                return httpx.Response(503, json={"detail": "Reload failed"})
            if not self.stuck:
                self.served = target
            return httpx.Response(200, json={"status": "reloaded", "model_version": self.served})
        if route == ("GET", "/health"):
            if self.health_html:
                return httpx.Response(200, text="<html>502 Bad Gateway</html>")
            return httpx.Response(200, json={"status": "ok", "model_version": self.served})
        if route == ("GET", "/samples"):
            return httpx.Response(200, json={"row": {"median_income": 3.0}, "true_price_usd": 1})
        if route == ("POST", "/predict"):
            price = None if self.served in self.bad_canary else 210_000.0
            return httpx.Response(200, json={"price_usd": price, "model_version": self.served})
        return httpx.Response(404, json={"detail": "not found"})


@pytest.fixture()
def world(monkeypatch, config_file):
    """Champion v1 served by the API; the real promote / reload logic without Prefect or MLflow."""
    registry = FakeRegistry("1")
    api = FakeAPI(registry, served="1")
    monkeypatch.setattr(deploy.tracking, "configure", lambda *a, **k: None)
    monkeypatch.setattr(deploy.tracking, "get_alias_version", registry.get_alias_version)
    monkeypatch.setattr(deploy.tracking, "set_alias", registry.set_alias)
    monkeypatch.setattr(deploy.tracking, "delete_alias", registry.delete_alias)
    monkeypatch.setattr(deploy, "promote", deploy.promote.fn)
    monkeypatch.setattr(deploy, "reload_api", deploy.reload_api.fn)
    monkeypatch.setattr(deploy.time, "sleep", lambda s: None)
    monkeypatch.setattr(deploy.mlflow, "start_run", lambda **k: contextlib.nullcontext())
    monkeypatch.setattr(deploy.mlflow, "log_param", lambda *a: None)
    monkeypatch.setattr(deploy.mlflow, "log_metric", lambda *a: None)
    monkeypatch.setattr(
        deploy,
        "_http_client",
        lambda url: httpx.Client(base_url=url, transport=httpx.MockTransport(api.handler)),
    )

    def config(require_api=False):
        return config_file(
            {
                "mlflow": {"tracking_uri": None, "experiment": "x"},
                "deploy": {
                    "alias": "champion",
                    "api_url": "http://api:8000",
                    "require_api": require_api,
                    "reload_retries": 3,
                },
            }
        )

    return SimpleNamespace(registry=registry, api=api, config=config)


def test_a_passing_version_is_promoted_reloaded_and_checked_with_a_canary(world):
    out = deploy.deploy_flow.fn(world.config(), "2", True)
    assert out["deployed"] is True and out["reloaded"] is True and out["model_version"] == "2"
    assert world.registry.champion == "2" and world.api.served == "2"
    canary = [r for r in world.api.requests if r[:2] == ("POST", "/predict")]
    assert len(canary) == 1 and canary[0][2]["source"] == deploy.CANARY_SOURCE


@pytest.mark.parametrize("failure", ["broken", "stuck", "html", "canary"])
def test_an_api_that_answers_but_cannot_serve_the_version_triggers_a_rollback(world, failure):
    api = world.api
    if failure == "broken":  # POST /reload answers 503
        api.broken.add("2")
    elif failure == "stuck":  # 200, but the API still serves v1
        api.stuck = True
    elif failure == "html":  # /health is not JSON (only for the new version's attempts)
        api.health_html = True
    else:  # the canary prediction is unusable
        api.bad_canary.add("2")

    out = deploy.deploy_flow.fn(world.config(), "2", True)
    assert out["deployed"] is False and out["reachable"] is True and out["rolled_back_to"] == "1"
    assert world.registry.champion == "1"
    # after the rollback the API was asked to reload, and it serves the old champion again
    assert api.reloads[0] == "2" and api.reloads[-1] == "1" and api.served == "1"
    # (an API whose /health is not JSON cannot confirm the restore either)
    assert out["api_restored"] is (failure != "html")


def test_a_required_api_that_fails_raises_after_the_rollback(world):
    world.api.broken.add("2")
    with pytest.raises(RuntimeError, match="champion restored to 1"):
        deploy.deploy_flow.fn(world.config(require_api=True), "2", True)
    assert world.registry.champion == "1" and world.api.served == "1"


def test_an_unreachable_optional_api_keeps_the_promotion(world):
    world.api.unreachable = True
    out = deploy.deploy_flow.fn(world.config(), "2", True)
    assert out["deployed"] is True and out["reloaded"] is False and out["reachable"] is False
    assert world.registry.champion == "2"  # the API loads it on its next registry poll


def test_an_unreachable_required_api_rolls_back(world):
    world.api.unreachable = True
    with pytest.raises(RuntimeError, match="not reachable"):
        deploy.deploy_flow.fn(world.config(require_api=True), "2", True)
    assert world.registry.champion == "1"


def test_an_unexpected_error_during_the_reload_rolls_back_and_is_raised(world):
    world.api.explode = True
    with pytest.raises(KeyError):
        deploy.deploy_flow.fn(world.config(), "2", True)
    assert world.registry.champion == "1"


def test_the_first_deployment_is_undone_by_removing_the_alias(world):
    world.registry.champion = None
    world.api.served = None
    world.api.broken.add("2")
    out = deploy.deploy_flow.fn(world.config(), "2", True)
    assert out["deployed"] is False and out["rolled_back_to"] is None
    assert world.registry.deleted and world.registry.champion is None


@pytest.mark.parametrize("passed", [False, None, "true", 1])
def test_nothing_is_promoted_without_a_passing_gate_result(world, passed):
    out = deploy.deploy_flow.fn(world.config(), "2", passed)
    assert out["deployed"] is False and world.registry.champion == "1" and not world.api.requests


def test_deploy_has_no_default_gate_result_and_runs_the_evaluation_on_its_own(monkeypatch):
    params = inspect.signature(deploy.deploy_flow.fn).parameters
    assert params["passed"].default is inspect.Parameter.empty
    assert params["model_version"].default is inspect.Parameter.empty

    calls = []
    monkeypatch.setattr(
        deploy, "eval_flow", lambda path, version=None: {"passed": False, "model_version": "5"}
    )
    monkeypatch.setattr(
        deploy,
        "deploy_flow",
        lambda path, version, passed: calls.append((path, version, passed)) or {"deployed": passed},
    )
    out = deploy.evaluate_and_deploy.fn("cfg.yaml")
    assert calls == [("cfg.yaml", "5", False)]
    assert out == {"gates_passed": False, "deployed": False}

    started = []
    monkeypatch.setattr(deploy, "evaluate_and_deploy", lambda path: started.append(path) or {})
    deploy.start("cfg.yaml")  # `run_flow.py --flow deploy` evaluates first
    assert started == ["cfg.yaml"]
