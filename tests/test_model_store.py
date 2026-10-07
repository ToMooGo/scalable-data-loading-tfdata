"""The API's model store against a fake registry: serialised loads, the alias check after a slow
load, one registry poll per interval, and error messages that do not leak internals."""

import threading
from types import SimpleNamespace

import pytest
from app.model_store import LOAD_FAILED, ModelBundle, ModelStore, NoChampionError


class FakeRegistry:
    """``@champion`` and the registered versions; ``hooks[v]`` runs while version ``v`` loads."""

    def __init__(self, champion: str | None):
        self.champion = champion
        self.loads: list[str] = []
        self.hooks: dict = {}
        self.fail_lookup = False

    def get_alias_version(self, name, alias="champion"):
        if self.fail_lookup:
            raise ConnectionError("registry down")
        return None if self.champion is None else SimpleNamespace(version=self.champion)

    def load_model(self, name, version):
        self.loads.append(version)
        if version in self.hooks:
            self.hooks[version]()
        return SimpleNamespace(
            model=f"model-{version}", profile=None, samples=[], version=version, run_id="r"
        )


@pytest.fixture()
def registry(monkeypatch):
    from tfdata_mlops import tracking

    reg = FakeRegistry("1")
    monkeypatch.setattr(tracking, "get_alias_version", reg.get_alias_version)
    monkeypatch.setattr(tracking, "load_model", reg.load_model)
    return reg


def serving(version: str) -> ModelStore:
    store = ModelStore()
    store.set_bundle(ModelBundle(f"model-{version}", None, [], version, "r"))
    return store


def test_reload_serves_the_champion_and_skips_a_version_already_served(registry):
    store = ModelStore()
    assert store.load_from_registry().version == "1" and store.bundle.model == "model-1"
    assert store.load_from_registry().version == "1" and registry.loads == ["1"]  # no second load
    registry.champion = "2"
    assert store.load_from_registry().version == "2" and registry.loads == ["1", "2"]

    registry.champion = None
    with pytest.raises(NoChampionError):
        store.load_from_registry()
    assert store.bundle.version == "2"  # a failed reload keeps the served model


def test_poll_reloads_only_when_the_champion_moves(registry):
    store = serving("1")
    assert store.refresh_if_stale() is False and registry.loads == []  # unchanged
    registry.champion = "2"
    assert store.refresh_if_stale() is True and store.bundle.version == "2"
    registry.champion = None
    assert store.refresh_if_stale() is False  # no champion at all: keep serving what we have
    registry.fail_lookup = True
    assert store.refresh_if_stale() is False  # a failed poll must not drop the loaded model
    assert store.bundle.version == "2"


def test_a_failed_load_keeps_the_model_and_hides_the_details(registry):
    def broken():
        raise OSError("/tmp/tmpab12/model.keras: secret internal path")

    registry.champion = "2"
    registry.hooks["2"] = broken
    store = serving("1")
    assert store.refresh_if_stale() is False and store.bundle.version == "1"
    assert "secret internal path" in store.last_error  # for the log
    assert store.public_error == LOAD_FAILED  # for anonymous callers

    registry.champion = None
    empty = ModelStore()
    assert empty.try_load() is False and "registry yet" in empty.public_error


def test_a_load_that_started_before_a_rollback_is_never_installed(registry):
    """A poll starts loading v2; meanwhile the deployment is rolled back to v1 and /reload is
    called. When the slow load finishes, v1 must still be served."""
    entered, release = threading.Event(), threading.Event()

    def slow():
        entered.set()
        assert release.wait(10)

    registry.champion = "2"
    registry.hooks["2"] = slow
    store = serving("1")
    poll = threading.Thread(target=store.refresh_if_stale)
    poll.start()
    assert entered.wait(10)
    registry.champion = "1"  # the deploy flow restores the previous champion ...
    reload = threading.Thread(target=store.load_from_registry)  # ... and calls POST /reload
    reload.start()
    release.set()
    poll.join(10)
    reload.join(10)
    assert store.bundle.version == "1" and store.bundle.model == "model-1"
    assert registry.loads == ["2"]  # v2 was loaded but not installed, v1 was still served


def test_an_alias_that_keeps_moving_is_not_followed_for_ever(registry):
    store = serving("1")

    def move():
        registry.champion = str(int(registry.champion) + 1)

    for v in range(2, 2 + ModelStore.MAX_LOADS):
        registry.hooks[str(v)] = move
    registry.champion = "2"
    with pytest.raises(RuntimeError, match="moved"):
        store.load_from_registry()
    assert store.bundle.version == "1" and len(registry.loads) == ModelStore.MAX_LOADS


def test_one_registry_poll_per_interval_however_many_threads_ask():
    now = [1000.0]
    store = ModelStore(clock=lambda: now[0])
    assert store.poll_due(60) is False  # the interval starts when the store is created
    now[0] += 61
    barrier, due = threading.Barrier(16), []

    def ask():
        barrier.wait()
        due.append(store.poll_due(60))

    threads = [threading.Thread(target=ask) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert due.count(True) == 1 and len(due) == 16
    assert store.poll_due(60) is False
    now[0] += 61
    assert store.poll_due(60) is True
    assert store.poll_due(0) is False  # polling switched off


def test_a_poll_does_not_wait_for_a_load_in_progress(registry):
    store = serving("1")
    registry.champion = "2"
    with store._load_lock:  # a /reload is loading right now
        assert store.refresh_if_stale() is False  # returns at once instead of queueing a load
    assert registry.loads == []
    assert store.refresh_if_stale() is True and store.bundle.version == "2"
