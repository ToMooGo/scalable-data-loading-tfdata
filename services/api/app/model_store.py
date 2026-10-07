"""Holds the champion model and swaps it atomically on /reload or when the registry poll sees a new
``@champion``.

Loads are serialised: one registry load runs at a time, and a loaded version is only installed if
the alias still points at it. A slow load that started before a rollback can therefore never
replace the restored champion. Requests keep using the bundle they picked up, so a load never
blocks a prediction; a poll that finds a load already running does not wait for it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from tfdata_mlops.inference import Profile

log = logging.getLogger(__name__)

LOAD_FAILED = "the registered model could not be loaded (details in the service log)"


class NoChampionError(LookupError):
    """The registry has no version under the alias yet (a normal state before the first run)."""


@dataclass
class ModelBundle:
    model: object
    profile: Profile
    samples: list[dict]
    version: str
    run_id: str
    loaded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class ModelStore:
    MAX_LOADS = 3  # the alias moved during this many loads in a row: give up, keep the old model

    def __init__(self, alias: str = "champion", clock: Callable[[], float] = time.monotonic):
        self.alias = alias
        self._bundle: ModelBundle | None = None
        self._lock = threading.Lock()  # guards the swap
        self._load_lock = threading.Lock()  # one registry load at a time
        self._poll_lock = threading.Lock()  # guards the poll timestamp
        self._clock = clock
        self._last_poll = clock()
        self.last_error: str | None = None  # full text, for the log
        self.public_error: str | None = None  # what anonymous callers may see

    @property
    def bundle(self) -> ModelBundle | None:
        return self._bundle

    def set_bundle(self, bundle: ModelBundle) -> None:
        with self._lock:
            self._bundle = bundle
            self.last_error = self.public_error = None

    def registry_version(self) -> str | None:
        from tfdata_mlops import tracking

        mv = tracking.get_alias_version(tracking.MODEL_NAME, self.alias)
        return str(mv.version) if mv is not None else None

    def load_from_registry(self) -> ModelBundle:
        """Serve the version ``@alias`` points at (``POST /reload``).

        Waits for a load that is already running, then checks the alias again, so the result is
        always what the alias points at now. Raises if nothing can be loaded; the served model
        then stays as it was.
        """
        with self._load_lock:
            try:
                return self._load_current()
            except Exception as exc:
                self._record_error(exc)
                raise

    def _load_current(self) -> ModelBundle:
        from tfdata_mlops import tracking

        for _ in range(self.MAX_LOADS):
            target = self.registry_version()
            if target is None:
                raise NoChampionError(
                    f"No '{self.alias}' version of {tracking.MODEL_NAME} in the registry yet"
                )
            current = self._bundle
            if current is not None and current.version == target:
                return current  # already served (for example loaded by a poll a moment ago)
            loaded = tracking.load_model(tracking.MODEL_NAME, target)
            # the alias may have moved while the model loaded (a rollback): install only what it
            # points at now, otherwise go round again
            if self.registry_version() == target:
                bundle = ModelBundle(
                    loaded.model, loaded.profile, loaded.samples, loaded.version, loaded.run_id
                )
                self.set_bundle(bundle)
                log.info("Loaded %s version %s", tracking.MODEL_NAME, loaded.version)
                return bundle
            log.warning(
                "@%s moved while version %s was loading - not installed", self.alias, target
            )
        raise RuntimeError(f"@{self.alias} moved during {self.MAX_LOADS} loads in a row")

    def refresh_if_stale(self) -> bool:
        """Reload when the registry's alias points at a different version than the one served.

        Lets several API replicas pick up a new champion without each receiving /reload. Never
        waits: if a load is already running, that load delivers the current champion.
        """
        if not self._load_lock.acquire(blocking=False):
            return False
        try:
            try:
                current = self.registry_version()
            except Exception as exc:
                log.warning("Registry poll failed: %s", exc)
                return False
            if current is None or (self._bundle is not None and current == self._bundle.version):
                return False
            log.info("Champion changed in the registry (v%s) - reloading", current)
            try:
                self._load_current()
                return True
            except Exception as exc:
                self._record_error(exc)
                return False
        finally:
            self._load_lock.release()

    def poll_due(self, interval: float) -> bool:
        """True for exactly one caller per ``interval`` seconds, however many threads ask."""
        if interval <= 0:
            return False
        with self._poll_lock:
            now = self._clock()
            if now - self._last_poll < interval:
                return False
            self._last_poll = now
            return True

    def maybe_refresh(self, interval: float) -> bool:
        """At most one registry poll per ``interval`` seconds (called on the request path)."""
        return self.poll_due(interval) and self.refresh_if_stale()

    def try_load(self) -> bool:
        try:
            self.load_from_registry()
            return True
        except Exception:
            return False

    def _record_error(self, exc: Exception) -> None:
        self.last_error = f"{type(exc).__name__}: {exc}"
        self.public_error = str(exc) if isinstance(exc, NoChampionError) else LOAD_FAILED
        log.warning("Model not loaded: %s", self.last_error)
