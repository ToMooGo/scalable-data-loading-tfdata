"""YAML configuration helpers.

A single YAML file drives every flow ("1 config, 1 command"). Values written as
``${ENV_VAR:default}`` are resolved from the environment, so the same config works
on a laptop and inside Docker Compose.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str):

        def repl(match: re.Match) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default if default is not None else "")

        resolved = _ENV_PATTERN.sub(repl, value)
        # keep YAML-like typing for fully substituted scalars
        if resolved != value and _ENV_PATTERN.fullmatch(value):
            try:
                return yaml.safe_load(resolved)
            except yaml.YAMLError:
                return resolved
        return resolved
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Load a YAML config, resolve ``${ENV:default}`` placeholders and apply overrides.

    A config may set ``extends: other.yaml`` (relative to itself) to inherit values.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_update(load_config(path.parent / parent), cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return _resolve_env(cfg)


def linspace(spec: dict | list) -> list[float]:
    """Turn ``{start, stop, num}`` (``np.linspace`` semantics) or a list into a list of floats."""
    if isinstance(spec, list):
        return [float(v) for v in spec]
    import numpy as np

    return [float(v) for v in np.linspace(spec["start"], spec["stop"], int(spec["num"]))]
