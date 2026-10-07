"""Single entry point: ``python run_flow.py --config configs/full_flow_config.yaml``.

The config's ``flow`` key picks which flow runs (full | train | eval | deploy | monitor).
``deploy`` on its own evaluates the newest version against the quality gates first.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FLOWS = {
    "full": "flows.full_flow",
    "train": "flows.train_flow",
    "eval": "flows.eval_flow",
    "deploy": "flows.deploy_flow",
    "monitor": "flows.monitor_flow",
}


def main(argv: list[str] | None = None) -> dict:
    # side effects belong to the command, not to importing this module (the tests import it)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)  # relative paths in configs (reports/, sqlite files) resolve from the repo root
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
    from tfdata_mlops.config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--flow", choices=sorted(FLOWS), help="override the config's `flow` key")
    args = parser.parse_args(argv)

    name = args.flow or load_config(args.config).get("flow", "full")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Only the config *path* is handed to Prefect, which stores flow parameters in plain text; each
    # flow loads the file itself (the resolved config contains the database URL and its password).
    result = importlib.import_module(FLOWS[name]).start(str(args.config))
    print(json.dumps(result, indent=2, default=str))
    return result


def exit_code(result: dict) -> int:
    """Non-zero when the candidate failed a quality gate or was not deployed (for CI and Docker)."""
    if result.get("gates_passed") is False or result.get("passed") is False:
        return 2
    if result.get("deployed") is False:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(exit_code(main()))
