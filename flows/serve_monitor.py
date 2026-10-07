"""Run the monitoring flow on a schedule: ``python -m flows.serve_monitor [--cron "0 * * * *"]``.

Registers a Prefect deployment (visible in the Prefect UI) and keeps serving it. In Docker Compose,
start it with ``docker compose --profile monitor up -d monitor``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flows.monitor_flow import scheduled_monitor  # noqa: E402


def main() -> None:
    os.chdir(ROOT)  # the config path and the relative paths inside it resolve from the repo root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/monitor_flow_config.yaml")
    parser.add_argument(
        "--cron", default=os.getenv("MONITOR_CRON", "0 * * * *"), help="default: hourly"
    )
    args = parser.parse_args()
    # only the config *path* is stored in Prefect (the flows load the file themselves): the resolved
    # config contains the database URL with its password
    scheduled_monitor.serve(
        name="scheduled-drift-check", cron=args.cron, parameters={"config_path": args.config}
    )


if __name__ == "__main__":
    main()
