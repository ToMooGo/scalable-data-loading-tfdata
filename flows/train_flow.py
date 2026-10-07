"""Train flow: parts A-E -> MLflow (params, metrics, tables, figures, model) -> Model Registry.

Every part runs as a Prefect task and logs to its own nested MLflow run under one parent run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import subprocess
import tempfile
from pathlib import Path

import mlflow
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NONE
from prefect.exceptions import MissingContextError

from tfdata_mlops import reporting, tracking
from tfdata_mlops.config import load_config
from tfdata_mlops.part_a_csv import run_part_a
from tfdata_mlops.part_b_tfrecord import run_part_b
from tfdata_mlops.part_c_features import run_part_c
from tfdata_mlops.part_d_tfds import run_part_d
from tfdata_mlops.part_e_model import run_part_e
from tfdata_mlops.prepare import Prepared, prepare_data
from tfdata_mlops.results import PartResult

ROOT = Path(__file__).resolve().parents[1]


def check_reports_writable(cfg: dict) -> None:
    """Fail in seconds, not after 30 minutes of training, if a report folder is read-only
    (typically a Docker bind mount owned by another user: run ``make up`` on Linux)."""
    for key in ("dir", "figures_dir", "tables_dir"):
        d = ROOT / cfg["reports"][key]
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write-test"
            probe.write_text("ok")
            probe.unlink()
        except OSError as exc:
            raise PermissionError(
                f"Cannot write to {d} ({exc}). On Linux start the stack with `make up` "
                "(it passes your user id) or set HOST_UID/HOST_GID in .env."
            ) from exc


def git_commit() -> str:
    """The commit of the code, for the lineage tags. Without git or a ``.git`` folder (the Docker
    image), the ``GIT_COMMIT`` environment variable is used, else ``unknown``."""
    try:
        commit = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, ValueError, subprocess.SubprocessError):  # no git, not a repository, timeout
        commit = ""
    return commit or os.getenv("GIT_COMMIT", "").strip() or "unknown"


def _nested(name: str, parent_run_id: str):
    return mlflow.start_run(run_name=name, nested=True, parent_run_id=parent_run_id)


def _logger():
    try:
        return get_run_logger()
    except MissingContextError:  # called outside a flow (for example in a unit test)
        return logging.getLogger(__name__)


def _fingerprint(cfg: dict, prep: Prepared, key: str) -> str:
    """What a saved part result depends on: its config section, the seed and the data."""
    text = json.dumps(
        {
            "section": cfg[f"part_{key.lower()}"],
            "seed": cfg["seed"],
            "data": prep.data_sha256,
            "shards": prep.shard_counts,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(text.encode()).hexdigest()


def _run_part(key: str, run_name: str, prep: Prepared, cfg: dict, pid: str, compute) -> PartResult:
    """Run one part in its own nested MLflow run and save its result next to the shards.

    With ``resume: true`` (environment variable ``RESUME``) a saved result whose config section,
    seed and data are unchanged is reused instead of computed again, so a long run that was
    interrupted (a restarted machine, a closed laptop) continues at the first unfinished part.
    Use it only while the code has not changed: the saved result does not record the code.
    """
    path = prep.root / "checkpoints" / f"{key}.pkl"
    fingerprint = _fingerprint(cfg, prep, key)
    resume = str(cfg.get("resume", False)).lower() == "true"
    with _nested(run_name, pid):
        res = None
        if resume and path.is_file():
            saved = pickle.loads(path.read_bytes())  # noqa: S301 - written by this flow, on this disk
            if saved.get("fingerprint") == fingerprint:
                res = saved["result"]
                _logger().info("Part %s resumed from %s", key, path.name)
        if res is None:
            res = compute()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pickle.dumps({"fingerprint": fingerprint, "result": res}))
            except Exception as exc:  # a part whose result cannot be pickled is simply not saved
                _logger().warning("Part %s result not saved: %s", key, exc)
        return _log_part(res)


def _log_part(res: PartResult) -> PartResult:
    """Write a part's params, metrics and tables to the active (nested) run."""
    tracking.log_params(res.params)
    tracking.log_metrics(res.metrics)
    for name, df in res.tables.items():
        tracking.log_table(df, f"tables/{name}.csv")
    res.run_id = mlflow.active_run().info.run_id
    return res


@task(name="prepare-data", cache_policy=NONE, retries=2, retry_delay_seconds=10)
def prepare(cfg: dict) -> Prepared:
    prep = prepare_data(cfg["data"], int(cfg["seed"]))
    get_run_logger().info("Split %s, shards %s", prep.split.sizes, prep.shard_counts)
    return prep


@task(name="A-csv-pipeline", cache_policy=NONE)
def part_a(prep, cfg, pid):
    res = _run_part(
        "A",
        "A-csv-pipeline",
        prep,
        cfg,
        pid,
        lambda: run_part_a(prep, cfg["part_a"], int(cfg["seed"])),
    )
    m = res.metrics
    get_run_logger().info(
        "A: book pipeline %.0f examples/s (strict), pandas/NumPy %.0f",
        m["strict_book_examples_per_s"],
        m["numpy_examples_per_s"],
    )
    return res


@task(name="B-tfrecord", cache_policy=NONE)
def part_b(prep, cfg, pid):
    res = _run_part(
        "B",
        "B-tfrecord",
        prep,
        cfg,
        pid,
        lambda: run_part_b(prep, cfg["part_b"], int(cfg["seed"])),
    )
    get_run_logger().info(
        "B: TFRecord.gz is %.2fx the size of CSV; corruption detected: %s",
        res.metrics["tfrecord_gz_over_csv"],
        bool(res.metrics["corruption_all_detected"]),
    )
    return res


@task(name="C-features", cache_policy=NONE)
def part_c(prep, cfg, pid):
    res = _run_part(
        "C",
        "C-features",
        prep,
        cfg,
        pid,
        lambda: run_part_c(prep, cfg["part_c"], int(cfg["seed"])),
    )
    get_run_logger().info(
        "C: best feature set on validation data: %s", res.extra["best_feature_set"]
    )
    return res


@task(name="D-tfds-mnist", cache_policy=NONE, retries=1, retry_delay_seconds=30)
def part_d(prep, cfg, pid):
    res = _run_part(
        "D",
        "D-tfds-mnist",
        prep,
        cfg,
        pid,
        lambda: run_part_d(cfg["part_d"], int(cfg["seed"]), prep.root / "mnist"),
    )
    get_run_logger().info(
        "D: MNIST via %s, test accuracy %.4f (tfds.load) / %.4f (TFRecord)",
        res.extra["source"],
        res.metrics["tfds_load_test_accuracy"],
        res.metrics["tfrecord_test_accuracy"],
    )
    return res


@task(name="E-deployed-model", cache_policy=NONE)
def part_e(prep, cfg, pid, best_feature_set):
    with _nested("E-deployed-model", pid):
        res = _log_part(run_part_e(prep, cfg["part_e"], int(cfg["seed"]), best_feature_set))
    get_run_logger().info(
        "E: %s -> test RMSE $%.0f (linear baseline $%.0f), serving skew %.2e USD",
        res.params["feature_set"],
        res.metrics["test_rmse"],
        res.metrics["baseline_linear_test_rmse"],
        res.metrics["skew_max_abs_usd"],
    )
    return res


@task(name="register-model", cache_policy=NONE)
def register_model(e: PartResult, lineage: dict) -> str:
    with mlflow.start_run(run_id=e.run_id, nested=True):
        version = tracking.log_model(e.extra["model"], e.extra["profile"], e.extra["samples"])
    tracking.set_version_tags(tracking.MODEL_NAME, version, lineage)  # which data, code and config
    get_run_logger().info("Registered %s version %s", tracking.MODEL_NAME, version)
    return version


@task(name="write-reports", cache_policy=NONE)
def write_reports(parts: dict[str, PartResult], cfg: dict, run_info: dict) -> dict:
    tables_dir = ROOT / cfg["reports"]["tables_dir"]
    reporting.write_tables(parts, tables_dir)
    figs = reporting.make_figures(tables_dir, ROOT / cfg["reports"]["figures_dir"], cfg)
    for p in figs.values():
        mlflow.log_artifact(str(p), "figures")
    rep = ROOT / cfg["reports"]["dir"]
    reporting.write_metrics_json(rep / "metrics.json", parts, run_info)
    reporting.write_results_md(rep / "RESULTS.md", parts, run_info, cfg)
    mlflow.log_artifact(str(rep / "metrics.json"))
    mlflow.log_artifact(str(rep / "RESULTS.md"))
    return {k: str(v) for k, v in figs.items()}


@flow(name="train-flow", log_prints=True)
def train_flow(config_path: str) -> dict:
    """Only the config *path* is a flow parameter: Prefect stores parameters in plain text and the
    resolved config contains the database URL with its password."""
    config = load_config(config_path)
    config["_config_path"] = config_path
    check_reports_writable(config)
    tracking.configure(config["mlflow"]["tracking_uri"], config["mlflow"]["experiment"])
    prep = prepare(config)
    safe_config = tracking.redact(config)  # no database passwords in MLflow artifacts
    config_text = json.dumps(safe_config, indent=2, sort_keys=True)
    lineage = {
        "data_sha256": prep.data_sha256,
        "config_md5": hashlib.md5(config_text.encode()).hexdigest(),
        "git_commit": git_commit(),
        "config_file": config.get("_config_path", "n/a"),
    }
    with mlflow.start_run(run_name="train", tags={"flow": "train", **lineage}) as parent:
        pid = parent.info.run_id
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            cfg_path.write_text(config_text)
            mlflow.log_artifact(str(cfg_path))
        mlflow.log_params({"seed": config["seed"], **prep.split.sizes})
        a = part_a(prep, config, pid)
        b = part_b(prep, config, pid)
        c = part_c(prep, config, pid)
        d = part_d(prep, config, pid)
        e = part_e(prep, config, pid, c.extra["best_feature_set"])
        version = register_model(e, lineage)
        run_info = {
            "parent_run_id": pid,
            "config": config.get("_config_path", "n/a"),
            "model_version": version,
            "split_sizes": prep.split.sizes,
            **lineage,
        }
        parts = {"A": a, "B": b, "C": c, "D": d, "E": e}
        write_reports(parts, config, run_info)
        mlflow.log_metrics(
            {
                "A_strict_book_examples_per_s": a.metrics["strict_book_examples_per_s"],
                "B_tfrecord_gz_over_csv": b.metrics["tfrecord_gz_over_csv"],
                "C_best_valid_rmse": c.metrics["best_feature_set_valid_rmse"],
                "D_tfds_load_test_accuracy": d.metrics["tfds_load_test_accuracy"],
                "E_test_rmse": e.metrics["test_rmse"],
            }
        )
    return {"parent_run_id": pid, "model_version": version, "e_test_rmse": e.metrics["test_rmse"]}


def start(config_path: str) -> dict:
    return train_flow(config_path)
