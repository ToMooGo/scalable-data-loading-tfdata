"""Write what a training run found: tables (CSV), metrics.json, RESULTS.md and the figures."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .results import PartResult


def write_tables(parts: dict[str, PartResult], directory: str | Path) -> list[Path]:
    """Every table of every part as ``<directory>/<table name>.csv``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = []
    for res in parts.values():
        for name, df in res.tables.items():
            path = directory / f"{name}.csv"
            df.to_csv(path, index=False)
            out.append(path)
    return out


def make_figures(tables_dir: str | Path, figures_dir: str | Path, cfg: dict | None = None) -> dict:
    from . import plots

    return plots.make_figures(Path(tables_dir), Path(figures_dir), cfg or {})


def write_metrics_json(path: str | Path, parts: dict[str, PartResult], run_info: dict) -> None:
    payload = {
        "run": run_info,
        "parts": {
            key: {"name": res.name, "params": res.params, "metrics": res.metrics}
            for key, res in parts.items()
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


# --------------------------------------------------------------------------------------------
# RESULTS.md
# --------------------------------------------------------------------------------------------
def _fmt(value, spec: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return format(value, spec) if spec else str(value)


def md_table(df: pd.DataFrame, columns: dict[str, tuple[str, str]]) -> str:
    """``columns``: source column -> (heading, format spec)."""
    head = "| " + " | ".join(h for h, _ in columns.values()) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(_fmt(row[c], spec) for c, (_, spec) in columns.items()) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([head, rule, *body])


def write_results_md(
    path: str | Path, parts: dict[str, PartResult], run_info: dict, cfg: dict
) -> None:
    a, b, c, d, e = (parts[k] for k in "ABCDE")
    am, cm, dm, em = a.metrics, c.metrics, d.metrics, e.metrics
    lines = [
        "# Results of the latest training run",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python run_flow.py`. "
        f"Config `{run_info.get('config', 'n/a')}`, seed {cfg.get('seed')}, "
        f"commit `{run_info.get('git_commit', 'n/a')}`, data SHA-256 `{run_info.get('data_sha256', '')[:12]}`.",
        "",
        "Every timing is the median of several passes after a warm-up pass on the machine that ran the "
        "flow; absolute numbers depend on that machine, ratios are what to compare.",
        "",
        "## A. Data API on sharded CSV",
        "",
        f"The pipeline reads {am['n_train_rows']:,} training rows and returns the same numbers as pandas "
        f"(largest difference {am['pipeline_max_abs_diff_inputs']:.1e}).",
        "",
    ]
    ab = a.tables["csv_ablation"]
    step0 = ab[ab.step_ms == 0]
    wide = step0.pivot(
        index="variant", columns="regime", values="examples_per_second"
    ).reset_index()
    wide = wide.sort_values("variant")
    lines += [
        "Reading speed (examples per second, no training step):",
        "",
        md_table(
            wide,
            {
                "variant": ("Pipeline", ""),
                "strict": ("Only the listed steps", ",.0f"),
                "default": ("tf.data defaults", ",.0f"),
            },
        ),
        "",
        "Shuffling sorted data (Spearman correlation of position and target: 1 = untouched, 0 = mixed):",
        "",
        md_table(
            a.tables["shuffle_quality"],
            {
                "label": ("Shuffling", ""),
                "spearman": ("Spearman", ".3f"),
                "predicted_spearman": ("Predicted", ".3f"),
                "batch_diversity": ("Batch diversity", ".3f"),
            },
        ),
        "",
        "## B. TFRecord",
        "",
        md_table(
            b.tables["format_benchmark"],
            {
                "format": ("Format", ""),
                "bytes": ("Bytes", ",.0f"),
                "size_vs_csv": ("Size vs CSV", ".2f"),
                "examples_per_second": ("Examples/s", ",.0f"),
            },
        ),
        "",
        "Scale test (the training set repeated N times):",
        "",
        md_table(
            b.tables["scale_test"],
            {
                "rows": ("Rows", ",.0f"),
                "csv_mb": ("CSV MB", ".1f"),
                "tfrecord_gz_mb": ("TFRecord.gz MB", ".1f"),
                "stream_growth_mb": ("tf.data memory growth MB", ".0f"),
                "pandas_growth_mb": ("pandas memory growth MB", ".0f"),
                "stream_examples_per_s": ("tf.data examples/s", ",.0f"),
            },
        ),
        "",
        "## C. Features inside the model",
        "",
        f"{cm['n_seeds']} seeds per cell, mean validation RMSE in dollars with a 95% interval; the last "
        "column is the paired difference from `+ ocean one-hot` (negative = better).",
        "",
    ]
    for head in c.tables["feature_summary"]["head"].unique():
        part = c.tables["feature_summary"].query("head == @head")
        lines += [
            f"Head: `{head}`",
            "",
            md_table(
                part,
                {
                    "feature_set": ("Feature set", ""),
                    "features": ("Inputs", ",.0f"),
                    "params": ("Parameters", ",.0f"),
                    "valid_rmse": ("Valid RMSE", ",.0f"),
                    "valid_rmse_lo": ("low", ",.0f"),
                    "valid_rmse_hi": ("high", ",.0f"),
                    "valid_diff_vs_baseline": ("vs baseline", "+,.0f"),
                },
            ),
            "",
        ]
    lines += [
        "## D. TensorFlow Datasets",
        "",
        f"`tfds.load('mnist', batch_size=32, as_supervised=True)` via **{d.extra['source']}**: "
        f"{dm['train_batches']} training batches and {dm['test_batches']} test batches. "
        f"Test accuracy after {d.params['epochs']} epochs: {dm['tfds_load_test_accuracy']:.4f} from `tfds.load`, "
        f"{dm['tfrecord_test_accuracy']:.4f} from TFRecord files.",
        "",
        md_table(
            d.tables["mnist_formats"].query("reading == 'one record at a time, serial'"),
            {
                "encoding": ("Encoding", ""),
                "compression": ("Compression", ""),
                "bytes_per_image": ("Bytes/image", ",.0f"),
                "examples_per_second": ("Images/s", ",.0f"),
            },
        ),
        "",
        "## E. Deployed model",
        "",
        f"Feature set `{e.params['feature_set']}`; test RMSE **${em['test_rmse']:,.0f}** "
        f"(R² {em['test_r2']:.3f}); linear baseline ${em['baseline_linear_test_rmse']:,.0f}; "
        f"always-predict-the-mean ${em['baseline_mean_test_rmse']:,.0f}. Largest difference between the "
        f"request path and the tf.data path: {em['skew_max_abs_usd']:.2e} USD.",
        "",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
