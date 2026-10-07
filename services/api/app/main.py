"""FastAPI model service + web UI.

* ``POST /predict``            - median house value of one census block group; logged to the database
* ``POST /predict/batch``      - up to 1,000 rows in one call (vectorised, one database commit)
* ``GET  /samples``            - a held-out example row, optionally damaged (missing / unseen / outlier)
* ``GET  /model-info``         - what the deployed model was trained on (ranges, categories, grid)
* ``GET  /pipeline/summary``   - the measurements of the latest training run
* ``POST /reload``             - hot-swap to the current ``champion`` in the MLflow registry
* ``GET  /monitoring/summary`` - live data-quality rates, prediction mix and recent requests
* ``GET  /``                   - single-page web UI

Run with ``uvicorn app.main:create_app --factory``.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import case, func, select

from tfdata_mlops import features as F
from tfdata_mlops.data import NUMERIC
from tfdata_mlops.db import Prediction, init_db, make_engine
from tfdata_mlops.inference import RowFlags, grid_cell, predict_rows

from .model_store import ModelBundle, ModelStore, NoChampionError
from .schemas import (
    BatchRequest,
    BatchResponse,
    PredictRequest,
    PredictResponse,
    Quality,
    Seen,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("api")
STATIC = Path(__file__).parent / "static"
KINDS = ("clean", "missing", "unseen", "outlier")
SUMMARY_TABLES = (
    "csv_ablation",
    "prefetch_sweep",
    "format_benchmark",
    "scale_test",
    "feature_summary",
    "shuffle_quality",
    "mnist_formats",
)


def _seen(row: dict, bundle: ModelBundle) -> Seen:
    stats = bundle.profile.stats
    z = {}
    for name, median, mean, std in zip(
        NUMERIC, stats.medians, stats.means, stats.stds, strict=True
    ):
        value = row.get(name)
        value = median if value is None else value
        z[name] = round((float(value) - mean) / std, 3)
    r, c = grid_cell(row["latitude"], row["longitude"])
    return Seen(
        z_scores=z,
        grid_row=r,
        grid_col=c,
        grid_cell=r * F.GRID + c,
        income_bucket=int(np.digitize(row["median_income"], F.INCOME_BOUNDARIES)),
    )


def _quality(flags: RowFlags) -> Quality:
    return Quality(
        missing=flags.missing,
        unknown_category=flags.unknown_category,
        out_of_range=flags.out_of_range,
    )


def _is_loopback(address: str) -> bool:
    if address.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(address.strip()).is_loopback
    except ValueError:
        return False


def _flag(condition):
    """1 where ``condition`` holds, else 0 (summed in SQL)."""
    return case((condition, 1), else_=0)


def create_app(
    store: ModelStore | None = None,
    database_url: str | None = None,
    load_models: bool = True,
    reports_dir: str | Path | None = None,
) -> FastAPI:
    store = store or ModelStore(alias=os.getenv("MODEL_ALIAS", "champion"))
    engine = make_engine(database_url or os.getenv("DATABASE_URL", "sqlite:///app.db"))
    Session = init_db(engine)
    poll_seconds = float(os.getenv("REGISTRY_POLL_SECONDS", "60"))
    admin_token = os.getenv("ADMIN_TOKEN", "")
    reports = Path(reports_dir or os.getenv("REPORTS_DIR", "reports"))
    bind_addr = os.getenv("BIND_ADDR", "127.0.0.1")
    if not admin_token and not _is_loopback(bind_addr):
        log.warning(
            "ADMIN_TOKEN is empty and BIND_ADDR=%s is not a loopback address: anyone who can reach "
            "the port can call POST /reload. Set ADMIN_TOKEN.",
            bind_addr,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_models and store.bundle is None:
            store.try_load()
        yield
        engine.dispose()

    app = FastAPI(
        title="tf.data pipeline MLOps - housing price service",
        version="1.0.0",
        description="A Keras model whose preprocessing layers (standardise, one-hot, bucketise, "
        "cross, embed) are part of the saved model, trained from sharded TFRecord files with "
        "tf.data and served from the MLflow registry.",
        lifespan=lifespan,
    )
    app.state.store = store
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc: RequestValidationError):
        # Drop the echoed input: it can contain NaN/inf, which is not valid JSON.
        detail = [{k: e[k] for k in ("loc", "msg", "type") if k in e} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    def require_model() -> ModelBundle:
        # at most once per poll interval (one thread polls): pick up a new @champion
        if load_models:
            store.maybe_refresh(poll_seconds)
        bundle = store.bundle
        if bundle is None:
            reason = store.public_error or "not loaded yet"
            raise HTTPException(503, f"No model deployed yet ({reason}). Run the pipeline first.")
        return bundle

    def run_model(bundle: ModelBundle, rows: list[dict]):
        with bundle.lock:  # one inference at a time per loaded model
            return predict_rows(bundle.model, rows, bundle.profile)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        b = store.bundle
        return {
            "status": "ok" if b else "no-model",
            "model_version": b.version if b else None,
            "feature_set": b.profile.feature_set if b else None,
            "loaded_at": b.loaded_at if b else None,
            "alias": store.alias,
            "detail": None if b else store.public_error,
        }

    @app.get("/ready")
    def ready():
        """Readiness probe: 200 only once the champion is loaded (``/health`` is liveness)."""
        b = store.bundle
        if b is None:
            raise HTTPException(503, f"No model loaded: {store.public_error or 'not loaded yet'}")
        return {"status": "ready", "model_version": b.version}

    @app.post("/reload")
    def reload(x_admin_token: str | None = Header(default=None)):
        """Hot-swap to the registry's current champion. If ADMIN_TOKEN is set, the request must
        carry it in the ``X-Admin-Token`` header (the deploy flow sends it)."""
        given = (x_admin_token or "").encode("utf-8")
        if admin_token and not hmac.compare_digest(given, admin_token.encode("utf-8")):
            raise HTTPException(401, "Missing or invalid X-Admin-Token")
        try:
            b = store.load_from_registry()
        except NoChampionError as exc:
            raise HTTPException(503, f"Reload failed: {exc}") from exc
        except (
            Exception
        ) as exc:  # the details stay in the server log (load_from_registry logs them)
            raise HTTPException(503, f"Reload failed: {store.public_error}") from exc
        return {"status": "reloaded", "model_version": b.version, "loaded_at": b.loaded_at}

    @app.get("/model-info")
    def model_info():
        b = require_model()
        p = b.profile
        return {
            "model_version": b.version,
            "feature_set": p.feature_set,
            "n_train": p.n_train,
            "numeric_columns": NUMERIC,
            "ranges": p.ranges,
            "categories": list(p.stats.vocab),
            "category_share": p.category_share,
            "missing_pct_in_training": p.missing_pct,
            "target_mean_usd": p.target_mean,
            "grid": {
                "size": F.GRID,
                "cell_counts": p.cell_counts,
                "lat_boundaries": F.LAT_BOUNDARIES,
                "lon_boundaries": F.LON_BOUNDARIES,
            },
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        b = require_model()
        t0 = time.perf_counter()
        row = req.model_dump(exclude={"source"})
        price, (flags,) = run_model(b, [row])
        latency = (time.perf_counter() - t0) * 1000
        pred_id = None
        try:
            with Session() as s:
                rec = Prediction(
                    source=req.source,
                    features=row,
                    prediction_usd=float(price[0]),
                    unknown_category=flags.unknown_category,
                    n_missing=len(flags.missing),
                    n_out_of_range=len(flags.out_of_range),
                    model_version=b.version,
                    latency_ms=latency,
                )
                s.add(rec)
                s.commit()
                pred_id = rec.id
        except Exception as exc:  # serving must not fail because logging failed
            log.error("Could not log prediction: %s", exc)
        return PredictResponse(
            prediction_id=pred_id,
            price_usd=round(float(price[0]), 2),
            quality=_quality(flags),
            seen=_seen(row, b),
            model_version=b.version,
            latency_ms=round(latency, 2),
        )

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(req: BatchRequest):
        b = require_model()
        t0 = time.perf_counter()
        rows = [r.model_dump() for r in req.rows]
        prices, flags = run_model(b, rows)
        latency = (time.perf_counter() - t0) * 1000
        try:
            with Session() as s:
                s.add_all(
                    Prediction(
                        source=req.source,
                        features=row,
                        prediction_usd=float(price),
                        unknown_category=f.unknown_category,
                        n_missing=len(f.missing),
                        n_out_of_range=len(f.out_of_range),
                        model_version=b.version,
                        latency_ms=latency / len(rows),
                    )
                    for row, price, f in zip(rows, prices, flags, strict=True)
                )
                s.commit()
        except Exception as exc:
            log.error("Could not log batch: %s", exc)
        return BatchResponse(
            prices_usd=[round(float(p), 2) for p in prices],
            n_missing=sum(bool(f.missing) for f in flags),
            n_unknown_category=sum(f.unknown_category for f in flags),
            n_out_of_range=sum(bool(f.out_of_range) for f in flags),
            model_version=b.version,
            latency_ms=round(latency, 2),
            rows_per_second=round(len(rows) / (latency / 1000), 1),
        )

    @app.get("/samples")
    def sample(
        kind: str = Query("clean", description=f"one of {', '.join(KINDS)}"),
        seed: int | None = Query(None, ge=0, description="fix the choice for a reproducible demo"),
    ):
        """A random held-out row from the training run's test set, optionally damaged."""
        if kind not in KINDS:
            raise HTTPException(422, f"kind must be one of {KINDS}")
        b = require_model()
        if not b.samples:
            raise HTTPException(404, "The deployed model has no sample rows.")
        rng = np.random.default_rng(seed)
        row = dict(b.samples[int(rng.integers(len(b.samples)))])
        true_price = row.pop("median_house_value")
        if kind == "missing":
            row["total_bedrooms"] = None
        elif kind == "unseen":
            row["ocean_proximity"] = "SUBURB"
        elif kind == "outlier":
            row["median_income"] = round(2 * b.profile.ranges["median_income"]["max"], 2)
        return {"kind": kind, "row": row, "true_price_usd": true_price}

    @app.get("/pipeline/summary")
    def pipeline_summary():
        """Tables and key numbers written by the latest training run (``reports/``)."""
        tables = {}
        for name in SUMMARY_TABLES:
            path = reports / "tables" / f"{name}.csv"
            if path.is_file():
                tables[name] = json.loads(pd.read_csv(path).to_json(orient="records"))
        metrics_path = reports / "metrics.json"
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
        )
        if not tables and metrics is None:
            raise HTTPException(404, "No reports found: run the training flow first.")
        return {"tables": tables, "metrics": metrics}

    @app.get("/monitoring/summary")
    def monitoring_summary(hours: float = Query(168, gt=0, le=24 * 365)):
        """Aggregates are computed in the database; only the 12 most recent rows are fetched."""
        since = datetime.now(UTC) - timedelta(hours=hours)
        window = Prediction.created_at >= since
        with Session() as s:
            n, unknown, outside, missing, mean_price, mean_latency = s.execute(
                select(
                    func.count(),
                    func.sum(_flag(Prediction.unknown_category)),
                    func.sum(_flag(Prediction.n_out_of_range > 0)),
                    func.sum(_flag(Prediction.n_missing > 0)),
                    func.avg(Prediction.prediction_usd),
                    func.avg(Prediction.latency_ms),
                ).where(window)
            ).one()
            source_rows = s.execute(
                select(
                    Prediction.source,
                    func.count(),
                    func.sum(
                        _flag(
                            Prediction.unknown_category
                            | (Prediction.n_out_of_range > 0)
                            | (Prediction.n_missing > 0)
                        )
                    ),
                )
                .where(window)
                .group_by(Prediction.source)
                .order_by(func.count().desc())
            ).all()
            recent = (
                s.execute(select(Prediction).where(window).order_by(Prediction.id.desc()).limit(12))
                .scalars()
                .all()
            )

        def pct(count):
            return round(100 * float(count or 0) / n, 2) if n else None

        b = store.bundle
        return {
            "window_hours": hours,
            "n_predictions": int(n),
            "unknown_category_pct": pct(unknown),
            "out_of_range_pct": pct(outside),
            "missing_pct": pct(missing),
            "expected_missing_pct": round(b.profile.missing_pct, 2) if b else None,
            "mean_prediction_usd": round(float(mean_price), 0) if n else None,
            "training_mean_usd": round(b.profile.target_mean, 0) if b else None,
            "mean_latency_ms": round(float(mean_latency), 2) if n else None,
            "by_source": {src: {"n": int(c), "flagged": int(f or 0)} for src, c, f in source_rows},
            "recent": [
                {
                    "id": r.id,
                    "created_at": r.created_at.isoformat(timespec="seconds")
                    if r.created_at
                    else None,
                    "source": r.source,
                    "prediction_usd": round(r.prediction_usd, 0),
                    "unknown_category": r.unknown_category,
                    "n_missing": r.n_missing,
                    "n_out_of_range": r.n_out_of_range,
                    "ocean_proximity": (r.features or {}).get("ocean_proximity"),
                    "median_income": (r.features or {}).get("median_income"),
                }
                for r in recent
            ],
        }

    return app
