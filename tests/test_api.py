"""API tests with an untrained in-memory model (no MLflow, no download)."""

import numpy as np
import pytest
from app.main import create_app
from app.model_store import ModelBundle, ModelStore
from fastapi.testclient import TestClient

from tfdata_mlops import features as F
from tfdata_mlops.data import TARGET
from tfdata_mlops.inference import build_profile


@pytest.fixture(scope="module")
def bundle(prep):
    import keras

    keras.utils.set_random_seed(0)
    model = F.build_model(F.FEATURE_SETS["all features"], prep.stats)
    profile = build_profile(prep.split.train, prep.stats, "all features")
    samples = [
        {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
        for r in prep.split.test.head(40).to_dict("records")
    ]
    return ModelBundle(model, profile, samples, version="7", run_id="abc")


@pytest.fixture()
def client(bundle, tmp_path, monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    store = ModelStore()
    store.set_bundle(bundle)
    (tmp_path / "reports" / "tables").mkdir(parents=True)
    app = create_app(
        store=store,
        database_url=f"sqlite:///{tmp_path}/app.db",
        load_models=False,
        reports_dir=tmp_path / "reports",
    )
    with TestClient(app) as c:
        c.reports = tmp_path / "reports"
        yield c


def payload(prep, i=0, **override):
    row = prep.split.test.iloc[i].drop(TARGET).to_dict()
    row = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
    return {**row, **override}


def test_health_ready_and_ui(client):
    assert client.get("/ready").json() == {"status": "ready", "model_version": "7"}
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_version"] == "7" and h["feature_set"] == "all features"
    page = client.get("/").text
    assert "Housing Lab" in page and "/static/app.js" in page
    assert client.get("/static/style.css").status_code == 200


def test_predict_returns_a_price_flags_and_what_the_model_saw(client, prep):
    body = {**payload(prep), "source": "test"}
    r = client.post("/predict", json=body).json()
    assert r["prediction_id"] == 1 and r["model_version"] == "7" and np.isfinite(r["price_usd"])
    expected_missing = ["total_bedrooms"] if body["total_bedrooms"] is None else []
    assert r["quality"]["missing"] == expected_missing
    assert r["quality"]["unknown_category"] is False
    seen = r["seen"]
    assert len(seen["z_scores"]) == 8
    assert (
        seen["grid_cell"] == seen["grid_row"] * 20 + seen["grid_col"]
        and 0 <= seen["grid_cell"] < 400
    )
    assert 0 <= seen["income_bucket"] <= 4


def test_the_same_request_gives_the_same_price(client, prep):
    a = client.post("/predict", json=payload(prep, 3)).json()["price_usd"]
    b = client.post("/predict", json=payload(prep, 3)).json()["price_usd"]
    assert a == b


def test_damaged_rows_are_answered_and_flagged(client, prep):
    base = payload(prep, 1)
    r = client.post(
        "/predict",
        json={**base, "ocean_proximity": "SUBURB", "total_bedrooms": None, "median_income": 60.0},
    ).json()
    assert r["quality"]["unknown_category"] is True
    assert (
        "total_bedrooms" in r["quality"]["missing"]
        and "median_income" in r["quality"]["out_of_range"]
    )
    assert np.isfinite(r["price_usd"])


def test_batch_endpoint_is_consistent_with_single_requests(client, prep):
    rows = [payload(prep, i) for i in range(10)]
    batch = client.post("/predict/batch", json={"rows": rows, "source": "batch-test"}).json()
    single = [client.post("/predict", json=r).json()["price_usd"] for r in rows]
    # float32 sums may differ by a few units in the last place between batch sizes (test_inference)
    assert np.allclose(batch["prices_usd"], single, rtol=1e-5, atol=0.1)
    assert batch["rows_per_second"] > 0 and batch["model_version"] == "7"
    assert client.get("/monitoring/summary").json()["by_source"]["batch-test"]["n"] == 10


@pytest.mark.parametrize(
    "change",
    [
        {"latitude": 123},
        {"housing_median_age": -1},
        {"median_income": -0.1},
        {"ocean_proximity": "<script>alert(1)</script>"},
        {"ocean_proximity": ""},
        {"source": "<b>"},
        {"households": 1e9},
        {"longitude": None},
    ],
)
def test_validation(client, prep, change):
    assert client.post("/predict", json={**payload(prep), **change}).status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_inf_are_rejected_without_crashing(client, prep, value):
    import json

    # json.dumps writes NaN / Infinity / -Infinity, which Python's JSON parser (FastAPI) accepts
    body = json.dumps({**payload(prep), "population": value})
    assert "NaN" in body or "Infinity" in body
    r = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["detail"][0]["loc"][-1] == "population"


def test_batch_size_limits(client, prep):
    too_many = {"rows": [payload(prep)] * 1001}
    assert client.post("/predict/batch", json=too_many).status_code == 422
    assert client.post("/predict/batch", json={"rows": []}).status_code == 422


def test_monitoring_summary_counts_flags(client, prep):
    client.post("/predict", json={**payload(prep, 0), "source": "ui"})
    client.post(
        "/predict", json={**payload(prep, 1, ocean_proximity="SUBURB"), "source": "sample-unseen"}
    )
    client.post(
        "/predict", json={**payload(prep, 2, total_bedrooms=None), "source": "sample-missing"}
    )
    m = client.get("/monitoring/summary").json()
    assert m["n_predictions"] == 3 and m["unknown_category_pct"] == pytest.approx(33.33, abs=0.01)
    assert m["missing_pct"] >= 33.33 and m["training_mean_usd"] > 0 and m["mean_latency_ms"] > 0
    assert m["by_source"]["sample-unseen"] == {"n": 1, "flagged": 1}
    assert len(m["recent"]) == 3 and m["recent"][0]["source"] == "sample-missing"
    assert client.get("/monitoring/summary?hours=0").status_code == 422


def test_samples(client):
    clean = client.get("/samples?seed=3").json()
    assert (
        clean["kind"] == "clean"
        and clean["true_price_usd"] > 0
        and "median_house_value" not in clean["row"]
    )
    assert client.get("/samples?seed=3").json() == clean  # a seed makes the demo reproducible
    assert client.get("/samples?kind=missing&seed=3").json()["row"]["total_bedrooms"] is None
    assert client.get("/samples?kind=unseen&seed=3").json()["row"]["ocean_proximity"] == "SUBURB"
    outlier = client.get("/samples?kind=outlier&seed=3").json()["row"]
    assert (
        outlier["median_income"]
        > client.get("/model-info").json()["ranges"]["median_income"]["max"]
    )
    assert client.get("/samples?kind=blurred").status_code == 422


def test_model_info_describes_the_training_data(client):
    info = client.get("/model-info").json()
    assert info["model_version"] == "7" and info["categories"][0] == "<1H OCEAN"
    assert info["grid"]["size"] == 20 and len(info["grid"]["cell_counts"]) == 400
    assert len(info["grid"]["lat_boundaries"]) == 19 and set(info["ranges"]["median_income"]) >= {
        "min",
        "max",
    }


def test_pipeline_summary_reads_the_latest_reports(client):
    assert client.get("/pipeline/summary").status_code == 404  # nothing written yet
    (client.reports / "tables" / "format_benchmark.csv").write_text(
        "format,bytes,size_vs_csv,examples_per_second\ncsv,100,1.0,1000\n"
    )
    (client.reports / "metrics.json").write_text('{"run": {"model_version": "7"}}')
    out = client.get("/pipeline/summary").json()
    assert (
        out["tables"]["format_benchmark"][0]["format"] == "csv"
        and out["metrics"]["run"]["model_version"] == "7"
    )


def test_no_model_gives_503(tmp_path):
    app = create_app(
        store=ModelStore(), database_url=f"sqlite:///{tmp_path}/x.db", load_models=False
    )
    row = {
        "longitude": -120,
        "latitude": 35,
        "housing_median_age": 10,
        "total_rooms": 100,
        "population": 100,
        "households": 30,
        "median_income": 3,
        "ocean_proximity": "INLAND",
    }
    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "no-model"
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json=row).status_code == 503
        assert c.get("/model-info").status_code == 503


def test_reload_requires_token(bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/empty_registry.db")
    store = ModelStore()
    store.set_bundle(bundle)
    app = create_app(store=store, database_url=f"sqlite:///{tmp_path}/y.db", load_models=False)
    with TestClient(app) as c:
        assert c.post("/reload").status_code == 401
        # a non-ASCII header must be rejected cleanly, not crash the comparison
        assert (
            c.post("/reload", headers={"X-Admin-Token": "s\xe9cret".encode("latin-1")}).status_code
            == 401
        )
        # right token, but no MLflow registry here -> 503 rather than a crash
        assert c.post("/reload", headers={"X-Admin-Token": "s3cret"}).status_code == 503


def test_load_errors_are_logged_but_not_shown_to_callers(tmp_path, monkeypatch, caplog):
    """Internal exception text (paths, URLs) stays in the server log."""
    from tfdata_mlops import tracking

    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    monkeypatch.setattr(
        tracking, "get_alias_version", lambda *a, **k: type("MV", (), {"version": "3"})()
    )

    def broken(*args, **kwargs):
        raise OSError("cannot open /var/lib/secret/model.keras for user mlops:pw")

    monkeypatch.setattr(tracking, "load_model", broken)
    app = create_app(store=ModelStore(), database_url=f"sqlite:///{tmp_path}/z.db")
    with caplog.at_level("WARNING"), TestClient(app) as c:  # the startup load fails as well
        answers = [
            c.post("/reload").json()["detail"],
            c.get("/health").json()["detail"],
            c.get("/ready").json()["detail"],
            c.get("/model-info").json()["detail"],
        ]
    assert all("secret" not in a and "could not be loaded" in a for a in answers), answers
    assert "/var/lib/secret/model.keras" in caplog.text


@pytest.mark.parametrize(
    ("bind", "token", "warned"),
    [
        ("0.0.0.0", "", True),
        ("127.0.0.1", "", False),
        ("localhost", "", False),
        ("0.0.0.0", "t", False),
    ],
)
def test_an_open_reload_endpoint_is_warned_about(
    tmp_path, monkeypatch, caplog, bind, token, warned
):
    monkeypatch.setenv("BIND_ADDR", bind)
    monkeypatch.setenv("ADMIN_TOKEN", token)
    with caplog.at_level("WARNING", logger="api"):
        create_app(store=ModelStore(), database_url=f"sqlite:///{tmp_path}/w.db", load_models=False)
    assert ("ADMIN_TOKEN is empty" in caplog.text) is warned
