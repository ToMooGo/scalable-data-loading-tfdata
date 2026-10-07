import hashlib
import io
import tarfile

import numpy as np
import pandas as pd
import pytest

from tfdata_mlops import data
from tfdata_mlops.config import linspace, load_config


def test_config_extends_overrides_and_env(tmp_path, monkeypatch):
    (tmp_path / "base.yaml").write_text(
        "a: 1\nnested: {x: 1, y: 2}\nurl: ${MY_URL:sqlite:///x.db}\n"
    )
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\nnested: {y: 3}\nport: ${MY_PORT:8000}\n"
    )
    monkeypatch.setenv("MY_URL", "postgresql://u:p@h/db")
    cfg = load_config(tmp_path / "child.yaml", overrides={"a": 5})
    assert cfg["a"] == 5 and cfg["nested"] == {"x": 1, "y": 3}
    assert cfg["url"] == "postgresql://u:p@h/db"
    assert cfg["port"] == 8000 and isinstance(
        cfg["port"], int
    )  # fully substituted scalars keep their type


def test_linspace():
    assert linspace({"start": 0, "stop": 1, "num": 3}) == [0.0, 0.5, 1.0]
    assert linspace([1, 2]) == [1.0, 2.0]


def test_shipped_configs_are_consistent():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    full = load_config(root / "full_flow_config.yaml")
    quick = load_config(root / "quick_flow_config.yaml")
    monitor = load_config(root / "monitor_flow_config.yaml")
    assert full["flow"] == "full" and monitor["flow"] == "monitor"
    assert full["data"]["shards"] == {"train": 20, "valid": 10, "test": 10}
    assert (
        quick["part_c"]["heads"]["linear"]["epochs"] < full["part_c"]["heads"]["linear"]["epochs"]
    )
    assert quick["part_c"]["heads"]["mlp"]["hidden"] == [64, 64]  # deep-merged, not replaced
    # the quick run never overwrites the committed results of the full run in reports/
    assert full["reports"]["dir"] == "reports"
    assert all(v.startswith("reports/quick") for v in quick["reports"].values())
    assert "deploy-canary" in monitor["monitor"]["exclude_source_prefixes"]
    assert set(monitor["monitor"]) >= {
        "window_hours",
        "min_predictions",
        "max_unknown_category_pct",
    }


def test_split_is_stratified_and_disjoint(housing):
    tagged = housing.assign(row_id=np.arange(len(housing)))  # coordinates are not unique row keys
    sp = data.split_housing(tagged, seed=3)
    assert len(sp.train) + len(sp.valid) + len(sp.test) == len(housing)
    ids = [set(part["row_id"]) for part in (sp.train, sp.valid, sp.test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert ids[0] | ids[1] | ids[2] == set(range(len(housing)))
    assert sp.sizes["n_test"] == pytest.approx(0.2 * len(housing), abs=2)
    assert sp.sizes["n_valid"] == pytest.approx(0.2 * len(housing), abs=2)
    # stratified on income: each part has about the same share of high-income blocks
    shares = [float((part.median_income > 4.5).mean()) for part in (sp.train, sp.valid, sp.test)]
    assert max(shares) - min(shares) < 0.03
    again = data.split_housing(tagged, seed=3)
    pd.testing.assert_frame_equal(sp.test, again.test)  # same seed, same split


def test_stats_come_from_the_training_rows_only(prep):
    train = prep.split.train
    filled = train[data.NUMERIC].fillna(train[data.NUMERIC].median())
    assert prep.stats.means == pytest.approx(tuple(filled.mean()))
    assert prep.stats.stds == pytest.approx(tuple(filled.std(ddof=0)))
    assert prep.stats.medians[data.NUMERIC.index("total_bedrooms")] == pytest.approx(
        train["total_bedrooms"].median()
    )
    assert data.Stats.from_dict(prep.stats.to_dict()) == prep.stats


def test_csv_shards_have_headers_equal_length_and_lose_nothing(prep):
    for name in ("train", "valid", "test"):
        paths = prep.csv[name]
        frames = [pd.read_csv(p) for p in paths]
        assert all(list(f.columns) == data.CSV_COLUMNS for f in frames)
        sizes = [len(f) for f in frames]
        assert max(sizes) - min(sizes) <= 1
        assert sum(sizes) == len(prep.table(name)) == data.count_csv_rows(paths)
    with pytest.raises(ValueError):
        data.write_csv_shards(prep.split.train, "/tmp/never", "x", 0)


def test_extract_csv_reads_only_the_expected_member():
    payload = b"a,b\n1,2\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("housing.csv")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    assert data._extract_csv(buf.getvalue()) == payload


def test_download_with_the_wrong_hash_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_download", lambda url, timeout=60.0: b"not the real archive")
    with pytest.raises(RuntimeError, match="SHA-256"):
        data.ensure_housing_csv(tmp_path)
    assert not (tmp_path / "housing" / "housing.csv").exists()


def test_cached_file_is_verified_before_use(tmp_path, monkeypatch):
    path = tmp_path / "housing" / "housing.csv"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"corrupted")
    calls = []

    def download(url, timeout=60.0):
        calls.append(url)
        return b"still wrong"

    monkeypatch.setattr(data, "_download", download)
    with pytest.raises(RuntimeError, match="SHA-256"):
        data.ensure_housing_csv(tmp_path)
    assert calls == [data.HOUSING_URL]  # a bad cache is not trusted: it triggers a download
    assert path.read_bytes() == b"corrupted"  # and a failed download replaces nothing

    # a cached file with the pinned digest is used as it is, without a download
    monkeypatch.setattr(data, "HOUSING_CSV_SHA256", hashlib.sha256(b"corrupted").hexdigest())
    assert data.ensure_housing_csv(tmp_path) == path and len(calls) == 1


def test_income_categories_cover_all_rows(housing):
    cats = data.income_category(housing["median_income"])
    assert cats.notna().all() and set(cats.astype(int)) <= {1, 2, 3, 4, 5}
    assert np.all(np.diff(cats.astype(int).to_numpy()[np.argsort(housing.median_income)]) >= 0)
