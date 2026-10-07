"""California housing data (Geron, Ch. 2), the book's stratified split and sharded CSV files (Ch. 13).

The file is the one the book downloads in Chapter 2 (``housing.tgz`` from the book's own GitHub
repository). It has 20,640 rows: eight numeric columns (``total_bedrooms`` has 207 missing values),
the five-category ``ocean_proximity`` and the target ``median_house_value``. Both the archive and
the extracted CSV are checked against pinned SHA-256 digests, so a changed or corrupted download is
refused.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

log = logging.getLogger(__name__)

HOUSING_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
)
HOUSING_TGZ_SHA256 = "d4cd501af90475f09b814c7447c7701f59bf28e8cf1180205ae5ace9737a0109"
HOUSING_CSV_SHA256 = "8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e"

NUMERIC = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
CATEGORICAL = "ocean_proximity"
TARGET = "median_house_value"
OCEAN_VOCAB = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
# CSV column order: the eight inputs, the category, then the target last (as in Ch. 13)
CSV_COLUMNS = [*NUMERIC, CATEGORICAL, TARGET]
N_INPUTS = len(NUMERIC)
TARGET_SCALE = 100_000.0  # the model predicts hundreds of thousands of dollars
EXPECTED_ROWS = 20_640


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _extract_csv(archive: bytes) -> bytes:
    """Read ``housing.csv`` from the archive without writing anything else to disk."""
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        member = tar.getmember("housing.csv")
        if not member.isfile():
            raise RuntimeError("housing.csv is not a regular file in the archive")
        fh = tar.extractfile(member)
        assert fh is not None
        return fh.read()


def ensure_housing_csv(cache_dir: str | Path = "data") -> Path:
    """Return the path of the verified ``housing.csv``, downloading it once."""
    cache_dir = Path(cache_dir)
    csv_path = cache_dir / "housing" / "housing.csv"
    if csv_path.exists() and sha256_file(csv_path) == HOUSING_CSV_SHA256:
        return csv_path
    log.info("Downloading %s", HOUSING_URL)
    archive = _download(HOUSING_URL)
    digest = hashlib.sha256(archive).hexdigest()
    if digest != HOUSING_TGZ_SHA256:
        raise RuntimeError(f"housing.tgz has SHA-256 {digest}, expected {HOUSING_TGZ_SHA256}")
    data = _extract_csv(archive)
    if hashlib.sha256(data).hexdigest() != HOUSING_CSV_SHA256:
        raise RuntimeError("housing.csv inside the archive does not match its pinned SHA-256")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_suffix(".part")
    tmp.write_bytes(data)
    os.replace(tmp, csv_path)  # atomic: never leave a half-written file behind
    return csv_path


def load_housing(cache_dir: str | Path = "data") -> pd.DataFrame:
    df = pd.read_csv(ensure_housing_csv(cache_dir))
    if len(df) != EXPECTED_ROWS or set(df.columns) != set(CSV_COLUMNS):
        raise RuntimeError(f"Unexpected housing data: {df.shape}, columns {list(df.columns)}")
    return df[CSV_COLUMNS]  # the file has the target before ocean_proximity; the CSV shards do not


def income_category(income: pd.Series) -> pd.Series:
    """The five income strata of Ch. 2 (used for stratified sampling)."""
    return pd.cut(income, bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])


@dataclass
class HousingSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> dict:
        return {"n_train": len(self.train), "n_valid": len(self.valid), "n_test": len(self.test)}


def split_housing(
    df: pd.DataFrame, seed: int = 42, test_size: float = 0.2, valid_fraction: float = 0.25
) -> HousingSplit:
    """Stratified train / validation / test split (Ch. 2: stratify on the income category).

    ``test_size`` of the data is held out first; ``valid_fraction`` of what is left becomes the
    validation set (0.2 and 0.25 give 60% / 20% / 20%). Rows come out shuffled.
    """
    strata = income_category(df["median_income"])
    first = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest_idx, test_idx = next(first.split(df, strata))
    rest = df.iloc[rest_idx]
    second = StratifiedShuffleSplit(n_splits=1, test_size=valid_fraction, random_state=seed)
    train_idx, valid_idx = next(second.split(rest, strata.iloc[rest_idx]))
    return HousingSplit(
        train=rest.iloc[train_idx].reset_index(drop=True),
        valid=rest.iloc[valid_idx].reset_index(drop=True),
        test=df.iloc[test_idx].reset_index(drop=True),
    )


@dataclass(frozen=True)
class Stats:
    """What the pipeline learns from the training set only (never from validation or test data).

    ``medians`` replace missing values (Ch. 2); ``means`` / ``stds`` standardise the numeric
    columns after that (Ch. 2 and 13). ``stds`` use ``ddof=0``, as ``StandardScaler`` does.
    """

    medians: tuple[float, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    vocab: tuple[str, ...] = tuple(OCEAN_VOCAB)

    def to_dict(self) -> dict:
        return {
            "numeric": NUMERIC,
            "medians": list(self.medians),
            "means": list(self.means),
            "stds": list(self.stds),
            "vocab": list(self.vocab),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Stats:
        return cls(tuple(d["medians"]), tuple(d["means"]), tuple(d["stds"]), tuple(d["vocab"]))


def compute_stats(train: pd.DataFrame) -> Stats:
    medians = train[NUMERIC].median()
    filled = train[NUMERIC].fillna(medians)
    return Stats(
        medians=tuple(float(v) for v in medians),
        means=tuple(float(v) for v in filled.mean()),
        stds=tuple(float(v) for v in filled.std(ddof=0)),
    )


def write_csv_shards(
    df: pd.DataFrame, directory: str | Path, prefix: str, n_shards: int
) -> list[Path]:
    """Split ``df`` into ``n_shards`` CSV files of (almost) equal length, each with a header row.

    Files of identical length interleave best (Ch. 13). Floats are written with their full
    precision and missing values as empty fields, so the file loses nothing.
    """
    if not 1 <= n_shards <= len(df):
        raise ValueError(f"n_shards must be between 1 and {len(df)}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, part in enumerate(np.array_split(np.arange(len(df)), n_shards)):
        path = directory / f"{prefix}_{i:02d}.csv"
        df.iloc[part][CSV_COLUMNS].to_csv(path, index=False)
        paths.append(path)
    return paths


def count_csv_rows(paths: list[str | Path]) -> int:
    """Data rows in the CSV shards (header rows excluded)."""
    total = 0
    for p in paths:
        with Path(p).open("rb") as fh:
            total += sum(1 for _ in fh) - 1
    return total
