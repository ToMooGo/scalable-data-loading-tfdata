"""Plain containers for what each part of the pipeline produces."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class PartResult:
    """``params`` and ``metrics`` go to MLflow, ``tables`` become CSV artifacts and report tables,
    ``extra`` carries objects (arrays, models) that the figures and later parts need."""

    name: str
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    run_id: str | None = None
