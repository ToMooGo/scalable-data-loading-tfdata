"""Request / response models."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

# letters, digits and a few separators; "<" and ">" are allowed because the book's category
# "<1H OCEAN" contains them (no quotes, "&", "/" or "="). The UI escapes every value it displays.
SAFE_TEXT = r"^[A-Za-z0-9 _.:@<>-]+$"
Number = Annotated[float, Field(allow_inf_nan=False)]


class House(BaseModel):
    """One census block group (Ch. 2 housing data). Bounds only reject impossible values; a value
    that is possible but outside the training range is accepted and flagged in the response."""

    longitude: Number = Field(..., ge=-180, le=180, description="degrees")
    latitude: Number = Field(..., ge=-90, le=90, description="degrees")
    housing_median_age: Number = Field(..., ge=0, le=500, description="years")
    total_rooms: Number = Field(..., ge=0, le=1e7)
    total_bedrooms: Number | None = Field(
        None,
        ge=0,
        le=1e7,
        description="optional: a missing value is replaced by the training median",
    )
    population: Number = Field(..., ge=0, le=1e7)
    households: Number = Field(..., ge=0, le=1e7)
    median_income: Number = Field(..., ge=0, le=1e3, description="in tens of thousands of dollars")
    ocean_proximity: str = Field(..., min_length=1, max_length=32, pattern=SAFE_TEXT)


class PredictRequest(House):
    source: str = Field(
        "api",
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9 _.:@-]+$",
        description="where the input came from (ui, sample-missing, ci, ...)",
    )


class BatchRequest(BaseModel):
    rows: list[House] = Field(..., min_length=1, max_length=1000)
    source: str = Field("batch", min_length=1, max_length=32, pattern=r"^[A-Za-z0-9 _.:@-]+$")


class Quality(BaseModel):
    missing: list[str]
    unknown_category: bool
    out_of_range: list[str]


class Seen(BaseModel):
    """What the preprocessing layers turn the request into (for display; the model computes it
    itself): standardised numbers and the cell of the 20 x 20 location grid."""

    z_scores: dict[str, float]
    grid_row: int
    grid_col: int
    grid_cell: int
    income_bucket: int


class PredictResponse(BaseModel):
    prediction_id: int | None
    price_usd: float
    quality: Quality
    seen: Seen
    model_version: str
    latency_ms: float


class BatchResponse(BaseModel):
    prices_usd: list[float]
    n_missing: int
    n_unknown_category: int
    n_out_of_range: int
    model_version: str
    latency_ms: float
    rows_per_second: float
