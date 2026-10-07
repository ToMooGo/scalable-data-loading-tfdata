"""Part C: turning raw columns into numbers a neural network can use (Geron, Ch. 13, "The Features API").

The book (Geron, Chapter 13) builds features with ``tf.feature_column``. TensorFlow now marks
that API as deprecated and Keras 3 removed ``DenseFeatures``, so this module builds the same
features with Keras preprocessing layers inside the model:

==============================================  ============================================
book (``tf.feature_column``)                    here (Keras 3 layers)
==============================================  ============================================
``numeric_column`` + ``normalizer_fn``          ``Normalization``
``bucketized_column``                           ``Discretization``
``categorical_column_with_vocabulary_list``     ``StringLookup`` (index 0 is the unknown bucket)
``indicator_column`` (one-hot)                  ``output_mode="one_hot"``
``crossed_column`` (hashed)                     ``HashedCrossing``
``embedding_column``                            ``Embedding``
``DenseFeatures`` layer                         ``Concatenate`` of the blocks
==============================================  ============================================

The boundaries, the vocabulary, the 20 x 20 grid and the hash sizes are the book's numbers. Because
the preprocessing is part of the model, the same Keras model that trains on batches from tf.data can
be served on raw rows with no second implementation of the preprocessing (the training/serving skew
that the book's TF Transform section warns about).
"""

from __future__ import annotations

from dataclasses import dataclass

import keras
import numpy as np
from keras import layers

from .data import CATEGORICAL, NUMERIC, Stats

INCOME_BOUNDARIES = [1.5, 3.0, 4.5, 6.0]  # 5 buckets (Ch. 13)
AGE_BOUNDARIES = [-1.0, -0.5, 0.0, 0.5, 1.0]  # on the standardised age: 6 buckets (Ch. 13, fn. 9)
GRID = 20  # 20 x 20 grid over California (Ch. 13)
LAT_BOUNDARIES = [float(v) for v in np.linspace(32.0, 42.0, GRID - 1)]
LON_BOUNDARIES = [float(v) for v in np.linspace(-125.0, -114.0, GRID - 1)]
AGE_X_OCEAN_BUCKETS = 100  # hash_bucket_size for age x ocean_proximity (Ch. 13)
LOCATION_HASH_BUCKETS = 1000  # hash_bucket_size for the location cross (Ch. 13)
N_OCEAN = 5


@dataclass(frozen=True)
class FeatureSet:
    """Which blocks of features the model gets. All blocks are optional.

    ``ocean``: ``None``, ``"onehot"`` (5 categories + 1 unknown bucket) or ``"embed"`` (2-D).
    ``age_x_ocean``: bucketised age crossed with ocean_proximity, hashed to 100 buckets.
    ``location``: the 20 x 20 grid as ``"grid_onehot"`` (400 exact cells), ``"grid_embed"``
    (the same 400 cells, learned 8-D embedding), ``"hash_onehot"`` or ``"hash_embed"`` (the book's
    hashing of the 400 cells into 1,000 buckets).
    """

    numeric: bool = True
    ocean: str | None = "onehot"
    ocean_embed_dim: int = 2
    income_buckets: bool = False
    age_x_ocean: str | None = None
    age_x_ocean_embed_dim: int = 4
    location: str | None = None
    location_embed_dim: int = 8

    def __post_init__(self) -> None:
        for field, allowed in (
            ("ocean", (None, "onehot", "embed")),
            ("age_x_ocean", (None, "onehot", "embed")),
            ("location", (None, "grid_onehot", "grid_embed", "hash_onehot", "hash_embed")),
        ):
            if getattr(self, field) not in allowed:
                raise ValueError(f"{field} must be one of {allowed}")


# The ablation of Part C: each set adds one idea to "ocean one-hot", except the first two.
FEATURE_SETS: dict[str, FeatureSet] = {
    "numeric only": FeatureSet(ocean=None),
    "+ ocean one-hot": FeatureSet(),
    "+ ocean embedding": FeatureSet(ocean="embed"),
    "+ income buckets": FeatureSet(income_buckets=True),
    "+ age x ocean": FeatureSet(age_x_ocean="embed"),
    "+ grid one-hot (400)": FeatureSet(location="grid_onehot"),
    "+ grid embedding (400)": FeatureSet(location="grid_embed"),
    "+ hashed grid (1000)": FeatureSet(location="hash_embed"),
    "all features": FeatureSet(income_buckets=True, age_x_ocean="embed", location="hash_embed"),
}
DEFAULT_FEATURE_SET = "all features"


def make_inputs() -> dict:
    """Keras inputs for the raw columns: eight numbers and the ocean_proximity string."""
    inputs = {name: keras.Input(shape=(1,), name=name, dtype="float32") for name in NUMERIC}
    inputs[CATEGORICAL] = keras.Input(shape=(1,), name=CATEGORICAL, dtype="string")
    return inputs


def _normalization(mean, std, name: str) -> layers.Layer:
    return layers.Normalization(
        mean=np.asarray(mean, dtype="float32"),
        variance=np.square(np.asarray(std, dtype="float32")),
        axis=-1,
        name=name,
    )


def build_preprocessing(fs: FeatureSet, stats: Stats, inputs: dict | None = None):
    """Wire the preprocessing layers for ``fs``.

    Returns ``(inputs, features, blocks)``: the raw inputs, one dense tensor per example and the
    ``(name, width)`` of each block in the order they are concatenated.
    """
    inputs = inputs or make_inputs()
    blocks: list[tuple[str, object, int]] = []

    def add(name: str, tensor, width: int) -> None:
        blocks.append((name, tensor, width))

    if fs.numeric:
        numbers = layers.Concatenate(name="numeric_concat")([inputs[n] for n in NUMERIC])
        add(
            "numeric", _normalization(stats.means, stats.stds, "standardize")(numbers), len(NUMERIC)
        )

    need_ocean_index = fs.ocean == "embed" or fs.age_x_ocean is not None
    ocean_index = None
    if need_ocean_index:
        ocean_index = layers.StringLookup(
            vocabulary=list(stats.vocab), num_oov_indices=1, output_mode="int", name="ocean_index"
        )(inputs[CATEGORICAL])
    if fs.ocean == "onehot":
        hot = layers.StringLookup(
            vocabulary=list(stats.vocab),
            num_oov_indices=1,
            output_mode="one_hot",
            name="ocean_onehot",
        )(inputs[CATEGORICAL])
        add("ocean_onehot", hot, N_OCEAN + 1)
    elif fs.ocean == "embed":
        emb = layers.Embedding(N_OCEAN + 1, fs.ocean_embed_dim, name="ocean_embedding")(ocean_index)
        add("ocean_embedding", layers.Flatten(name="ocean_embedding_flat")(emb), fs.ocean_embed_dim)

    if fs.income_buckets:
        income = layers.Discretization(
            bin_boundaries=INCOME_BOUNDARIES, output_mode="one_hot", name="income_bucket"
        )(inputs["median_income"])
        add("income_bucket", income, len(INCOME_BOUNDARIES) + 1)

    if fs.age_x_ocean is not None:
        age_i = NUMERIC.index("housing_median_age")
        age_z = _normalization([stats.means[age_i]], [stats.stds[age_i]], "age_standardize")(
            inputs["housing_median_age"]
        )
        age_bucket = layers.Discretization(
            bin_boundaries=AGE_BOUNDARIES, output_mode="int", name="age_bucket"
        )(age_z)
        cross = layers.HashedCrossing(num_bins=AGE_X_OCEAN_BUCKETS, name="age_x_ocean_cross")(
            [age_bucket, ocean_index]
        )
        if fs.age_x_ocean == "embed":
            emb = layers.Embedding(
                AGE_X_OCEAN_BUCKETS, fs.age_x_ocean_embed_dim, name="age_x_ocean_embedding"
            )(cross)
            add(
                "age_x_ocean",
                layers.Flatten(name="age_x_ocean_flat")(emb),
                fs.age_x_ocean_embed_dim,
            )
        else:
            hot = layers.CategoryEncoding(
                AGE_X_OCEAN_BUCKETS, output_mode="one_hot", name="age_x_ocean_onehot"
            )(cross)
            add("age_x_ocean", hot, AGE_X_OCEAN_BUCKETS)

    if fs.location is not None:
        if fs.location.startswith("hash"):
            lat_i = layers.Discretization(
                bin_boundaries=LAT_BOUNDARIES, output_mode="int", name="lat_bucket"
            )(inputs["latitude"])
            lon_i = layers.Discretization(
                bin_boundaries=LON_BOUNDARIES, output_mode="int", name="lon_bucket"
            )(inputs["longitude"])
            cross = layers.HashedCrossing(num_bins=LOCATION_HASH_BUCKETS, name="location_cross")(
                [lat_i, lon_i]
            )
            if fs.location == "hash_embed":
                emb = layers.Embedding(
                    LOCATION_HASH_BUCKETS, fs.location_embed_dim, name="location_embedding"
                )(cross)
                add("location", layers.Flatten(name="location_flat")(emb), fs.location_embed_dim)
            else:
                hot = layers.CategoryEncoding(
                    LOCATION_HASH_BUCKETS, output_mode="one_hot", name="location_onehot"
                )(cross)
                add("location", hot, LOCATION_HASH_BUCKETS)
        else:
            # exact cell: the outer product of two one-hot vectors is a one-hot vector of 400 cells
            lat_h = layers.Discretization(
                bin_boundaries=LAT_BOUNDARIES, output_mode="one_hot", name="lat_onehot"
            )(inputs["latitude"])
            lon_h = layers.Discretization(
                bin_boundaries=LON_BOUNDARIES, output_mode="one_hot", name="lon_onehot"
            )(inputs["longitude"])
            grid = layers.Dot(axes=(2, 1), name="grid_outer_product")(
                [layers.Reshape((GRID, 1))(lat_h), layers.Reshape((1, GRID))(lon_h)]
            )
            grid = layers.Flatten(name="grid_cell")(grid)
            if fs.location == "grid_embed":
                # an embedding is a one-hot vector times a matrix: a Dense layer without bias
                emb = layers.Dense(fs.location_embed_dim, use_bias=False, name="grid_embedding")(
                    grid
                )
                add("location", emb, fs.location_embed_dim)
            else:
                add("location", grid, GRID * GRID)

    if not blocks:
        raise ValueError("the feature set is empty")
    features = (
        blocks[0][1]
        if len(blocks) == 1
        else layers.Concatenate(name="features")([tensor for _, tensor, _ in blocks])
    )
    return inputs, features, [(name, width) for name, _, width in blocks]


def build_model(
    fs: FeatureSet,
    stats: Stats,
    hidden: tuple[int, ...] = (64, 64),
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Preprocessing layers followed by a small fully connected regressor.

    The model takes the raw columns (a dict of arrays shaped ``(n, 1)``) and returns the median
    house value in hundreds of thousands of dollars.
    """
    inputs, features, _ = build_preprocessing(fs, stats)
    x = features
    for i, units in enumerate(hidden):
        x = layers.Dense(units, activation="relu", name=f"hidden_{i}")(x)
    output = layers.Dense(1, name="price")(x)
    model = keras.Model(inputs, output, name="housing_regressor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


def feature_width(fs: FeatureSet, stats: Stats) -> int:
    _, _, blocks = build_preprocessing(fs, stats)
    return sum(width for _, width in blocks)


def reference_features(df, stats: Stats, fs: FeatureSet) -> np.ndarray:
    """The same features computed with plain NumPy, to test the layers against.

    Only blocks without a learned part (one-hot, buckets, exact grid) are supported.
    """
    n = len(df)
    parts = []
    if fs.numeric:
        x = df[NUMERIC].fillna(dict(zip(NUMERIC, stats.medians, strict=True))).to_numpy(np.float64)
        parts.append((x - np.asarray(stats.means)) / np.asarray(stats.stds))
    if fs.ocean == "onehot":
        idx = np.array(
            [stats.vocab.index(v) + 1 if v in stats.vocab else 0 for v in df[CATEGORICAL]]
        )
        parts.append(np.eye(N_OCEAN + 1)[idx])
    if fs.income_buckets:
        idx = np.digitize(df["median_income"].to_numpy(), INCOME_BOUNDARIES)
        parts.append(np.eye(len(INCOME_BOUNDARIES) + 1)[idx])
    if fs.location == "grid_onehot":
        lat = np.digitize(df["latitude"].to_numpy(), LAT_BOUNDARIES)
        lon = np.digitize(df["longitude"].to_numpy(), LON_BOUNDARIES)
        cell = lat * GRID + lon
        parts.append(np.eye(GRID * GRID)[cell])
    if (
        fs.ocean == "embed"
        or fs.age_x_ocean
        or fs.location in ("grid_embed", "hash_onehot", "hash_embed")
    ):
        raise ValueError("reference_features supports one-hot and bucket blocks only")
    return np.concatenate(parts, axis=1) if parts else np.empty((n, 0))
