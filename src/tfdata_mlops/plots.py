"""Figures for the technical report and the README.

Every figure is rebuilt from the CSV tables a training run writes (``reports/tables``), so a
figure can never disagree with the numbers next to it. The style is plain: white background,
dark text, one muted colour per series, labels written out in full (no legend codes).
"""

from __future__ import annotations

import re
import textwrap
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .data import TARGET_SCALE  # noqa: E402

NAVY = "#1f4e79"
ORANGE = "#d98324"
GREEN = "#3b7d4f"
GRAY = "#7a7a7a"
LIGHT = "#9db4cc"
BASELINE_SET = "+ ocean one-hot"  # the baseline of the feature-set tables
DPI = 300
BAR_LABEL = 9.5  # bar annotations; printed at 7 pt or more

plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e3e3e3",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _dash(label: str) -> str:
    """Range labels such as ``1.5-3`` with an en dash, so the hyphen is not read as a minus."""
    return re.sub(r"(?<=\d)-(?=\d)", "\u2013", str(label))


def _bar_label(ax, container, **kwargs):
    """Bar annotations with a white halo, so a grid line never runs through a digit."""
    texts = ax.bar_label(container, **kwargs)
    for text in texts:
        text.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white")])
    return texts


THOUSANDS = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
PLAIN_COUNT = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
IN_K = mticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}")


def _wrap(text: str, width: int) -> str:
    return textwrap.fill(" ".join(str(text).split()), width=width, break_long_words=False)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def _read(tables_dir: Path, name: str) -> pd.DataFrame | None:
    path = tables_dir / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def _headroom(ax, values, factor: float = 1.22) -> None:
    ax.set_xlim(0, float(np.nanmax(values)) * factor)


# --------------------------------------------------------------------------------------------
# A. Data API
# --------------------------------------------------------------------------------------------
def fig_csv_throughput(t: dict, out: Path) -> Path | None:
    df = t["csv_ablation"]
    if df is None:
        return None
    df = df[df.step_ms == 0]
    variants = list(dict.fromkeys(df.variant))
    strict = df[df.regime == "strict"].set_index("variant").examples_per_second.reindex(variants)
    default = df[df.regime == "default"].set_index("variant").examples_per_second.reindex(variants)
    y = np.arange(len(variants))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 0.62 * len(variants) + 1.7))
    b1 = ax.barh(y - h / 2, strict.values, h, color=NAVY, label="Only the steps listed")
    b2 = ax.barh(y + h / 2, default.values, h, color=ORANGE, label="tf.data default optimisations")
    _bar_label(ax, b1, fmt="{:,.0f}", padding=3, fontsize=BAR_LABEL)
    _bar_label(ax, b2, fmt="{:,.0f}", padding=3, fontsize=BAR_LABEL)
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(v, 30) for v in variants])
    ax.invert_yaxis()
    ax.set_xlabel("Examples read per second (higher is faster, no training step)")
    _headroom(ax, np.r_[strict.values, default.values], 1.16)
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2, frameon=False)
    ax.set_title("Reading sharded CSV files: effect of each step of the pipeline")
    return _save(fig, out / "a_csv_throughput.png")


def fig_prefetch(t: dict, out: Path) -> Path | None:
    df = t["prefetch_sweep"]
    if df is None:
        return None
    fig, ax = plt.subplots(figsize=(6.9, 4.5))
    spec = {0: ("No prefetch", NAVY, "o"), 1: ("With prefetch", ORANGE, "s")}
    for prefetch, (label, colour, marker) in spec.items():
        d = df[df.prefetch == prefetch].sort_values("step_ms")
        if d.empty:
            continue
        ax.plot(
            d.step_ms,
            d.ms_per_batch,
            marker=marker,
            ms=8,
            color=colour,
            lw=1.8,
            label=f"{label}: measured",
        )
        ax.plot(
            d.step_ms,
            d.predicted_ms_per_batch,
            ls="--",
            color=colour,
            lw=1.2,
            marker=marker,
            ms=4.5,
            mfc="white",
            label=f"{label}: formula",
        )
    ax.set_xlabel("Time of one training step (ms per batch)")
    ax.set_ylabel("Time per batch (ms)")
    ax.set_title("Prefetching hides the input time behind the training step")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    return _save(fig, out / "a_prefetch_sweep.png")


def fig_shuffle(t: dict, out: Path) -> Path | None:
    df = t["shuffle_quality"]
    if df is None:
        return None
    labels = [_wrap(v, 20) for v in df.label]
    y = np.arange(len(df))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 0.47 * len(df) + 2.0), sharey=True)
    a1.barh(y, df.spearman, 0.6, color=NAVY, label="Measured")
    pred = df.predicted_spearman
    for yi, v, pv in zip(y, df.spearman, pred, strict=True):  # label to the right of bar and marker
        x_end = max(v, 0.0 if pd.isna(pv) else pv, 0.0)
        a1.text(
            x_end + 0.045,
            yi,
            f"{v:.2f}".replace("-", "\u2212"),
            va="center",
            fontsize=BAR_LABEL,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )
    mask = pred.notna().to_numpy()
    if mask.any():
        a1.plot(
            pred[mask],
            y[mask],
            "D",
            color=ORANGE,
            mec="black",
            mew=0.6,
            ms=5.5,
            label="Formula",
        )
    a1.set_xlim(-0.2, 1.25)
    a1.axvline(0, color="black", lw=0.8)
    a1.set_xlabel("Spearman correlation of position\nand target (1 = untouched, 0 = mixed)")
    a1.set_yticks(y)
    a1.set_yticklabels(labels)
    a1.invert_yaxis()
    a1.grid(axis="y", visible=False)
    handles, names = a1.get_legend_handles_labels()
    order = [names.index("Measured"), names.index("Formula")] if "Formula" in names else [0]
    a1.legend(
        [handles[i] for i in order],
        [names[i] for i in order],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        frameon=False,
    )
    bars2 = a2.barh(y, df.batch_diversity, 0.6, color=GREEN)
    _bar_label(
        a2, bars2, labels=[f"{v:.2f}" for v in df.batch_diversity], padding=3, fontsize=BAR_LABEL
    )
    a2.set_xlim(0, 1.25)
    a2.set_xlabel("Batch diversity\n(1 = each batch looks like the whole set)")
    a2.grid(axis="y", visible=False)
    a1.set_title("Order of the data")
    a2.set_title("Variety inside a batch")
    fig.suptitle("Shuffling a training set that is stored in sorted order", y=1.02, fontsize=11)
    fig.tight_layout()
    return _save(fig, out / "a_shuffle_quality.png")


# --------------------------------------------------------------------------------------------
# B. TFRecord
# --------------------------------------------------------------------------------------------
def fig_formats(t: dict, out: Path) -> Path | None:
    df = t["format_benchmark"]
    if df is None:
        return None
    df = df[df.step_ms == 0].reset_index(drop=True)
    labels = [_wrap(v, 22) for v in df.format]
    y = np.arange(len(df))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 0.6 * len(df) + 2.0), sharey=True)
    b1 = a1.barh(y, df.size_vs_csv, 0.6, color=NAVY)
    _bar_label(
        a1,
        b1,
        fmt="{:.2f}x",
        padding=3,
        fontsize=BAR_LABEL,
    )
    _headroom(a1, df.size_vs_csv, 1.22)
    a1.set_xlabel("Size relative to the CSV shards\n(1.00 = the size of the CSV)")
    a1.set_yticks(y)
    a1.set_yticklabels(labels)
    a1.invert_yaxis()
    a1.grid(axis="y", visible=False)
    b2 = a2.barh(y, df.examples_per_second, 0.6, color=ORANGE)
    _bar_label(a2, b2, fmt="{:,.0f}", padding=3, fontsize=BAR_LABEL)
    _headroom(a2, df.examples_per_second, 1.22)
    a2.xaxis.set_major_formatter(THOUSANDS)
    a2.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, steps=[1, 2, 4, 5, 10]))
    a2.set_xlabel("Examples read per second\n(higher is faster)")
    a2.grid(axis="y", visible=False)
    a1.set_title("Size")
    a2.set_title("Reading speed")
    fig.suptitle("CSV and TFRecord shards holding the same training rows", y=1.02, fontsize=11)
    fig.tight_layout()
    return _save(fig, out / "b_formats.png")


def _plain_log_axis(ax, axis: str, ticks=None) -> None:
    """Log axis with ordinary numbers (1, 2, 5, 10 ...) instead of powers of ten."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    if ticks is not None:
        target.set_major_locator(mticker.FixedLocator(list(ticks)))
    else:
        target.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
    target.set_major_formatter(THOUSANDS)
    target.set_minor_formatter(mticker.NullFormatter())
    target.set_minor_locator(mticker.NullLocator())


def fig_scale(t: dict, out: Path) -> Path | None:
    df = t["scale_test"]
    if df is None or len(df) < 2:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 4.0))
    a1.plot(
        df.rows,
        df.stream_growth_mb,
        "o-",
        color=NAVY,
        lw=1.8,
        label="tf.data streaming (TFRecord.gz)",
    )
    a1.plot(
        df.rows,
        df.pandas_growth_mb,
        "s-",
        color=ORANGE,
        lw=1.8,
        label="pandas: read the CSV shards and standardise",
    )
    a1.set_xscale("log")
    a1.set_yscale("log")
    _plain_log_axis(a1, "x", df.rows)
    _plain_log_axis(a1, "y")
    a1.set_xlabel("Training rows (log scale)")
    a1.set_ylabel("Memory growth (MB, log scale)")
    a1.set_title("Memory")
    pandas_rate = df.rows / df.pandas_seconds
    a2.plot(
        df.rows,
        df.stream_examples_per_s,
        "o-",
        color=NAVY,
        lw=1.8,
        label="tf.data streaming (TFRecord.gz)",
    )
    a2.plot(
        df.rows,
        pandas_rate,
        "s-",
        color=ORANGE,
        lw=1.8,
        label="pandas: read the CSV shards and standardise",
    )
    a2.set_xscale("log")
    _plain_log_axis(a2, "x", df.rows)
    a2.yaxis.set_major_formatter(THOUSANDS)
    a2.set_xlabel("Training rows (log scale)")
    a2.set_ylabel("Examples per second")
    a2.set_ylim(bottom=0)
    a2.set_title("Speed")
    handles, names = a1.get_legend_handles_labels()
    fig.legend(
        handles, names, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02)
    )
    fig.suptitle(
        "The training set repeated 1, 10 and 100 times: memory and speed of one pass",
        y=1.0,
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, out / "b_scale_test.png")


# --------------------------------------------------------------------------------------------
# C. Features
# --------------------------------------------------------------------------------------------
def fig_features(t: dict, out: Path) -> Path | None:
    df = t["feature_summary"]
    if df is None:
        return None
    heads = [h for h in ("linear", "mlp") if h in set(df["head"])]
    titles = {"linear": "Linear head", "mlp": "Two hidden layers of 64 units"}
    sets = list(dict.fromkeys(df.feature_set))
    y = np.arange(len(sets))
    fig, axes = plt.subplots(
        1,
        len(heads),
        figsize=(7.5, 0.5 * len(sets) + 2.5),
        sharey=True,
        squeeze=False,
    )
    for ax, head in zip(axes[0], heads, strict=True):
        d = df[df["head"] == head].set_index("feature_set").reindex(sets)
        lo = (d.test_rmse - d.test_rmse_lo).to_numpy()
        hi = (d.test_rmse_hi - d.test_rmse).to_numpy()
        ax.errorbar(
            d.test_rmse.to_numpy(),
            y,
            xerr=[lo, hi],
            fmt="o",
            color=NAVY,
            ecolor=NAVY,
            capsize=2.5,
            ms=4,
        )
        ax.axvline(d.test_rmse[BASELINE_SET], color=GRAY, lw=1.0, ls=":", zorder=1)
        ax.set_xlabel("Test RMSE in USD (lower is better)\nmean over seeds and 95% interval")
        ax.set_title(titles.get(head, head))
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap(s, 18) for s in sets])
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(THOUSANDS)
    axes[0][0].invert_yaxis()  # the y axis is shared: invert it once, first set at the top
    handles = [
        Line2D([], [], marker="o", color=NAVY, ms=5, lw=1.2, label="Mean over seeds, 95% interval"),
        Line2D([], [], color=GRAY, lw=1.0, ls=":", label=f"Baseline: {BASELINE_SET}"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01)
    )
    fig.suptitle(
        "Test error by feature set (note the different horizontal scales)", y=1.0, fontsize=10.5
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, out / "c_feature_ablation.png")


def fig_location_grid(t: dict, out: Path, buckets: int = 1000) -> Path | None:
    df = t["location_cells"]
    if df is None:
        return None
    grid = df.sort_values("cell").train_rows.to_numpy().reshape(20, 20).astype(float)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 3.7), gridspec_kw={"width_ratios": [1.05, 1]})
    shown = np.ma.masked_equal(grid, 0)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")  # cells without training rows
    im = a1.imshow(
        shown,
        origin="lower",
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=max(2, grid.max())),
        interpolation="nearest",
    )
    a1.grid(False)
    a1.set_xlabel("Longitude cell (west to east)")
    a1.set_ylabel("Latitude cell (south to north)")
    a1.set_xticks(range(0, 20, 4))
    a1.set_yticks(range(0, 20, 4))
    a1.set_title("Training rows per grid cell\n(light grey = none)")
    cb = fig.colorbar(im, ax=a1, fraction=0.046, pad=0.03)
    cb.ax.yaxis.set_major_formatter(PLAIN_COUNT)  # 1, 10, 100, 1,000 instead of powers of ten
    cb.set_label("Training rows (log scale)")

    # hashing the 400 cells into `buckets` buckets: how many cells land in the same bucket?
    n_cells = len(df)
    measured = df.bucket.value_counts().value_counts().sort_index()
    measured = measured.reindex(range(1, int(measured.index.max()) + 1), fill_value=0)
    p = 1.0 / buckets
    ks = np.arange(1, len(measured) + 1)
    expected = np.array([buckets * comb(n_cells, k) * p**k * (1 - p) ** (n_cells - k) for k in ks])
    m_bars = a2.bar(
        ks - 0.2, measured.to_numpy(), 0.4, color=NAVY, label="Measured (HashedCrossing)"
    )
    e_bars = a2.bar(ks + 0.2, expected, 0.4, color=ORANGE, label="Expected for random hashing")
    _bar_label(a2, m_bars, labels=[f"{v:,.0f}" for v in measured], padding=2, fontsize=BAR_LABEL)
    _bar_label(a2, e_bars, labels=[f"{v:,.0f}" for v in expected], padding=2, fontsize=BAR_LABEL)
    a2.set_ylim(0, float(max(measured.max(), expected.max())) * 1.5)
    a2.set_xticks(ks)
    a2.set_xlabel("Number of grid cells sharing one bucket")
    a2.set_ylabel(f"Number of buckets (out of {buckets:,})")
    a2.set_title(f"400 cells hashed into {buckets:,} buckets")
    a2.legend(loc="upper right", frameon=False)
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, out / "c_location_grid.png")


# --------------------------------------------------------------------------------------------
# D. TFDS and image records
# --------------------------------------------------------------------------------------------
def fig_mnist_formats(t: dict, out: Path) -> Path | None:
    df = t["mnist_formats"]
    if df is None:
        return None
    combos = list(dict.fromkeys(zip(df.encoding, df.compression, strict=True)))
    names = {"none": "uncompressed", "GZIP": "GZIP"}
    labels = [f"{e}, {names.get(c, c)}" for e, c in combos]
    y = np.arange(len(combos))
    sizes = [
        df[(df.encoding == e) & (df.compression == c)].bytes_per_image.iloc[0] for e, c in combos
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 0.78 * len(combos) + 2.9), sharey=True)
    b1 = a1.barh(y, sizes, 0.6, color=GRAY)
    _bar_label(a1, b1, fmt="{:,.0f} B", padding=3, fontsize=BAR_LABEL)
    _headroom(a1, sizes, 1.26)
    a1.set_yticks(y)
    a1.set_yticklabels(labels)
    a1.invert_yaxis()
    a1.set_xlabel("Bytes per image in the file")
    a1.set_title("Size of one 28 x 28 image")
    a1.grid(axis="y", visible=False)

    modes = list(dict.fromkeys(df.reading))
    colours = dict(zip(modes, [NAVY, LIGHT, ORANGE, GREEN], strict=False))
    h = 0.8 / len(modes)
    top = 0.0
    for row, (e, c) in enumerate(combos):
        have = [
            m
            for m in modes
            if len(df[(df.encoding == e) & (df.compression == c) & (df.reading == m)])
        ]
        for i, mode in enumerate(have):  # the bars of one row are centred on its tick
            val = df[(df.encoding == e) & (df.compression == c) & (df.reading == mode)]
            val = float(val.examples_per_second.iloc[0])
            top = max(top, val)
            offset = (i - (len(have) - 1) / 2) * h
            bar = a2.barh(row + offset, val, h * 0.92, color=colours[mode])
            _bar_label(a2, bar, fmt="{:,.0f}", padding=2, fontsize=BAR_LABEL)
    a2.set_xlim(0, top * 1.28)
    a2.set_xlabel("Images decoded per second (higher is faster)")
    a2.set_title("Reading speed")
    a2.grid(axis="y", visible=False)
    a2.xaxis.set_major_formatter(THOUSANDS)
    a2.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, steps=[1, 2, 4, 5, 10]))
    handles = [Patch(color=colours[m], label=m) for m in modes]
    fig.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0)
    )
    fig.text(
        0.5,
        -0.045,
        "Decoding a batch at a time is only possible for the raw encoding.",
        ha="center",
        fontsize=BAR_LABEL,
    )
    fig.suptitle("The same MNIST training images stored in three encodings", y=1.0, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return _save(fig, out / "d_mnist_formats.png")


# --------------------------------------------------------------------------------------------
# E. The trained model
# --------------------------------------------------------------------------------------------
def fig_model(t: dict, out: Path) -> Path | None:
    preds, groups, hist = t["test_predictions"], t["error_by_group"], t["training_history"]
    if preds is None or groups is None:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.4))
    (a1, a2), (a3, a4) = axes
    if hist is not None:
        a1.plot(hist.epoch, hist.rmse * TARGET_SCALE, color=NAVY, lw=1.8, label="Training batches")
        a1.plot(
            hist.epoch, hist.val_rmse * TARGET_SCALE, color=ORANGE, lw=1.8, label="Validation set"
        )
        a1.set_xlabel("Epoch")
        a1.set_ylabel("RMSE (thousand USD)")
        a1.legend(frameon=False)
        a1.yaxis.set_major_formatter(IN_K)
    a1.set_title("(a) Learning curve")
    hb = a2.hexbin(
        preds.median_house_value / 1e3,
        preds.prediction / 1e3,
        gridsize=40,
        cmap="Blues",
        mincnt=1,
        bins="log",
    )
    lim = [0, 540]
    a2.plot(lim, lim, color="black", lw=0.9, ls="--")
    a2.set_xlim(lim)
    a2.set_ylim(lim)
    a2.set_xlabel("Actual median house value (thousand USD)")
    a2.set_ylabel("Predicted (thousand USD)")
    a2.set_title("(b) Test set: predicted against actual")
    cb = fig.colorbar(hb, ax=a2, fraction=0.046, pad=0.03)
    cb.ax.yaxis.set_major_formatter(PLAIN_COUNT)  # 1, 10, 100, 1,000 instead of powers of ten
    cb.set_label("Number of block groups (log scale)")
    for ax, kind, title, colour, ylabel in (
        (
            a3,
            "ocean_proximity",
            "(c) Test error by area type",
            NAVY,
            "Area type",
        ),
        (
            a4,
            "median_income",
            "(d) Test error by income band",
            ORANGE,
            "Median income (x 10,000 USD)",
        ),
    ):
        d = groups[groups.group_by == kind]
        yy = np.arange(len(d))
        bars = ax.barh(yy, d.rmse, 0.6, color=colour)
        ax.set_yticks(yy)
        ax.set_yticklabels([f"{_dash(g)} (n={n:,})" for g, n in zip(d.group, d.rows, strict=True)])
        ax.set_ylabel(ylabel)
        ax.invert_yaxis()
        _bar_label(
            ax, bars, labels=[f"{v / 1e3:,.1f}" for v in d.rmse], padding=3, fontsize=BAR_LABEL
        )
        _headroom(ax, d.rmse, 1.25)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(IN_K)
        ax.set_xlabel("RMSE (thousand USD)")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    fig.tight_layout(h_pad=1.8, w_pad=1.6)
    return _save(fig, out / "e_model_quality.png")


FIGURES = (
    fig_csv_throughput,
    fig_prefetch,
    fig_shuffle,
    fig_formats,
    fig_scale,
    fig_features,
    fig_location_grid,
    fig_mnist_formats,
    fig_model,
)
TABLES = (
    "csv_ablation",
    "prefetch_sweep",
    "shuffle_quality",
    "format_benchmark",
    "scale_test",
    "feature_summary",
    "location_cells",
    "mnist_formats",
    "test_predictions",
    "error_by_group",
    "training_history",
)


def make_figures(tables_dir: Path, figures_dir: Path, cfg: dict) -> dict[str, Path]:
    """Draw every figure whose table exists; returns ``{figure file stem: path}``."""
    tables_dir, figures_dir = Path(tables_dir), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables = {name: _read(tables_dir, name) for name in TABLES}
    drawn = {}
    for draw in FIGURES:
        path = draw(tables, figures_dir)
        if path is not None:
            drawn[path.stem] = path
    return drawn
