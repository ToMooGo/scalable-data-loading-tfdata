"""Turn the result tables of a training run into LaTeX, so no number in the report is typed by hand.

Reads  reports/tables/*.csv  and  reports/metrics.json
Writes reports/technical_report/generated/tab_*.tex   (the tabular environment only)
       reports/technical_report/generated/numbers.tex (\\metric{part}{name} and \\param{part}{name} macros)

Run it from anywhere:  python reports/technical_report/make_tables.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TABLES = ROOT / "reports" / "tables"
METRICS = ROOT / "reports" / "metrics.json"
GATES = ROOT / "reports" / "quality_gates.json"
OUT = HERE / "generated"

_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex(text: object) -> str:
    """Escape a value for LaTeX text and collapse repeated spaces."""
    s = re.sub(r"\s+", " ", str(text)).strip()
    return "".join(_ESCAPES.get(ch, ch) for ch in s)


def num(x: float, decimals: int = 0) -> str:
    """Thousands separators with a plain comma; a real minus sign."""
    if x is None or pd.isna(x):
        return ""
    s = f"{x:,.{decimals}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def signed(x: float, decimals: int = 0) -> str:
    if x is None or pd.isna(x):
        return ""
    return ("$+$" if x > 0 else "") + num(x, decimals)


def usd(x: float) -> str:
    """A dollar amount with the sign before the dollar sign, as the text prints it: -$955."""
    return ("$-$" if x < 0 else "") + "\\$" + num(abs(x))


def t_interval(values, confidence: float = 0.95) -> tuple[float, float, float]:
    """Mean and t-interval of the mean, as ``stats.mean_ci`` and Chapter 2 of the book."""
    v = np.asarray(values, dtype=float)
    lo, hi = st.t.interval(confidence, len(v) - 1, loc=v.mean(), scale=st.sem(v))
    return float(v.mean()), float(lo), float(hi)


def sci(x: float) -> str:
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"${mantissa}\\times 10^{{{int(exponent)}}}$"


def table(spec: str, header: list[str], rows: list[list[str]], top: list[str] | None = None) -> str:
    """A booktabs tabular. ``top`` lines (a grouped header, ``\\cmidrule``) go above ``header``."""
    lines = [
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        *(top or []),
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body, encoding="utf-8")


# --------------------------------------------------------------------------- Part A
def tab_csv_speed() -> None:
    ab = read("csv_ablation")
    cols = {}
    for regime in ("strict", "default"):
        for step in (0.0, 4.0):
            part = ab[(ab.regime == regime) & (ab.step_ms == step)].set_index("variant")
            cols[(regime, step)] = part.examples_per_second
    order = list(ab[ab.regime == "strict"].variant.drop_duplicates())
    short = {
        "one file at a time, no shuffle, serial parse": "one file, no shuffle, serial parse",
        "prefetch(1)  [the book's function]": "prefetch(1), the book's function",
    }

    def label(v: str) -> str:
        for long, brief in short.items():
            v = v.replace(long, brief)
        return tex(v)

    rows = [
        [label(v)]
        + [
            num(cols[k][v])
            for k in [("strict", 0.0), ("default", 0.0), ("strict", 4.0), ("default", 4.0)]
        ]
        for v in order
    ]
    np_ = read("numpy_baseline").set_index("step_ms").examples_per_second
    rows.append(
        [
            "pandas and NumPy: read the shards, standardise, slice batches",
            num(np_[0.0]),
            num(np_[0.0]),
            num(np_[4.0]),
            num(np_[4.0]),
        ]
    )
    write(
        "tab_csv_speed",
        table(
            "L{6.2cm}R{1.95cm}R{1.95cm}R{1.95cm}R{1.95cm}",
            [
                "Pipeline",
                "Only listed steps",
                "tf.data defaults",
                "Only listed steps",
                "tf.data defaults",
            ],
            rows,
            top=[
                " & \\multicolumn{2}{c}{No training step} & \\multicolumn{2}{c}{4 ms training step} \\\\",
                "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
            ],
        ),
    )


def tab_prefetch() -> None:
    sw = read("prefetch_sweep")
    rows = []
    for step in sorted(sw.step_ms.unique()):
        a = sw[(sw.step_ms == step) & (sw.prefetch == 0)].iloc[0]
        b = sw[(sw.step_ms == step) & (sw.prefetch == 1)].iloc[0]
        rows.append(
            [
                num(step, 1),
                num(a.ms_per_batch, 2),
                num(a.predicted_ms_per_batch, 2),
                num(b.ms_per_batch, 2),
                num(b.predicted_ms_per_batch, 2),
            ]
        )
    write(
        "tab_prefetch",
        table(
            "R{1.6cm}R{2.6cm}R{2.6cm}R{2.6cm}R{2.6cm}",
            [
                "Step (ms)",
                "No prefetch, measured",
                "No prefetch, formula",
                "prefetch(1), measured",
                "prefetch(1), formula",
            ],
            rows,
        ),
    )


def tab_shuffle() -> None:
    sh = read("shuffle_quality")
    rows = []
    for _, r in sh.iterrows():
        rows.append(
            [
                tex(r.label),
                str(int(r.n_readers)),
                num(r.buffer),
                f"{r.spearman:.3f}".replace("-", "$-$"),
                "" if pd.isna(r.predicted_spearman) else f"{r.predicted_spearman:.3f}",
                f"{r.batch_diversity:.3f}",
            ]
        )
    write(
        "tab_shuffle",
        table(
            "L{4.4cm}R{1.4cm}R{1.4cm}R{1.8cm}R{1.8cm}R{2.4cm}",
            ["Shuffling", "Readers", "Buffer", "Spearman", "Formula", "Batch diversity"],
            rows,
        ),
    )


# --------------------------------------------------------------------------- Part B
def tab_formats() -> None:
    fb = read("format_benchmark")
    rows = [
        [tex(r.format), num(r.bytes), f"{r.size_vs_csv:.2f}", num(r.examples_per_second)]
        for _, r in fb.iterrows()
        if r.step_ms == 0
    ]
    write(
        "tab_formats",
        table(
            "L{7.0cm}R{2.2cm}R{2.0cm}R{2.6cm}",
            ["Format and reading", "Bytes", "Size vs CSV", "Examples per second"],
            rows,
        ),
    )


def tab_wire() -> None:
    w = read("wire_size").iloc[0]
    rows = [
        ["Records", num(w.records)],
        [
            "Serialised Example, smallest and largest (bytes)",
            f"{num(w.min_example_bytes)} and {num(w.max_example_bytes)}",
        ],
        ["Serialised Example, mean (bytes)", num(w.mean_example_bytes, 2)],
        ["File size from the formula (bytes)", num(w.predicted_file_bytes)],
        ["File size on disk (bytes)", num(w.file_bytes)],
        [
            "Formula equals the length of every record",
            "yes" if bool(w.formula_matches_all) else "no",
        ],
    ]
    write("tab_wire", table("p{9cm}r", ["Quantity", "Value"], rows))


def tab_corruption() -> None:
    c = read("corruption")
    rows = [
        [tex(r.file), tex(r.case), tex(r.result), "yes" if bool(r.error_detected) else "no"]
        for _, r in c.iterrows()
    ]
    write("tab_corruption", table("llll", ["File", "Case", "Result", "Error raised"], rows))


def tab_scale() -> None:
    s = read("scale_test")
    rows = [
        [
            num(r.copies),
            num(r.rows),
            num(r.csv_mb, 1),
            num(r.tfrecord_gz_mb, 1),
            num(r.stream_growth_mb),
            num(r.pandas_growth_mb),
            num(r.rows / r.stream_examples_per_s, 1),
            num(r.pandas_seconds, 1),
        ]
        for _, r in s.iterrows()
    ]
    write(
        "tab_scale",
        table(
            "R{1.2cm}R{1.8cm}R{1.4cm}R{2.1cm}R{1.7cm}R{1.7cm}R{1.6cm}R{1.6cm}",
            [
                "Copies",
                "Rows",
                "CSV (MB)",
                "TFRecord.gz (MB)",
                "tf.data growth (MB)",
                "pandas growth (MB)",
                "tf.data pass (s)",
                "pandas load (s)",
            ],
            rows,
        ),
    )


# --------------------------------------------------------------------------- Part C
def tab_features() -> None:
    fs = read("feature_summary")
    for head in fs["head"].unique():
        part = fs[fs["head"] == head]
        rows = []
        for _, r in part.iterrows():
            if pd.isna(r.valid_diff_vs_baseline):
                diff = "baseline"
            else:
                star = "*" if bool(r.significant_vs_baseline) else ""
                diff = (
                    f"{signed(r.valid_diff_vs_baseline)} "
                    f"[{signed(r.valid_diff_lo)}, {signed(r.valid_diff_hi)}]{star}"
                )
            rows.append(
                [
                    tex(r.feature_set).replace(" x ", " $\\times$ "),
                    num(r.features),
                    num(r.params),
                    num(r.valid_rmse),
                    diff,
                    num(r.test_rmse),
                    f"{r.test_r2:.3f}",
                ]
            )
        write(
            f"tab_features_{head}",
            table(
                "L{3.9cm}R{1.0cm}R{1.2cm}R{1.3cm}L{4.0cm}R{1.3cm}R{1.0cm}",
                [
                    "Feature set",
                    "Inputs",
                    "Params",
                    "Valid. RMSE",
                    "Diff. from baseline\\newline [95\\% interval]",
                    "Test RMSE",
                    "Test $R^2$",
                ],
                rows,
            ),
        )


def tab_hash() -> None:
    h = read("hash_collisions").iloc[0]
    rows = [
        ["Grid cells", num(h.cells)],
        ["Hash buckets", num(h.buckets)],
        ["Distinct buckets used by all cells (measured)", num(h.distinct_buckets_all_cells)],
        [
            "Distinct buckets expected for a uniform random hash",
            num(h.expected_distinct_buckets_all_cells, 1),
        ],
        ["Cells that contain training rows", num(h.cells_with_training_rows)],
        [
            "Occupied cells that share a bucket with another occupied cell",
            num(h.occupied_cells_sharing_a_bucket),
        ],
        [
            "Training rows in such shared buckets",
            f"{num(h.training_rows_in_shared_buckets)} ({h.training_rows_in_shared_buckets_pct:.2f}\\%)",
        ],
    ]
    write("tab_hash", table("p{10cm}r", ["Quantity", "Value"], rows))


# --------------------------------------------------------------------------- Part D
def tab_mnist_formats() -> None:
    mf = read("mnist_formats")
    columns = [
        "one record at a time, serial",
        "one record at a time, parallel",
        "a batch at a time, serial",
        "a batch at a time, parallel",
    ]
    rows = []
    for (enc, comp), part in mf.groupby(["encoding", "compression"], sort=False):
        speed = part.set_index("reading").examples_per_second
        rows.append(
            [tex(enc), tex(comp), num(part.iloc[0].bytes_per_image)]
            + [num(speed[c]) if c in speed.index else "--" for c in columns]
        )
    lines = [
        "\\begin{tabular}{llR{2.1cm}R{1.6cm}R{1.6cm}R{1.6cm}R{1.6cm}}",
        "\\toprule",
        " & & & \\multicolumn{2}{c}{One record at a time} "
        "& \\multicolumn{2}{c}{A batch at a time} \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Encoding & Compression & Bytes/image & Serial & Parallel & Serial & Parallel \\\\",
        "\\midrule",
    ]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_mnist_formats", "\n".join(lines) + "\n")


def tab_mnist_fits() -> None:
    m = read("mnist_fits")
    g = m.groupby("input", sort=False).test_accuracy.agg(["mean", "std", "count"]).reset_index()
    rows = [
        [
            "\\texttt{tfds.load}" if r.input == "tfds.load" else tex(r.input),
            num(r["count"]),
            f"{r['mean']:.4f}",
            "" if pd.isna(r["std"]) else f"{r['std']:.4f}",
        ]
        for _, r in g.iterrows()
    ]
    write(
        "tab_mnist_fits",
        table("lrrr", ["Input", "Seeds", "Mean test accuracy", "Standard deviation"], rows),
    )


# --------------------------------------------------------------------------- Part E
def tab_error_groups() -> None:
    e = read("error_by_group")
    for group_by, name in (("ocean_proximity", "ocean"), ("median_income", "income")):
        part = e[e.group_by == group_by]
        rows = [
            [
                re.sub(r"(?<=\d)-(?=\d)", "--", tex(r.group)),  # 1.5-3 -> en dash, not a minus
                num(r.rows),
                num(r.rmse),
                signed(r.mean_error),
            ]
            for _, r in part.iterrows()
        ]
        write(
            f"tab_error_{name}",
            table(
                "lrrr",
                ["Group", "Test rows", "RMSE (USD)", "Mean error (USD)"],
                rows,
            ),
        )


def tab_gates() -> None:
    path = GATES
    if not path.is_file():
        return
    checks = json.loads(path.read_text(encoding="utf-8"))["checks"]
    rules = (
        ("test R-squared", "at least"),
        ("RMSE gain", "at least"),
        ("unseen category", "equals"),
        ("missing number", "equals"),
    )
    rows = []
    for c in checks:
        rule = next((r for prefix, r in rules if c["check"].startswith(prefix)), "at most")
        decimals = 3 if c["limit"] < 10 else 0
        if rule == "equals":  # a yes/no check stored as 1 or 0
            value, required = ("yes" if c["value"] == c["limit"] else "no"), "yes"
        else:
            value = sci(c["value"]) if 0 < c["value"] < 1e-2 else num(c["value"], decimals)
            required = f"{rule} {num(c['limit'], decimals)}"
        label = c["check"]
        if label.startswith("test RMSE vs champion"):
            label = "test RMSE against the current champion (USD)"
        if label.startswith("registered model reproduces"):
            label = "reproduces the training RMSE, |difference| (USD)"
        rows.append([tex(label), value, required, "pass" if c["passed"] else "fail"])
    write("tab_gates", table("p{8cm}rrl", ["Check", "Value", "Required", "Result"], rows))


# --------------------------------------------------------------------------- derived numbers
def derived() -> dict[str, str]:
    """Numbers that the text quotes and that are computed from the tables (ratios, differences).

    Each is written as ``\\val{name}``; names may contain any character."""
    v: dict[str, str] = {}

    def ratio(x: float, d: int = 2) -> str:
        return f"{x:.{d}f}"

    # Part A: the ablation, with the variants numbered 1 to 6 as in the table
    ab = read("csv_ablation")
    ab["k"] = ab.variant.str.extract(r"^(\d+)\.").astype(int)
    for (regime, step), part in ab.groupby(["regime", "step_ms"]):
        speed = part.set_index("k").examples_per_second
        tag = f"{regime}:{int(step)}"
        for k, val in speed.items():
            v[f"abl:{tag}:{k}"] = num(val)
        for a, b in ((2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (5, 1), (5, 6)):
            v[f"ablr:{tag}:{a}:{b}"] = ratio(speed[a] / speed[b])
    s0 = ab[(ab.regime == "strict") & (ab.step_ms == 0)].set_index("k").examples_per_second
    gains = [s0[a] / s0[a - 1] for a in (2, 3, 4, 5)]
    v["gain:strict:0:min"], v["gain:strict:0:max"] = ratio(min(gains)), ratio(max(gains))
    s4 = ab[(ab.regime == "strict") & (ab.step_ms == 4)].set_index("k").examples_per_second
    v["abl:strict:4:min14"], v["abl:strict:4:max14"] = num(s4[:4].min()), num(s4[:4].max())
    d4 = ab[(ab.regime == "default") & (ab.step_ms == 4)].examples_per_second
    v["abl:default:4:spread"] = f"{100 * (d4.max() / d4.min() - 1):.1f}"
    d0 = ab[(ab.regime == "default") & (ab.step_ms == 0)].examples_per_second
    v["abl:default:0:min"], v["abl:default:0:max"] = num(d0.min()), num(d0.max())
    v["abl:0:max"] = num(ab[ab.step_ms == 0].examples_per_second.max())  # fastest CSV pipeline
    npb = read("numpy_baseline").set_index("step_ms").examples_per_second
    for step, val in npb.items():
        v[f"numpy:{int(step)}"] = num(val)
    book = ab[(ab.regime == "strict") & (ab.k == 5)].set_index("step_ms").examples_per_second
    v["numpy:over:book:0"] = ratio(npb[0.0] / book[0.0], 1)
    v["numpy:book:share:4"] = f"{100 * book[4.0] / npb[4.0]:.0f}"
    sw = read("prefetch_sweep")
    batch = 32
    v["bound:4"] = num(batch / 0.004)
    over = sw[sw.prefetch == 1]
    over = over[over.step_ms >= 2].eval("ms_per_batch - predicted_ms_per_batch")
    v["sweep:over:min"] = ratio(over.min())
    v["sweep:over:max"] = ratio(over.max())
    serial = sw[(sw.prefetch == 0) & (sw.step_ms >= 1)]
    serial = serial.eval("ms_per_batch - predicted_ms_per_batch")
    v["sweep:serial:over:min"] = ratio(serial.min())
    v["sweep:serial:over:max"] = ratio(serial.max())
    eq = read("csv_equivalence").iloc[0]
    v["eq:missing"] = num(eq.rows_with_missing_filled)
    v["eq:rows"] = num(eq.rows)
    n_batches = int(sw.batches.iloc[0])
    v["sweep:batches"] = num(n_batches)
    sh = read("shuffle_quality").set_index("label")
    for label, row in sh.iterrows():
        key = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
        v[f"shuf:{key}:rho"] = f"{row.spearman:.3f}".replace("-", "$-$")
        v[f"shuf:{key}:div"] = f"{row.batch_diversity:.3f}"
        if not pd.isna(row.predicted_spearman):
            v[f"shuf:{key}:pred"] = f"{row.predicted_spearman:.3f}"
    v["shuf:buf10000:frac"] = f"{100 * 10000 / eq.rows:.0f}"
    pred = sh.dropna(subset=["predicted_spearman"])
    v["shuf:npred"] = str(len(pred))
    v["shuf:maxdev"] = f"{(pred.spearman - pred.predicted_spearman).abs().max():.3f}"

    # Part C
    fs = read("feature_summary")
    for head, part in fs.groupby("head"):
        base = part[part.feature_set == "+ ocean one-hot"].iloc[0]
        for _, r in part.iterrows():
            key = re.sub(r"[^a-z0-9]+", "-", r.feature_set.lower()).strip("-")
            tag = f"c:{head}:{key}"
            v[f"{tag}:valid"] = num(r.valid_rmse)
            v[f"{tag}:test"] = num(r.test_rmse)
            v[f"{tag}:r2"] = f"{r.test_r2:.3f}"
            v[f"{tag}:params"] = num(r.params)
            v[f"{tag}:features"] = num(r.features)
            if not pd.isna(r.valid_diff_vs_baseline):
                v[f"{tag}:diff"] = num(abs(r.valid_diff_vs_baseline))
                v[f"{tag}:difflo"] = num(abs(r.valid_diff_lo))
                v[f"{tag}:diffhi"] = num(abs(r.valid_diff_hi))
                v[f"{tag}:pct"] = f"{100 * abs(r.valid_diff_vs_baseline) / base.valid_rmse:.1f}"
        v[f"c:{head}:base"] = num(base.valid_rmse)
        v[f"c:{head}:base:test"] = num(base.test_rmse)
        sig = part.significant_vs_baseline.fillna(False).astype(bool)
        better = part.valid_diff_vs_baseline < 0
        v[f"c:{head}:nbetter"] = str(int((sig & better).sum()))
        v[f"c:{head}:nworse"] = str(int((sig & ~better).sum()))
        v[f"c:{head}:ncompared"] = str(int(part.valid_diff_vs_baseline.notna().sum()))
        best = part.loc[part.valid_rmse.idxmin()]
        v[f"c:{head}:best:name"] = tex(best.feature_set)
        v[f"c:{head}:best"] = num(best.valid_rmse)
    lin_base = fs[(fs["head"] == "linear") & (fs.feature_set == "+ ocean one-hot")].valid_rmse.iloc[
        0
    ]
    mlp_base = fs[(fs["head"] == "mlp") & (fs.feature_set == "+ ocean one-hot")].valid_rmse.iloc[0]
    v["c:mlp:over:linear:pct"] = f"{100 * (1 - mlp_base / lin_base):.1f}"
    hc = read("hash_collisions").iloc[0]
    v["hash:shared:cells"] = num(hc.cells - hc.distinct_buckets_all_cells)
    v["hash:expected:shared"] = num(hc.cells - hc.expected_distinct_buckets_all_cells, 1)
    v["hash:expected"] = num(hc.expected_distinct_buckets_all_cells, 1)
    v["hash:rows:pct"] = f"{hc.training_rows_in_shared_buckets_pct:.2f}"

    # Part B
    fb = read("format_benchmark").set_index("format")
    speed = fb.examples_per_second
    one, many, many_gz = (
        speed["tfrecord, parse one record at a time"],
        speed["tfrecord, parse a batch at a time"],
        speed["tfrecord.gz, parse a batch at a time"],
    )
    v["fmt:batch:over:csv"] = ratio(many / speed["csv"], 1)
    v["fmt:batchgz:over:csv"] = ratio(many_gz / speed["csv"], 1)
    v["fmt:batch:over:single"] = ratio(many / one, 1)
    v["fmt:single:over:csv"] = ratio(one / speed["csv"], 2)
    v["fmt:max"] = num(speed.max())
    size = fb.size_vs_csv
    v["fmt:csvgz:size"] = ratio(size["csv.gz"])
    v["fmt:tfr:size"] = ratio(size["tfrecord, parse a batch at a time"])
    v["fmt:tfrgz:size"] = ratio(size["tfrecord.gz, parse a batch at a time"])
    v["fmt:tfrgz:over:csvgz"] = ratio(
        fb.bytes["tfrecord.gz, parse a batch at a time"] / fb.bytes["csv.gz"]
    )
    wi = read("wire_size").iloc[0]
    v["wire:min"], v["wire:max"] = num(wi.min_example_bytes), num(wi.max_example_bytes)
    co = read("corruption")
    broken = co[co.case != "intact"]
    v["corr:cases"] = str(len(broken))
    v["corr:detected"] = str(int(broken.error_detected.sum()))
    sc = read("scale_test")
    last = sc.iloc[-1]
    v["scale:growth:ratio"] = ratio(last.pandas_growth_mb / last.stream_growth_mb, 1)
    v["scale:speed:min"] = num(sc.stream_examples_per_s.min())
    v["scale:speed:max"] = num(sc.stream_examples_per_s.max())
    v["scale:rows:max"] = num(last.rows)
    v["scale:pass:max"] = num(last.rows / last.stream_examples_per_s, 1)
    v["scale:pandas:max"] = num(last.pandas_seconds, 1)
    v["scale:growth:stream:min"], v["scale:growth:stream:max"] = (
        num(sc.stream_growth_mb.min()),
        num(sc.stream_growth_mb.max()),
    )
    v["scale:growth:pandas:min"], v["scale:growth:pandas:max"] = (
        num(sc.pandas_growth_mb.min()),
        num(sc.pandas_growth_mb.max()),
    )
    v["scale:csv:max"] = num(last.csv_mb, 1)

    # run-to-run noise: the same CSV pipeline timed in Part A (row 6, defaults) and in Part B (csv)
    noise_a = float(
        ab[(ab.regime == "default") & (ab.step_ms == 0) & (ab.k == 6)].examples_per_second.iloc[0]
    )
    noise_b = float(read("format_benchmark").set_index("format").examples_per_second["csv"])
    v["noise:a"], v["noise:b"] = num(noise_a), num(noise_b)
    v["noise:pct"] = f"{100 * abs(noise_b / noise_a - 1):.0f}"
    # prefetch sweep: with prefetch, steps 0 and 1 ms
    pf = sw[sw.prefetch == 1].set_index("step_ms")
    v["sweep:pf:0"], v["sweep:pf:1"] = num(pf.ms_per_batch[0.0], 2), num(pf.ms_per_batch[1.0], 2)
    v["sweep:pred:0"] = num(pf.predicted_ms_per_batch[0.0], 2)
    # scale: growth factors of memory while the data grow 100 times
    v["scale:ratio:stream"] = ratio(sc.stream_growth_mb.iloc[-1] / sc.stream_growth_mb.iloc[0], 1)
    v["scale:ratio:pandas"] = ratio(sc.pandas_growth_mb.iloc[-1] / sc.pandas_growth_mb.iloc[0], 0)
    # hashed grid against the exact grid and the embedding of the exact cells (linear head)
    lin = fs[fs["head"] == "linear"].set_index("feature_set").valid_rmse
    hashed = round(
        lin["+ hashed grid (1000)"]
    )  # differences of the rounded values that the text prints
    v["hash:lin:worse"] = num(hashed - round(lin["+ grid one-hot (400)"]))
    v["hash:lin:worse:emb"] = num(hashed - round(lin["+ grid embedding (400)"]))

    # Part C: paired differences between two feature sets of one model over the same seeds
    # (the method of stats.paired_diff_ci); the text prints the difference of the rounded means
    # of the tables and the interval of the paired differences.
    fr = read("feature_runs")

    def runs(head: str, name: str) -> pd.Series:
        part = fr[(fr["head"] == head) & (fr.feature_set == name)]
        return part.set_index("seed").valid_rmse.sort_index()

    def pair(tag: str, head: str, a: str, b: str) -> tuple[float, float, float]:
        """``b`` minus ``a``: mean, low and high of the 95% interval."""
        m, lo, hi = t_interval(runs(head, b) - runs(head, a))
        v[f"{tag}:lo"], v[f"{tag}:hi"] = usd(lo), usd(hi)
        return m, lo, hi

    mlp = fs[fs["head"] == "mlp"].set_index("feature_set").valid_rmse
    onehot, emb = "+ grid one-hot (400)", "+ grid embedding (400)"
    hashed_set = "+ hashed grid (1000)"
    pair("pair:mlp:gridemb", "mlp", onehot, emb)
    v["pair:mlp:gridemb:diff"] = num(abs(round(mlp[emb]) - round(mlp[onehot])))
    pair("pair:mlp:hash", "mlp", onehot, hashed_set)
    v["pair:mlp:hash:diff"] = num(abs(round(mlp[hashed_set]) - round(mlp[onehot])))
    pair("pair:lin:hash", "linear", onehot, hashed_set)
    pair("pair:lin:hashemb", "linear", emb, hashed_set)
    deployed = json.loads(METRICS.read_text(encoding="utf-8"))["parts"]["E"]["params"]
    deployed = deployed["feature_set"]
    pair("pair:mlp:num", "mlp", deployed, "numeric only")
    v["pair:mlp:num:diff"] = num(round(mlp["numeric only"]) - round(mlp[deployed]))
    others = [s for s in mlp.index if s != "numeric only"]
    v["c:mlp:vsnum:n"] = str(len(others))
    v["c:mlp:vsnum:nbetter"] = str(
        sum(t_interval(runs("mlp", s) - runs("mlp", "numeric only"))[2] < 0 for s in others)
    )
    v["c:mlp:valid:min"], v["c:mlp:valid:max"] = num(mlp.min()), num(mlp.max())

    # Part E
    pe = json.loads(METRICS.read_text(encoding="utf-8"))["parts"]["E"]["metrics"]
    v["e:gain:linear:pct"] = f"{100 * (1 - pe['test_rmse'] / pe['baseline_linear_test_rmse']):.1f}"
    v["e:gain:mean:pct"] = f"{100 * (1 - pe['test_rmse'] / pe['baseline_mean_test_rmse']):.1f}"
    # 95% interval of the test RMSE as in Chapter 2 of the book: a t-interval for the mean of the
    # squared errors of the test rows, then the square root of both ends
    pr = read("test_predictions")
    err = pr.prediction - pr.median_house_value
    sq = err**2
    _, lo, hi = t_interval(sq)
    v["e:rmse"] = num(np.sqrt(sq.mean()))
    v["e:rmse:lo"], v["e:rmse:hi"] = num(np.sqrt(lo)), num(np.sqrt(hi))
    v["e:ntest"] = num(len(pr))
    # the test rows at the largest value of the target (the cap that Chapter 2 points out)
    top = pr.median_house_value.max()
    capped = pr.median_house_value == top
    v["cap:value"] = num(top)
    v["cap:n"], v["cap:pct"] = num(capped.sum()), f"{100 * capped.mean():.1f}"
    v["cap:rest:n"] = num((~capped).sum())
    v["cap:rmse"] = num(np.sqrt(sq[capped].mean()))
    v["cap:rest:rmse"] = num(np.sqrt(sq[~capped].mean()))
    v["cap:mean:err"] = usd(err[capped].mean())

    # Part D
    mf = read("mnist_formats")
    pixels = 28 * 28

    def pick(e: str, c: str, r: str) -> pd.Series:
        return mf[(mf.encoding == e) & (mf.compression == c) & (mf.reading == r)].iloc[0]

    one_s, one_p = "one record at a time, serial", "one record at a time, parallel"
    bat_s, bat_p = "a batch at a time, serial", "a batch at a time, parallel"
    raw = pick("raw", "none", one_s).bytes_per_image
    v["mnist:pixels"] = num(pixels)
    v["mnist:raw"] = num(raw)
    v["mnist:raw:overhead"] = num(raw - pixels)
    v["mnist:raw:gz"] = num(pick("raw", "GZIP", one_s).bytes_per_image)
    v["mnist:raw:gz:ratio"] = ratio(raw / pick("raw", "GZIP", one_s).bytes_per_image, 1)
    v["mnist:tensor:gz:ratio"] = ratio(
        pick("tensor", "none", one_s).bytes_per_image
        / pick("tensor", "GZIP", one_s).bytes_per_image,
        1,
    )
    v["mnist:png"] = num(pick("png", "none", one_s).bytes_per_image)
    v["mnist:png:gz"] = num(pick("png", "GZIP", one_s).bytes_per_image)
    v["mnist:png:over:raw"] = ratio(pick("png", "none", one_s).bytes_per_image / raw)
    v["mnist:tensor"] = num(pick("tensor", "none", one_s).bytes_per_image)
    v["mnist:tensor:extra"] = num(pick("tensor", "none", one_s).bytes_per_image - raw)
    v["mnist:tensor:gz"] = num(pick("tensor", "GZIP", one_s).bytes_per_image)
    single = mf[mf.reading == one_s].examples_per_second
    v["mnist:single:min"], v["mnist:single:max"] = num(single.min()), num(single.max())
    v["mnist:single:spread"] = f"{100 * (single.max() / single.min() - 1):.0f}"
    for comp, tag in (("none", "plain"), ("GZIP", "gz")):
        v[f"mnist:batch:{tag}"] = num(pick("raw", comp, bat_s).examples_per_second)
        v[f"mnist:batch:over:single:{tag}"] = ratio(
            pick("raw", comp, bat_s).examples_per_second
            / pick("raw", comp, one_s).examples_per_second,
            1,
        )
    ser = mf[mf.reading.isin([one_s, bat_s])].reset_index(drop=True)
    par = mf[mf.reading.isin([one_p, bat_p])].reset_index(drop=True)
    pairs = []
    for _, r in par.iterrows():
        twin = r.reading.replace("parallel", "serial")
        base = ser[
            (ser.encoding == r.encoding)
            & (ser.compression == r.compression)
            & (ser.reading == twin)
        ].iloc[0]
        pairs.append(r.examples_per_second / base.examples_per_second)
    v["mnist:par:over:ser:min"], v["mnist:par:over:ser:max"] = ratio(min(pairs)), ratio(max(pairs))
    v["mnist:par:n"] = str(len(pairs))
    v["mnist:par:slower"] = str(sum(p < 1 for p in pairs))
    fits = read("mnist_fits")
    a = fits[fits.input == "tfds.load"].sort_values("seed").test_accuracy.to_numpy()
    b = fits[fits.input != "tfds.load"].sort_values("seed").test_accuracy.to_numpy()
    diff = a - b  # tfds.load minus TFRecord, same seed
    m = float(diff.mean())
    lo, hi = st.t.interval(0.95, len(diff) - 1, loc=m, scale=st.sem(diff))
    for tag, val in (("mean", m), ("lo", lo), ("hi", hi)):
        v[f"mnist:acc:diff:{tag}"] = f"{val:.4f}".replace("-", "$-$")
    v["mnist:acc:tfds"], v["mnist:acc:tfrecord"] = f"{a.mean():.4f}", f"{b.mean():.4f}"
    v["mnist:acc:min"] = f"{min(a.min(), b.min()):.4f}"
    v["mnist:acc:max"] = f"{max(a.max(), b.max()):.4f}"
    v["mnist:acc:n"] = str(len(a))
    return v


# --------------------------------------------------------------------------- macros
def fmt_value(v: object) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return fmt_value(int(v))
    if isinstance(v, float):
        a = abs(v)
        if a == 0:
            return "0"
        if a >= 100:
            s = f"{v:,.0f}"
        elif a >= 1:
            s = f"{v:.2f}"
        elif a >= 1e-3:
            s = f"{v:.4f}"
        else:
            return sci(v)
        return s.replace("-", "$-$") if s.startswith("-") else s
    return tex(v)


def numbers() -> None:
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    lines = [
        "% Generated by make_tables.py. Do not edit.",
        "\\newcommand{\\metric}[2]{\\csname m:#1:#2\\endcsname}",
    ]
    for key, part in payload["parts"].items():
        for name, value in part["metrics"].items():
            lines.append(
                f"\\expandafter\\newcommand\\csname m:{key}:{name}\\endcsname{{{fmt_value(value)}}}"
            )
        for name, value in part["params"].items():
            lines.append(
                f"\\expandafter\\newcommand\\csname p:{key}:{name}\\endcsname{{{fmt_value(value)}}}"
            )
    lines.append("\\newcommand{\\val}[1]{\\csname v:#1\\endcsname}")
    for name, value in derived().items():
        lines.append(f"\\expandafter\\newcommand\\csname v:{name}\\endcsname{{{value}}}")
    run = payload["run"]
    lines.append("\\newcommand{\\param}[2]{\\csname p:#1:#2\\endcsname}")
    lines.append(f"\\newcommand{{\\runSeed}}{{{tex(run.get('seed', 42))}}}")
    lines.append(f"\\newcommand{{\\runCommit}}{{{tex(run.get('git_commit', 'n/a'))}}}")
    lines.append(f"\\newcommand{{\\runDataSha}}{{{tex(str(run.get('data_sha256', ''))[:12])}}}")
    sizes = run.get("split_sizes", {})
    for name, value in sizes.items():
        macro = "runSplit" + "".join(p.capitalize() for p in re.split(r"[^a-zA-Z]+", name) if p)
        lines.append(f"\\newcommand{{\\{macro}}}{{{fmt_value(int(value))}}}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global TABLES, METRICS, OUT
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, default=TABLES)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    TABLES, METRICS, OUT = args.tables, args.metrics, args.out
    for fn in (
        tab_csv_speed,
        tab_prefetch,
        tab_shuffle,
        tab_formats,
        tab_wire,
        tab_corruption,
        tab_scale,
        tab_features,
        tab_hash,
        tab_mnist_formats,
        tab_mnist_fits,
        tab_error_groups,
        tab_gates,
        numbers,
    ):
        fn()
    print(f"wrote {len(list(OUT.glob('*.tex')))} files to {OUT}")


if __name__ == "__main__":
    main()
