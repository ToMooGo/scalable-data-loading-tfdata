"""Draw the two diagrams used in the README and the report.

    python docs/make_architecture.py

writes ``docs/images/architecture.png`` (the services and flows) and ``docs/images/pipeline.png``
(the tf.data stages). The drawing code checks its own result and raises an error instead of saving
a picture in which

* a text comes within 0.12 inch of the edge of its box,
* a label touches a box, another label or the border of a dashed group,
* a text would print smaller than 7 pt when the picture is scaled to the 16 cm text width of the
  report (``PRINT_WIDTH``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK = "#1a1a1a"
MUTED = "#3a3a3a"
NAVY, NAVY_SOFT = "#1f4e79", "#dce8f5"
ORANGE, ORANGE_SOFT = "#c9731b", "#fbeedd"
GREEN, GREEN_SOFT = "#3b7d4f", "#e0efe5"
PURPLE, PURPLE_SOFT = "#5b3f8c", "#ebe4f5"
GRAY, GRAY_SOFT = "#6f6f6f", "#f2f2f2"
OUT = Path(__file__).resolve().parent / "images"
PRINT_WIDTH = 6.3  # inches: 16 cm, the text width of the report
MIN_PRINT_PT = 7.0
PAD = 0.15  # inch of white around the saved picture


class Canvas:
    def __init__(self, width: float, height: float):
        self.width = width
        self.fig, self.ax = plt.subplots(figsize=(width, height))
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.boxes: list[tuple[FancyBboxPatch, list]] = []
        self.groups: list[FancyBboxPatch] = []
        self.labels: list = []
        self.texts: list = []

    # -- shapes ---------------------------------------------------------------------------------
    def box(
        self,
        x,
        y,
        w,
        h,
        title,
        lines=(),
        edge=NAVY,
        fill=NAVY_SOFT,
        size=12.5,
        title_size=14,
        lw=1.5,
    ):
        """A rounded box. The title sits at a fixed distance from the top edge, so the titles of
        neighbouring boxes line up whatever the number of lines under them."""
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1", lw=lw, ec=edge, fc=fill
        )
        self.ax.add_patch(patch)
        title_h = title_size / 72 * 1.3
        step = size / 72 * 1.45
        top = y + h - 0.17
        texts = [
            self.ax.text(
                x + w / 2,
                top,
                title,
                fontsize=title_size,
                fontweight="bold",
                color=INK,
                ha="center",
                va="top",
            )
        ]
        for i, line in enumerate(lines):
            texts.append(
                self.ax.text(
                    x + w / 2,
                    top - title_h - 0.08 - step * i,
                    line,
                    fontsize=size,
                    color=INK,
                    ha="center",
                    va="top",
                )
            )
        self.boxes.append((patch, texts))
        self.texts += texts

    def group(self, x, y, w, h, label, size=12.5):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.15",
            lw=1.2,
            ec=GRAY,
            fc="none",
            ls="--",
        )
        self.ax.add_patch(patch)
        self.groups.append(patch)
        self.texts.append(
            self.ax.text(
                x + w - 0.2, y + h - 0.12, label, fontsize=size, color=MUTED, ha="right", va="top"
            )
        )

    def label(self, text, at, ha="center", va="center", size=12.5, pad=0.08):
        """Text next to an arrow, on a white patch so a line never runs through it."""
        t = self.ax.text(
            *at,
            text,
            fontsize=size,
            color=INK,
            ha=ha,
            va=va,
            linespacing=1.15,
            bbox={"boxstyle": f"round,pad={pad}", "fc": "white", "ec": "none"},
        )
        self.labels.append(t)
        self.texts.append(t)

    def text(self, x, y, s, size=12.5, color=INK, **kw):
        t = self.ax.text(x, y, s, fontsize=size, color=color, **kw)
        self.texts.append(t)
        return t

    def arrow(self, p, q, both=False, lw=1.5):
        style = "<|-|>" if both else "-|>"
        self.ax.add_patch(
            FancyArrowPatch(
                p, q, arrowstyle=style, mutation_scale=15, lw=lw, color=INK, shrinkA=1, shrinkB=1
            )
        )

    def elbow(self, points, lw=1.5):
        """A right-angled connector through ``points`` with an arrow head on the last segment."""
        xs, ys = zip(*points[:-1], strict=True)
        self.ax.plot(xs, ys, color=INK, lw=lw, solid_capstyle="butt")
        self.ax.add_patch(
            FancyArrowPatch(
                points[-2],
                points[-1],
                arrowstyle="-|>",
                mutation_scale=15,
                lw=lw,
                color=INK,
                shrinkA=0,
                shrinkB=1,
            )
        )

    # -- checks and output ----------------------------------------------------------------------
    def _extent(self, artist, renderer, inv):
        return artist.get_window_extent(renderer).transformed(inv).extents

    def check_and_save(self, path: Path) -> None:
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()
        problems = []
        for patch, texts in self.boxes:
            bx0, by0, bx1, by1 = patch.get_extents().transformed(inv).extents
            for t in texts:
                x0, y0, x1, y1 = self._extent(t, renderer, inv)
                if x0 < bx0 + 0.12 or x1 > bx1 - 0.12 or y0 < by0 + 0.08 or y1 > by1 - 0.08:
                    problems.append(f"{t.get_text()!r} too close to the edge of its box")
        boxes = [p.get_extents().transformed(inv).extents for p, _ in self.boxes]
        extents = [(t, self._extent(t, renderer, inv)) for t in self.labels]
        for i, (label, (lx0, ly0, lx1, ly1)) in enumerate(extents):
            for bx0, by0, bx1, by1 in boxes:
                if lx0 < bx1 + 0.04 and lx1 > bx0 - 0.04 and ly0 < by1 + 0.04 and ly1 > by0 - 0.04:
                    problems.append(f"label {label.get_text()!r} touches a box")
            for other, (ox0, oy0, ox1, oy1) in extents[i + 1 :]:
                if lx0 < ox1 + 0.04 and lx1 > ox0 - 0.04 and ly0 < oy1 + 0.04 and ly1 > oy0 - 0.04:
                    problems.append(f"labels {label.get_text()!r} and {other.get_text()!r} touch")
            for g in self.groups:
                gx0, gy0, gx1, gy1 = g.get_extents().transformed(inv).extents
                crosses_x = any(lx0 - 0.04 < gx < lx1 + 0.04 for gx in (gx0, gx1))
                crosses_y = any(ly0 - 0.04 < gy < ly1 + 0.04 for gy in (gy0, gy1))
                if (crosses_x and gy0 < ly1 and ly0 < gy1) or (
                    crosses_y and gx0 < lx1 and lx0 < gx1
                ):
                    problems.append(f"label {label.get_text()!r} sits on a group border")
        saved_width = self.fig.get_tightbbox(renderer).width + 2 * PAD  # what savefig writes
        scale = PRINT_WIDTH / saved_width
        for t in self.texts:
            if t.get_fontsize() * scale < MIN_PRINT_PT:
                problems.append(
                    f"{t.get_text()!r}: {t.get_fontsize() * scale:.1f} pt at the report's width"
                )
        if problems:
            raise ValueError(f"{path.name}:\n  " + "\n  ".join(problems))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=PAD, facecolor="white")
        plt.close(self.fig)


def architecture() -> None:
    """Three columns: the flows (left), MLflow (middle), the service (right)."""
    c = Canvas(10.0, 9.9)
    W, GAP = 2.8, 0.4
    x1, x2, x3 = 0.4, 0.4 + W + GAP, 0.4 + 2 * (W + GAP)  # the three columns
    FLOW_Y, FLOW_H = 5.85, 1.55

    c.box(
        x1,
        8.4,
        W,
        1.25,
        "Data",
        ["housing.csv, SHA-256 pinned", "MNIST through tfds.load"],
        edge=GRAY,
        fill=GRAY_SOFT,
    )
    c.group(0.15, 5.6, 9.7, 2.3, "Flows in the pipeline container")
    flows = [
        (
            x1,
            "Train flow",
            ["splits and shards data,", "runs parts A to E,", "registers the model"],
        ),
        (
            x2,
            "Evaluate flow",
            ["reloads the version,", "applies the quality gates,", "champion or challenger"],
        ),
        (
            x3,
            "Deploy flow",
            ["moves the champion alias,", "calls /reload,", "rolls back on failure"],
        ),
    ]
    for x, title, lines in flows:
        c.box(x, FLOW_Y, W, FLOW_H, title, lines, size=12.5)
    mid = FLOW_Y + FLOW_H / 2
    c.arrow((x1 + W, mid), (x2, mid))
    c.arrow((x2 + W, mid), (x3, mid))
    c.arrow((x1 + W / 2, 8.4), (x1 + W / 2, FLOW_Y + FLOW_H))

    # MLflow under the first two flows, the service under the third
    ML_Y, ML_H, ML_W = 3.55, 1.3, 4.85
    c.box(
        x1,
        ML_Y,
        ML_W,
        ML_H,
        "MLflow",
        ["tracking server and model registry", "runs, figures, versions, alias champion"],
        size=12.5,
    )
    c.arrow((x1 + W / 2, FLOW_Y), (x1 + W / 2, ML_Y + ML_H))
    c.label("logs,\nregisters", (x1 + W / 2 + 0.12, 5.2), ha="left")
    c.arrow((x2 + 1.0, FLOW_Y), (x2 + 1.0, ML_Y + ML_H))
    c.label("reads and\nlogs gates", (x2 + 0.88, 5.2), ha="right")
    c.arrow((x3 + 0.5, FLOW_Y), (x1 + ML_W - 0.5, ML_Y + ML_H))
    c.label("sets the\nalias", (6.2, 5.0), ha="center")

    API_Y, API_H = 2.75, 1.95
    c.box(
        x3,
        API_Y,
        W,
        API_H,
        "FastAPI service",
        [
            "predict, health, ready",
            "and reload endpoints;",
            "serves the champion",
            "model and swaps it",
            "without a restart",
        ],
        edge=ORANGE,
        fill=ORANGE_SOFT,
        size=12.5,
    )
    c.arrow((x3 + 1.4, FLOW_Y), (x3 + 1.4, API_Y + API_H))
    c.label("POST /reload", (x3 + 1.55, 5.2), ha="left")
    c.arrow((x3, ML_Y + ML_H / 2 + 0.1), (x1 + ML_W, ML_Y + ML_H / 2 + 0.1))
    c.label("loads\nchampion", ((x1 + ML_W + x3) / 2, ML_Y + ML_H / 2 - 0.22), size=12.5)

    # bottom row
    ROW_Y, ROW_H = 0.35, 1.3
    c.box(
        x1,
        ROW_Y,
        W,
        ROW_H,
        "Monitor flow",
        ["scheduled drift check", "of logged predictions"],
        edge=GREEN,
        fill=GREEN_SOFT,
        size=12.5,
    )
    c.arrow((x1 + W / 2, ROW_Y + ROW_H), (x1 + W / 2, ML_Y))
    c.label("reads the\ntraining profile", (x1 + W / 2 + 0.12, 2.6), ha="left")

    pg_x, pg_w = x2 + 0.5, 2.3
    c.box(
        pg_x,
        ROW_Y,
        pg_w,
        ROW_H,
        "PostgreSQL 16",
        ["predictions table,", "MLflow backend"],
        edge=GRAY,
        fill=GRAY_SOFT,
        size=12.5,
    )
    c.arrow((pg_x + 0.5, ML_Y), (pg_x + 0.5, ROW_Y + ROW_H))
    c.label("backend\nstore", (pg_x + 0.62, 2.6), ha="left")
    c.elbow([(x3 + 0.45, API_Y), (x3 + 0.45, ROW_Y + 0.65), (pg_x + pg_w, ROW_Y + 0.65)])
    c.label("stores\npredictions", (x3 + 0.33, 2.15), ha="right")
    # the monitor reads the logged predictions: along the bottom edge, under the database
    bottom = 0.12
    c.elbow(
        [
            (x1 + 0.5, ROW_Y),
            (x1 + 0.5, bottom),
            (pg_x + 1.65, bottom),
            (pg_x + 1.65, ROW_Y),
        ]
    )
    c.label("reads the logged predictions", (x1 + 1.9, bottom), ha="left", va="center", pad=0.5)

    c.box(
        x3 + 0.6,
        ROW_Y,
        W - 0.6,
        ROW_H,
        "Browser",
        ["web page of the", "service"],
        edge=GRAY,
        fill=GRAY_SOFT,
        size=12.5,
    )
    c.arrow((x3 + 0.6 + (W - 0.6) / 2, ROW_Y + ROW_H), (x3 + 0.6 + (W - 0.6) / 2, API_Y))
    c.label("requests", (x3 + 0.6 + (W - 0.6) / 2 + 0.12, 2.15), ha="left")

    c.check_and_save(OUT / "architecture.png")


def pipeline() -> None:
    """Two lanes side by side; the same row is the same stage, so the swap of two steps shows."""
    lane_w, h, gap = 4.4, 1.2, 0.4
    c = Canvas(9.6, 10.0)
    xa, xb = 0.15, 0.15 + lane_w + 0.5
    top = 9.3
    swap = ORANGE
    lanes = [
        (
            xa,
            "Part A: sharded CSV files",
            NAVY,
            NAVY_SOFT,
            [
                ("read", ["list_files, interleave of 5", "files, header skipped"], None),
                ("shuffle", ["buffer of 10,000 lines"], None),
                ("map", ["decode_csv and standardise,", "5 parallel threads"], swap),
                ("batch", ["32 to 128 examples"], swap),
                ("prefetch", ["prepares the next batch", "while the model trains"], None),
                ("model.fit", ["8 numeric features per row"], None),
            ],
        ),
        (
            xb,
            "Part B: GZIP-compressed TFRecord files",
            PURPLE,
            PURPLE_SOFT,
            [
                ("read", ["list_files, interleave of 5", "GZIP TFRecord files"], None),
                ("shuffle", ["buffer of 10,000 records"], None),
                ("batch", ["32 to 128 examples"], swap),
                ("map", ["parse_example parses the", "whole batch in one call"], swap),
                ("prefetch", ["prepares the next batch", "while the model trains"], None),
                ("model.fit", ["layers inside the model", "prepare the features"], None),
            ],
        ),
    ]
    for x, title, edge, fill, stages in lanes:
        c.text(x, top + 0.45, title, size=14.5, fontweight="bold", ha="left", va="center")
        for i, (name, lines, accent) in enumerate(stages):
            y = top - (i + 1) * h - i * gap
            last = i == len(stages) - 1
            c.box(
                x,
                y,
                lane_w,
                h,
                name,
                lines,
                edge=accent or (GREEN if last else edge),
                fill=GREEN_SOFT if last else fill,
                lw=3.0 if accent else 1.5,
            )
            if not last:
                c.arrow((x + lane_w / 2, y), (x + lane_w / 2, y - gap))
    c.check_and_save(OUT / "pipeline.png")


if __name__ == "__main__":
    architecture()
    pipeline()
    print("wrote", OUT / "architecture.png", "and", OUT / "pipeline.png")
