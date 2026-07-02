#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "plotly>=6.1",
#   "kaleido>=1.0",
# ]
# ///
"""Render the README batch-size throughput curve from a batch-curve benchmark record.

Reads the JSON that ``scripts/modal_esme_batch_curve.py --command curve`` writes and
exports one static SVG card into ``assets/``:

- fig-esme-batch-curve.svg: llm_infer tok/s vs concurrent requests, against the flat
  naive HF-sequential floor. The record's mature-engine rows are deliberately not
  plotted; the public story measures distance from the naive baseline only.

The committed input lives at ``assets/esme-batch-curve.json`` (a curated copy of the
bench-results record), so the figure regenerates from repo state alone:

    uv run scripts/plot_benchmark_curve.py
    uv run scripts/plot_benchmark_curve.py --input bench-results/esme-batch-curve-curve-<stamp>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import plotly.graph_objects as go

# Style contract: match esme-pretrain scripts/plot_run_telemetry.py (the shared card spec).
FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
TITLE_COLOR = "#1f2937"
SUBTITLE_COLOR = "#6b7280"
AXIS_COLOR = "#cfd5df"
GRID_COLOR = "#eef1f6"
BORDER_COLOR = "#d9dde7"
TICK_COLOR = "#6b7280"
BLUE = "#636efa"
GREEN = "#00cc96"
RED = "#ef553b"
CARD_WIDTH = 920
CARD_HEIGHT = 560


def load_rows(path: Path) -> dict[str, list[dict]]:
    """Plotted rows per system, sorted by batch size; a diverged (ungated) row is a hard error.

    Only ``llm_infer`` and ``hf_sequential`` are plotted. The record's vLLM rows are
    deliberately not loaded — mature-engine numbers stay out of the public story
    (docs/benchmark.md states that framing once).
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    by_system: dict[str, list[dict]] = {"llm_infer": [], "hf_sequential": []}
    for row in record["rows"]:
        system = row["system"]
        if system not in by_system:
            continue
        if row["tokens_per_second"] is None:
            raise ValueError(
                f"{path}: {system} batch={row['batch_size']} reports no tok/s "
                "(reference gate failed); the figure only plots gated rows"
            )
        by_system[system].append(row)
    for system, rows in by_system.items():
        if not rows:
            raise ValueError(f"{path}: no rows for {system}")
        rows.sort(key=lambda r: r["batch_size"])
    return by_system


def rounded_border_path(radius_px: float) -> str:
    """Rounded-rect border in paper coordinates (plotly paths have no arc command)."""
    rx = radius_px / CARD_WIDTH
    ry = radius_px / CARD_HEIGHT
    x0, x1 = 0.5 / CARD_WIDTH, 1 - 0.5 / CARD_WIDTH
    y0, y1 = 0.5 / CARD_HEIGHT, 1 - 0.5 / CARD_HEIGHT
    return (
        f"M {x0 + rx},{y0} L {x1 - rx},{y0} Q {x1},{y0} {x1},{y0 + ry} "
        f"L {x1},{y1 - ry} Q {x1},{y1} {x1 - rx},{y1} "
        f"L {x0 + rx},{y1} Q {x0},{y1} {x0},{y1 - ry} "
        f"L {x0},{y0 + ry} Q {x0},{y0} {x0 + rx},{y0} Z"
    )


def card_layout(title: str, subtitle: str, conclusion: str) -> go.Layout:
    return go.Layout(
        width=CARD_WIDTH,
        height=CARD_HEIGHT,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": FONT_FAMILY, "size": 12, "color": TICK_COLOR},
        margin={"l": 84, "r": 84, "t": 118, "b": 96},
        showlegend=False,
        shapes=[
            {
                "type": "path",
                "path": rounded_border_path(radius_px=8),
                "xref": "paper",
                "yref": "paper",
                "line": {"color": BORDER_COLOR, "width": 1},
                "layer": "above",
            }
        ],
        annotations=[
            {
                "text": f"<b>{title}</b>",
                "xref": "paper",
                "yref": "paper",
                "x": -0.045,
                "y": 1.24,
                "xanchor": "left",
                "showarrow": False,
                "font": {"family": FONT_FAMILY, "size": 22, "color": TITLE_COLOR},
            },
            {
                "text": subtitle,
                "xref": "paper",
                "yref": "paper",
                "x": -0.045,
                "y": 1.135,
                "xanchor": "left",
                "showarrow": False,
                "font": {"family": FONT_FAMILY, "size": 14, "color": SUBTITLE_COLOR},
            },
            {
                "text": conclusion,
                "xref": "paper",
                "yref": "paper",
                "x": -0.045,
                "y": -0.185,
                "xanchor": "left",
                "showarrow": False,
                "font": {"family": FONT_FAMILY, "size": 12, "color": SUBTITLE_COLOR},
            },
        ],
    )


def styled_axis(**overrides: object) -> dict[str, object]:
    axis: dict[str, object] = {
        "showgrid": True,
        "gridcolor": GRID_COLOR,
        "gridwidth": 1,
        "zeroline": False,
        "showline": True,
        "linecolor": AXIS_COLOR,
        "linewidth": 1,
        "ticks": "outside",
        "tickcolor": AXIS_COLOR,
        "tickfont": {"family": FONT_FAMILY, "size": 12, "color": TICK_COLOR},
        "title": {"font": {"family": FONT_FAMILY, "size": 13, "color": "#374151"}},
    }
    axis.update(overrides)
    return axis


def _series(rows: list[dict]) -> tuple[list[int], list[float]]:
    return [r["batch_size"] for r in rows], [r["tokens_per_second"] for r in rows]


def build_curve_figure(by_system: dict[str, list[dict]]) -> go.Figure:
    """The engine curve against the naive-HF floor — the repo's claim is distance from
    the naive baseline, so nothing else is drawn."""
    llm_x, llm_y = _series(by_system["llm_infer"])
    hf_x, hf_y = _series(by_system["hf_sequential"])
    hf_floor = hf_y[0]  # the anchor row (full warmup+3-iteration protocol)
    gain_vs_floor = llm_y[-1] / hf_floor

    figure = go.Figure(
        layout=card_layout(
            title="Esme-214M-Chat: serving throughput vs concurrency",
            subtitle=(
                "Total tok/s vs concurrent chat requests - A100-80GB, up to 256 new tokens, "
                "greedy, every point oracle-gated"
            ),
            conclusion=(
                "Conclusion: one shared paged-KV engine turns concurrency into throughput -"
                f" {llm_y[0]:,.0f} tok/s at {llm_x[0]} requests to {llm_y[-1]:,.0f} at"
                f" {llm_x[-1]}, {gain_vs_floor:.0f}x the flat naive-HF floor."
            ),
        )
    )

    # HF floor: dashed line through its measured points — one at every batch level, so
    # the flat line is measurement, not extrapolation.
    figure.add_trace(
        go.Scatter(
            x=hf_x,
            y=hf_y,
            mode="lines+markers",
            name="naive HF sequential",
            line={"color": RED, "width": 1.4, "dash": "dash"},
            marker={"size": 7, "color": RED},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=llm_x,
            y=llm_y,
            mode="lines+markers",
            name="llm_infer",
            line={"color": BLUE, "width": 2.6},
            marker={"size": 8, "color": BLUE},
        )
    )

    y_max = max(llm_y) * 1.12
    figure.update_layout(
        xaxis=styled_axis(
            title={"text": "concurrent requests"},
            type="log",
            tickvals=llm_x,
            ticktext=[str(v) for v in llm_x],
            range=[math.log10(llm_x[0] / 1.25), math.log10(llm_x[-1] * 1.25)],
        ),
        yaxis=styled_axis(
            title={"text": "output tok/s"},
            range=[0, y_max],
        ),
    )

    # Annotations: x is log10 (log axis), y is raw (linear axis).
    llm_mid = min(2, len(llm_x) - 1)
    figure.add_annotation(
        {
            "text": "llm_infer (this repo)",
            "x": math.log10(llm_x[llm_mid]),
            "y": llm_y[llm_mid] + y_max * 0.13,
            "showarrow": False,
            "xanchor": "center",
            "font": {"family": FONT_FAMILY, "size": 13, "color": BLUE},
        }
    )
    figure.add_annotation(
        {
            "text": (
                f"naive HF sequential (floor) - {min(hf_y):.0f}-{max(hf_y):.0f} tok/s"
                " across all levels"
            ),
            "x": math.log10(hf_x[len(hf_x) // 2]),
            "y": hf_floor + y_max * 0.05,
            "showarrow": False,
            "xanchor": "left",
            "xshift": 14,
            "font": {"family": FONT_FAMILY, "size": 13, "color": RED},
        }
    )
    figure.add_annotation(
        {
            "text": f"<b>{gain_vs_floor:.0f}x the floor</b>",
            "x": math.log10(llm_x[-1]),
            "y": llm_y[-1] - y_max * 0.075,
            "showarrow": False,
            "xanchor": "right",
            "xshift": -10,
            "font": {"family": FONT_FAMILY, "size": 14, "color": TITLE_COLOR},
        }
    )
    return figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the README batch-curve SVG.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("assets/esme-batch-curve.json"),
        help="Batch-curve benchmark record (modal_esme_batch_curve.py output).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    args = parser.parse_args(argv)

    try:
        by_system = load_rows(args.input)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "fig-esme-batch-curve.svg"
    build_curve_figure(by_system).write_image(
        output, format="svg", width=CARD_WIDTH, height=CARD_HEIGHT
    )
    print(f"batch_curve: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
