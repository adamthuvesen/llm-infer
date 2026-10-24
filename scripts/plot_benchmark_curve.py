#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "plotly>=6.1",
#   "kaleido>=1.0",
# ]
# ///
"""Render the README benchmark figures from committed benchmark records.

Exports two static SVG cards into ``assets/``:

- fig-esme-batch-curve.svg: llm_infer tok/s vs concurrent requests against the flat naive
  HF-sequential floor, from ``scripts/modal_esme_batch_curve.py --command curve``. It shows
  batch scaling; the headline comparisons are HF batched generate() and vLLM.
- fig-esme-ablation-ladder.svg: what each engine technique adds over HF batched generate(),
  one panel per batch size, from ``scripts/modal_esme_batch_curve.py --command ladder``.

The committed inputs live at ``assets/esme-batch-curve.json`` and
``assets/esme-ablation-ladder.json`` (curated copies of bench-results records), so the
figures regenerate from repo state alone:

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
from plotly.subplots import make_subplots

# Style contract: match esme-pretrain scripts/plot_run_telemetry.py (the shared card spec).
FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
TITLE_COLOR = "#1f2937"
SUBTITLE_COLOR = "#6b7280"
AXIS_COLOR = "#cfd5df"
GRID_COLOR = "#eef1f6"
BORDER_COLOR = "#d9dde7"
TICK_COLOR = "#6b7280"
BLUE = "#2563eb"
NEUTRAL = "#64748b"
CARD_WIDTH = 920
CARD_HEIGHT = 560

# Ladder rows top to bottom: the HF batched baseline, then each cumulative engine rung.
LADDER_SYSTEMS = (
    ("hf_batched", "HF batched generate()"),
    ("paged_continuous", "paged KV + continuous batching"),
    ("+batched_prefill", "+ batched prefill"),
    ("+decode_window", "+ decode window"),
    ("+piecewise_graphs", "+ piecewise CUDA graphs"),
    ("+grouped_graphs", "+ grouped graphs"),
)


def _record_rows(record: dict[str, object], path: Path) -> list[dict]:
    """Read the policy-v2 benchmark rows."""
    rows = record.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected a rows list")
    return rows


def _require_headline_row(path: Path, row: dict) -> None:
    if row.get("policy_version") != 2 or row.get("headline_eligible") is not True:
        raise ValueError(
            f"{path}: {row['system']} batch={row['batch_size']} is not a headline-eligible "
            "policy-v2 row"
        )


def load_rows(path: Path) -> dict[str, list[dict]]:
    """Plotted rows per system, sorted by batch size; a diverged (ungated) row is a hard error.

    Only ``llm_infer`` and ``hf_sequential`` are plotted. The record's vLLM rows are
    deliberately not loaded — mature-engine numbers stay out of the public story
    (docs/benchmark.md states that framing once).
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    by_system: dict[str, list[dict]] = {"llm_infer": [], "hf_sequential": []}
    for row in _record_rows(record, path):
        system = row["system"]
        if system not in by_system:
            continue
        _require_headline_row(path, row)
        if row.get("tokens_per_second") is None:
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
        margin={"l": 84, "r": 84, "t": 118, "b": 118},
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
                "x": 0.5,
                "y": -0.245,
                "xanchor": "center",
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
    """The engine curve against the naive-HF floor: batch-scaling evidence, not the headline."""
    llm_x, llm_y = _series(by_system["llm_infer"])
    hf_x, hf_y = _series(by_system["hf_sequential"])
    hf_by_batch = dict(zip(hf_x, hf_y, strict=True))
    final_hf = hf_by_batch[llm_x[-1]]
    gain_vs_same_row_hf = llm_y[-1] / final_hf

    figure = go.Figure(
        layout=card_layout(
            title="Esme-214M-Chat: batch-sweep serving throughput",
            subtitle=(
                "Same-run A100-80GB sweep - total output tok/s by concurrent chat requests, "
                "up to 256 new tokens"
            ),
            conclusion=(
                f"{llm_y[0]:,.0f} -> {llm_y[-1]:,.0f} tok/s from {llm_x[0]} to"
                f" {llm_x[-1]} requests; {gain_vs_same_row_hf:.1f}x same-row naive HF."
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
            line={"color": NEUTRAL, "width": 1.4, "dash": "dash"},
            marker={"size": 7, "color": "#ffffff", "line": {"color": NEUTRAL, "width": 1.5}},
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
            "text": "llm_infer",
            "x": math.log10(llm_x[llm_mid]),
            "y": llm_y[llm_mid] + y_max * 0.13,
            "showarrow": False,
            "xanchor": "center",
            "font": {"family": FONT_FAMILY, "size": 13, "color": BLUE},
        }
    )
    figure.add_annotation(
        {
            "text": (f"naive HF sequential floor: {min(hf_y):.0f}-{max(hf_y):.0f} tok/s"),
            "x": math.log10(hf_x[len(hf_x) // 2]),
            "y": final_hf + y_max * 0.05,
            "showarrow": False,
            "xanchor": "left",
            "xshift": 14,
            "font": {"family": FONT_FAMILY, "size": 13, "color": NEUTRAL},
        }
    )
    figure.add_annotation(
        {
            "text": f"<b>{gain_vs_same_row_hf:.1f}x vs same-row HF</b>",
            "x": math.log10(llm_x[-1]),
            "y": llm_y[-1] + y_max * 0.02,
            "showarrow": False,
            "xanchor": "right",
            "xshift": -22,
            "font": {"family": FONT_FAMILY, "size": 14, "color": TITLE_COLOR},
        }
    )
    return figure


def load_ladder_rows(path: Path) -> dict[int, dict[str, float]]:
    """Raw tok/s per batch and ladder system; any ungated row is a hard error."""
    record = json.loads(path.read_text(encoding="utf-8"))
    wanted = {system for system, _ in LADDER_SYSTEMS}
    by_batch: dict[int, dict[str, float]] = {}
    for row in _record_rows(record, path):
        system = row["system"]
        if system not in wanted:
            continue
        _require_headline_row(path, row)
        by_batch.setdefault(int(row["batch_size"]), {})[system] = float(row["tokens_per_second"])
    if not by_batch:
        raise ValueError(f"{path}: no ladder rows")
    for batch, systems in by_batch.items():
        missing = wanted - set(systems)
        if missing:
            raise ValueError(f"{path}: batch {batch} is missing {sorted(missing)}")
    return dict(sorted(by_batch.items()))


def build_ladder_figure(by_batch: dict[int, dict[str, float]]) -> go.Figure:
    """One horizontal-bar panel per batch size; each panel keeps its own tok/s scale."""
    batches = list(by_batch)
    labels = [label for _, label in LADDER_SYSTEMS]
    gains = [values["+grouped_graphs"] / values["hf_batched"] for values in by_batch.values()]
    figure = make_subplots(rows=1, cols=len(batches), shared_yaxes=True, horizontal_spacing=0.08)
    figure.update_layout(
        card_layout(
            title="Esme-214M-Chat: what each engine technique adds",
            subtitle=(
                "One A100-80GB container - output tok/s on chat requests, real EOS, "
                "every row reference-gated"
            ),
            conclusion=(
                "Full engine vs HF batched generate(): "
                + " / ".join(f"{gain:.1f}x" for gain in gains)
                + f" at {' / '.join(str(b) for b in batches)} requests."
                " CUDA graphs are the largest single step."
            ),
        )
    )
    margin = {"l": 206, "r": 48, "t": 118, "b": 118}
    figure.update_layout(margin=margin, bargap=0.28)
    # The wide label margin would push the card's title and subtitle right; pin them to
    # the same 36 px left inset the batch-curve card's title sits at.
    title_x = (36 - margin["l"]) / (CARD_WIDTH - margin["l"] - margin["r"])
    for annotation in figure.layout.annotations[:2]:
        annotation.x = title_x
    for column, (batch, values) in enumerate(by_batch.items(), start=1):
        tok_s = [values[system] for system, _ in LADDER_SYSTEMS]
        text = [
            f"{value:,.0f}" if system in ("hf_batched", "+grouped_graphs") else ""
            for (system, _), value in zip(LADDER_SYSTEMS, tok_s, strict=True)
        ]
        figure.add_trace(
            go.Bar(
                x=tok_s,
                y=labels,
                orientation="h",
                marker={
                    "color": [NEUTRAL] + [BLUE] * (len(LADDER_SYSTEMS) - 1),
                    "cornerradius": 4,
                },
                text=text,
                textposition="outside",
                textfont={"family": FONT_FAMILY, "size": 12, "color": TITLE_COLOR},
                cliponaxis=False,
                hovertemplate="%{y}: %{x:,.0f} tok/s<extra></extra>",
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(
            styled_axis(
                title={"text": f"{batch} requests"},
                range=[0, max(tok_s) * 1.3],
                tickformat="~s",
                nticks=4,
            ),
            row=1,
            col=column,
        )
    figure.update_yaxes(styled_axis(showgrid=False, autorange="reversed", ticks=""))
    return figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the README benchmark SVGs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("assets/esme-batch-curve.json"),
        help="Batch-curve benchmark record (modal_esme_batch_curve.py output).",
    )
    parser.add_argument(
        "--ladder-input",
        type=Path,
        default=Path("assets/esme-ablation-ladder.json"),
        help="Ablation-ladder benchmark record (modal_esme_batch_curve.py --command ladder).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    args = parser.parse_args(argv)

    try:
        by_system = load_rows(args.input)
        ladder = load_ladder_rows(args.ladder_input)
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "fig-esme-batch-curve.svg"
    build_curve_figure(by_system).write_image(
        output, format="svg", width=CARD_WIDTH, height=CARD_HEIGHT
    )
    print(f"batch_curve: {output}")
    ladder_output = args.output_dir / "fig-esme-ablation-ladder.svg"
    build_ladder_figure(ladder).write_image(
        ladder_output, format="svg", width=CARD_WIDTH, height=CARD_HEIGHT
    )
    print(f"ablation_ladder: {ladder_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
