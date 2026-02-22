"""
Reusable Plotly chart builders for the BIST Quant Research Cockpit.

All figures use `plotly.graph_objects` and follow the design system defined
in `app.ui` for consistent theming across the app.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from app.ui import (
    ACCENT,
    BG_SURFACE,
    BORDER_DEFAULT,
    DANGER,
    FONT_MONO,
    FONT_SANS,
    PLOTLY_TEMPLATE,
    REGIME_COLORS,
    SUCCESS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
    apply_chart_style,
    base_fig,
)

# ── colour constants (re-exported for backward compat) ────────────────────────
# Use REGIME_COLORS from ui.py as the single source of truth.


def _regime_band_color(regime: str) -> str:
    return REGIME_COLORS.get(regime.lower(), REGIME_COLORS["unknown"])


# ── public chart builders ─────────────────────────────────────────────────────


def equity_curve(
    dates: list[Any],
    values: list[float],
    benchmark_dates: list[Any] | None = None,
    benchmark_values: list[float] | None = None,
    title: str = "Equity Curve",
    regime_bands: list[dict[str, Any]] | None = None,
) -> go.Figure:
    """
    Line chart with optional benchmark overlay and coloured regime background bands.

    Args:
        dates: X-axis dates for the strategy equity curve.
        values: Normalised equity values (e.g. starting at 100).
        benchmark_dates: Optional X-axis dates for the benchmark series.
        benchmark_values: Optional benchmark equity values.
        title: Chart title.
        regime_bands: Optional list of dicts ``{start, end, regime}`` used to
            shade the background by market regime.

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    fig = base_fig(title)

    # Regime background bands
    if regime_bands:
        for band in regime_bands:
            color = _regime_band_color(band.get("regime", "unknown"))
            fig.add_vrect(
                x0=band["start"],
                x1=band["end"],
                fillcolor=color,
                opacity=0.12,
                layer="below",
                line_width=0,
            )

    # Benchmark
    if benchmark_dates is not None and benchmark_values is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark_dates,
                y=benchmark_values,
                mode="lines",
                name="Benchmark (XU100)",
                line=dict(color=TEXT_MUTED, width=1.5, dash="dot"),
            )
        )

    # Strategy
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            name="Strategy",
            line=dict(color=ACCENT, width=2),
            fill=None,
        )
    )

    return fig


def drawdown_chart(
    dates: list[Any],
    drawdowns: list[float],
    title: str = "Drawdown",
) -> go.Figure:
    """
    Filled area chart for drawdown, filled red below zero.

    Args:
        dates: X-axis dates.
        drawdowns: Drawdown values as negative percentages (e.g. -15.4).
        title: Chart title.

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    fig = base_fig(title)

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdowns,
            mode="lines",
            name="Drawdown",
            line=dict(color=DANGER, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,49,49,0.2)",
        )
    )

    # Horizontal zero line
    fig.add_hline(y=0, line_color=BORDER_DEFAULT, line_width=1)

    fig.update_yaxes(ticksuffix="%")
    return fig


def monthly_returns_heatmap(
    monthly_returns: dict[int | str, dict[int | str, float]],
) -> go.Figure:
    """
    Calendar heatmap: months as columns, years as rows.
    Green = positive, red = negative.

    Args:
        monthly_returns: Nested dict ``{year: {month_int: return_pct}}``.
            Month integers are 1-based.  Values are percentages (e.g. 3.5
            means +3.5 %).

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    years = sorted(monthly_returns.keys())
    z: list[list[float | None]] = []
    text: list[list[str]] = []

    for year in years:
        row: list[float | None] = []
        text_row: list[str] = []
        for m in range(1, 13):
            val = monthly_returns[year].get(m) or monthly_returns[year].get(str(m))
            row.append(val)
            text_row.append(f"{val:+.1f}%" if val is not None else "")
        z.append(row)
        text.append(text_row)

    fig = base_fig("Monthly Returns (%)")

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=month_labels,
            y=[str(y) for y in years],
            text=text,
            texttemplate="%{text}",
            colorscale=[
                [0.0, DANGER],
                [0.5, BG_SURFACE],
                [1.0, SUCCESS],
            ],
            zmid=0,
            showscale=True,
            colorbar=dict(title="%", ticksuffix="%"),
        )
    )

    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def bar_metrics(
    metrics: dict[str, float],
) -> go.Figure:
    """
    Horizontal bar chart for comparing metric values.

    Args:
        metrics: ``{label: value}`` pairs.

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    labels = list(metrics.keys())
    values = list(metrics.values())
    colors = [SUCCESS if v >= 0 else DANGER for v in values]

    fig = base_fig("Metrics")

    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
        )
    )

    fig.update_yaxes(showgrid=False)
    return fig


def regime_timeline(
    dates: list[Any],
    regimes: list[str],
) -> go.Figure:
    """
    Step chart showing market regime over time, colour-coded per regime.

    Args:
        dates: X-axis dates (same length as *regimes*).
        regimes: Regime label strings per date.

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    fig = base_fig("Regime Timeline")

    # Encode regimes as numeric for the step line
    regime_order = ["Bull", "Recovery", "Bear", "Stress"]
    regime_to_int = {r: i for i, r in enumerate(regime_order)}

    y_values = [regime_to_int.get(r, -1) for r in regimes]

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y_values,
            mode="lines",
            line=dict(shape="hv", width=2, color=ACCENT),
            name="Regime",
            showlegend=False,
        )
    )

    # Colour-coded background bands per unique contiguous regime segment
    current = None
    start = None
    for i, (d, r) in enumerate(zip(dates, regimes)):
        if r != current:
            if current is not None and start is not None:
                fig.add_vrect(
                    x0=start,
                    x1=d,
                    fillcolor=_regime_band_color(current),
                    opacity=0.18,
                    layer="below",
                    line_width=0,
                    annotation_text=current if i < 10 else "",
                    annotation_position="top left",
                )
            current = r
            start = d
    # Close last band
    if current is not None and start is not None and dates:
        fig.add_vrect(
            x0=start,
            x1=dates[-1],
            fillcolor=_regime_band_color(current),
            opacity=0.18,
            layer="below",
            line_width=0,
        )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(len(regime_order))),
        ticktext=regime_order,
        showgrid=False,
    )
    return fig


__all__ = [
    "equity_curve",
    "drawdown_chart",
    "monthly_returns_heatmap",
    "bar_metrics",
    "regime_timeline",
    "REGIME_COLORS",
]
