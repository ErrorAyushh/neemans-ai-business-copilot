"""
charts.py

Production-ready Plotly visualizations for Neeman's AI Business Copilot.

Architecture:
    analytics/metrics.py
            ↓
        app.py / agent
            ↓
        charts.py
            ↓
        Plotly figures

This module is visualization-only.

IMPORTANT:
- No business calculations are performed here.
- No raw dataframe is accepted.
- No KPI is recalculated.
- All numbers must already come from analytics/metrics.py.
- Missing/invalid data returns None instead of breaking the UI.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================================
# THEME
# ============================================================================

PRIMARY = "#4F46E5"
PRIMARY_LIGHT = "#818CF8"

POSITIVE = "#16A34A"
NEGATIVE = "#DC2626"
WARNING = "#D97706"
NEUTRAL = "#94A3B8"

TEXT = "#0F172A"
MUTED_TEXT = "#64748B"
GRID = "#E2E8F0"
BACKGROUND = "#FFFFFF"


_BASE_LAYOUT = {
    "paper_bgcolor": BACKGROUND,
    "plot_bgcolor": BACKGROUND,
    "font": {
        "family": "Inter, Arial, sans-serif",
        "size": 12,
        "color": TEXT,
    },
    "margin": {
        "l": 20,
        "r": 20,
        "t": 55,
        "b": 25,
    },
}


# ============================================================================
# HELPERS
# ============================================================================

def _empty(value: Any) -> bool:
    """Return True when a value is missing or empty."""

    if value is None:
        return True

    if isinstance(value, (list, tuple, dict, pd.DataFrame)):
        return len(value) == 0

    return False


def _records_to_df(records: list[dict] | None) -> pd.DataFrame | None:
    """Safely convert metric records into a dataframe."""

    if not records:
        return None

    try:
        df = pd.DataFrame(records)
    except Exception:
        return None

    return None if df.empty else df


def _style(
    fig: go.Figure,
    *,
    height: int = 340,
    show_legend: bool = False,
) -> go.Figure:
    """Apply the shared dashboard visual style."""

    fig.update_layout(
        **_BASE_LAYOUT,
        height=height,
        showlegend=show_legend,
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=GRID,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
    )

    return fig


def _change_color(value: float | None) -> str:
    """Return a semantic color for a percentage change."""

    if value is None:
        return NEUTRAL

    if value < 0:
        return NEGATIVE

    if value > 0:
        return POSITIVE

    return NEUTRAL


def _format_currency(value: float | None) -> str:
    """Format INR values for chart labels."""

    if value is None:
        return "—"

    value = float(value)

    if abs(value) >= 10_000_000:
        return f"₹{value / 10_000_000:.2f} Cr"

    if abs(value) >= 100_000:
        return f"₹{value / 100_000:.2f} L"

    return f"₹{value:,.0f}"


# ============================================================================
# REVENUE
# ============================================================================

def revenue_comparison_chart(
    previous_value: float | None,
    current_value: float | None,
    previous_label: str,
    current_label: str,
) -> go.Figure | None:
    """
    Compare revenue between two periods.

    Inputs are already-calculated revenue values from metrics.py.
    """

    if previous_value is None or current_value is None:
        return None

    df = pd.DataFrame(
        {
            "Period": [previous_label, current_label],
            "Revenue": [previous_value, current_value],
        }
    )

    fig = px.bar(
        df,
        x="Period",
        y="Revenue",
        text="Revenue",
        category_orders={
            "Period": [previous_label, current_label]
        },
    )

    fig.update_traces(
        marker_color=[NEUTRAL, PRIMARY],
        texttemplate=[
            _format_currency(previous_value),
            _format_currency(current_value),
        ],
        textposition="outside",
        hovertemplate="%{x}<br>Revenue: ₹%{y:,.0f}<extra></extra>",
    )

    fig.update_layout(
        title="Revenue Comparison",
        xaxis_title=None,
        yaxis_title="Revenue (₹)",
    )

    return _style(fig, height=320)


# ============================================================================
# GENERIC CHANGE CHART
# ============================================================================

def change_pct_chart(
    records: list[dict] | None,
    label_key: str,
    change_key: str,
    title: str,
) -> go.Figure | None:
    """
    Display percentage changes for categories, channels, or
    marketing channels.

    The values are taken directly from metrics.py.
    """

    df = _records_to_df(records)

    if df is None:
        return None

    if label_key not in df.columns or change_key not in df.columns:
        return None

    df = df.dropna(subset=[change_key]).copy()

    if df.empty:
        return None

    df[change_key] = pd.to_numeric(
        df[change_key],
        errors="coerce",
    )

    df = df.dropna(subset=[change_key])

    if df.empty:
        return None

    df = df.sort_values(change_key)

    fig = px.bar(
        df,
        x=change_key,
        y=label_key,
        orientation="h",
        text=change_key,
        title=title,
    )

    fig.update_traces(
        marker_color=[
            _change_color(value)
            for value in df[change_key]
        ],
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            "%{y}<br>"
            "Change: %{x:.2f}%"
            "<extra></extra>"
        ),
    )

    fig.add_vline(
        x=0,
        line_width=1,
        line_color=GRID,
    )

    fig.update_xaxes(
        title="Change (%)",
        zeroline=False,
    )

    fig.update_yaxes(
        title=None,
        autorange="reversed",
    )

    return _style(
        fig,
        height=max(280, 48 * len(df)),
    )


# ============================================================================
# CATEGORY
# ============================================================================

def category_revenue_change_chart(
    categories: list[dict] | None,
) -> go.Figure | None:
    """Revenue change by product category."""

    return change_pct_chart(
        categories,
        label_key="category",
        change_key="revenue_change_pct",
        title="Category Revenue Change",
    )


# ============================================================================
# CHANNEL
# ============================================================================

def channel_revenue_change_chart(
    channels: list[dict] | None,
) -> go.Figure | None:
    """Revenue change by sales channel."""

    return change_pct_chart(
        channels,
        label_key="channel",
        change_key="revenue_change_pct",
        title="Channel Revenue Change",
    )


def revenue_share_chart(
    channels: list[dict] | None,
) -> go.Figure | None:
    """Current-period revenue share by sales channel."""

    df = _records_to_df(channels)

    if df is None:
        return None

    required = {
        "channel",
        "current_revenue_share_pct",
    }

    if not required.issubset(df.columns):
        return None

    df = df.dropna(
        subset=["current_revenue_share_pct"]
    )

    if df.empty:
        return None

    fig = px.pie(
        df,
        names="channel",
        values="current_revenue_share_pct",
        hole=0.58,
    )

    fig.update_traces(
        textinfo="label+percent",
        textposition="outside",
        hovertemplate=(
            "%{label}<br>"
            "Revenue share: %{value:.1f}%"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        title="Revenue Share by Channel",
    )

    return _style(
        fig,
        height=360,
        show_legend=False,
    )


# ============================================================================
# INVENTORY
# ============================================================================

def inventory_kpi_comparison_chart(
    current: dict | None,
    previous: dict | None,
) -> go.Figure | None:
    """
    Compare the three core inventory KPIs.

    Uses values already calculated by metrics.py:
        average_availability_pct
        average_closing_stock
        stockout_days
    """

    if not current or not previous:
        return None

    specifications = [
        (
            "average_availability_pct",
            "Availability",
            "%",
        ),
        (
            "average_closing_stock",
            "Avg Closing Stock",
            "",
        ),
        (
            "stockout_days",
            "Stockout Days",
            "",
        ),
    ]

    available = [
        item
        for item in specifications
        if current.get(item[0]) is not None
        and previous.get(item[0]) is not None
    ]

    if not available:
        return None

    frames = []

    for key, label, suffix in available:
        frames.append(
            pd.DataFrame(
                {
                    "Metric": label,
                    "Period": [
                        "Previous",
                        "Current",
                    ],
                    "Value": [
                        previous[key],
                        current[key],
                    ],
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)

    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        color="Period",
        barmode="group",
        text="Value",
        color_discrete_map={
            "Previous": NEUTRAL,
            "Current": PRIMARY,
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        hovertemplate=(
            "%{x}<br>"
            "%{fullData.name}: %{y:.2f}"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        title="Inventory Performance",
        xaxis_title=None,
        yaxis_title=None,
        legend_title=None,
    )

    return _style(
        fig,
        height=350,
        show_legend=True,
    )


def availability_by_category_chart(
    availability_by_category: dict | None,
) -> go.Figure | None:
    """Display inventory availability by category."""

    if not availability_by_category:
        return None

    df = pd.DataFrame(
        [
            {
                "Category": category,
                "Availability": value,
            }
            for category, value
            in availability_by_category.items()
        ]
    )

    if df.empty:
        return None

    df["Availability"] = pd.to_numeric(
        df["Availability"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Availability"]
    ).sort_values("Availability")

    if df.empty:
        return None

    fig = px.bar(
        df,
        x="Availability",
        y="Category",
        orientation="h",
        text="Availability",
        title="Inventory Availability by Category",
    )

    fig.update_traces(
        marker_color=[
            NEGATIVE if value < 80 else POSITIVE
            for value in df["Availability"]
        ],
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            "%{y}<br>"
            "Availability: %{x:.1f}%"
            "<extra></extra>"
        ),
    )

    fig.update_xaxes(
        title="Availability (%)",
        range=[0, 105],
    )

    fig.update_yaxes(title=None)

    return _style(
        fig,
        height=max(280, 48 * len(df)),
    )


# ============================================================================
# MARKETING
# ============================================================================

def roas_chart(
    channels: list[dict] | None,
) -> go.Figure | None:
    """Display current ROAS by marketing channel."""

    df = _records_to_df(channels)

    if df is None:
        return None

    required = {
        "channel",
        "current_roas",
    }

    if not required.issubset(df.columns):
        return None

    df["current_roas"] = pd.to_numeric(
        df["current_roas"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["current_roas"]
    ).sort_values("current_roas")

    if df.empty:
        return None

    fig = px.bar(
        df,
        x="current_roas",
        y="channel",
        orientation="h",
        text="current_roas",
        title="Current ROAS by Channel",
    )

    fig.update_traces(
        marker_color=[
            NEGATIVE if value < 1 else PRIMARY
            for value in df["current_roas"]
        ],
        texttemplate="%{text:.2f}x",
        textposition="outside",
        hovertemplate=(
            "%{y}<br>"
            "ROAS: %{x:.2f}x"
            "<extra></extra>"
        ),
    )

    fig.add_vline(
        x=1,
        line_dash="dash",
        line_color=WARNING,
        annotation_text="1.0x",
        annotation_position="top",
    )

    fig.update_xaxes(
        title="ROAS",
        rangemode="tozero",
    )

    fig.update_yaxes(title=None)

    return _style(
        fig,
        height=max(280, 48 * len(df)),
    )


def marketing_roas_change_chart(
    channels: list[dict] | None,
) -> go.Figure | None:
    """Display ROAS percentage change by marketing channel."""

    return change_pct_chart(
        channels,
        label_key="channel",
        change_key="roas_change_pct",
        title="ROAS Change by Marketing Channel",
    )


def marketing_revenue_change_chart(
    channels: list[dict] | None,
) -> go.Figure | None:
    """Display attributed revenue change by marketing channel."""

    return change_pct_chart(
        channels,
        label_key="channel",
        change_key="attributed_revenue_change_pct",
        title="Attributed Revenue Change",
    )


# ============================================================================
# EXECUTIVE OVERVIEW
# ============================================================================

def executive_kpi_change_chart(
    kpis: dict | None,
) -> go.Figure | None:
    """
    Compare percentage changes across the four executive KPIs.

    Expected structure:

        {
            "revenue": {
                "current": ...,
                "previous": ...,
                "pct_change": ...
            },
            ...
        }
    """

    if not kpis:
        return None

    labels = {
        "revenue": "Revenue",
        "orders": "Orders",
        "units": "Units Sold",
        "average_order_value": "Average Order Value",
    }

    rows = []

    for key, label in labels.items():
        item = kpis.get(key)

        if not item:
            continue

        change = item.get("pct_change")

        if change is None:
            continue

        rows.append(
            {
                "KPI": label,
                "Change": change,
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("Change")

    fig = px.bar(
        df,
        x="Change",
        y="KPI",
        orientation="h",
        text="Change",
        title="Executive KPI Movement",
    )

    fig.update_traces(
        marker_color=[
            _change_color(value)
            for value in df["Change"]
        ],
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate=(
            "%{y}<br>"
            "Change: %{x:.2f}%"
            "<extra></extra>"
        ),
    )

    fig.add_vline(
        x=0,
        line_color=GRID,
        line_width=1,
    )

    fig.update_xaxes(
        title="Change (%)"
    )

    fig.update_yaxes(
        title=None,
        autorange="reversed",
    )

    return _style(
        fig,
        height=max(280, 55 * len(df)),
    )


# ============================================================================
# GENERAL PURPOSE COMPARISON
# ============================================================================

def metric_comparison_chart(
    previous_value: float | None,
    current_value: float | None,
    previous_label: str,
    current_label: str,
    title: str,
    y_axis_title: str | None = None,
) -> go.Figure | None:
    """
    Generic two-period comparison chart.

    Useful for inventory, sales, or any already-calculated KPI.
    """

    if previous_value is None or current_value is None:
        return None

    df = pd.DataFrame(
        {
            "Period": [
                previous_label,
                current_label,
            ],
            "Value": [
                previous_value,
                current_value,
            ],
        }
    )

    fig = px.bar(
        df,
        x="Period",
        y="Value",
        text="Value",
        category_orders={
            "Period": [
                previous_label,
                current_label,
            ]
        },
    )

    fig.update_traces(
        marker_color=[
            NEUTRAL,
            PRIMARY,
        ],
        texttemplate="%{text:,.2f}",
        textposition="outside",
    )

    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title=y_axis_title,
    )

    return _style(
        fig,
        height=300,
    )


# ============================================================================
# EMPTY STATE HELPER
# ============================================================================

def no_data_message(
    title: str = "No data available",
    message: str = "There is not enough data to display this chart.",
) -> go.Figure:
    """
    Return a clean Plotly empty-state figure.

    This allows app.py to render a consistent UI even when
    a metric is unavailable.
    """

    fig = go.Figure()

    fig.add_annotation(
        text=(
            f"<b>{title}</b><br>"
            f"<span style='color:{MUTED_TEXT}'>{message}</span>"
        ),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        align="center",
        font={
            "size": 14,
            "color": TEXT,
        },
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return _style(
        fig,
        height=240,
        show_legend=False,
    )


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "revenue_comparison_chart",
    "change_pct_chart",
    "category_revenue_change_chart",
    "channel_revenue_change_chart",
    "revenue_share_chart",
    "inventory_kpi_comparison_chart",
    "availability_by_category_chart",
    "roas_chart",
    "marketing_roas_change_chart",
    "marketing_revenue_change_chart",
    "executive_kpi_change_chart",
    "metric_comparison_chart",
    "no_data_message",
]