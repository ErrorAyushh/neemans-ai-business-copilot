"""
Neeman's AI Business Copilot
----------------------------

Streamlit frontend. Owns UI, state, user interaction, orchestration and
presentation only — every business number displayed here comes from
analytics/metrics.py (via the RCA agent's tool trace on the RCA page, or
directly on the dashboard pages), and every chart comes from charts.py.

Architecture:

    CSV -> analytics/metrics.py -> agent/tools.py -> RCA agent (LangGraph
    + Sarvam AI) -> app.py -> charts.py / Plotly -> Streamlit UI

app.py does NOT recalculate revenue, percentage changes, ROAS, inventory
changes, or evidence strength, and does NOT compute month/period
boundaries itself — those all come from metrics.resolve_named_period().

VISUAL IDENTITY
----------------
The product's entire value proposition is that nothing is invented —
every number is deterministic, and every claim is checked against
evidence before it reaches the user. The UI is built around that idea:
a restrained "ledger" aesthetic (paper surface, hairline rules, a single
verified/caution/alert color system) with an ink-stamp motif for
verification and confidence states, instead of decorative icons or
emoji. Structure carries meaning here (numbered navigation reflects the
actual overview -> drill-down -> investigation workflow) rather than
decorating it.
"""

from __future__ import annotations

import re
from datetime import date

import pandas as pd
import streamlit as st

import charts
from analytics import metrics


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Neeman's AI Business Copilot",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# STYLING
# =============================================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --ink: #1B1912;
        --ink-soft: #57523F;
        --paper: #ECE8DC;
        --paper-raised: #FAF8F1;
        --line: #D2CBB4;
        --line-strong: #A99F80;
        --verified: #2F6F4E;
        --verified-soft: #E1EADD;
        --caution: #A0741C;
        --caution-soft: #F1E6C8;
        --alert: #AE4327;
        --alert-soft: #F3E0D6;
        --font-display: 'Fraunces', Georgia, serif;
        --font-body: 'IBM Plex Sans', -apple-system, sans-serif;
        --font-mono: 'IBM Plex Mono', 'Courier New', monospace;
    }

    html, body, [class*="css"] {
        font-family: var(--font-body);
        color: var(--ink);
    }

    .stApp {
        background: var(--paper);
    }

    .block-container {
        max-width: 1360px;
        padding-top: 1.4rem;
        padding-bottom: 3.5rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--line);
        background: var(--paper-raised);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.1rem;
    }

    /* ---- Masthead ---------------------------------------------------- */

    .masthead {
        border-bottom: 1px solid var(--line-strong);
        padding-bottom: 14px;
        margin-bottom: 1.6rem;
    }

    .masthead-eyebrow {
        font-family: var(--font-mono);
        font-size: .72rem;
        font-weight: 500;
        letter-spacing: .18em;
        text-transform: uppercase;
        color: var(--ink-soft);
        margin-bottom: 4px;
    }

    .main-title {
        font-family: var(--font-display);
        font-size: 2.4rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: var(--ink);
        line-height: 1.1;
    }

    .masthead-rule {
        height: 3px;
        margin: 12px 0 10px;
        background: repeating-linear-gradient(
            90deg, var(--ink) 0px, var(--ink) 22px, transparent 22px, transparent 26px
        );
        opacity: .22;
    }

    .masthead-meta {
        font-family: var(--font-mono);
        font-size: .78rem;
        color: var(--ink-soft);
    }

    .masthead-dot { margin: 0 8px; opacity: .5; }

    /* ---- Generic section chrome --------------------------------------- */

    .panel-eyebrow {
        font-family: var(--font-mono);
        font-size: .68rem;
        font-weight: 500;
        letter-spacing: .14em;
        text-transform: uppercase;
        color: var(--ink-soft);
        margin: .2rem 0 .5rem;
    }

    .section-title {
        font-family: var(--font-display);
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--ink);
        margin-top: .6rem;
        margin-bottom: .15rem;
    }

    .muted {
        color: var(--ink-soft);
        font-size: .89rem;
    }

    /* ---- Sidebar brand block ------------------------------------------ */

    .brand-card {
        padding: 16px 15px;
        border: 1px solid var(--line-strong);
        border-radius: 4px;
        background: var(--paper-raised);
        margin-bottom: 10px;
    }

    .brand-wordmark {
        font-family: var(--font-mono);
        font-size: .72rem;
        font-weight: 600;
        letter-spacing: .16em;
        color: var(--ink);
    }

    .brand-name {
        font-family: var(--font-display);
        font-weight: 600;
        font-size: 1.08rem;
        color: var(--ink);
        line-height: 1.25;
        margin-top: 2px;
    }

    .brand-sub {
        margin-top: 6px;
        color: var(--ink-soft);
        font-size: .78rem;
        line-height: 1.5;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        margin-top: 11px;
        padding: 4px 10px;
        border-radius: 3px;
        border: 1px solid var(--line);
        color: var(--ink-soft);
        font-family: var(--font-mono);
        font-size: .7rem;
        letter-spacing: .04em;
        text-transform: uppercase;
    }

    .status-pill::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--verified);
        display: inline-block;
    }

    /* ---- Signal / evidence cards --------------------------------------- */

    .signal {
        padding: 15px 17px;
        border: 1px solid var(--line);
        border-left: 3px solid var(--line-strong);
        border-radius: 3px;
        margin-bottom: 11px;
        background: var(--paper-raised);
    }

    .signal-title {
        font-weight: 620;
        margin-bottom: 5px;
        font-size: .93rem;
        color: var(--ink);
        display: flex;
        align-items: center;
    }

    .signal-body {
        color: var(--ink-soft);
        line-height: 1.55;
        font-size: .89rem;
    }

    .tick {
        display: inline-block;
        width: 0;
        height: 0;
        margin-right: 9px;
        flex-shrink: 0;
    }

    .tick-up {
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-bottom: 8px solid var(--verified);
    }

    .tick-down {
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 8px solid var(--alert);
    }

    .kpi-caption {
        color: var(--ink-soft);
        font-family: var(--font-mono);
        font-size: .78rem;
        margin-top: -.55rem;
    }

    /* ---- RCA result cards ------------------------------------------------ */

    .rca-card {
        padding: 16px 18px;
        border-radius: 4px;
        border: 1px solid var(--line);
        background: var(--paper-raised);
        min-height: 104px;
    }

    .rca-card-label {
        font-family: var(--font-mono);
        color: var(--ink-soft);
        font-size: .7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: .1em;
    }

    .rca-card-value {
        font-family: var(--font-mono);
        font-variant-numeric: tabular-nums;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 7px;
        color: var(--ink);
        line-height: 1.2;
    }

    .rca-card-sub {
        color: var(--ink-soft);
        font-size: .8rem;
        margin-top: 5px;
    }

    /* ---- Stamp components (confidence + grounding) ---------------------- */

    .confidence-badge {
        display: inline-flex;
        align-items: center;
        padding: 5px 12px;
        border-radius: 2px;
        font-family: var(--font-mono);
        font-weight: 600;
        font-size: .74rem;
        letter-spacing: .08em;
        text-transform: uppercase;
        border: 1.5px solid currentColor;
        transform: rotate(-1deg);
    }

    .confidence-high    { color: var(--verified); background: var(--verified-soft); }
    .confidence-medium  { color: var(--caution);  background: var(--caution-soft); }
    .confidence-low     { color: var(--alert);    background: var(--alert-soft); }
    .confidence-unknown { color: var(--ink-soft); background: var(--paper); }

    .verify-stamp {
        display: inline-block;
        padding: 7px 14px;
        border: 1.5px solid var(--verified);
        border-radius: 2px;
        color: var(--verified);
        background: var(--verified-soft);
        font-family: var(--font-mono);
        font-weight: 600;
        font-size: .76rem;
        letter-spacing: .1em;
        text-transform: uppercase;
        transform: rotate(-1deg);
        margin: 4px 0 8px;
    }

    .rca-question-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 16px;
        margin-bottom: 8px;
    }

    .rca-question-title {
        font-family: var(--font-display);
        font-size: 1.08rem;
        font-weight: 600;
        color: var(--ink);
    }

    .period-chip {
        display: inline-block;
        padding: 5px 10px;
        border: 1px solid var(--line-strong);
        border-radius: 2px;
        background: var(--paper);
        color: var(--ink-soft);
        font-family: var(--font-mono);
        font-size: .72rem;
        font-weight: 500;
        white-space: nowrap;
    }

    /* ---- Nav / inputs ----------------------------------------------------- */

    div[data-testid="stRadio"] > div[role="radiogroup"] label {
        border: 1px solid var(--line);
        border-radius: 3px;
        padding: 7px 10px;
        margin-bottom: 6px;
        background: var(--paper-raised);
    }

    div[data-testid="stRadio"] label p {
        font-family: var(--font-mono);
        font-size: .84rem;
        letter-spacing: .01em;
    }

    div[data-testid="stButton"] > button {
        border-radius: 3px;
        border: 1px solid var(--line-strong);
        font-family: var(--font-mono);
        font-weight: 500;
        letter-spacing: .02em;
        color: var(--ink);
        background: var(--paper-raised);
        transition: all .15s ease;
    }

    div[data-testid="stButton"] > button:hover {
        border-color: var(--ink);
        color: var(--ink);
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        border: 1px solid var(--ink);
        background: var(--ink);
        color: var(--paper-raised);
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: var(--ink-soft);
    }

    div[data-testid="stTextArea"] textarea {
        border-radius: 3px;
        border: 1px solid var(--line-strong);
        background: var(--paper-raised);
        font-size: .95rem;
        line-height: 1.55;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: var(--ink);
        box-shadow: 0 0 0 1px var(--ink);
    }

    div[data-testid="stMetric"] {
        background: var(--paper-raised);
        border: 1px solid var(--line);
        border-radius: 3px;
        padding: 10px 13px;
    }

    div[data-testid="stMetricValue"] {
        font-family: var(--font-mono);
        font-variant-numeric: tabular-nums;
    }

    .trace-summary {
        padding: 10px 12px;
        border: 1px solid var(--line);
        border-radius: 3px;
        background: var(--paper-raised);
        color: var(--ink-soft);
        font-family: var(--font-mono);
        font-size: .82rem;
    }

    .sidebar-caption {
        color: var(--ink-soft);
        font-size: .76rem;
        line-height: 1.5;
    }

    .app-footer {
        border-top: 1px solid var(--line);
        padding-top: 12px;
        margin-top: 8px;
        font-family: var(--font-mono);
        font-size: .74rem;
        letter-spacing: .03em;
        color: var(--ink-soft);
        text-transform: uppercase;
    }

    hr {
        border-color: var(--line) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# =============================================================================
# DATA LOADING (with production error handling)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data():
    return metrics.load_all()


try:
    sales, products, inventory, marketing = load_data()
except FileNotFoundError as exc:
    st.error(
        "The business dataset could not be found. Make sure the CSV files "
        "exist under the `data/` folder before running the app."
    )
    with st.expander("Technical detail"):
        st.code(str(exc))
    st.stop()

if sales.empty:
    st.error("The sales dataset is empty — there's nothing to show yet.")
    st.stop()

DATASET_START_DATE: date = pd.to_datetime(sales["date"]).min().date()
DATASET_END_DATE: date = pd.to_datetime(sales["date"]).max().date()


# =============================================================================
# HELPERS
# =============================================================================

def money(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) >= 10_000_000:
        return f"₹{value / 10_000_000:.2f} Cr"
    if abs(value) >= 100_000:
        return f"₹{value / 100_000:.2f} L"
    return f"₹{value:,.0f}"


def pct(value, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}f}%"


def records_df(data: dict, key: str) -> pd.DataFrame:
    records = data.get(key, [])
    return pd.DataFrame(records) if records else pd.DataFrame()


def section(title: str, description: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="muted">{description}</div>', unsafe_allow_html=True)


def rca_card(label: str, value: str, sub: str | None = None) -> None:
    st.markdown(
        f"""
        <div class="rca-card">
            <div class="rca-card-label">{label}</div>
            <div class="rca-card-value">{value}</div>
            {f'<div class="rca-card-sub">{sub}</div>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def signal_badge(strength: str | None, change: float | None) -> str:
    """A short plain-text evidence marker used consistently in dashboard
    tables (st.dataframe, which renders text only — no HTML/emoji) and in
    HTML signal cards. Direction is conveyed with a triangle glyph and
    magnitude with strength wording, deliberately without emoji or color
    dependence, so it degrades gracefully anywhere it's printed."""
    if not strength or strength == "INSUFFICIENT" or change is None:
        return "— n/a"
    if strength == "WEAK":
        return "\u00b7 weak"
    arrow = "\u25b2" if change > 0 else "\u25bc" if change < 0 else "\u2014"
    if strength == "STRONG":
        return f"{arrow}{arrow} strong"
    if strength == "MODERATE":
        return f"{arrow} moderate"
    return "— n/a"


def render_chart(fig, empty_title: str = "No data available", empty_message: str = "Nothing to show for this selection.") -> None:
    """Every chart on the page goes through this so missing chart data never
    looks like a broken app — it degrades to a clean empty state instead."""
    st.plotly_chart(
        fig if fig is not None else charts.no_data_message(empty_title, empty_message),
        use_container_width=True,
        config={"displayModeBar": False},
    )


# =============================================================================
# PERIOD OPTIONS
# =============================================================================
# The only thing computed here is WHICH calendar months exist in the data,
# for populating the dropdown — never their start/end boundaries. Every
# actual date range used anywhere in the app comes from
# metrics.resolve_named_period(), so there is exactly one place period
# language becomes concrete dates.

def _period_options(sales_df: pd.DataFrame) -> list[tuple[str, str]]:
    """Returns (display_label, resolve_named_period key) pairs."""
    months = pd.period_range(
        pd.Timestamp(sales_df["date"].min()).to_period("M"),
        pd.Timestamp(sales_df["date"].max()).to_period("M"),
        freq="M",
    )
    options = [("Latest Month", "latest_month")]
    if len(months) >= 2:
        options.append(("Previous Month", "previous_month"))
    options += [(m.strftime("%B %Y"), m.strftime("%Y-%m")) for m in months]
    return options


PERIOD_OPTIONS = _period_options(sales)
PERIOD_LABELS = [label for label, _ in PERIOD_OPTIONS]
PERIOD_KEY_BY_LABEL = dict(PERIOD_OPTIONS)


def resolve_period(label: str) -> dict:
    return metrics.resolve_named_period(sales, PERIOD_KEY_BY_LABEL[label])


# =============================================================================
# NAVIGATION
# =============================================================================
# Numbered labels are used deliberately here, not decoratively — they
# reflect the actual investigative order the tool is built around:
# scan the headline numbers, drill into a dimension, then investigate a
# root cause. The canonical page names below are unchanged so every
# `if page == "..."` check further down the file stays exactly as-is.

PAGE_DISPLAY = {
    "Executive Overview": "01 · Overview",
    "Business Performance": "02 · Performance",
    "Root Cause Analysis": "03 · Root Cause",
}
_PAGE_BY_DISPLAY = {v: k for k, v in PAGE_DISPLAY.items()}


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand-card">
            <div class="brand-wordmark">NEEMAN'S</div>
            <div class="brand-name">AI Business Copilot</div>
            <div class="brand-sub">
                Evidence-based retail analytics with deterministic metrics and
                Sarvam-powered root cause analysis.
            </div>
            <div class="status-pill">Analytics ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown('<div class="panel-eyebrow">Navigate</div>', unsafe_allow_html=True)
    page_display = st.radio(
        "Navigate",
        list(PAGE_DISPLAY.values()),
        label_visibility="collapsed",
    )
    page = _PAGE_BY_DISPLAY[page_display]

    st.divider()
    st.markdown('<div class="panel-eyebrow">Reporting Period</div>', unsafe_allow_html=True)

    current_label = st.selectbox(
        "Current period",
        PERIOD_LABELS,
        index=0,
        key="current_period_label",
    )
    default_comparison_index = 1 if len(PERIOD_LABELS) > 1 else 0
    comparison_label = st.selectbox(
        "Comparison period",
        PERIOD_LABELS,
        index=default_comparison_index,
        key="comparison_period_label",
    )

    current_resolved = resolve_period(current_label)
    comparison_resolved = resolve_period(comparison_label)

    if current_resolved["status"] != "ok" or comparison_resolved["status"] != "ok":
        st.error("One of the selected periods isn't available in the dataset.")
        st.stop()

    CURRENT_START, CURRENT_END = current_resolved["start_date"], current_resolved["end_date"]
    CURRENT_LABEL = current_resolved["label"]
    COMPARISON_START, COMPARISON_END = comparison_resolved["start_date"], comparison_resolved["end_date"]
    COMPARISON_LABEL = comparison_resolved["label"]

    st.caption(f"**Current** · {CURRENT_START} → {CURRENT_END}")
    st.caption(f"**Compared against** · {COMPARISON_START} → {COMPARISON_END}")

    st.divider()
    st.markdown('<div class="panel-eyebrow">Dataset Coverage</div>', unsafe_allow_html=True)
    st.info(
        f"**{DATASET_START_DATE.strftime('%b %Y')} → {DATASET_END_DATE.strftime('%b %Y')}**\n\n"
        f"Months available: **{len([o for o in PERIOD_OPTIONS if o[1] not in ('latest_month', 'previous_month')])}**"
    )
    st.caption("Dashboard calculations powered by pandas (analytics/metrics.py).")


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    f"""
    <div class="masthead">
        <div class="masthead-eyebrow">Evidence-Grounded Retail Analytics</div>
        <div class="main-title">Neeman's AI Business Copilot</div>
        <div class="masthead-rule"></div>
        <div class="masthead-meta">
            Coverage {DATASET_START_DATE.strftime('%b %Y')} – {DATASET_END_DATE.strftime('%b %Y')}
            <span class="masthead-dot">•</span>
            Deterministic analytics, Sarvam-interpreted root cause analysis
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# EXECUTIVE OVERVIEW
# =============================================================================

if page == "Executive Overview":

    section("Executive Overview", f"{CURRENT_LABEL} compared with {COMPARISON_LABEL}")

    kpi_result = metrics.compare_sales_kpis(sales, CURRENT_START, CURRENT_END, COMPARISON_START, COMPARISON_END)

    if kpi_result.get("status") != "ok":
        st.warning("No sales data is available for the selected periods.")
        st.stop()

    kpis = kpi_result["kpis"]
    revenue, orders, units, aov = kpis["revenue"], kpis["orders"], kpis["units"], kpis["average_order_value"]

    columns = st.columns(4)
    cards = [
        ("Revenue", money(revenue["current"]), revenue["pct_change"]),
        ("Orders", f'{orders["current"]:,.0f}', orders["pct_change"]),
        ("Units Sold", f'{units["current"]:,.0f}', units["pct_change"]),
        ("Average Order Value", money(aov["current"]), aov["pct_change"]),
    ]
    for column, (label, value, change) in zip(columns, cards):
        with column:
            st.metric(label, value, f"{change:+.2f}%" if change is not None else None)
    st.markdown(f'<div class="kpi-caption">vs {COMPARISON_LABEL}</div>', unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        render_chart(
            charts.revenue_comparison_chart(revenue["previous"], revenue["current"], COMPARISON_LABEL, CURRENT_LABEL)
        )
    with c2:
        render_chart(charts.executive_kpi_change_chart(kpis))

    # -------------------------------------------------------------------------
    # Business signals — biggest movers only, sourced straight from metrics.py
    # -------------------------------------------------------------------------

    section("Business Signals", "The most important movements — see Business Performance for full detail.")

    category_cmp = metrics.compare_category_performance(sales, CURRENT_START, CURRENT_END, COMPARISON_START, COMPARISON_END)
    channel_cmp = metrics.compare_channel_performance(sales, CURRENT_START, CURRENT_END, COMPARISON_START, COMPARISON_END)
    inventory_cmp = metrics.compare_inventory_performance(inventory, CURRENT_START, CURRENT_END, COMPARISON_START, COMPARISON_END)
    marketing_cmp = metrics.compare_marketing_performance(marketing, CURRENT_START, CURRENT_END, COMPARISON_START, COMPARISON_END)

    def _extreme(records: list[dict], key: str, pick_max: bool) -> dict | None:
        valid = [r for r in records if r.get(key) is not None]
        if not valid:
            return None
        return max(valid, key=lambda r: r[key]) if pick_max else min(valid, key=lambda r: r[key])

    category_records = category_cmp.get("categories", [])
    worst_category = _extreme(category_records, "revenue_change_pct", pick_max=False)
    best_category = _extreme(category_records, "revenue_change_pct", pick_max=True)

    channel_records = channel_cmp.get("channels", [])
    best_channel = _extreme(channel_records, "revenue_change_pct", pick_max=True)

    marketing_records = marketing_cmp.get("channels", [])
    worst_marketing = _extreme(marketing_records, "roas_change_pct", pick_max=False)

    if worst_category and worst_category["revenue_change_pct"] is not None and worst_category["revenue_change_pct"] < 0:
        st.markdown(
            f"""<div class="signal"><div class="signal-title"><span class="tick tick-down"></span>Biggest Category Decline</div>
            <div class="signal-body"><b>{worst_category['category']}</b> declined by
            <b>{pct(worst_category['revenue_change_pct'])}</b> — {signal_badge(worst_category.get('signal_strength'), worst_category['revenue_change_pct'])}</div></div>""",
            unsafe_allow_html=True,
        )

    if best_category and best_category["revenue_change_pct"] is not None and best_category["revenue_change_pct"] > 0:
        st.markdown(
            f"""<div class="signal"><div class="signal-title"><span class="tick tick-up"></span>Strongest Category</div>
            <div class="signal-body"><b>{best_category['category']}</b> grew by
            <b>{pct(best_category['revenue_change_pct'])}</b> — {signal_badge(best_category.get('signal_strength'), best_category['revenue_change_pct'])}</div></div>""",
            unsafe_allow_html=True,
        )

    if best_channel and best_channel["revenue_change_pct"] is not None and best_channel["revenue_change_pct"] > 0:
        st.markdown(
            f"""<div class="signal"><div class="signal-title"><span class="tick tick-up"></span>Strongest Channel</div>
            <div class="signal-body"><b>{best_channel['channel']}</b> grew by
            <b>{pct(best_channel['revenue_change_pct'])}</b> — {signal_badge(best_channel.get('signal_strength'), best_channel['revenue_change_pct'])}</div></div>""",
            unsafe_allow_html=True,
        )

    if inventory_cmp.get("status") == "ok":
        availability_change = inventory_cmp.get("availability_change_pct_points")
        stockout_change = inventory_cmp.get("stockout_days_change")
        if availability_change is not None and (availability_change < 0 or (stockout_change or 0) > 0):
            st.markdown(
                f"""<div class="signal"><div class="signal-title"><span class="tick tick-down"></span>Inventory Deterioration</div>
                <div class="signal-body">Availability changed by <b>{availability_change:+.2f} pp</b>,
                stockout days changed by <b>{stockout_change:+d}</b> —
                {signal_badge(inventory_cmp.get('signal_strength'), availability_change)}</div></div>""",
                unsafe_allow_html=True,
            )

    if worst_marketing and worst_marketing["roas_change_pct"] is not None and worst_marketing["roas_change_pct"] < 0:
        st.markdown(
            f"""<div class="signal"><div class="signal-title"><span class="tick tick-down"></span>Marketing Efficiency Decline</div>
            <div class="signal-body"><b>{worst_marketing['channel']}</b> ROAS declined by
            <b>{pct(worst_marketing['roas_change_pct'])}</b> — {signal_badge(worst_marketing.get('signal_strength'), worst_marketing['roas_change_pct'])}</div></div>""",
            unsafe_allow_html=True,
        )


# =============================================================================
# BUSINESS PERFORMANCE  (Category / Channel / Inventory / Marketing tabs)
# =============================================================================

elif page == "Business Performance":

    section("Business Performance", f"{CURRENT_LABEL} compared with {COMPARISON_LABEL} — what happened, by dimension.")

    tab_category, tab_channel, tab_inventory, tab_marketing = st.tabs(
        ["Category", "Channel", "Inventory", "Marketing"]
    )

    # ---- Category ----
    with tab_category:
        comparison = metrics.compare_category_performance(sales, CURRENT_START, CURRENT_END, COMPARISON_START, COMPARISON_END)
        df = records_df(comparison, "categories")
        render_chart(charts.category_revenue_change_chart(comparison.get("categories")), "No category data", "No category data for this selection.")
        if not df.empty:
            display = df.copy()
            display["Current Revenue"] = display["current_revenue"].map(money)
            display["Comparison Revenue"] = display["previous_revenue"].map(money)
            display["Revenue Change"] = display["revenue_change_pct"].map(pct)
            display["Signal"] = [signal_badge(r.get("signal_strength"), r.get("revenue_change_pct")) for r in comparison.get("categories", [])]
            st.dataframe(
                display[["category", "Current Revenue", "Comparison Revenue", "Revenue Change", "Signal"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No category data available for the selected periods.")

    # ---- Channel ----
    with tab_channel:
        comparison = metrics.compare_channel_performance(sales, CURRENT_START, CURRENT_END, COMPARISON_START, COMPARISON_END)
        df = records_df(comparison, "channels")
        c1, c2 = st.columns(2)
        with c1:
            render_chart(charts.channel_revenue_change_chart(comparison.get("channels")), "No channel data", "No channel data for this selection.")
        with c2:
            render_chart(charts.revenue_share_chart(comparison.get("channels")), "No channel data", "No revenue-share data for this selection.")
        if not df.empty:
            display = df.copy()
            display["Current Revenue"] = display["current_revenue"].map(money)
            display["Comparison Revenue"] = display["previous_revenue"].map(money)
            display["Revenue Change"] = display["revenue_change_pct"].map(pct)
            display["Current Share"] = display["current_revenue_share_pct"].map(lambda x: f"{x:.1f}%")
            display["Signal"] = [signal_badge(r.get("signal_strength"), r.get("revenue_change_pct")) for r in comparison.get("channels", [])]
            st.dataframe(
                display[["channel", "Current Revenue", "Comparison Revenue", "Revenue Change", "Current Share", "Signal"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No channel data available for the selected periods.")

    # ---- Inventory ----
    with tab_inventory:
        comparison = metrics.compare_inventory_performance(inventory, CURRENT_START, CURRENT_END, COMPARISON_START, COMPARISON_END)
        if comparison.get("status") != "ok":
            st.info("No inventory data available for the selected periods.")
        else:
            current_period, previous_period = comparison["current_period"], comparison["previous_period"]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Availability",
                    f"{current_period['average_availability_pct']:.2f}%" if current_period["average_availability_pct"] is not None else "—",
                    f"{comparison['availability_change_pct_points']:+.2f} pp" if comparison["availability_change_pct_points"] is not None else None,
                )
            with c2:
                st.metric(
                    "Avg Closing Stock",
                    f"{current_period['average_closing_stock']:.2f}" if current_period["average_closing_stock"] is not None else "—",
                    f"{comparison['stock_change_pct']:+.2f}%" if comparison["stock_change_pct"] is not None else None,
                )
            with c3:
                st.metric(
                    "Stockout Days",
                    f"{current_period['stockout_days']:,}",
                    f"{comparison['stockout_days_change']:+,} days",
                )
            st.caption(f"Signal: {signal_badge(comparison.get('signal_strength'), comparison.get('availability_change_pct_points'))}")

            render_chart(
                charts.inventory_kpi_comparison_chart(current_period, previous_period),
                "No inventory comparison", "Not enough inventory data to compare periods.",
            )

            current_metrics = metrics.get_inventory_metrics(inventory, CURRENT_START, CURRENT_END)
            render_chart(
                charts.availability_by_category_chart(current_metrics.get("availability_by_category")),
                "No availability breakdown", "No per-category availability data for this period.",
            )

            if current_period["average_availability_pct"] is not None and current_period["average_availability_pct"] < 90:
                st.warning(f"Inventory availability is {current_period['average_availability_pct']:.2f}%, indicating a potential availability risk.")
            if current_period["stockout_days"] > previous_period["stockout_days"]:
                st.error(f"Stockout days increased from {previous_period['stockout_days']} to {current_period['stockout_days']}.")

    # ---- Marketing ----
    with tab_marketing:
        comparison = metrics.compare_marketing_performance(marketing, CURRENT_START, CURRENT_END, COMPARISON_START, COMPARISON_END)
        df = records_df(comparison, "channels")
        c1, c2 = st.columns(2)
        with c1:
            render_chart(charts.roas_chart(comparison.get("channels")), "No ROAS data", "No ROAS data for this selection.")
        with c2:
            render_chart(charts.marketing_roas_change_chart(comparison.get("channels")), "No ROAS change data", "No ROAS change data for this selection.")
        render_chart(charts.marketing_revenue_change_chart(comparison.get("channels")), "No attributed revenue data", "No attributed revenue change data for this selection.")
        if not df.empty:
            display = df.copy()
            display["Current Spend"] = display["current_spend"].map(money)
            display["Current Attributed Revenue"] = display["current_attributed_revenue"].map(money)
            display["Current ROAS"] = display["current_roas"].map(lambda x: f"{x:.2f}x" if pd.notna(x) else "—")
            display["ROAS Change"] = display["roas_change_pct"].map(pct)
            display["Attributed Revenue Change"] = display["attributed_revenue_change_pct"].map(pct)
            display["Signal"] = [signal_badge(r.get("signal_strength"), r.get("attributed_revenue_change_pct")) for r in comparison.get("channels", [])]
            st.dataframe(
                display[["channel", "Current Spend", "Current Attributed Revenue", "Current ROAS", "ROAS Change", "Attributed Revenue Change", "Signal"]],
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No marketing data available for the selected periods.")


# =============================================================================
# ROOT CAUSE ANALYSIS
# =============================================================================

elif page == "Root Cause Analysis":

    RCA_SECTION_HEADERS = [
        "Executive Summary", "What Changed", "Evidence", "Root Cause Ranking",
        "Growth Drivers", "Recommendations", "Data Limitations", "Confidence",
    ]

    def parse_rca_sections(text: str) -> dict:
        """Split the agent's structured text into named sections. Falls back
        to one "Full Response" section if headers aren't found (e.g. a
        degraded/fallback answer) — the agent's reasoning is never modified
        here, only how it's laid out."""
        pattern = r"^(" + "|".join(re.escape(h) for h in RCA_SECTION_HEADERS) + r")\s*$"
        sections, current, buffer = {}, None, []
        for line in text.splitlines():
            if re.match(pattern, line.strip(), flags=re.IGNORECASE):
                if current:
                    sections[current] = "\n".join(buffer).strip()
                current = line.strip()
                buffer = []
            else:
                buffer.append(line)
        if current:
            sections[current] = "\n".join(buffer).strip()
        if not sections:
            sections["Full Response"] = text.strip()
        return sections

    def extract_revenue_change(trace: list) -> tuple:
        """Pull the headline revenue change straight from the trace — never
        from the LLM's prose — so the summary card can't drift from what was
        actually measured."""
        for step in reversed(trace):
            if step["tool"] == "compare_sales_kpis":
                revenue_kpi = step["result"].get("kpis", {}).get("revenue")
                if revenue_kpi:
                    return revenue_kpi.get("pct_change"), revenue_kpi.get("current"), revenue_kpi.get("previous")
            if step["tool"] == "compare_periods" and step["input"].get("metric") == "revenue":
                result = step["result"]
                if result.get("status") == "ok":
                    return result["pct_change"], result["period_a"]["value"], result["period_b"]["value"]
        return None, None, None

    def extract_result_signal(result) -> dict | None:
        """
        Use the RCA agent's canonical deterministic strongest signal.
        The agent explicitly exposes result.strongest_signal so the UI does
        not maintain a second, potentially inconsistent ranking implementation.
        """
        signal = getattr(result, "strongest_signal", None)
        return signal if isinstance(signal, dict) else None

    def extract_confidence(sections: dict) -> str:
        match = re.search(r"\b(HIGH|MEDIUM|LOW)\b", sections.get("Confidence", ""), flags=re.IGNORECASE)
        return match.group(1).upper() if match else "—"

    def confidence_badge(confidence: str) -> str:
        css_class = {"HIGH": "confidence-high", "MEDIUM": "confidence-medium", "LOW": "confidence-low"}.get(confidence, "confidence-unknown")
        return f'<span class="confidence-badge {css_class}">{confidence}</span>'

    def render_dimension_evidence(trace: list) -> None:
        """Compact, signal-labeled tables per dimension actually
        investigated — built straight from tool output, never from prose."""
        for step in trace:
            tool, result = step["tool"], step["result"]

            if tool == "compare_category_performance" and result.get("categories"):
                st.markdown("**Category Performance**")
                rows = [
                    {"Category": r["category"], "Revenue Change": pct(r.get("revenue_change_pct")), "Signal": signal_badge(r.get("signal_strength"), r.get("revenue_change_pct"))}
                    for r in result["categories"]
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            elif tool == "compare_channel_performance" and result.get("channels"):
                st.markdown("**Channel Performance**")
                rows = [
                    {"Channel": r["channel"], "Revenue Change": pct(r.get("revenue_change_pct")), "Signal": signal_badge(r.get("signal_strength"), r.get("revenue_change_pct"))}
                    for r in result["channels"]
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            elif tool == "compare_marketing_performance" and result.get("channels"):
                st.markdown("**Marketing Performance**")
                rows = [
                    {
                        "Channel": r["channel"],
                        "Attributed Revenue Change": pct(r.get("attributed_revenue_change_pct")),
                        "ROAS Change": pct(r.get("roas_change_pct")),
                        "Signal": signal_badge(r.get("signal_strength"), r.get("attributed_revenue_change_pct")),
                    }
                    for r in result["channels"]
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            elif tool == "compare_inventory_performance" and result.get("status") == "ok":
                st.markdown("**Inventory Signal**")
                st.write(
                    f"{signal_badge(result.get('signal_strength'), result.get('availability_change_pct_points'))} — "
                    f"availability {pct(result.get('availability_change_pct_points'), 1)} points, "
                    f"stockout days change {result.get('stockout_days_change', '—')}"
                )

    RATE_LIMIT_MARKERS = ("429", "rate_limit", "rate limit")

    # -------------------------------------------------------------------------
    # PAGE BODY
    # -------------------------------------------------------------------------

    section("Root Cause Analysis", "Ask the AI Copilot why a business metric changed.")

    st.markdown(
        f"""
        <div class="signal">
            <div class="rca-question-header">
                <div class="rca-question-title">Evidence-Based Root Cause Analysis</div>
                <div class="period-chip">{COMPARISON_LABEL} → {CURRENT_LABEL}</div>
            </div>
            <div class="signal-body">
                Sarvam AI interprets evidence returned by deterministic analytics tools.
                Business calculations and evidence-strength labels remain controlled by
                pandas, not the language model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
# Every date-sensitive example must explicitly carry the currently
    # selected periods (chronologically: comparison -> current), since
    # these questions are otherwise ambiguous to the agent. This list is
    # rebuilt from CURRENT_LABEL / COMPARISON_LABEL on every script rerun,
    # so it always reflects the sidebar's current period selection with no
    # separate date-resolution logic here.
    examples = [
        f"Why did revenue change from {COMPARISON_LABEL} to {CURRENT_LABEL}?",
        f"Which category contributed most to the revenue change from {COMPARISON_LABEL} to {CURRENT_LABEL}?",
        f"Is inventory availability contributing to the revenue change from {COMPARISON_LABEL} to {CURRENT_LABEL}?",
        f"Which marketing channel deteriorated the most from {COMPARISON_LABEL} to {CURRENT_LABEL}?",
    ]
    example_columns = st.columns(2)
    for i, example in enumerate(examples):
        if example_columns[i % 2].button(example, key=f"rca_example_{i}", use_container_width=True):
            st.session_state["rca_question"] = example

    question = st.text_area(
        "Investigation question",
        value=st.session_state.get("rca_question", ""),
        placeholder="Example: Why did revenue decline from June to July?",
        height=110,
    )

    if st.button("Run Investigation", type="primary", use_container_width=True):
        clean_question = question.strip()

        if not clean_question:
            st.warning("Please enter an investigation question.")
        else:
            try:
                # Lazy import keeps the dashboard pages independent from the
                # Sarvam/RCA stack, while still surfacing a useful import error
                # if the RCA dependency is misconfigured in deployment.
                from agent.rca_agent import run_investigation

                with st.spinner("Investigating business performance with Sarvam AI..."):
                    st.session_state["last_rca_result"] = run_investigation(
                        clean_question,
                        current_period=current_resolved,
                        comparison_period=comparison_resolved,
                    )

            except Exception as exc:
                st.session_state["last_rca_result"] = None

                # The previous version swallowed the real exception and always
                # displayed the same generic message. That made deployment
                # failures indistinguishable from model/tool failures.
                st.error("RCA investigation could not be started.")

                with st.expander("Technical diagnostic", expanded=True):
                    st.code(
                        f"{type(exc).__name__}: {exc}",
                        language="text",
                    )
                    st.caption(
                        "If this is a deployment issue, check SARVAM_API_KEY, "
                        "langchain-sarvam, LangGraph, and the agent/tools imports."
                    )

    # -------------------------------------------------------------------------
    # RCA RESULT
    # -------------------------------------------------------------------------

    result = st.session_state.get("last_rca_result")

    if result:
        st.divider()

        status = getattr(result, "status", "ok")
        answer = getattr(result, "final_answer", None) or ""
        trace = getattr(result, "trace", []) or []
        grounding_warning = getattr(result, "grounding_warning", None) or ""

        is_rate_limited = (
            status == "model_rate_limited"
            or any(marker in grounding_warning.lower() for marker in RATE_LIMIT_MARKERS)
        )

        if status == "no_api_key":
            st.error(
                "SARVAM_API_KEY is not configured. Add SARVAM_API_KEY to your "
                ".env/secrets configuration and restart the app."
            )
        elif status == "auth_error":
            st.error(
                "Sarvam authentication failed. Verify that SARVAM_API_KEY is "
                "correct, active, and available to the running process."
            )
        elif is_rate_limited:
            st.warning(
                "Sarvam is temporarily rate-limited. The analytics trace is "
                "preserved when available; please retry after the limit resets."
            )
        elif status in ("synthesis_fallback", "api_error", "grounding_failed"):
            st.warning(
                "The analytics evidence was collected, but the AI explanation "
                "did not complete normally. Showing the evidence-backed result."
            )

        if grounding_warning and status not in ("ok", "model_rate_limited", "auth_error"):
            with st.expander("Investigation diagnostic"):
                st.code(grounding_warning, language="text")

        if status == "ok" and answer:
            sections = parse_rca_sections(answer)
            revenue_change, _, _ = extract_revenue_change(trace)
            strongest = extract_result_signal(result)
            confidence = getattr(result, "confidence", None) or extract_confidence(sections)

            result_current_period = getattr(result, "current_period", None)
            result_previous_period = getattr(result, "previous_period", None)

            actual_current_label = (
                result_current_period.get("label")
                if isinstance(result_current_period, dict) and result_current_period.get("label")
                else CURRENT_LABEL
            )
            actual_previous_label = (
                result_previous_period.get("label")
                if isinstance(result_previous_period, dict) and result_previous_period.get("label")
                else COMPARISON_LABEL
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                rca_card(
                    "Revenue Change",
                    pct(revenue_change) if revenue_change is not None else "—",
                    f"{actual_previous_label} → {actual_current_label}",
                )
            with c2:
                if strongest:
                    rca_card("Strongest Signal", f"{strongest['dimension']}: {strongest['label']}", f"{strongest['strength']} · {pct(strongest['change'])}")
                else:
                    rca_card("Strongest Signal", "—", "No conclusive dimension evidence")
            with c3:
                st.markdown(
                    f'<div class="rca-card"><div class="rca-card-label">Confidence</div>'
                    f'<div style="margin-top:8px;">{confidence_badge(confidence)}</div></div>',
                    unsafe_allow_html=True,
                )

            st.write("")

            if "Executive Summary" in sections:
                st.markdown("#### Executive Summary")
                st.markdown(sections["Executive Summary"])

            if "What Changed" in sections:
                with st.expander("What Changed", expanded=True):
                    st.markdown(sections["What Changed"])

            if "Evidence" in sections or any(t["tool"].startswith("compare_") for t in trace):
                with st.expander("Evidence", expanded=True):
                    if "Evidence" in sections:
                        st.markdown(sections["Evidence"])
                        st.markdown("---")
                    render_dimension_evidence(trace)

            ranking_key = "Growth Drivers" if "Growth Drivers" in sections else "Root Cause Ranking"
            if ranking_key in sections:
                with st.expander(ranking_key, expanded=True):
                    st.markdown(sections[ranking_key])

            if "Recommendations" in sections:
                with st.expander("Recommendations", expanded=True):
                    st.markdown(sections["Recommendations"])

            if "Data Limitations" in sections:
                with st.expander("Data Limitations"):
                    st.markdown(sections["Data Limitations"])

            if "Full Response" in sections:
                st.markdown(sections["Full Response"])

        elif answer:
            st.markdown(answer)

        if status == "ok":
            if grounding_warning:
                st.warning(grounding_warning)
            else:
                st.markdown(
                    '<div class="verify-stamp">Evidence Verified</div>',
                    unsafe_allow_html=True,
                )

        # ---- Investigation trace ----
        st.divider()
        if trace:
            with st.expander(f"Investigation Trace — {len(trace)} tool call(s) used"):
                for i, step in enumerate(trace, start=1):
                    st.write(f"{i}. `{step.get('tool', 'unknown')}`")
                st.markdown("---")
                if st.checkbox("Show tool inputs & outputs", key="rca_trace_details"):
                    for i, step in enumerate(trace, start=1):
                        st.markdown(f"**{i}. {step.get('tool', 'unknown')}**")
                        st.json({"input": step.get("input", {}), "result": step.get("result", {})})
        else:
            st.info("The agent did not execute any tools.")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown(
    f"""
    <div class="app-footer">
        Neeman's AI Business Copilot · Dataset ending {DATASET_END_DATE} ·
        Metrics powered by pandas · RCA powered by LangGraph + Sarvam AI
    </div>
    """,
    unsafe_allow_html=True,
)
