"""
Neeman's AI Business Copilot
----------------------------

Streamlit frontend. Owns UI, state, user interaction, orchestration and
presentation only — every business number displayed here comes from
analytics/metrics.py (via the RCA agent's tool trace on the RCA page, or
directly on the dashboard pages), and every chart comes from charts.py.

Architecture:

    CSV -> analytics/metrics.py -> agent/tools.py -> RCA agent (LangGraph
    + Groq) -> app.py -> charts.py / Plotly -> Streamlit UI

app.py does NOT recalculate revenue, percentage changes, ROAS, inventory
changes, or evidence strength, and does NOT compute month/period
boundaries itself — those all come from metrics.resolve_named_period().
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
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# STYLING
# =============================================================================

st.markdown(
    """
    <style>
    .block-container { max-width: 1400px; padding-top: 1.5rem; padding-bottom: 3rem; }

    .main-title { font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 0.1rem; }
    .subtitle { color: #64748b; margin-bottom: 1.4rem; font-size: 0.95rem; }
    .section-title { font-size: 1.25rem; font-weight: 700; margin-top: 0.6rem; margin-bottom: 0.2rem; }
    .muted { color: #64748b; font-size: 0.88rem; }

    .signal {
        padding: 14px 16px; border: 1px solid #e2e8f0; border-radius: 12px;
        margin-bottom: 10px; background: #ffffff;
    }
    .signal-title { font-weight: 700; margin-bottom: 4px; font-size: 0.92rem; }
    .signal-body { color: #475569; line-height: 1.5; font-size: 0.9rem; }

    .kpi-caption { color: #64748b; font-size: 0.82rem; margin-top: -0.6rem; }

    .rca-card {
        padding: 16px 18px; border-radius: 14px; border: 1px solid #e2e8f0;
        background: #ffffff; box-shadow: 0 2px 12px rgba(15, 23, 42, 0.04); min-height: 100px;
    }
    .rca-card-label {
        color: #64748b; font-size: 0.78rem; font-weight: 650;
        text-transform: uppercase; letter-spacing: 0.03em;
    }
    .rca-card-value { font-size: 1.4rem; font-weight: 800; margin-top: 5px; color: #0f172a; }
    .rca-card-sub { color: #64748b; font-size: 0.82rem; margin-top: 3px; }

    .confidence-badge {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        font-weight: 700; font-size: 0.78rem; letter-spacing: 0.02em;
    }
    .confidence-high { background: #dcfce7; color: #166534; }
    .confidence-medium { background: #fef3c7; color: #92400e; }
    .confidence-low { background: #fee2e2; color: #991b1b; }
    .confidence-unknown { background: #f1f5f9; color: #475569; }
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
    """A short emoji+label used consistently in dashboard tables and RCA evidence."""
    if not strength or strength == "INSUFFICIENT" or change is None:
        return "⚪ —"
    if change < 0:
        return {"STRONG": "🔴 STRONG", "MODERATE": "🟠 MODERATE", "WEAK": "🟡 WEAK"}.get(strength, "⚪ —")
    return {"STRONG": "🟢 STRONG", "MODERATE": "🟢 MODERATE", "WEAK": "🟡 WEAK"}.get(strength, "⚪ —")


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
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 🧠 Neeman's AI Business Copilot")
    st.caption("AI-powered retail analytics • RCA powered by LangGraph + Groq")

    st.divider()

    page = st.radio(
        "Navigate",
        ["Executive Overview", "Business Performance", "Root Cause Analysis"],
    )

    st.divider()
    st.markdown("### 📅 Period")

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

    st.caption(f"**Current:** {CURRENT_START} → {CURRENT_END}")
    st.caption(f"**Compared against:** {COMPARISON_START} → {COMPARISON_END}")

    st.divider()
    st.markdown("### Dataset")
    st.info(
        f"**Available data**\n\n"
        f"{DATASET_START_DATE.strftime('%b %Y')} → {DATASET_END_DATE.strftime('%b %Y')}\n\n"
        f"Months available: **{len([o for o in PERIOD_OPTIONS if o[1] not in ('latest_month', 'previous_month')])}**"
    )
    st.caption("Dashboard calculations powered by pandas (analytics/metrics.py).")


# =============================================================================
# HEADER
# =============================================================================

st.markdown('<div class="main-title">Neeman\'s AI Business Copilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered retail analytics • RCA powered by LangGraph + Groq</div>',
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
            f"""<div class="signal"><div class="signal-title">🔻 Biggest Category Decline</div>
            <div class="signal-body"><b>{worst_category['category']}</b> declined by
            <b>{pct(worst_category['revenue_change_pct'])}</b> — {signal_badge(worst_category.get('signal_strength'), worst_category['revenue_change_pct'])}</div></div>""",
            unsafe_allow_html=True,
        )

    if best_category and best_category["revenue_change_pct"] is not None and best_category["revenue_change_pct"] > 0:
        st.markdown(
            f"""<div class="signal"><div class="signal-title">📈 Strongest Category</div>
            <div class="signal-body"><b>{best_category['category']}</b> grew by
            <b>{pct(best_category['revenue_change_pct'])}</b> — {signal_badge(best_category.get('signal_strength'), best_category['revenue_change_pct'])}</div></div>""",
            unsafe_allow_html=True,
        )

    if best_channel and best_channel["revenue_change_pct"] is not None and best_channel["revenue_change_pct"] > 0:
        st.markdown(
            f"""<div class="signal"><div class="signal-title">🛒 Strongest Channel</div>
            <div class="signal-body"><b>{best_channel['channel']}</b> grew by
            <b>{pct(best_channel['revenue_change_pct'])}</b> — {signal_badge(best_channel.get('signal_strength'), best_channel['revenue_change_pct'])}</div></div>""",
            unsafe_allow_html=True,
        )

    if inventory_cmp.get("status") == "ok":
        availability_change = inventory_cmp.get("availability_change_pct_points")
        stockout_change = inventory_cmp.get("stockout_days_change")
        if availability_change is not None and (availability_change < 0 or (stockout_change or 0) > 0):
            st.markdown(
                f"""<div class="signal"><div class="signal-title">📦 Inventory Deterioration</div>
                <div class="signal-body">Availability changed by <b>{availability_change:+.2f} pp</b>,
                stockout days changed by <b>{stockout_change:+d}</b> —
                {signal_badge(inventory_cmp.get('signal_strength'), availability_change)}</div></div>""",
                unsafe_allow_html=True,
            )

    if worst_marketing and worst_marketing["roas_change_pct"] is not None and worst_marketing["roas_change_pct"] < 0:
        st.markdown(
            f"""<div class="signal"><div class="signal-title">📣 Marketing Efficiency Decline</div>
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
        ["📦 Category", "🛒 Channel", "📊 Inventory", "📣 Marketing"]
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
                st.warning(f"⚠️ Inventory availability is {current_period['average_availability_pct']:.2f}%, indicating a potential availability risk.")
            if current_period["stockout_days"] > previous_period["stockout_days"]:
                st.error(f"🚨 Stockout days increased from {previous_period['stockout_days']} to {current_period['stockout_days']}.")

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

    def extract_strongest_signal(trace: list) -> dict | None:
        """Scan every dimension actually investigated and surface the single
        strongest signal_strength result — INSUFFICIENT never qualifies as
        "strongest", it means there was nothing conclusive to report."""
        strength_rank = {"STRONG": 3, "MODERATE": 2, "WEAK": 1}
        best, best_rank = None, 0

        def consider(dimension: str, label: str, change, strength: str | None):
            nonlocal best, best_rank
            rank = strength_rank.get(strength, 0)
            if rank > best_rank:
                best, best_rank = {"dimension": dimension, "label": label, "change": change, "strength": strength}, rank

        for step in trace:
            tool, result = step["tool"], step["result"]
            if tool == "compare_category_performance":
                for r in result.get("categories", []):
                    consider("Category", r["category"], r.get("revenue_change_pct"), r.get("signal_strength"))
            elif tool == "compare_channel_performance":
                for r in result.get("channels", []):
                    consider("Channel", r["channel"], r.get("revenue_change_pct"), r.get("signal_strength"))
            elif tool == "compare_marketing_performance":
                for r in result.get("channels", []):
                    consider("Marketing", r["channel"], r.get("attributed_revenue_change_pct"), r.get("signal_strength"))
            elif tool == "compare_inventory_performance" and result.get("status") == "ok":
                consider("Inventory", "Availability / stockouts", result.get("availability_change_pct_points"), result.get("signal_strength"))

        return best

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
        """
        <div class="signal" style="background: linear-gradient(135deg, #f8fafc, #eef2ff); border-color:#cbd5e1;">
            <div class="signal-title">🧠 Evidence-based RCA</div>
            <div class="signal-body">
                The agent investigates using dedicated analytical tools and produces an
                evidence-backed diagnosis, with every evidence-strength label computed
                deterministically by pandas — never judged by the LLM.
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

    if st.button("🔎 Investigate", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter an investigation question.")
        else:
            try:
                from agent.rca_agent import run_investigation
                with st.spinner("Investigating business performance..."):
                    st.session_state["last_rca_result"] = run_investigation(question.strip())
            except Exception:
                st.session_state["last_rca_result"] = None
                st.error("RCA investigation could not be completed. Please retry in a moment.")

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

        is_rate_limited = status == "api_error" and any(marker in grounding_warning.lower() for marker in RATE_LIMIT_MARKERS)

        if status == "no_api_key":
            st.info("No GROQ_API_KEY is configured, so the agent can't run live. Add it to your `.env` file to enable investigations.")
        elif is_rate_limited:
            st.warning("AI investigation is temporarily unavailable because the model rate limit has been reached. Please try again later.")
        elif status in ("synthesis_fallback", "api_error"):
            st.warning("The analytics data was available, but the AI reasoning step didn't complete normally. Showing the raw evidence gathered instead.")

        if status == "ok" and answer:
            sections = parse_rca_sections(answer)
            revenue_change, _, _ = extract_revenue_change(trace)
            strongest = extract_strongest_signal(trace)
            confidence = extract_confidence(sections)

            c1, c2, c3 = st.columns(3)
            with c1:
                rca_card("Revenue Change", pct(revenue_change) if revenue_change is not None else "—", f"{COMPARISON_LABEL} → {CURRENT_LABEL}")
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
                st.warning(f"⚠️ {grounding_warning}")
            else:
                st.success("✓ Numeric claims verified")

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

st.divider()
st.caption(
    f"Neeman's AI Business Copilot • Dataset ending {DATASET_END_DATE} "
    "• Metrics powered by pandas • RCA powered by LangGraph + Groq"
)