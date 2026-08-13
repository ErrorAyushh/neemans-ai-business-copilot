"""
agent/tools.py

LangChain tool wrappers around analytics/metrics.py.

IMPORTANT:
This file contains NO business calculations.

All business calculations remain inside:
    analytics/metrics.py

These functions only:
1. Receive JSON-compatible arguments from the LLM.
2. Validate/forward those arguments.
3. Call the corresponding deterministic metrics function.
4. Return the resulting dictionary.

Tool-selection guidance:
- resolve_named_period() must be used before named-period analysis.
- compare_sales_kpis() is preferred when multiple headline sales KPIs
  are required.
- compare_periods() MUST remain available for targeted comparison of
  one specific metric.
- Category, channel, inventory, and marketing comparisons are separate
  analytical dimensions and should only be called when relevant.
"""

from __future__ import annotations

from langchain_core.tools import tool

from analytics import metrics


# ============================================================
# LOAD DATA
# ============================================================

_sales, _products, _inventory, _marketing = metrics.load_all()


# ============================================================
# DATASET METADATA
# ============================================================

DATASET_END_DATE = (
    _sales["date"]
    .max()
    .strftime("%Y-%m-%d")
)


# ============================================================
# PERIOD RESOLUTION
# ============================================================

@tool
def resolve_named_period(period: str) -> dict:
    """
    Resolve a named period into an exact start_date and end_date.

    This tool MUST be called before performing analysis involving
    named periods such as:

        latest_month
        previous_month
        June
        June 2026
        2026-06

    Never calculate month boundaries manually.

    Supported inputs:
        - "latest_month"
        - "previous_month"
        - "YYYY-MM"
        - "Month"
        - "Month YYYY"

    Examples:
        "latest_month"
        "previous_month"
        "June"
        "June 2026"
        "2026-03"

    If the requested period does not exist in the dataset, the
    underlying metrics function returns status="unavailable".

    An unavailable period is a legitimate analytical result and
    must NOT be worked around by guessing another period.
    """

    if not isinstance(period, str) or not period.strip():
        return {
            "status": "unavailable",
            "reason": "Period must be a non-empty string.",
        }

    return metrics.resolve_named_period(
        _sales,
        period.strip(),
    )


# ============================================================
# SALES TOOLS
# ============================================================

@tool
def get_sales_metrics(
    start_date: str,
    end_date: str,
) -> dict:
    """
    Get overall sales metrics for a specific date range.

    Returns:
        - revenue
        - orders
        - units
        - average_order_value

    Dates must use:
        YYYY-MM-DD

    Use this when the absolute sales performance of one period is
    required rather than a comparison.
    """

    return metrics.get_sales_metrics(
        _sales,
        start_date,
        end_date,
    )


@tool
def compare_periods(
    metric: str,
    period_a_start: str,
    period_a_end: str,
    period_b_start: str,
    period_b_end: str,
) -> dict:
    """
    Compare ONE sales metric between two periods.

    Allowed metrics:
        - revenue
        - units
        - orders

    period_a:
        Current/investigated period.

    period_b:
        Previous/baseline period.

    Use this tool when the investigation specifically requires
    ONE metric.

    Examples:
        - "How did revenue change?"
        - "Compare orders between June and July."
        - "Did units increase?"

    When multiple headline sales KPIs are required, prefer
    compare_sales_kpis() instead to avoid unnecessary tool calls.

    IMPORTANT:
    Do NOT remove or replace this tool with compare_sales_kpis().
    This tool is intentionally retained for targeted analysis.
    """

    allowed_metrics = {
        "revenue",
        "units",
        "orders",
    }

    metric_normalized = metric.strip().lower()

    if metric_normalized not in allowed_metrics:
        return {
            "status": "invalid",
            "metric": metric,
            "allowed_metrics": sorted(allowed_metrics),
        }

    return metrics.compare_periods(
        _sales,
        metric_normalized,
        period_a_start,
        period_a_end,
        period_b_start,
        period_b_end,
    )


@tool
def compare_sales_kpis(
    current_start_date: str,
    current_end_date: str,
    previous_start_date: str,
    previous_end_date: str,
) -> dict:
    """
    Compare ALL headline sales KPIs between two periods in ONE call.

    Returns comparisons for:
        - revenue
        - orders
        - units
        - average_order_value

    Each KPI includes its period values and percentage change.

    Prefer this tool when the investigation requires a broad
    headline-sales comparison.

    Do NOT call compare_periods() repeatedly for revenue, orders,
    and units when this tool can provide the same information
    in one call.

    Use compare_periods() instead when only ONE specific metric
    is required.
    """

    return metrics.compare_sales_kpis(
        _sales,
        current_start_date,
        current_end_date,
        previous_start_date,
        previous_end_date,
    )


# ============================================================
# CATEGORY TOOLS
# ============================================================

@tool
def get_category_performance(
    start_date: str,
    end_date: str,
) -> dict:
    """
    Get category-level sales performance for a date range.

    Returns category-level:
        - revenue
        - units
        - orders

    Use this when absolute category performance is required.
    """

    return metrics.get_category_performance(
        _sales,
        start_date,
        end_date,
    )


@tool
def compare_category_performance(
    current_start_date: str,
    current_end_date: str,
    previous_start_date: str,
    previous_end_date: str,
) -> dict:
    """
    Compare category performance between two periods.

    Returns, per category:
        - current revenue
        - previous revenue
        - revenue percentage change
        - current units
        - previous units
        - units percentage change
        - current orders
        - previous orders
        - orders percentage change
        - deterministic signal_strength

    signal_strength is produced by metrics.py.

    Allowed signal values may include:
        STRONG
        MODERATE
        WEAK
        INSUFFICIENT

    The agent MUST use the supplied signal_strength rather than
    inventing its own severity classification.

    Use this tool when investigating whether category-level
    performance contributed to an overall business change.
    """

    return metrics.compare_category_performance(
        _sales,
        current_start_date,
        current_end_date,
        previous_start_date,
        previous_end_date,
    )


# ============================================================
# CHANNEL TOOLS
# ============================================================

@tool
def get_channel_performance(
    start_date: str,
    end_date: str,
) -> dict:
    """
    Get sales-channel performance for a date range.

    Returns, by channel:
        - revenue
        - units
        - orders
        - revenue share

    Use this for absolute channel performance.
    """

    return metrics.get_channel_performance(
        _sales,
        start_date,
        end_date,
    )


@tool
def compare_channel_performance(
    current_start_date: str,
    current_end_date: str,
    previous_start_date: str,
    previous_end_date: str,
) -> dict:
    """
    Compare sales-channel performance between two periods.

    Returns, by channel:
        - current revenue
        - previous revenue
        - revenue percentage change
        - units comparison
        - orders comparison
        - revenue-share comparison
        - deterministic signal_strength

    Use the signal_strength returned by metrics.py as-is.

    Use this tool when investigating whether channel performance
    contributed to an overall revenue or sales change.
    """

    return metrics.compare_channel_performance(
        _sales,
        current_start_date,
        current_end_date,
        previous_start_date,
        previous_end_date,
    )


# ============================================================
# INVENTORY TOOLS
# ============================================================

@tool
def get_inventory_metrics(
    start_date: str,
    end_date: str,
    category: str = "",
) -> dict:
    """
    Get inventory performance for a date range.

    Returns inventory indicators such as:
        - availability
        - closing stock
        - low-stock SKU count
        - stockout information

    category is optional.

    Supported categories include:
        - Running
        - Casual
        - Formal
        - Lounge
        - Slip-On

    Leave category empty when overall inventory performance is
    required.
    """

    normalized_category = category.strip() if category else ""

    return metrics.get_inventory_metrics(
        _inventory,
        start_date,
        end_date,
        normalized_category or None,
    )


@tool
def compare_inventory_performance(
    current_start_date: str,
    current_end_date: str,
    previous_start_date: str,
    previous_end_date: str,
    category: str = "",
) -> dict:
    """
    Compare inventory conditions between two periods.

    Returns inventory comparisons including:
        - availability
        - closing stock
        - low-stock SKU count
        - stockout information
        - deterministic signal_strength

    category is optional.

    Leave category empty for overall inventory comparison.

    Use this tool when inventory deterioration could plausibly
    explain a sales or revenue change.

    IMPORTANT:
    Inventory deterioration is evidence of a possible contributing
    factor, not automatic proof of causation.

    The RCA agent must distinguish:
        observed evidence
    from:
        inferred/possible causes.
    """

    normalized_category = category.strip() if category else ""

    return metrics.compare_inventory_performance(
        _inventory,
        current_start_date,
        current_end_date,
        previous_start_date,
        previous_end_date,
        normalized_category,
    )


# ============================================================
# MARKETING TOOLS
# ============================================================

@tool
def get_marketing_metrics(
    start_date: str,
    end_date: str,
    channel: str = "",
) -> dict:
    """
    Get marketing performance for a date range.

    Returns available marketing metrics such as:
        - spend
        - attributed revenue
        - ROAS
        - campaign-level metrics when available

    channel is optional.

    Supported channels may include:
        - Google
        - Meta
        - Influencer

    Leave channel empty for overall marketing performance.
    """

    normalized_channel = channel.strip() if channel else ""

    return metrics.get_marketing_metrics(
        _marketing,
        start_date,
        end_date,
        normalized_channel or None,
    )


@tool
def compare_marketing_performance(
    current_start_date: str,
    current_end_date: str,
    previous_start_date: str,
    previous_end_date: str,
) -> dict:
    """
    Compare marketing performance between two periods.

    Returns, by marketing channel:
        - spend change
        - attributed revenue change
        - ROAS change
        - deterministic signal_strength

    Use signal_strength from metrics.py as-is.

    Use this tool when investigating whether marketing performance
    changed materially between two periods.

    IMPORTANT:
    Marketing changes should not automatically be treated as the
    cause of a revenue change unless the available evidence supports
    that interpretation.
    """

    return metrics.compare_marketing_performance(
        _marketing,
        current_start_date,
        current_end_date,
        previous_start_date,
        previous_end_date,
    )


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOLS = [
    # --------------------------------------------------------
    # Period resolution
    # --------------------------------------------------------
    resolve_named_period,

    # --------------------------------------------------------
    # Sales
    # --------------------------------------------------------
    get_sales_metrics,
    compare_periods,
    compare_sales_kpis,

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------
    get_category_performance,
    compare_category_performance,

    # --------------------------------------------------------
    # Sales channels
    # --------------------------------------------------------
    get_channel_performance,
    compare_channel_performance,

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------
    get_inventory_metrics,
    compare_inventory_performance,

    # --------------------------------------------------------
    # Marketing
    # --------------------------------------------------------
    get_marketing_metrics,
    compare_marketing_performance,
]