"""
analytics/metrics.py

All business calculations live in this file.

IMPORTANT DESIGN PRINCIPLE:
The LLM never performs business calculations itself.

Pandas calculates:
    - sales metrics
    - period comparisons
    - category performance
    - channel performance
    - inventory metrics
    - marketing metrics
    - cross-period (monthly) ranking
    - revenue trend / time series

The AI agent only interprets the results returned by these functions.
"""

import calendar
import os
import re
import pandas as pd


# ============================================================
# DATA LOCATION
# ============================================================

DATA_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
    )
)


# ============================================================
# DATA LOADING
# ============================================================

def _load_csv(name: str) -> pd.DataFrame:
    """
    Load one CSV from the data directory.

    All datasets except products.csv are expected to contain
    a 'date' column.
    """

    path = os.path.join(DATA_DIR, name)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    if name == "products.csv":
        return pd.read_csv(path)

    return pd.read_csv(
        path,
        parse_dates=["date"],
    )


def load_all():
    """
    Load all four business datasets.

    Returns:
        sales
        products
        inventory
        marketing
    """

    sales = _load_csv("sales.csv")
    products = _load_csv("products.csv")
    inventory = _load_csv("inventory.csv")
    marketing = _load_csv("marketing.csv")

    return (
        sales,
        products,
        inventory,
        marketing,
    )


# ============================================================
# DATE FILTERING
# ============================================================

def _filter_dates(
    df: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Filter a DataFrame to an inclusive date range.
    """

    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    mask = (
        (df["date"] >= start_date)
        & (df["date"] <= end_date)
    )

    return df.loc[mask].copy()


# ============================================================
# GENERIC PERCENT CHANGE
# ============================================================

def _pct_change(
    current: float,
    previous: float,
):
    """
    Calculate percentage change:

        (current - previous) / previous * 100

    Returns None if the previous value is zero (or None) —
    callers must treat None as "not computable", not 0%.
    """

    if not previous:
        return None

    return round(
        (current - previous) / previous * 100,
        2,
    )


# ============================================================
# DETERMINISTIC EVIDENCE STRENGTH
# ============================================================
# Fixed, auditable thresholds — computed here, never left to the LLM to
# eyeball. The RCA agent narrates these labels; it does not assign them.

def _signal_strength(pct_change: float | None) -> str:
    """Classify a percentage change's magnitude into a fixed evidence tier."""
    if pct_change is None:
        return "INSUFFICIENT"
    magnitude = abs(pct_change)
    if magnitude >= 25:
        return "STRONG"
    if magnitude >= 10:
        return "MODERATE"
    return "WEAK"


def _inventory_signal_strength(
    availability_change_pp: float | None,
    stockout_days_change: int | None,
) -> str:
    """Inventory has two independent risk signals (availability points lost,
    stockout days gained) — take whichever is more severe."""
    if availability_change_pp is None and stockout_days_change is None:
        return "INSUFFICIENT"

    availability_severity = abs(availability_change_pp) if availability_change_pp is not None else 0
    stockout_severity = abs(stockout_days_change) if stockout_days_change is not None else 0

    if availability_severity >= 20 or stockout_severity >= 20:
        return "STRONG"
    if availability_severity >= 10 or stockout_severity >= 10:
        return "MODERATE"
    return "WEAK"


# ============================================================
# NAMED PERIOD RESOLUTION
# ============================================================
# The RCA agent must resolve every period reference through this function
# before calling any comparison tool. This is the single place period
# language ("June", "latest_month", "2026-03"...) becomes concrete dates —
# keeping date math out of the LLM's hands entirely.

_MONTH_NAME_TO_NUM = {
    name.lower(): i for i, name in enumerate(calendar.month_name) if name
}
_MONTH_ABBR_TO_NUM = {
    name.lower(): i for i, name in enumerate(calendar.month_abbr) if name
}


def _available_months(sales: pd.DataFrame) -> list:
    if sales.empty:
        return []
    start = pd.Timestamp(sales["date"].min()).to_period("M")
    end = pd.Timestamp(sales["date"].max()).to_period("M")
    return list(pd.period_range(start, end, freq="M"))


def resolve_named_period(sales: pd.DataFrame, period: str) -> dict:
    """
    Resolve a natural period reference into concrete start_date/end_date
    that actually exist in the dataset.

    Accepts:
        "latest_month"          -> most recent complete month in the data
        "previous_month"        -> the month immediately before that
        "YYYY-MM"                -> e.g. "2026-03"
        a month name             -> e.g. "June" (most recent matching
                                     occurrence in the dataset) or
                                     "June 2026" (exact)

    Returns {"status": "ok", "start_date", "end_date", "label"} or
    {"status": "unavailable", "reason": "..."} — never guesses a fallback
    period silently.
    """
    months = _available_months(sales)
    if not months:
        return {"status": "unavailable", "reason": "No sales data available."}

    key = period.strip().lower()
    target = None

    if key == "latest_month":
        target = months[-1]
    elif key == "previous_month":
        target = months[-2] if len(months) >= 2 else None
    else:
        iso_match = re.match(r"^(\d{4})-(\d{1,2})$", key)
        if iso_match:
            year, month_num = int(iso_match.group(1)), int(iso_match.group(2))
            target = pd.Period(year=year, month=month_num, freq="M")
        else:
            name_match = re.match(r"^([a-zA-Z]+)\s*(\d{4})?$", key)
            if name_match:
                month_str, year_str = name_match.group(1), name_match.group(2)
                month_num = _MONTH_NAME_TO_NUM.get(month_str) or _MONTH_ABBR_TO_NUM.get(month_str)
                if month_num:
                    if year_str:
                        target = pd.Period(year=int(year_str), month=month_num, freq="M")
                    else:
                        matches = [p for p in months if p.month == month_num]
                        target = matches[-1] if matches else None

    if target is None or target not in months:
        return {
            "status": "unavailable",
            "reason": (
                f"'{period}' does not resolve to a month present in the dataset "
                f"({months[0]} to {months[-1]})."
            ),
        }

    start_date = target.start_time.date()
    end_date = min(target.end_time.date(), pd.Timestamp(sales["date"].max()).date())

    return {
        "status": "ok",
        "start_date": str(start_date),
        "end_date": str(end_date),
        "label": target.strftime("%B %Y"),
    }


# ============================================================
# SALES METRICS
# ============================================================

def get_sales_metrics(
    sales: pd.DataFrame,
    start: str,
    end: str,
) -> dict:
    """
    Revenue, orders, units and average order value
    for a date range.
    """

    window = _filter_dates(
        sales,
        start,
        end,
    )

    if window.empty:
        return {
            "status": "no_data",
            "start": start,
            "end": end,
        }

    revenue = float(
        window["revenue"].sum()
    )

    orders = int(
        window["order_id"].nunique()
    )

    units = int(
        window["units"].sum()
    )

    average_order_value = (
        revenue / orders
        if orders
        else 0.0
    )

    return {
        "status": "ok",
        "start": start,
        "end": end,
        "revenue": round(
            revenue,
            2,
        ),
        "orders": orders,
        "units": units,
        "average_order_value": round(
            average_order_value,
            2,
        ),
    }


# ============================================================
# SALES KPI COMPARISON (current vs previous, all 4 headline KPIs)
# ============================================================

def compare_sales_kpis(
    sales: pd.DataFrame,
    current_start: str,
    current_end: str,
    previous_start: str,
    previous_end: str,
) -> dict:
    """
    Compare all four Executive Overview KPIs (revenue, orders,
    units, average order value) between two periods in one call.

    This exists so the UI never has to re-derive percentage
    changes itself — every KPI card on the Executive Overview
    page should be built from this single result.
    """

    current = get_sales_metrics(
        sales,
        current_start,
        current_end,
    )

    previous = get_sales_metrics(
        sales,
        previous_start,
        previous_end,
    )

    if (
        current.get("status") != "ok"
        or previous.get("status") != "ok"
    ):
        return {
            "status": "no_data"
        }

    kpis = {}

    for key in (
        "revenue",
        "orders",
        "units",
        "average_order_value",
    ):

        current_value = current[key]
        previous_value = previous[key]

        kpis[key] = {
            "current": current_value,
            "previous": previous_value,
            "pct_change": _pct_change(
                current_value,
                previous_value,
            ),
        }

    return {
        "status": "ok",

        "current_period": {
            "start": current_start,
            "end": current_end,
        },

        "previous_period": {
            "start": previous_start,
            "end": previous_end,
        },

        "kpis": kpis,
    }


# ============================================================
# GENERIC SALES PERIOD COMPARISON
# ============================================================

def compare_periods(
    sales: pd.DataFrame,
    metric: str,
    period_a_start: str,
    period_a_end: str,
    period_b_start: str,
    period_b_end: str,
) -> dict:
    """
    Compare a sales metric between two periods.

    Supported metrics:
        revenue
        units
        orders

    period_a = current/investigated period
    period_b = previous/baseline period
    """

    valid_metrics = {
        "revenue": "revenue",
        "units": "units",
        "orders": "order_id",
    }

    if metric not in valid_metrics:
        return {
            "status": "error",
            "reason": (
                f"Unknown metric '{metric}'. "
                f"Use one of {list(valid_metrics)}."
            ),
        }

    period_a = _filter_dates(
        sales,
        period_a_start,
        period_a_end,
    )

    period_b = _filter_dates(
        sales,
        period_b_start,
        period_b_end,
    )

    if period_a.empty or period_b.empty:
        return {
            "status": "no_data"
        }

    if metric == "orders":

        value_a = float(
            period_a["order_id"].nunique()
        )

        value_b = float(
            period_b["order_id"].nunique()
        )

    else:

        column = valid_metrics[metric]

        value_a = float(
            period_a[column].sum()
        )

        value_b = float(
            period_b[column].sum()
        )

    change = _pct_change(
        value_a,
        value_b,
    )

    return {
        "status": "ok",
        "metric": metric,

        "period_a": {
            "start": period_a_start,
            "end": period_a_end,
            "value": round(
                value_a,
                2,
            ),
        },

        "period_b": {
            "start": period_b_start,
            "end": period_b_end,
            "value": round(
                value_b,
                2,
            ),
        },

        "pct_change": change,
    }


# ============================================================
# CROSS-PERIOD MONTHLY RANKING
# ============================================================
# Deterministic ranking of every calendar month present in the dataset,
# by total revenue. This is what a "which month has the best/worst
# revenue?" question actually needs — NOT a two-period comparison. Before
# this function existed, the agent had no tool for this shape of
# question and was forced to invent a fake second period to satisfy
# compare_sales_kpis/compare_periods, which is how a nonexistent
# "January 2025" got hallucinated into an answer for a Jan-Jul 2026
# dataset. Reuses _available_months/_filter_dates — no new date logic.

def rank_months_by_revenue(
    sales: pd.DataFrame,
    year: str | None = None,
) -> dict:
    """
    Rank every calendar month present in the dataset by total revenue,
    descending (best month first).

    year: optional 4-digit year string (e.g. "2026") to restrict the
    ranking to months within that year. None/empty ranks across the
    entire available dataset.

    Returns:
        {"status": "ok", "months": [ {period, label, start_date, end_date,
         revenue, orders, units}, ... ] sorted descending by revenue,
         "best_month": months[0], "worst_month": months[-1]}
        or {"status": "no_data", ...} if there is nothing to rank.
    """
    months = _available_months(sales)
    if not months:
        return {"status": "no_data"}

    if year:
        try:
            year_int = int(year)
        except (TypeError, ValueError):
            return {"status": "error", "reason": f"Invalid year: {year!r}"}
        months = [m for m in months if m.year == year_int]
        if not months:
            return {
                "status": "no_data",
                "reason": f"No data available for {year}.",
            }

    dataset_max_date = pd.Timestamp(sales["date"].max()).date()
    records = []

    for m in months:
        start_date = m.start_time.date()
        end_date = min(m.end_time.date(), dataset_max_date)
        window = _filter_dates(sales, str(start_date), str(end_date))
        if window.empty:
            continue

        revenue = float(window["revenue"].sum())
        orders = int(window["order_id"].nunique())
        units = int(window["units"].sum())

        records.append({
            "period": str(m),
            "label": m.strftime("%B %Y"),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "revenue": round(revenue, 2),
            "orders": orders,
            "units": units,
        })

    if not records:
        return {"status": "no_data"}

    records.sort(key=lambda r: r["revenue"], reverse=True)

    return {
        "status": "ok",
        "months": records,
        "best_month": records[0],
        "worst_month": records[-1],
    }


# ============================================================
# REVENUE TREND / TIME SERIES
# ============================================================
# A chronological, month-bucketed series across a date range — distinct
# from compare_periods (exactly two points) and rank_months_by_revenue
# (unordered by date, ordered by value). Used for "show the trend" /
# "how has revenue changed over X" style questions.

def get_revenue_trend(
    sales: pd.DataFrame,
    start: str,
    end: str,
) -> dict:
    """
    Month-by-month revenue/orders/units across a date range, as a
    chronologically ordered series of monthly data points.

    Returns:
        {"status": "ok", "start", "end",
         "points": [ {period, label, revenue, orders, units}, ... ]
         in chronological order}
        or {"status": "no_data"} if the range has no sales.
    """
    window = _filter_dates(sales, start, end)
    if window.empty:
        return {"status": "no_data"}

    window = window.copy()
    window["_month"] = window["date"].dt.to_period("M")

    grouped = (
        window
        .groupby("_month")
        .agg(
            revenue=("revenue", "sum"),
            units=("units", "sum"),
            orders=("order_id", "nunique"),
        )
        .round(2)
        .reset_index()
        .sort_values("_month")
    )

    points = []
    for _, row in grouped.iterrows():
        period = row["_month"]
        points.append({
            "period": str(period),
            "label": period.strftime("%B %Y"),
            "revenue": float(row["revenue"]),
            "orders": int(row["orders"]),
            "units": int(row["units"]),
        })

    return {
        "status": "ok",
        "start": start,
        "end": end,
        "points": points,
    }


# ============================================================
# CATEGORY PERFORMANCE
# ============================================================

def get_category_performance(
    sales: pd.DataFrame,
    start: str,
    end: str,
) -> dict:
    """
    Revenue, units and orders by category.
    """

    window = _filter_dates(
        sales,
        start,
        end,
    )

    if window.empty:
        return {
            "status": "no_data"
        }

    grouped = (
        window
        .groupby("category")
        .agg(
            revenue=("revenue", "sum"),
            units=("units", "sum"),
            orders=("order_id", "nunique"),
        )
        .round(2)
        .reset_index()
    )

    return {
        "status": "ok",
        "start": start,
        "end": end,
        "categories": grouped.to_dict(
            orient="records"
        ),
    }


# ============================================================
# CATEGORY PERIOD COMPARISON
# ============================================================

def compare_category_performance(
    sales: pd.DataFrame,
    current_start: str,
    current_end: str,
    previous_start: str,
    previous_end: str,
) -> dict:
    """
    Compare category-level revenue, units and orders
    between two periods.

    Results are sorted from largest revenue decline
    to strongest performance. Categories with no previous
    revenue (pct_change is None) sort last, not first —
    an unpriced/undefined change is not "the biggest decline".

    Also exposes, per category:
        - absolute_revenue_change      (current_revenue - previous_revenue)
        - contribution_to_total_change_pct
          (this category's share of the TOTAL revenue delta across all
          categories — NOT the same thing as this category's own growth
          rate. A category can have a huge growth_rate_pct off a tiny
          base and still contribute little to the overall change; a
          category with modest growth off a large base can dominate the
          total change. Both figures are exposed explicitly so nobody
          conflates "highest percentage growth" with "biggest
          contributor".)
    """

    current = _filter_dates(
        sales,
        current_start,
        current_end,
    )

    previous = _filter_dates(
        sales,
        previous_start,
        previous_end,
    )

    if current.empty or previous.empty:
        return {
            "status": "no_data"
        }

    current_grouped = (
        current
        .groupby("category")
        .agg(
            revenue=("revenue", "sum"),
            units=("units", "sum"),
            orders=("order_id", "nunique"),
        )
        .reset_index()
    )

    previous_grouped = (
        previous
        .groupby("category")
        .agg(
            revenue=("revenue", "sum"),
            units=("units", "sum"),
            orders=("order_id", "nunique"),
        )
        .reset_index()
    )

    merged = current_grouped.merge(
        previous_grouped,
        on="category",
        how="outer",
        suffixes=(
            "_current",
            "_previous",
        ),
    ).fillna(0)

    total_current_revenue = float(merged["revenue_current"].sum())
    total_previous_revenue = float(merged["revenue_previous"].sum())
    total_revenue_delta = total_current_revenue - total_previous_revenue

    records = []

    for _, row in merged.iterrows():

        current_revenue = float(row["revenue_current"])
        previous_revenue = float(row["revenue_previous"])
        current_units = float(row["units_current"])
        previous_units = float(row["units_previous"])
        current_orders = float(row["orders_current"])
        previous_orders = float(row["orders_previous"])

        absolute_revenue_change = round(current_revenue - previous_revenue, 2)
        contribution_to_total_change_pct = (
            round(absolute_revenue_change / total_revenue_delta * 100, 2)
            if total_revenue_delta
            else None
        )

        records.append(
            {
                "category": row["category"],
                "current_revenue": round(current_revenue, 2),
                "previous_revenue": round(previous_revenue, 2),
                "revenue_change_pct": _pct_change(current_revenue, previous_revenue),
                "absolute_revenue_change": absolute_revenue_change,
                "contribution_to_total_change_pct": contribution_to_total_change_pct,
                "current_units": int(current_units),
                "previous_units": int(previous_units),
                "units_change_pct": _pct_change(current_units, previous_units),
                "current_orders": int(current_orders),
                "previous_orders": int(previous_orders),
                "orders_change_pct": _pct_change(current_orders, previous_orders),
                "signal_strength": _signal_strength(_pct_change(current_revenue, previous_revenue)),
            }
        )

    # None (undefined change) sorts after every real number, ascending.
    records.sort(
        key=lambda x: (
            x["revenue_change_pct"] is None,
            x["revenue_change_pct"] or 0,
        )
    )

    return {
        "status": "ok",

        "current_period": {
            "start": current_start,
            "end": current_end,
        },

        "previous_period": {
            "start": previous_start,
            "end": previous_end,
        },

        "categories": records,
    }


# ============================================================
# CHANNEL PERFORMANCE
# ============================================================

def get_channel_performance(
    sales: pd.DataFrame,
    start: str,
    end: str,
) -> dict:
    """
    Revenue, units, orders and revenue share by channel.
    """

    window = _filter_dates(
        sales,
        start,
        end,
    )

    if window.empty:
        return {
            "status": "no_data"
        }

    grouped = (
        window
        .groupby("channel")
        .agg(
            revenue=("revenue", "sum"),
            units=("units", "sum"),
            orders=("order_id", "nunique"),
        )
        .round(2)
        .reset_index()
    )

    total_revenue = float(grouped["revenue"].sum())

    grouped["revenue_share_pct"] = (
        (grouped["revenue"] / total_revenue * 100).round(1)
        if total_revenue
        else 0.0
    )

    return {
        "status": "ok",
        "start": start,
        "end": end,
        "channels": grouped.to_dict(
            orient="records"
        ),
    }


# ============================================================
# CHANNEL PERIOD COMPARISON
# ============================================================

def compare_channel_performance(
    sales: pd.DataFrame,
    current_start: str,
    current_end: str,
    previous_start: str,
    previous_end: str,
) -> dict:
    """
    Compare channel revenue, units and orders between
    two periods.
    """

    current = _filter_dates(
        sales,
        current_start,
        current_end,
    )

    previous = _filter_dates(
        sales,
        previous_start,
        previous_end,
    )

    if current.empty or previous.empty:
        return {
            "status": "no_data"
        }

    current_grouped = (
        current
        .groupby("channel")
        .agg(
            revenue=("revenue", "sum"),
            units=("units", "sum"),
            orders=("order_id", "nunique"),
        )
        .reset_index()
    )

    previous_grouped = (
        previous
        .groupby("channel")
        .agg(
            revenue=("revenue", "sum"),
            units=("units", "sum"),
            orders=("order_id", "nunique"),
        )
        .reset_index()
    )

    merged = current_grouped.merge(
        previous_grouped,
        on="channel",
        how="outer",
        suffixes=(
            "_current",
            "_previous",
        ),
    ).fillna(0)

    current_total = float(current_grouped["revenue"].sum())

    records = []

    for _, row in merged.iterrows():

        current_revenue = float(row["revenue_current"])
        previous_revenue = float(row["revenue_previous"])
        current_units = float(row["units_current"])
        previous_units = float(row["units_previous"])
        current_orders = float(row["orders_current"])
        previous_orders = float(row["orders_previous"])

        current_share = (
            round(current_revenue / current_total * 100, 1)
            if current_total
            else 0.0
        )

        records.append(
            {
                "channel": row["channel"],
                "current_revenue": round(current_revenue, 2),
                "previous_revenue": round(previous_revenue, 2),
                "revenue_change_pct": _pct_change(current_revenue, previous_revenue),
                "current_units": int(current_units),
                "previous_units": int(previous_units),
                "units_change_pct": _pct_change(current_units, previous_units),
                "current_orders": int(current_orders),
                "previous_orders": int(previous_orders),
                "orders_change_pct": _pct_change(current_orders, previous_orders),
                "current_revenue_share_pct": current_share,
                "signal_strength": _signal_strength(_pct_change(current_revenue, previous_revenue)),
            }
        )

    records.sort(
        key=lambda x: (
            x["revenue_change_pct"] is None,
            x["revenue_change_pct"] or 0,
        )
    )

    return {
        "status": "ok",

        "current_period": {
            "start": current_start,
            "end": current_end,
        },

        "previous_period": {
            "start": previous_start,
            "end": previous_end,
        },

        "channels": records,
    }


# ============================================================
# INVENTORY METRICS
# ============================================================

def get_inventory_metrics(
    inventory: pd.DataFrame,
    start: str,
    end: str,
    category: str | None = None,
) -> dict:
    """
    Inventory availability and stockout metrics.

    Primary schema (preferred):
        date, product_id, category,
        availability_pct, closing_stock, stockout_flag

    Compatibility fallback (older datasets):
        stock_available in place of availability_pct/closing_stock.

    low_stock_sku_count uses closing_stock (or stock_available as a
    fallback) — a SKU is "low stock" if its average closing stock in
    the window is at or below 15 units.
    """

    window = _filter_dates(
        inventory,
        start,
        end,
    )

    if category:
        window = window[
            window["category"] == category
        ]

    if window.empty:
        return {
            "status": "no_data"
        }

    # --------------------------------------------------------
    # Availability
    # --------------------------------------------------------

    availability_col = (
        "availability_pct"
        if "availability_pct" in window.columns
        else "stock_available"
        if "stock_available" in window.columns
        else None
    )

    if availability_col:
        average_availability = round(float(window[availability_col].mean()), 2)
        availability_by_category = (
            window.groupby("category")[availability_col].mean().round(2).to_dict()
            if category is None
            else None
        )
    else:
        average_availability = None
        availability_by_category = None

    # --------------------------------------------------------
    # Closing stock
    # --------------------------------------------------------

    stock_col = (
        "closing_stock"
        if "closing_stock" in window.columns
        else "stock_available"
        if "stock_available" in window.columns
        else None
    )

    average_closing_stock = (
        round(float(window[stock_col].mean()), 2)
        if stock_col
        else None
    )

    # --------------------------------------------------------
    # Low-stock SKUs (uses the same stock_col resolved above)
    # --------------------------------------------------------

    if stock_col and "product_id" in window.columns:
        sku_avg_stock = window.groupby("product_id")[stock_col].mean()
        low_stock_sku_count = int((sku_avg_stock <= 15).sum())
    else:
        low_stock_sku_count = 0

    # --------------------------------------------------------
    # Stockouts
    # --------------------------------------------------------

    if "stockout_flag" in window.columns:
        stockout_days = int(window["stockout_flag"].sum())
        stockouts_by_category = (
            window.groupby("category")["stockout_flag"].sum().astype(int).to_dict()
            if category is None
            else None
        )
    else:
        stockout_days = 0
        stockouts_by_category = None

    return {
        "status": "ok",
        "start": start,
        "end": end,
        "category_filter": category,
        "average_availability_pct": average_availability,
        "average_closing_stock": average_closing_stock,
        "low_stock_sku_count": low_stock_sku_count,
        "stockout_days": stockout_days,
        "availability_by_category": availability_by_category,
        "stockouts_by_category": stockouts_by_category,
    }


# ============================================================
# INVENTORY PERIOD COMPARISON
# ============================================================

def compare_inventory_performance(
    inventory: pd.DataFrame,
    current_start: str,
    current_end: str,
    previous_start: str,
    previous_end: str,
    category: str = "",
) -> dict:
    """
    Compare inventory conditions between two periods.

    signal_strength here is a MAGNITUDE tier, not a statistical test —
    see _inventory_signal_strength. Callers (agent prompts, UI copy)
    must not describe it as "statistically significant"; the deterministic
    wording below distinguishes "material" evidence from "very small
    observed change" without ever claiming a significance test was run.
    """

    current = get_inventory_metrics(
        inventory,
        current_start,
        current_end,
        category or None,
    )

    previous = get_inventory_metrics(
        inventory,
        previous_start,
        previous_end,
        category or None,
    )

    if (
        current.get("status") != "ok"
        or previous.get("status") != "ok"
    ):
        return {
            "status": "no_data"
        }

    current_availability = current["average_availability_pct"]
    previous_availability = previous["average_availability_pct"]
    current_stock = current["average_closing_stock"]
    previous_stock = previous["average_closing_stock"]
    current_low_stock = current["low_stock_sku_count"]
    previous_low_stock = previous["low_stock_sku_count"]
    current_stockouts = current["stockout_days"]
    previous_stockouts = previous["stockout_days"]

    if current_availability is not None and previous_availability is not None:
        availability_change_points = round(current_availability - previous_availability, 2)
        availability_change_relative = _pct_change(current_availability, previous_availability)
    else:
        availability_change_points = None
        availability_change_relative = None

    stock_change_relative = (
        _pct_change(current_stock, previous_stock)
        if current_stock is not None and previous_stock is not None
        else None
    )

    stockout_days_change = current_stockouts - previous_stockouts
    strength = _inventory_signal_strength(availability_change_points, stockout_days_change)

    # Deterministic, non-statistical wording. NEVER "not statistically
    # significant" (no test was run) — describe magnitude only.
    high_availability_both_periods = (
        current_availability is not None
        and previous_availability is not None
        and current_availability >= 99
        and previous_availability >= 99
    )
    no_stockouts = current_stockouts == 0 and previous_stockouts == 0

    if high_availability_both_periods and no_stockouts:
        evidence_note = (
            "Availability remained above 99% in both periods with zero stockouts recorded; "
            "inventory availability is unlikely to be a material contributor to the revenue change."
        )
    elif strength == "STRONG":
        evidence_note = "The observed change in availability/stockouts is large relative to fixed thresholds."
    elif strength == "MODERATE":
        evidence_note = "The observed change in availability/stockouts is moderate; the evidence is worth noting but not conclusive."
    elif strength == "WEAK":
        evidence_note = "The observed change is very small; the evidence is weak."
    else:
        evidence_note = "There is not enough inventory data to evaluate this signal."

    return {
        "status": "ok",
        "category_filter": category or None,

        "current_period": {
            "start": current_start,
            "end": current_end,
            "average_availability_pct": current_availability,
            "average_closing_stock": current_stock,
            "low_stock_sku_count": current_low_stock,
            "stockout_days": current_stockouts,
        },

        "previous_period": {
            "start": previous_start,
            "end": previous_end,
            "average_availability_pct": previous_availability,
            "average_closing_stock": previous_stock,
            "low_stock_sku_count": previous_low_stock,
            "stockout_days": previous_stockouts,
        },

        "availability_change_pct_points": availability_change_points,
        "availability_change_relative_pct": availability_change_relative,
        "stock_change_pct": stock_change_relative,
        "low_stock_sku_change": current_low_stock - previous_low_stock,
        "stockout_days_change": stockout_days_change,
        "signal_strength": strength,
        "evidence_note": evidence_note,
    }


# ============================================================
# MARKETING METRICS
# ============================================================

def get_marketing_metrics(
    marketing: pd.DataFrame,
    start: str,
    end: str,
    channel: str | None = None,
) -> dict:
    """
    Spend, attributed revenue, impressions,
    clicks/conversions and ROAS by marketing channel.
    """

    window = _filter_dates(
        marketing,
        start,
        end,
    )

    if channel:
        window = window[
            window["channel"] == channel
        ]

    if window.empty:
        return {
            "status": "no_data"
        }

    aggregations = {
        "spend": ("spend", "sum"),
        "attributed_revenue": ("attributed_revenue", "sum"),
        "impressions": ("impressions", "sum"),
    }

    if "clicks" in window.columns:
        aggregations["clicks"] = ("clicks", "sum")

    if "conversions" in window.columns:
        aggregations["conversions"] = ("conversions", "sum")

    grouped = (
        window
        .groupby("channel")
        .agg(**aggregations)
        .round(2)
        .reset_index()
    )

    grouped["roas"] = grouped.apply(
        lambda r: round(r["attributed_revenue"] / r["spend"], 2) if r["spend"] else None,
        axis=1,
    )

    return {
        "status": "ok",
        "start": start,
        "end": end,
        "channels": grouped.to_dict(
            orient="records"
        ),
    }


# ============================================================
# MARKETING PERIOD COMPARISON
# ============================================================

def compare_marketing_performance(
    marketing: pd.DataFrame,
    current_start: str,
    current_end: str,
    previous_start: str,
    previous_end: str,
) -> dict:
    """
    Compare marketing performance between two periods.

    "Deteriorated the most" is an EFFICIENCY question, not a revenue-
    growth question — so channels are ranked by roas_change_pct (most
    negative first), and signal_strength is likewise derived from
    roas_change_pct, never from attributed_revenue_change_pct. A channel
    can grow attributed revenue while still becoming less efficient
    (rising spend outpacing rising revenue), and that is the case this
    function is specifically responsible for surfacing correctly.
    """

    current = _filter_dates(
        marketing,
        current_start,
        current_end,
    )

    previous = _filter_dates(
        marketing,
        previous_start,
        previous_end,
    )

    if current.empty or previous.empty:
        return {
            "status": "no_data"
        }

    current_grouped = (
        current
        .groupby("channel")
        .agg(
            spend=("spend", "sum"),
            attributed_revenue=("attributed_revenue", "sum"),
        )
        .reset_index()
    )

    previous_grouped = (
        previous
        .groupby("channel")
        .agg(
            spend=("spend", "sum"),
            attributed_revenue=("attributed_revenue", "sum"),
        )
        .reset_index()
    )

    merged = current_grouped.merge(
        previous_grouped,
        on="channel",
        how="outer",
        suffixes=("_current", "_previous"),
    ).fillna(0)

    records = []

    for _, row in merged.iterrows():

        current_spend = float(row["spend_current"])
        previous_spend = float(row["spend_previous"])
        current_revenue = float(row["attributed_revenue_current"])
        previous_revenue = float(row["attributed_revenue_previous"])

        current_roas = current_revenue / current_spend if current_spend else None
        previous_roas = previous_revenue / previous_spend if previous_spend else None

        roas_change_pct = (
            _pct_change(current_roas, previous_roas)
            if current_roas is not None and previous_roas is not None
            else None
        )

        records.append(
            {
                "channel": row["channel"],
                "current_spend": round(current_spend, 2),
                "previous_spend": round(previous_spend, 2),
                "spend_change_pct": _pct_change(current_spend, previous_spend),
                "current_attributed_revenue": round(current_revenue, 2),
                "previous_attributed_revenue": round(previous_revenue, 2),
                "attributed_revenue_change_pct": _pct_change(current_revenue, previous_revenue),
                "current_roas": round(current_roas, 2) if current_roas is not None else None,
                "previous_roas": round(previous_roas, 2) if previous_roas is not None else None,
                "roas_change_pct": roas_change_pct,
                # Deterministic evidence strength for DETERIORATION is
                # based on the efficiency signal (ROAS), not revenue growth.
                "signal_strength": _signal_strength(roas_change_pct),
            }
        )

    # Ranked by efficiency deterioration: most negative roas_change_pct
    # first. None (undefined — e.g. zero spend in one period) sorts last,
    # since "deteriorated the most" requires an actual computed change.
    records.sort(
        key=lambda x: (
            x["roas_change_pct"] is None,
            x["roas_change_pct"] if x["roas_change_pct"] is not None else 0,
        )
    )

    most_deteriorated = next(
        (r for r in records if r["roas_change_pct"] is not None and r["roas_change_pct"] < 0),
        None,
    )

    return {
        "status": "ok",

        "current_period": {
            "start": current_start,
            "end": current_end,
        },

        "previous_period": {
            "start": previous_start,
            "end": previous_end,
        },

        "channels": records,
        "most_deteriorated_channel": most_deteriorated,
    }


# ============================================================
# GROWTH CONVENIENCE FUNCTION
# ============================================================

def get_growth(
    sales: pd.DataFrame,
    current_start: str,
    current_end: str,
    previous_start: str,
    previous_end: str,
) -> dict:
    """
    Revenue growth between two periods.
    """

    return compare_periods(
        sales,
        "revenue",
        current_start,
        current_end,
        previous_start,
        previous_end,
    )