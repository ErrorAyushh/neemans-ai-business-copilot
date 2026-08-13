"""
agent/intent_router.py

Deterministic (non-LLM) question-intent classifier for Neeman's AI
Business Copilot.

WHY THIS EXISTS

Previously every question was forced through the same two-period RCA
graph. A question with no natural two-period shape (e.g. "Which month
has the best revenue?") has no explicit periods to resolve, but the only
tools available to the model required two periods — so the model was
forced to invent one, producing a hallucinated period (e.g.
"January 2025").

This router runs BEFORE any LLM call. It never calls the model and never
touches analytics directly. It only classifies the question so that
agent/rca_agent.py's handle_question() can send it down the correct
deterministic path:

(a) SUPPORTED_INTENTS (TWO_PERIOD_RCA / DIMENSION_RCA /
    DIMENSION_RANKING / PERIOD_COMPARISON) -> the existing LangGraph
    + Sarvam tool-calling RCA graph (run_investigation()), unchanged.

(b) DIRECT_ANALYTICS_INTENTS (CROSS_PERIOD_RANKING /
    SINGLE_PERIOD_LOOKUP / TIME_SERIES_TREND) -> direct deterministic
    pandas analytics (analytics/metrics.rank_months_by_revenue /
    get_revenue_trend / resolve_named_period + get_sales_metrics),
    bypassing the two-period RCA graph entirely so the model is never
    given the opportunity to invent a period to force-fit the question.

Both branches are implemented in agent/rca_agent.py's handle_question() —
this module is a pure classifier and has no side effects.
"""

from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    TWO_PERIOD_RCA = "TWO_PERIOD_RCA"
    DIMENSION_RCA = "DIMENSION_RCA"
    DIMENSION_RANKING = "DIMENSION_RANKING"
    PERIOD_COMPARISON = "PERIOD_COMPARISON"
    CROSS_PERIOD_RANKING = "CROSS_PERIOD_RANKING"
    SINGLE_PERIOD_LOOKUP = "SINGLE_PERIOD_LOOKUP"
    TIME_SERIES_TREND = "TIME_SERIES_TREND"


# Intents handled by the existing agent/tools.py + rca_agent.py LangGraph
# tool-calling graph — proceed through run_investigation() exactly as
# before, unchanged.
SUPPORTED_INTENTS = frozenset({
    Intent.TWO_PERIOD_RCA,
    Intent.DIMENSION_RCA,
    Intent.DIMENSION_RANKING,
    Intent.PERIOD_COMPARISON,
})

# Intents handled by direct deterministic pandas analytics calls — NEVER
# routed through the two-period RCA graph, so the LLM never gets a chance
# to invent a period for a question shape it wasn't asked.
DIRECT_ANALYTICS_INTENTS = frozenset({
    Intent.CROSS_PERIOD_RANKING,
    Intent.SINGLE_PERIOD_LOOKUP,
    Intent.TIME_SERIES_TREND,
})


# --------------------------------------------------------------------
# Pattern definitions — checked in order from most to least specific.
# First match wins; TWO_PERIOD_RCA is the fallback.
# --------------------------------------------------------------------

_CROSS_PERIOD_RANKING_PATTERNS = (
    r"\bwhich month\b",
    r"\bbest (revenue |sales )?month\b",
    r"\bworst (revenue |sales )?month\b",
    r"\btop month\b",
    r"\brank(ing)? (the )?months\b",
    r"\bhighest revenue\b.*\bmonth\b",
    r"\blowest revenue\b.*\bmonth\b",
)

# Trend must win over a two-period marker ONLY when the user is asking
# for a trend/series shape ("show revenue trend from April to June" is
# still a trend, not a two-point comparison) — checked before dimension/
# two-period classification.
_TREND_PATTERNS = (
    r"\btrend\b",
    r"\bover time\b",
    r"\btime.?series\b",
    r"\bhow has\b.*\bchanged over\b",
    r"\bmonthly revenue\b",
)

_SINGLE_LOOKUP_PATTERNS = (
    r"^what was\b",
    r"^what is\b",
    r"^how many\b",
    r"^how much\b",
)

# A single-period lookup question must NOT also contain a two-period
# marker ("from X to Y", "between X and Y") — those stay TWO_PERIOD_RCA.
_TWO_PERIOD_MARKERS = (
    r"\bfrom\b.*\bto\b",
    r"\bbetween\b.*\band\b",
    r"\bvs\.?\b",
    r"\bversus\b",
)

_EXPLICIT_COMPARE_PATTERNS = (
    r"^compare\b",
    r"\bcompare\b.*\band\b",
)

_RANKING_MARKERS = (
    r"\bwhich (category|channel)\b",
    r"\bmost\b",
    r"\bdeteriorated the most\b",
    r"\bbest performing\b",
    r"\bworst performing\b",
)

_DIMENSION_KEYWORDS = (
    "category", "categories", "channel", "channels",
    "inventory", "stock", "stockout", "availability",
    "marketing", "roas", "campaign", "advertising",
)


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def classify_intent(question: str) -> Intent:
    """Deterministic, regex-based classification — no LLM call, no
    analytics call. Order matters: narrower patterns are checked before
    the general TWO_PERIOD_RCA fallback.

    1. CROSS_PERIOD_RANKING  ("which month has the best revenue")
    2. TIME_SERIES_TREND     ("show revenue trend", "...from April to June")
    3. SINGLE_PERIOD_LOOKUP  ("what was revenue in June 2026") — only
       if there is no two-period marker in the question.
    4. PERIOD_COMPARISON     ("compare April 2026 and June 2026") — only
       if it isn't actually a revenue-change/RCA question in disguise.
    5. DIMENSION_RANKING / DIMENSION_RCA — a dimension keyword plus/minus
       a ranking marker ("which category contributed most" vs "is
       inventory contributing").
    6. TWO_PERIOD_RCA — the fallback for general "why did revenue change"
       questions.
    """
    q = re.sub(r"\s+", " ", (question or "").lower()).strip()

    if _matches_any(_CROSS_PERIOD_RANKING_PATTERNS, q):
        return Intent.CROSS_PERIOD_RANKING

    if _matches_any(_TREND_PATTERNS, q):
        return Intent.TIME_SERIES_TREND

    has_two_period_marker = _matches_any(_TWO_PERIOD_MARKERS, q)

    if _matches_any(_SINGLE_LOOKUP_PATTERNS, q) and not has_two_period_marker:
        return Intent.SINGLE_PERIOD_LOOKUP

    if _matches_any(_EXPLICIT_COMPARE_PATTERNS, q) and not any(
        kw in q for kw in ("revenue change", "why did", "contributing")
    ):
        return Intent.PERIOD_COMPARISON

    dimension_hit = any(kw in q for kw in _DIMENSION_KEYWORDS)
    ranking_hit = _matches_any(_RANKING_MARKERS, q)

    if dimension_hit and ranking_hit:
        return Intent.DIMENSION_RANKING
    if dimension_hit and not ranking_hit:
        return Intent.DIMENSION_RCA

    return Intent.TWO_PERIOD_RCA


def unsupported_intent_message(intent: Intent) -> str:
    """Kept for defensive use only (e.g. an intent value that somehow
    isn't in either SUPPORTED_INTENTS or DIRECT_ANALYTICS_INTENTS). Under
    normal operation every Intent value is handled by one of the two
    branches in agent/rca_agent.py's handle_question(), so this should
    never actually be shown to a user."""
    label = intent.value.replace("_", " ").title()
    return (
        f"This looks like a {label} question, but no handler is registered for it. "
        "This is an internal routing gap — please report it."
    )