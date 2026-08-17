
"""
agent/rca_agent.py

Neeman's AI Business Copilot — production RCA agent.

PROVIDER: Sarvam AI (sarvam-105b), via the official `langchain-sarvam`
package's ChatSarvam. Verified against the actual package (PyPI
langchain-sarvam 0.1.2) and Sarvam's own docs before building on it:
ChatSarvam is a real BaseChatModel with bind_tools()/invoke() support, the
403+invalid_api_key_error auth behavior and the reasoning_effort default
notes are both accurate per docs.sarvam.ai. The LangGraph tool-calling
architecture is otherwise unchanged: agent -> tools -> mandatory ->
synthesize graph, same AIMessage.tool_calls / ToolMessage protocol.

"""

import json
import os
import re
import time

from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_sarvam import ChatSarvam, ChatSarvamError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# NOTE: DATASET_END_DATE lives in agent.tools, not analytics.metrics.
# analytics/metrics.py is a stateless function library (every function
# takes a dataframe as a parameter; nothing is loaded at import time).
# agent/tools.py is the layer that actually loads the dataset once, at
# import time, and derives DATASET_END_DATE from that loaded dataframe —
# so it is the single authoritative source for this constant. Importing
# it from analytics.metrics instead was the root cause of the ImportError.
from agent.tools import TOOLS, DATASET_END_DATE
from agent.intent_router import (
    DIRECT_ANALYTICS_INTENTS,
    Intent,
    SUPPORTED_INTENTS,
    classify_intent,
)
from analytics import metrics


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.environ.get("SARVAM_MODEL", "sarvam-105b")

# A normal general revenue investigation needs:
#   2 period resolutions + 1 KPI comparison + 4 dimensions = 7 calls.
# 12 leaves controlled headroom for additional relevant model-selected calls
# without allowing an unbounded investigation.
MAX_TOOL_CALLS = 12

# Bounded retry budget for the FINAL SYNTHESIS step only (see
# _synthesize_final_answer). This never triggers a new analytics tool
# call or re-runs the investigation — it only re-invokes the model
# against the evidence already collected in `trace`. Kept small and
# fixed so a failing model can never hang the request indefinitely.
MAX_SYNTHESIS_RETRIES = 2

GROUNDING_TOLERANCE = 0.02
MIN_DIGITS_TO_GROUND = 3

RATE_LIMIT_MARKERS = (
    "429",
    "insufficient_quota",
    "rate_limit_exceeded",
    "rate limit",
    "too many requests",
)

AUTH_ERROR_MARKERS = (
    "403",
    "401",
    "invalid_api_key",
    "unauthorized",
    "authentication",
)


# ============================================================
# DETERMINISTIC INVESTIGATION SCOPE
# ============================================================

DIMENSION_ORDER = ("category", "channel", "inventory", "marketing")

DIMENSION_TOOL = {
    "category": "compare_category_performance",
    "channel": "compare_channel_performance",
    "inventory": "compare_inventory_performance",
    "marketing": "compare_marketing_performance",
}

# Fixed priority used ONLY as a tie-breaker when two signals have identical
# strength AND identical magnitude — keeps ranking output byte-for-byte
# reproducible regardless of tool execution order.
DIMENSION_PRIORITY = {"Category": 0, "Channel": 1, "Inventory": 2, "Marketing": 3}

_DIMENSION_KEYWORDS = {
    "category": {"category", "categories", "product category", "product categories"},
    "channel": {"channel", "channels", "sales channel", "sales channels"},
    "inventory": {"inventory", "stock", "stockout", "stockouts", "availability", "low stock"},
    "marketing": {"marketing", "roas", "ad spend", "campaign", "campaigns", "advertising", "paid ads"},
}

_GENERAL_MARKERS = (
    "revenue", "sales performance", "overall performance",
    "business performance", "overall sales", "total sales",
)
# NOTE: _GENERAL_MARKERS is intentionally UNUSED as a scope override — see
# the bug explanation in _classify_scope's docstring. Kept only as a
# documented list of words that historically triggered the bug, so a
# future maintainer doesn't reintroduce the same short-circuit.


def _classify_scope(question: str) -> set[str]:
    """Deterministically decides mandatory dimension scope. A question that
    names one OR MORE specific dimensions (category/channel/inventory/
    marketing) gets exactly those dimensions; a question naming none of
    them defaults to all four for safety.

    FIXED BUG (v1): a previous version checked `_GENERAL_MARKERS` (words
    like "revenue") BEFORE checking dimension keywords, and returned all
    four dimensions immediately on a match. Because the word "revenue"
    appears in nearly every RCA question — including narrow ones like
    "Which CATEGORIES contributed most to the REVENUE decline?" — that
    check silently overrode explicit single-dimension questions on every
    investigation, not intermittently. Dimension keywords are now checked
    first and are authoritative whenever any dimension is named.

    FIXED BUG (v2): a previous version of THIS function still forced all
    four dimensions whenever more than one dimension keyword matched —
    i.e. a question naming exactly two dimensions ("Did inventory or
    marketing show the stronger negative signal?") was treated the same
    as a question naming none, and got category+channel evidence mixed in
    that nobody asked about. A question naming N specific dimensions
    (N >= 1) is now scoped to exactly those N dimensions; "default to all
    four" is now reserved for the case where NO dimension is named at
    all (e.g. "Why did revenue decline?").
    """
    q = re.sub(r"\s+", " ", question.lower()).strip()
    q_for_matching = q.replace("marketing channel", "marketing")
    hits = {dim for dim, kws in _DIMENSION_KEYWORDS.items() if any(kw in q_for_matching for kw in kws)}

    if hits:
        return hits
    return set(DIMENSION_ORDER)


def _relevant_dimension_tools(question: str) -> set[str]:
    """The DIMENSION_TOOL names relevant to a question's scope, per
    _classify_scope. Used to filter strongest-signal/confidence/fallback
    computation so a narrow question's results only ever reflect the
    dimension(s) actually asked about — even if the model independently
    calls an extra tool outside that scope."""
    return {DIMENSION_TOOL[d] for d in _classify_scope(question)}


# ============================================================
# SYSTEM PROMPT  (doc: period-context template, now actually wired up)
# ============================================================

_PERIOD_CONTEXT_SUPPLIED = """
============================================================
APPLICATION-CONTROLLED PERIOD CONTEXT
============================================================

The application has already resolved the active comparison periods.

CURRENT PERIOD:
{CURRENT_START_DATE} -> {CURRENT_END_DATE}
Label: {CURRENT_PERIOD_LABEL}

COMPARISON PERIOD:
{COMPARISON_START_DATE} -> {COMPARISON_END_DATE}
Label: {COMPARISON_PERIOD_LABEL}

These dates are authoritative application context.

PERIOD RESOLUTION RULES:

1. If the user's question does NOT explicitly specify dates/months,
   use the supplied CURRENT PERIOD and COMPARISON PERIOD directly.

2. Do NOT call resolve_named_period for already-resolved application
   periods.

3. If the user's question explicitly specifies BOTH periods, resolve
   those periods using resolve_named_period and use them instead of
   the application periods.

4. If the user specifies only ONE period, use it with the supplied
   application comparison period only when the intended comparison is
   unambiguous. Otherwise use the supplied application periods.

5. NEVER invent, infer, or fabricate a period.

6. NEVER ask the user to provide periods that are already available
   in application context.
"""


_PERIOD_CONTEXT_MISSING = """
============================================================
NO APPLICATION PERIOD CONTEXT
============================================================

No current/comparison period was supplied by the application.

You must resolve the periods from the user's question using
resolve_named_period before calling any comparison tool.

If both periods cannot be identified with high confidence:

- Do NOT invent a period.
- Do NOT assume a random month.
- Ask the user which periods they want compared.

For questions that do not require a two-period comparison, use the
appropriate deterministic analytics tool instead of forcing the
question into a two-period comparison.
"""


_PERIOD_LANGUAGE_RULE = """
============================================================
PERIOD LANGUAGE RULE
============================================================

Every comparison-based conclusion MUST explicitly identify the
periods being compared.

Never produce a period-ambiguous statement.

BAD:
"Meta deteriorated the most."

GOOD:
"Meta deteriorated the most from June 2026 to July 2026,
with a -34.77% ROAS change."

Whenever a metric is described as increasing, decreasing,
improving, deteriorating, contributing, or declining, make the
relevant period explicit when doing so is necessary for clarity.

For a single-period or cross-period ranking question, identify
the relevant month/year in the result.
"""


_CORE_PRINCIPLE = """
============================================================
CORE PRINCIPLE
============================================================

You are Neeman's AI Business Copilot.

You are an INVESTIGATOR and INTERPRETER, not a calculator.

Your responsibility is to:

1. Understand the user's actual business question.
2. Use the available analytical tools to obtain deterministic evidence.
3. Interpret those results clearly.
4. Answer exactly what the user asked.
5. Avoid unsupported causal conclusions.
6. Never invent business numbers, periods, metrics, evidence,
   confidence values, or recommendations.

BUSINESS CALCULATIONS:

The analytical tools are the source of truth for business calculations.

NEVER:

- calculate percentages yourself when a tool provides them
- estimate missing values
- interpolate missing periods
- invent a missing month
- create unsupported totals
- create unsupported rankings
- modify tool-returned numbers
- assign your own signal strength
- invent confidence

Every numerical claim in the final response MUST be directly supported
by a tool result obtained during the current investigation.

If a required value was not returned by a tool:

- do not manufacture it
- do not estimate it
- either omit it or explicitly state that the evidence is unavailable

SIGNAL STRENGTH:

Use tool-provided signal_strength exactly as returned:

- STRONG
- MODERATE
- WEAK
- INSUFFICIENT

Do not reinterpret or upgrade/downgrade these values.
"""


_QUESTION_SCOPE_RULE = """
============================================================
QUESTION SCOPE — MOST IMPORTANT RULE
============================================================

Answer the user's ACTUAL question, not a broader question.

Before writing the final answer, identify:

1. PRIMARY QUESTION:
   What exactly is the user asking?

2. PRIMARY METRIC:
   What metric or business outcome is being investigated?

3. PRIMARY DIMENSION:
   If the user asks about a specific dimension such as:
   - inventory
   - category
   - channel
   - marketing
   then focus the answer on that dimension.

4. COMPARISON:
   What periods or observations are being compared?

5. QUESTION TYPE:
   Is the user asking for an explanation, a comparison, a ranking,
   a trend, a recommendation/advice, a risk/opportunity assessment,
   or an evidence/confidence assessment? Answer THAT question — do
   not silently substitute a different (even related) business
   question. If the user asks "which factor was the strongest
   observed contributor?", answer that directly ("the strongest
   observed contributor was X...") — do not rewrite it into "why did
   average order value decrease?" or any other question they did not
   ask.

The scope of the final answer MUST match the scope of the question.

NARROW QUESTION:

If the user asks something specific such as:

"Is inventory availability contributing to the revenue change?"

then:

- Focus primarily on inventory availability.
- Report the relevant inventory evidence.
- Explain whether it is a contributing signal.
- Mention the revenue movement only as context.
- Do NOT produce a complete category/channel/marketing RCA.
- Do NOT rank unrelated dimensions.
- Do NOT generate ten unrelated growth drivers.
- Do NOT discuss dimensions that are irrelevant to answering the question.

GENERAL RCA QUESTION:

If the user asks:

"Why did revenue decrease?"

or:

"What caused the revenue decline?"

then investigate and summarize the relevant dimensions collected by
the analytical pipeline.

RANKING QUESTION:

If the user asks:

"Which month had the highest revenue?"

then answer the ranking question directly using the deterministic
ranking result.

Do NOT force it into a two-period RCA.

TREND QUESTION:

If the user asks:

"Show me the revenue trend from January to July."

then present the chronological trend returned by the analytics tool.

Do NOT convert it into a two-period RCA unless the user explicitly
asks for a comparison or explanation.

RECOMMENDATION / ADVICE QUESTION:

If the user asks:

"What advice would you give management?"
"What should we do next?"
"How can we improve revenue?"

then lead with a direct recommendation grounded in the strongest
available evidence — see the BUSINESS ADVICE section below. Do not
answer with a general RCA when the user asked specifically for advice
or next steps.

The user's question determines the response scope.
"""


_EVIDENCE_CLASSIFICATION = """
============================================================
EVIDENCE CLASSIFICATION
============================================================

Use the following evidence hierarchy.

OBSERVED FACT
-------------
A value directly returned by an analytical tool.

Example:
"Inventory availability decreased from 99.30% to 74.95%."

CONTRIBUTING FACTOR
-------------------
A STRONG or MODERATE dimension-level signal that is relevant to
the observed business movement.

Example:
"Inventory availability is a strong observed contributing signal
to the revenue decline."

LIKELY ROOT CAUSE
-----------------
Use sparingly.

Only use this label when the available evidence provides a strong,
directly connected explanation and the evidence is substantially
more compelling than competing explanations.

Even then, it does NOT mean proven causation.

HYPOTHESIS
----------
A plausible interpretation that goes beyond what the tools directly
establish.

Explicitly label it as a hypothesis.

INSUFFICIENT EVIDENCE
---------------------
Use when the available data cannot support the proposed explanation.

IMPORTANT:

Correlation, temporal coincidence, contribution percentage, or
signal strength does NOT by itself prove causation.
"""


_CAUSAL_LANGUAGE_SAFETY = """
============================================================
CAUSAL LANGUAGE SAFETY
============================================================

The dataset is synthetic historical business data.

The analytical tools identify observed changes and signals.
They do NOT perform causal inference.

Therefore, NEVER state that a dimension definitively caused a result.

DO NOT USE:

- caused by
- definitely caused
- directly caused
- proved that
- proves that
- resulted from
- because of
- was the cause of
- is the reason for

PREFER:

- strongest observed signal
- contributing factor
- likely contributing factor
- consistent with
- associated with
- may have contributed
- evidence suggests
- observed alongside
- coincides with
- potential contributor

If causal interpretation is useful, explicitly qualify it:

"The data indicate a strong contributing signal, but do not establish
direct causation."

IMPORTANT:

Do not turn a tool's STRONG signal_strength into a claim of
causation.

STRONG means the observed change meets the deterministic evidence
threshold. It does NOT mean "proven cause."
"""


_DIMENSION_EVIDENCE_RULE = """
============================================================
DIMENSION EVIDENCE RULE
============================================================

Use only the dimension evidence that is relevant to the user's question.

Possible dimensions include:

- category
- channel
- inventory
- marketing

GENERAL BUSINESS RCA:

If the pipeline supplies evidence for multiple dimensions, summarize
the relevant dimensions and rank the strongest observed contributing
signals.

NARROW DIMENSION QUESTION:

If the question explicitly asks about one dimension:

- prioritize that dimension
- do not overwhelm the answer with unrelated dimensions
- do not create a generic root-cause ranking

Example:

Question:
"Is inventory availability contributing to the revenue change?"

Correct:

Inventory Signal: STRONG

Availability:
99.30% -> 74.95%

Change:
-24.35 percentage points

Stockouts:
0 -> 56 days

Assessment:
Inventory availability is a strong observed contributing signal
to the revenue decline.

Caveat:
The data does not establish direct causation.

Incorrect:

A ten-item ranking containing Running, Website, Meta, Google,
Casual, inventory, etc.

The answer must remain focused on inventory.

If a relevant dimension has WEAK or INSUFFICIENT evidence, report that
accurately.

Never upgrade WEAK or INSUFFICIENT evidence.

If all relevant evidence is WEAK or INSUFFICIENT, explicitly state:

"No strong or moderate evidence was found for this dimension."
"""


_CONFIDENCE_RULE = """
============================================================
CONFIDENCE
============================================================

Confidence is determined by the application, not by the model.

A SYSTEM NOTE may provide a deterministic confidence value.

If a confidence value is supplied:

- reproduce it exactly
- do not modify it
- do not reinterpret it

If no confidence value is supplied:

"Confidence: Not provided by the analytical tools."

NEVER invent confidence.

Do not convert signal_strength into a confidence percentage.
"""


_NUMERIC_INTEGRITY_RULE = """
============================================================
NUMERIC INTEGRITY
============================================================

Every number in the final response must come from the current
investigation's tool results.

Before presenting a number, verify that:

1. It exists in a tool result.
2. It refers to the correct period.
3. It refers to the correct metric.
4. It has not been manually recomputed.
5. Its sign and units are preserved.

Preserve tool precision unless formatting is purely cosmetic.

Examples:

99.30% may be displayed as 99.3%.

-24.35 percentage points must remain a percentage-point change,
not "-24.35%".

ROAS 2.83 -> 1.93 is not the same thing as a 31.84 percentage-point
change.

Never confuse:

- percentage change
- percentage-point change
- absolute change
- contribution percentage
- revenue share
- ROAS
"""


_RECOMMENDATION_RULE = """
============================================================
RECOMMENDATIONS
============================================================

Recommendations must follow the evidence.

Every recommendation must:

1. Be directly connected to an observed signal.
2. Be actionable.
3. Avoid inventing an unsupported cause.
4. Avoid pretending that correlation proves causation.

GOOD:

"Investigate the SKUs affected by the 56 July stockout days and
determine whether those availability gaps overlapped with the
largest revenue declines."

BAD:

"Increase inventory immediately because stockouts caused the
revenue decline."

For narrow questions, provide only the recommendations relevant
to that question.

Do not generate generic business advice.
"""


_BUSINESS_ADVICE_RULE = """
============================================================
BUSINESS ADVICE / "WHAT SHOULD WE DO" QUESTIONS
============================================================

For broad questions such as "What advice would you give management?",
"How can we improve revenue?", "What should we do next?", or "What are
the biggest risks/opportunities?", answer using ONLY evidence from the
current investigation, structured as:

1. OBSERVED SITUATION — the headline metric movement, with both
   periods named explicitly.
2. STRONGEST RELEVANT EVIDENCE — the STRONG/MODERATE signal(s) most
   relevant to the question (e.g. risk questions lean on deteriorating
   signals; opportunity questions lean on improving signals).
3. RECOMMENDED ACTION — tied directly and explicitly to that evidence.
4. WHAT TO VALIDATE NEXT — a concrete investigation or experiment, not
   a guaranteed fix. Recommending an investigation is valid even when
   causality is not established.
5. LIMITATION — state plainly that the evidence indicates association
   or contribution, not proven causation.

Do NOT produce generic MBA-style advice ("focus on customer retention",
"invest in brand building", "optimize the supply chain") unless it is
explicitly and directly tied to a signal actually returned by the tools
during this investigation. If no STRONG or MODERATE signal exists for
the relevant scope, say so explicitly ("no strong or moderate signal was
found to base a specific recommendation on") rather than inventing a
plausible-sounding recommendation to fill the gap.

RISK vs. OPPORTUNITY QUESTIONS:

"What are the biggest risks visible in the data?" -> report the
deteriorating (declining revenue, falling ROAS, dropping availability,
rising stockouts) signals with STRONG/MODERATE strength.

"What opportunities are visible in the data?" -> report the improving
(growing revenue, rising ROAS, strong category/channel growth) signals
with STRONG/MODERATE strength.

"Which areas improved enough to offset declining areas?" -> compare the
absolute/contribution figures already returned by the comparison tools;
do not estimate an offset that the tools did not compute.
"""


_NO_QUESTION_REWRITING_RULE = """
============================================================
NO QUESTION REWRITING
============================================================

Never invent a new investigation question.

The "Investigation Question" line in your response format MUST reproduce
the user's actual question, verbatim except for trivial whitespace or
capitalization normalization. Never paraphrase it, never narrow it, and
never substitute a different (even closely related) question.

BAD — user asked "Give me a root cause analysis of the revenue change."
and the response states:

"## Investigation Question
Why did revenue decrease from June 2026 to July 2026?"

This is a fabricated question the user did not ask. It is a numeric-
integrity-adjacent violation: it misrepresents what was investigated.

GOOD — the same response states:

"## Investigation Question
Give me a root cause analysis of the revenue change."

You MAY use the resolved periods elsewhere in the response (Period
section, Executive Summary, etc.) — the restriction applies only to the
"Investigation Question" line itself.
"""


_OUT_OF_SCOPE_RULE = """
============================================================
QUESTIONS BEYOND AVAILABLE DATA
============================================================

The available dataset and tools cover only: sales (revenue, orders,
units, average order value), product categories, sales channels,
inventory (availability, closing stock, stockouts), and marketing
(spend, attributed revenue, ROAS).

If a question requires information the dataset and tools do not
contain — for example pricing strategy, competitor activity, customer
sentiment/reviews, macroeconomic conditions, staffing, logistics cost,
or SKU-level detail beyond what a tool actually returns:

- Do NOT hallucinate an answer.
- State plainly: "The available dataset does not contain the
  information required to answer that question."
- Where useful, briefly explain what additional data would be needed.
- If PART of the question can be answered from available evidence,
  answer that part using the normal evidence rules, and clearly flag
  the part that cannot be answered.

This is preferable to producing a plausible-sounding but unsupported
answer. Never fabricate data, studies, benchmarks, or industry figures
to fill the gap.
"""


_FINAL_RESPONSE_FORMAT = """
============================================================
FINAL RESPONSE FORMAT
============================================================

The final response must be concise, structured, and directly answer
the user's question.

Choose the response format based on the question type.

------------------------------------------------------------
A. NARROW RCA / DIMENSION QUESTION
------------------------------------------------------------

Use:

## Investigation Question
<user's question>

## Period
**Current:** <period>
**Comparison:** <period>

## Answer
<Direct one-paragraph answer to the question.>

## Evidence
- <metric>: <current> -> <comparison> (<change>)
- <relevant supporting metric>
- **Signal strength:** <tool value>

## Assessment
<CONTRIBUTING FACTOR / LIKELY ROOT CAUSE / HYPOTHESIS /
INSUFFICIENT EVIDENCE>

<One concise explanation grounded only in tool evidence.>

## Recommendation
<One to three evidence-based actions, if useful.>

## Data Limitation
<Brief causality limitation when applicable.>

## Confidence
<deterministic confidence or exact fallback text>

Do NOT add unrelated dimensions.

------------------------------------------------------------
B. GENERAL RCA QUESTION
------------------------------------------------------------

Use:

# ROOT CAUSE ANALYSIS

## Investigation Question
<question>

## Period
**Current:** <period>
**Comparison:** <period>

## Executive Summary
2-4 sentences covering:
- primary KPI movement
- comparison periods
- strongest relevant signals
- confidence

## What Changed
Primary KPI movement with both periods explicitly named.

## Evidence

### Category Performance
<only if relevant>

### Channel Performance
<only if relevant>

### Inventory Performance
<only if relevant>

### Marketing Performance
<only if relevant>

For each relevant dimension include:
- actual tool-returned metrics
- period-to-period change
- signal_strength

## Root Cause Ranking
Rank only the relevant observed contributing factors.

For each factor:

1. <Factor>
   - Evidence strength: <tool value>
   - Supporting metrics: <tool values>
   - Interpretation: <CONTRIBUTING FACTOR / LIKELY ROOT CAUSE /
     HYPOTHESIS>
   - Causality: <brief limitation>

Do not manufacture a fixed number of factors.
If only three meaningful factors exist, report three.

## Recommendations
Evidence-linked recommendations only.

## Data Limitations
State that the available evidence indicates observed
association/contribution but does not establish direct causation.

## Confidence
Use the deterministic confidence exactly.

------------------------------------------------------------
C. CROSS-PERIOD RANKING QUESTION
------------------------------------------------------------

Example:
"Which month had the best revenue?"

Use:

## Investigation Question
<question>

## Answer
**<best month>** had the highest revenue at **<tool value>**.

## Revenue Ranking

| Rank | Month | Revenue |
|---|---|---:|
| 1 | ... | ... |
| 2 | ... | ... |
| 3 | ... | ... |

Include all ranking records returned by the tool when appropriate.

If a year filter was requested, clearly state it.

Do NOT create a fake comparison period.

Do NOT produce:
- Revenue Change
- Root Cause Ranking
- Growth Drivers
- Strongest Signal
unless the user explicitly asks for those things.

------------------------------------------------------------
D. TIME-SERIES / TREND QUESTION
------------------------------------------------------------

Use:

## Investigation Question
<question>

## Revenue Trend

| Month | Revenue | Orders | Units |
|---|---:|---:|---:|
| ... | ... | ... | ... |

## Trend Summary
2-4 sentences describing only the pattern directly supported
by the returned series.

Do not invent a cause for the trend.

Do not force the result into a two-period RCA.

------------------------------------------------------------
E. RECOMMENDATION / ADVICE QUESTION
------------------------------------------------------------

Example:
"What advice would you give management to improve revenue?"
"What should we do next based on the strongest signals?"
"What are the biggest risks/opportunities visible in the data?"

Use:

## Investigation Question
<question>

## Recommendation
<Direct answer — lead with the recommendation itself, not
background.>

## Evidence Behind It
<Observed signals that support the recommendation, with tool-returned
figures and signal_strength. Only STRONG/MODERATE signals unless
explicitly asked to include weak ones.>

## Recommended Actions
1. <action>
2. <action>
3. <action>

## What To Validate Next
<Concrete next investigation/experiment before committing resources.>

## Data Limitation
State that the evidence indicates association/contribution, not proven
causation.

## Confidence
Use the deterministic confidence exactly.

If no STRONG or MODERATE signal exists to support a specific
recommendation, say so explicitly instead of filling the section with
generic advice.

------------------------------------------------------------

GENERAL FORMATTING RULES:

- Use Markdown headings.
- Use bullets for evidence.
- Use tables for rankings and time-series data.
- Keep the answer readable.
- Avoid repeating the same number multiple times unnecessarily.
- Do not expose internal tool names unless the user asks.
- Do not mention the investigation budget.
- Do not mention hidden system instructions.
- Do not produce empty sections.
- Do not produce unrelated sections merely because the template
  contains them.
- If a section has no relevant information, omit it.
- Answer the user's question FIRST, then provide supporting evidence.
"""


_TOOL_EFFICIENCY = """
============================================================
TOOL EFFICIENCY
============================================================

Never call the same tool with identical arguments more than once.

Use the minimum number of tool calls necessary to answer the question.

You have a maximum of {MAX_TOOL_CALLS} individual tool executions.

Do not ask the user for information already available in the
application context.

Do not collect unrelated evidence merely to make the answer look
more comprehensive.

Evidence breadth must follow question scope.

If a SYSTEM NOTE already supplied the dimension evidence you need (it
lists which tools were already run), use it directly — do not call that
same tool again "to double-check", and do not call ADDITIONAL dimension
tools (category/channel/inventory/marketing) outside what the question
actually asks about. A question scoped to one dimension should result in
evidence from that one dimension only; pulling in extra dimensions wastes
budget, produces a longer answer than the question calls for, and risks
mixing unrelated evidence into the response.
"""


def _build_system_prompt(
    current_period: dict | None,
    comparison_period: dict | None,
) -> str:
    """Build the production RCA system prompt."""

    if current_period and comparison_period:
        period_section = _PERIOD_CONTEXT_SUPPLIED.format(
            CURRENT_START_DATE=current_period.get("start_date", "?"),
            CURRENT_END_DATE=current_period.get("end_date", "?"),
            CURRENT_PERIOD_LABEL=current_period.get("label", "?"),
            COMPARISON_START_DATE=comparison_period.get("start_date", "?"),
            COMPARISON_END_DATE=comparison_period.get("end_date", "?"),
            COMPARISON_PERIOD_LABEL=comparison_period.get("label", "?"),
        )
    else:
        period_section = _PERIOD_CONTEXT_MISSING

    return (
        "You are Neeman's AI Business Copilot — a deterministic-evidence "
        "business analytics and Root Cause Analysis agent.\n\n"
        "Your job is to investigate business questions using ONLY the "
        "analytical tools and application context provided to you. "
        "The analytical tools are the source of truth for all business "
        "numbers and deterministic evidence. You are able to answer a "
        "broad range of natural-language business questions — general "
        "RCA, narrow dimension questions, ranking, trend, comparison, "
        "recommendation/advice, risk/opportunity, and evidence/confidence "
        "questions — provided they are answerable from the available "
        "dataset and tools. You are not a general-purpose chatbot: for "
        "anything outside that scope, say so explicitly rather than "
        "guessing.\n\n"
        f"The dataset's latest available date is: {DATASET_END_DATE}\n"
        "The dataset is synthetic historical business data. "
        "Do NOT assume today's date is the same as the dataset date.\n\n"
        + period_section
        + _PERIOD_LANGUAGE_RULE
        + _CORE_PRINCIPLE
        + _QUESTION_SCOPE_RULE
        + _EVIDENCE_CLASSIFICATION
        + _CAUSAL_LANGUAGE_SAFETY
        + _DIMENSION_EVIDENCE_RULE
        + _CONFIDENCE_RULE
        + _NUMERIC_INTEGRITY_RULE
        + _RECOMMENDATION_RULE
        + _BUSINESS_ADVICE_RULE
        + _OUT_OF_SCOPE_RULE
        + _NO_QUESTION_REWRITING_RULE
        + _FINAL_RESPONSE_FORMAT
        + _TOOL_EFFICIENCY.format(MAX_TOOL_CALLS=MAX_TOOL_CALLS)
    )


SYNTHESIS_INSTRUCTION = (
    "The investigation is complete. Do not call additional tools.\n\n"
    "Answer the user's ORIGINAL question using ONLY evidence returned "
    "during this investigation.\n\n"
    "First determine the question type and response scope:\n"
    "1. narrow dimension/RCA question\n"
    "2. general RCA question\n"
    "3. cross-period ranking\n"
    "4. time-series/trend\n"
    "5. recommendation/advice question (including risk/opportunity)\n\n"
    "Use the corresponding final response format exactly.\n\n"
    "Prioritize the direct answer over background explanation.\n"
    "Do not add unrelated dimensions.\n"
    "Do not invent numbers, periods, rankings, confidence, or causes.\n"
    "Use signal_strength values exactly as returned by the tools.\n"
    "Do not convert correlation or signal strength into proven causation.\n"
    "If the question is narrow, keep the answer narrow.\n"
    "If evidence is insufficient, say so explicitly.\n"
    "If the question asks for something the dataset/tools cannot provide, "
    "say so explicitly instead of guessing.\n"
    "Use the deterministic confidence value from the SYSTEM NOTE if "
    "provided; otherwise write exactly: "
    "\"Confidence: Not provided by the analytical tools.\""
)
# ============================================================
# LANGGRAPH STATE
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_count: int
    mandatory_evidence_injected: bool


# ============================================================
# INVESTIGATION RESULT
# ============================================================

@dataclass
class InvestigationResult:
    question: str
    trace: list = field(default_factory=list)
    final_answer: str | None = None
    grounding_warning: str | None = None

    # "ok" | "no_api_key" | "model_rate_limited" | "auth_error"
    # | "synthesis_fallback" | "grounding_failed" | "api_error"
    status: str = "ok"

    confidence: str | None = None
    confidence_reason: str | None = None

    # The single deterministic strongest signal — the UI should display
    # THIS, not recompute its own version (see module docstring).
    strongest_signal: dict | None = None

    # Actual periods used by the investigation, so the UI never displays
    # stale selector state for an RCA generated from different periods.
    current_period: dict | None = None
    previous_period: dict | None = None


# ============================================================
# SARVAM (via langchain-sarvam's ChatSarvam)
# ============================================================

def _get_llm(with_tools: bool):
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY not set. Add SARVAM_API_KEY to your .env file.")

    llm = ChatSarvam(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.2,
        max_tokens=4096,
        reasoning_effort="low",
    )
    return llm.bind_tools(TOOLS) if with_tools else llm


def _extract_text(content) -> str:
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content or "")


def _parse_tool_args(raw_args) -> tuple[dict, str | None]:
    if raw_args is None:
        return {}, None
    if isinstance(raw_args, dict):
        return raw_args, None
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except (json.JSONDecodeError, TypeError):
            return {}, f"Tool arguments were not valid JSON: {raw_args!r}"
        if not isinstance(parsed, dict):
            return {}, f"Tool arguments parsed to a non-object JSON value: {raw_args!r}"
        return parsed, None
    return {}, f"Unexpected tool argument type: {type(raw_args).__name__}"


def _is_rate_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


def _is_auth_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in AUTH_ERROR_MARKERS)


# ============================================================
# AGENT NODE
# ============================================================

def _agent_node(state: AgentState, llm) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ============================================================
# TOOL NODE
# ============================================================

def _find_cached_result(trace: list, tool_name: str, tool_args: dict) -> dict | None:
    for step in trace:
        if step["tool"] == tool_name and step["input"] == tool_args:
            return step["result"]
    return None


def _tools_node(state: AgentState, trace: list) -> dict:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return {"messages": [], "tool_call_count": state["tool_call_count"]}

    tool_lookup = {tool.name: tool for tool in TOOLS}
    results = []
    executed = 0
    remaining_budget = max(0, MAX_TOOL_CALLS - state["tool_call_count"])

    for call in last_message.tool_calls:
        tool_name = call["name"]
        tool_args, parse_error = _parse_tool_args(call.get("args"))

        if parse_error is not None:
            output = {"status": "error", "reason": parse_error}
            trace.append({"tool": tool_name, "input": call.get("args"), "result": output, "trace_status": "failed"})
            results.append(ToolMessage(content=json.dumps(output, default=str), tool_call_id=call["id"]))
            continue

        cached = _find_cached_result(trace, tool_name, tool_args)
        if cached is not None:
            trace.append({"tool": tool_name, "input": tool_args, "result": cached, "trace_status": "deduplicated"})
            results.append(ToolMessage(content=json.dumps(cached, default=str), tool_call_id=call["id"]))
            continue

        if executed >= remaining_budget:
            output = {"status": "skipped", "reason": "Tool-call budget exhausted for this investigation."}
            trace_status = "skipped"
        else:
            tool_fn = tool_lookup.get(tool_name)
            if tool_fn is None:
                output = {"status": "error", "reason": f"Unknown tool: {tool_name}"}
                trace_status = "failed"
            else:
                try:
                    output = tool_fn.invoke(tool_args)
                    trace_status = "success" if output.get("status") == "ok" else "failed" if output.get("status") == "error" else "no_data"
                except Exception as exc:
                    output = {"status": "error", "reason": f"Tool execution failed: {type(exc).__name__}: {exc}"}
                    trace_status = "failed"
            executed += 1

        trace.append({"tool": tool_name, "input": tool_args, "result": output, "trace_status": trace_status})
        results.append(ToolMessage(content=json.dumps(output, default=str), tool_call_id=call["id"]))

    return {"messages": results, "tool_call_count": state["tool_call_count"] + executed}


# ============================================================
# KPI / PERIOD EXTRACTION
# ============================================================

def _has_kpi_comparison(trace: list) -> bool:
    for step in trace:
        if step["tool"] == "compare_sales_kpis":
            return True
        if step["tool"] == "compare_periods" and step["input"].get("metric") == "revenue":
            return True
    return False


def _extract_period_args(trace: list) -> dict | None:
    for step in reversed(trace):
        tool, result = step["tool"], step["result"]
        if tool == "compare_sales_kpis" and result.get("status") == "ok":
            args = step["input"]
            required = ("current_start_date", "current_end_date", "previous_start_date", "previous_end_date")
            if not all(key in args for key in required):
                return None
            return {k: args[k] for k in required}
        if tool == "compare_periods" and step["input"].get("metric") == "revenue" and result.get("status") == "ok":
            args = step["input"]
            required = ("period_a_start", "period_a_end", "period_b_start", "period_b_end")
            if not all(key in args for key in required):
                return None
            return {
                "current_start_date": args["period_a_start"],
                "current_end_date": args["period_a_end"],
                "previous_start_date": args["period_b_start"],
                "previous_end_date": args["period_b_end"],
            }
    return None


# ============================================================
# BOOTSTRAP EVIDENCE (application-supplied periods)
# ============================================================

def _bootstrap_period_evidence(
    trace: list,
    question: str,
    current_period: dict | None,
    comparison_period: dict | None,
) -> str | None:
    """Deterministically pre-populate the revenue KPI headline AND the
    required dimension evidence for the application-supplied periods,
    BEFORE the model's first turn — rather than waiting for the model to
    choose to call compare_sales_kpis first.

    WHY THIS EXISTS (real production bug):
    `_mandatory_evidence_node` only fires after a KPI comparison
    (`compare_sales_kpis` or `compare_periods(metric="revenue")`) already
    appears in the trace — see `_after_tools_route` / `_has_kpi_comparison`.
    That was meant to reuse the model's already-resolved periods, but it
    silently assumed the model would always call the KPI tool first. A
    real investigation ("What were the strongest observed contributors to
    the revenue change?") showed the model jumping straight to
    compare_category_performance without ever calling compare_sales_kpis —
    so mandatory dimension evidence never triggered, the UI's "Revenue
    Change" card was left blank, and only one dimension was investigated
    for what was really a general-scope question.

    Since app.py always resolves and passes current_period/
    comparison_period before calling run_investigation, we already know
    the periods the investigation should use in the common case (no
    explicitly different period named in the question) — there is no
    reason to wait on the model. This bootstrap makes the KPI headline and
    dimension coverage guarantees actually unconditional instead of
    conditional on model behavior.

    This does NOT prevent the model from independently investigating a
    DIFFERENT, explicitly-named period (system prompt rule 3 under
    APPLICATION-CONTROLLED PERIOD CONTEXT) — if it does, the existing
    `_mandatory_evidence_node` still fires for that different period
    (mandatory_evidence_injected starts False), and tool-call
    deduplication means nothing here is ever executed twice.
    """
    if not current_period or not comparison_period:
        return None

    period_args = {
        "current_start_date": current_period.get("start_date"),
        "current_end_date": current_period.get("end_date"),
        "previous_start_date": comparison_period.get("start_date"),
        "previous_end_date": comparison_period.get("end_date"),
    }
    if not all(period_args.values()):
        return None

    tool_lookup = {tool.name: tool for tool in TOOLS}
    kpi_tool = tool_lookup.get("compare_sales_kpis")
    if kpi_tool is None:
        return None

    try:
        kpi_result = kpi_tool.invoke(period_args)
    except Exception as exc:
        kpi_result = {"status": "error", "reason": f"Tool execution failed: {type(exc).__name__}: {exc}"}

    kpi_trace_status = (
        "success" if kpi_result.get("status") == "ok"
        else "no_data" if kpi_result.get("status") == "no_data"
        else "failed"
    )
    trace.append({"tool": "compare_sales_kpis", "input": period_args, "result": kpi_result, "trace_status": kpi_trace_status})

    injected_results = {"compare_sales_kpis": kpi_result}
    required_dims = _classify_scope(question)

    for dim in DIMENSION_ORDER:
        if dim not in required_dims:
            continue
        tool_name = DIMENSION_TOOL[dim]
        tool_fn = tool_lookup.get(tool_name)
        if tool_fn is None:
            continue
        try:
            output = tool_fn.invoke(dict(period_args))
            trace_status = (
                "success" if output.get("status") == "ok"
                else "no_data" if output.get("status") == "no_data"
                else "failed"
            )
        except Exception as exc:
            output = {"status": "error", "reason": f"Tool execution failed: {type(exc).__name__}: {exc}"}
            trace_status = "failed"

        trace.append({"tool": tool_name, "input": dict(period_args), "result": output, "trace_status": trace_status})
        injected_results[tool_name] = output

    # Scoped to the question's own dimensions (required_dims), so this
    # confidence value is consistent with the same evidence the model is
    # told to base its answer on — never inflated/deflated by a dimension
    # outside what the question actually asked about.
    allowed_tools = {DIMENSION_TOOL[d] for d in required_dims}
    confidence, confidence_reason = _deterministic_confidence(trace, allowed_tools=allowed_tools)

    note_parts = [
        "SYSTEM NOTE: The following evidence has been collected automatically, "
        "before your first turn, for the application-controlled current/"
        "comparison periods. This IS the investigation's revenue headline and "
        "dimension evidence — do not call these same tools again for these "
        "exact periods. Use these exact figures and signal_strength values in "
        "your synthesis. Report a failed or no_data entry as \"investigation "
        "failed\" or \"no data available\", never as \"not investigated\". If "
        "the user's question explicitly names a different period, you may "
        "still call resolve_named_period/comparison tools for that different "
        "period.\n\n" + json.dumps(injected_results, default=str)
    ]
    if confidence:
        note_parts.append(
            f"SYSTEM NOTE: The deterministic confidence for this investigation is {confidence} "
            f"— {confidence_reason}. State this exact confidence value in your final answer; "
            "do not infer or invent a different one."
        )

    return "\n\n".join(note_parts)


# ============================================================
# MANDATORY EVIDENCE NODE
# ============================================================

def _mandatory_evidence_node(state: AgentState, trace: list, question: str) -> dict:
    """Deterministically executes required dimension tools not already run,
    reusing the locked period. Also computes deterministic confidence right
    here — the earliest point at which it's known — and always
    communicates it to the model in the same note, so every subsequent
    model turn (whether it ends naturally or via forced synthesis) already
    knows the confidence value and never has to invent one.
    """
    period_args = _extract_period_args(trace)
    if period_args is None:
        return {"messages": [], "tool_call_count": state["tool_call_count"], "mandatory_evidence_injected": True}

    required_dims = _classify_scope(question)
    already_run = {step["tool"] for step in trace}
    tool_lookup = {tool.name: tool for tool in TOOLS}
    remaining_budget = max(0, MAX_TOOL_CALLS - state["tool_call_count"])

    injected_results = {}
    for dim in DIMENSION_ORDER:
        if dim not in required_dims:
            continue
        tool_name = DIMENSION_TOOL[dim]
        if tool_name in already_run:
            continue
        if len(injected_results) >= remaining_budget:
            break

        cached = _find_cached_result(trace, tool_name, dict(period_args))
        if cached is not None:
            trace.append({"tool": tool_name, "input": dict(period_args), "result": cached, "trace_status": "deduplicated"})
            injected_results[tool_name] = cached
            continue

        tool_fn = tool_lookup.get(tool_name)
        if tool_fn is None:
            output = {"status": "error", "reason": f"Required tool not available: {tool_name}"}
            trace_status = "failed"
        else:
            try:
                output = tool_fn.invoke(dict(period_args))
                trace_status = "success" if output.get("status") == "ok" else "failed" if output.get("status") == "error" else "no_data"
            except Exception as exc:
                output = {"status": "error", "reason": f"Tool execution failed: {type(exc).__name__}: {exc}"}
                trace_status = "failed"

        trace.append({"tool": tool_name, "input": dict(period_args), "result": output, "trace_status": trace_status})
        injected_results[tool_name] = output

    # Scoped exactly like the bootstrap's confidence computation — see
    # that function's comment for why this must match the question's
    # actual dimension scope rather than the whole trace.
    allowed_tools = {DIMENSION_TOOL[d] for d in required_dims}
    confidence, confidence_reason = _deterministic_confidence(trace, allowed_tools=allowed_tools)

    note_parts = []
    if injected_results:
        note_parts.append(
            "SYSTEM NOTE: The following mandatory dimension evidence has been collected "
            "automatically for the exact periods already resolved. These dimensions ARE "
            "investigated. Do not call these same tools again. Use their exact figures and "
            "signal_strength values in your synthesis. Report a failed or no_data dimension "
            "as \"investigation failed\" or \"no data available\", never as \"not investigated\":\n\n"
            + json.dumps(injected_results, default=str)
        )
    if confidence:
        note_parts.append(
            f"SYSTEM NOTE: The deterministic confidence for this investigation is {confidence} "
            f"— {confidence_reason}. State this exact confidence value in your final answer; "
            "do not infer or invent a different one."
        )

    if not note_parts:
        return {"messages": [], "tool_call_count": state["tool_call_count"], "mandatory_evidence_injected": True}

    return {
        "messages": [HumanMessage(content="\n\n".join(note_parts))],
        "tool_call_count": state["tool_call_count"] + len(injected_results),
        "mandatory_evidence_injected": True,
    }


def _after_tools_route(state: AgentState, trace: list) -> str:
    if not state.get("mandatory_evidence_injected") and _has_kpi_comparison(trace):
        return "mandatory"
    return "agent"


# ============================================================
# SYNTHESIZE / ROUTING / GRAPH
# ============================================================

def _synthesize_node(state: AgentState, llm_no_tools) -> dict:
    last_message = state["messages"][-1]
    already_has_answer = (
        isinstance(last_message, AIMessage)
        and not last_message.tool_calls
        and bool(_extract_text(last_message.content).strip())
    )
    if already_has_answer:
        return {"messages": []}
    response = llm_no_tools.invoke(state["messages"] + [HumanMessage(content=SYNTHESIS_INSTRUCTION)])
    return {"messages": [response]}


def _should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    has_tool_calls = isinstance(last_message, AIMessage) and bool(last_message.tool_calls)
    if has_tool_calls and state["tool_call_count"] < MAX_TOOL_CALLS:
        return "tools"
    return "synthesize"


def _build_graph(llm, llm_no_tools, trace: list, question: str):
    graph = StateGraph(AgentState)
    graph.add_node("agent", lambda state: _agent_node(state, llm))
    graph.add_node("tools", lambda state: _tools_node(state, trace))
    graph.add_node("mandatory", lambda state: _mandatory_evidence_node(state, trace, question))
    graph.add_node("synthesize", lambda state: _synthesize_node(state, llm_no_tools))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", "synthesize": "synthesize"})
    graph.add_conditional_edges(
        "tools", lambda state: _after_tools_route(state, trace), {"mandatory": "mandatory", "agent": "agent"}
    )
    graph.add_edge("mandatory", "agent")
    graph.add_edge("synthesize", END)
    return graph.compile()


# ============================================================
# GROUNDING
# ============================================================

_NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize_number(token: str) -> float | None:
    cleaned = token.replace(",", "").replace("%", "").replace("₹", "").strip()
    try:
        return round(float(cleaned), 2)
    except (TypeError, ValueError):
        return None


def _collect_evidence_numbers(trace: list) -> set:
    text = json.dumps([step["result"] for step in trace], default=str)
    return {n for tok in _NUMBER_PATTERN.findall(text) if (n := _normalize_number(tok)) is not None}


def _check_grounding(final_text: str, trace: list) -> list[str]:
    if not trace:
        return []
    evidence = _collect_evidence_numbers(trace)
    suspicious = []
    for token in _NUMBER_PATTERN.findall(final_text):
        digits_only = token.replace(",", "").replace("-", "").replace(".", "")
        if len(digits_only) < MIN_DIGITS_TO_GROUND:
            continue
        value = _normalize_number(token)
        if value is None:
            continue
        candidates = {value, -value, abs(value)}
        if not any(abs(c - e) <= GROUNDING_TOLERANCE for c in candidates for e in evidence):
            suspicious.append(token)
    return suspicious


def _attempt_synthesis_recovery(
    llm_no_tools,
    system_prompt: str,
    trace: list,
    question: str,
    reason: str,
    suspicious: list[str] | None = None,
) -> str | None:
    """One bounded recovery attempt: re-invokes the no-tools model directly
    against the SAME already-collected evidence (no new analytics/tool
    calls, no re-investigation) — either because the previous attempt
    returned empty/no content, or because it contained numbers that
    couldn't be verified against the tool evidence. Returns cleaned text,
    or None if this attempt also failed to produce anything usable.

    FIXED BUG: a previous version never included the actual user
    `question` text in this recovery call — only the evidence and a
    generic "answer the user's original question" instruction. Since this
    call is a brand-new, standalone LLM invocation (deliberately NOT
    reusing the original conversation's message history, to avoid
    resending a possibly-malformed prior turn), the model had no way to
    know what the original question actually was, and would default to
    answering a generic "why did revenue decline" narrative regardless of
    what was really asked — a real, observed case of the recovered answer
    silently addressing the wrong question. The question is now restated
    explicitly and verbatim in every recovery prompt.
    """
    evidence_text = json.dumps(
        [{"tool": step["tool"], "input": step["input"], "result": step["result"]} for step in trace],
        default=str,
    )
    if reason == "empty":
        instruction = (
            "Your previous response was empty or did not complete. Produce the final answer "
            "NOW, using ONLY the evidence below. Do not call any tools — the investigation is "
            "already complete. Answer the user's original question directly and concisely, using "
            "whichever response format from the system instructions matches the question type. "
            "Do not invent numbers, causes, or assumptions. Prefer 'contributing factor', "
            "'strongest observed signal', or 'consistent with' over causal certainty."
        )
    else:
        instruction = (
            "Your previous answer contained numeric figures that could not be verified against "
            f"the tool evidence: {', '.join(suspicious or [])}.\n\n"
            "Rewrite the answer using ONLY the evidence below. Every numeric claim must match a "
            "value appearing in that evidence. Do not invent calculations, numbers, causes, or "
            "assumptions. Do not use causal certainty. Prefer 'contributing factor', 'strongest "
            "observed signal', or 'consistent with'. If causation is not established, explicitly "
            "say: 'The available data does not establish direct causation.'\n\n"
            "Use the response format from the system instructions that matches the question type."
        )
    prompt = (
        f"{instruction}\n\n"
        f"THE USER'S ORIGINAL INVESTIGATION QUESTION (answer exactly this question, "
        f"verbatim — do not substitute a different question):\n\"{question}\"\n\n"
        f"EVIDENCE:\n{evidence_text}"
    )
    try:
        response = llm_no_tools.invoke([SystemMessage(content=system_prompt), HumanMessage(content=prompt)])
        return _extract_text(response.content).strip() or None
    except Exception:
        return None


def _synthesize_final_answer(
    llm_no_tools,
    system_prompt: str,
    trace: list,
    question: str,
    initial_answer: str,
) -> tuple[str | None, str]:
    """Bounded, evidence-reusing recovery loop for the final synthesis step.

    Handles the two ways synthesis can go wrong — an empty/incomplete
    response, or a non-empty response with ungrounded numbers — with a
    single unified retry loop instead of only handling the grounding case
    (a prior version had no retry at all for an empty response, going
    straight to the deterministic fallback on the very first empty
    completion, which is a real contributor to the "works on the second
    try" symptom: the SAME model, given the SAME evidence again, quite
    often succeeds on the very next attempt).

    Never re-runs analytics tools and never calls the LLM more than
    MAX_SYNTHESIS_RETRIES additional times — bounded, no infinite loop.

    Returns (final_text_or_None, outcome) where outcome is one of:
      "ok"                  — initial_answer was already good, no retry needed
      "recovered"           — a bounded retry produced a good answer
      "failed"               — all bounded attempts exhausted; caller must
                                use the deterministic fallback
    """
    if initial_answer:
        suspicious = _check_grounding(initial_answer, trace)
        if not suspicious:
            return initial_answer, "ok"
        reason = "grounding"
    else:
        suspicious = []
        reason = "empty"

    for _ in range(MAX_SYNTHESIS_RETRIES):
        retried = _attempt_synthesis_recovery(llm_no_tools, system_prompt, trace, question, reason, suspicious)
        if not retried:
            # The retry itself came back empty/errored — try again (still
            # bounded by the loop) treating it as another empty case.
            reason, suspicious = "empty", []
            continue
        retried_suspicious = _check_grounding(retried, trace)
        if not retried_suspicious:
            return retried, "recovered"
        reason, suspicious = "grounding", retried_suspicious

    return None, "failed"


# ============================================================
# DETERMINISTIC SIGNALS / STRONGEST SIGNAL / CONFIDENCE
# ============================================================

def _collect_signals(trace: list, allowed_tools: set[str] | None = None) -> list[tuple]:
    """(dimension, label, change, signal_strength, rupee_impact) for every
    STRONG/MODERATE signal, deterministically ranked by:
      1. signal strength (STRONG > MODERATE)
      2. absolute rupee revenue impact, where computable (larger first) —
         see note below
      3. absolute magnitude of the dimension's own relative change, as a
         fallback for dimensions with no rupee-comparable figure
      4. fixed dimension priority (Category > Channel > Inventory > Marketing)
    NEVER by trace insertion order.

    allowed_tools: when provided (typically `_relevant_dimension_tools(question)`),
    restricts consideration to signals from those tool names only — e.g. a
    question scoped to "category" never surfaces a Marketing or Inventory
    signal here, even if that tool happens to appear elsewhere in the
    trace. None (the default) considers every dimension tool in the trace,
    unchanged from prior behavior.

    WHY RUPEE IMPACT, NOT RELATIVE %:
    A prior version ranked purely by each dimension's own relative percentage
    change (e.g. category revenue_change_pct vs marketing
    attributed_revenue_change_pct vs inventory availability_change_pct_points).
    Those are different bases and not comparable — a real production bug
    surfaced this: Meta's attributed revenue moved -34.77% off a small
    ~3.3M base (a ~1.15M rupee swing) while Running category moved "only"
    -30.88% off a much larger ~6.98M base (a ~2.15M rupee swing, and 63.96%
    of the ENTIRE revenue delta per compare_category_performance's own
    contribution_to_total_change_pct). The old ranker picked Meta as
    "strongest" purely because -34.77 > -30.88, directly contradicting the
    model's own Root Cause Ranking (which correctly led with Running) and
    undermining trust in the deterministic UI card. Ranking by rupee impact
    where a rupee figure exists (category: absolute_revenue_change; channel
    and marketing: current-vs-previous revenue/attributed-revenue
    difference, computed here from tool-returned figures only — never
    estimated) puts every dimension on the same footing as the revenue
    change actually being investigated. Inventory has no revenue-equivalent
    figure (availability points, stockout days) and is never converted into
    one; it instead ranks by its own relative magnitude, below any
    dimension that DOES have a computed rupee impact, so it is never
    silently placed above (or below) a revenue mover using invented
    numbers.
    """
    signals = []
    for step in trace:
        tool, result = step["tool"], step["result"]
        if allowed_tools is not None and tool not in allowed_tools:
            continue
        if tool == "compare_category_performance":
            for r in result.get("categories", []):
                if r.get("signal_strength") in ("STRONG", "MODERATE"):
                    rupee_impact = r.get("absolute_revenue_change")
                    signals.append((
                        "Category", r.get("category", "Unknown"),
                        r.get("revenue_change_pct"), r["signal_strength"],
                        abs(rupee_impact) if rupee_impact is not None else None,
                    ))
        elif tool == "compare_channel_performance":
            for r in result.get("channels", []):
                if r.get("signal_strength") in ("STRONG", "MODERATE"):
                    current_rev = r.get("current_revenue")
                    previous_rev = r.get("previous_revenue")
                    rupee_impact = (
                        abs(current_rev - previous_rev)
                        if current_rev is not None and previous_rev is not None
                        else None
                    )
                    signals.append((
                        "Channel", r.get("channel", "Unknown"),
                        r.get("revenue_change_pct"), r["signal_strength"],
                        rupee_impact,
                    ))
        elif tool == "compare_marketing_performance":
            for r in result.get("channels", []):
                if r.get("signal_strength") in ("STRONG", "MODERATE"):
                    current_rev = r.get("current_attributed_revenue")
                    previous_rev = r.get("previous_attributed_revenue")
                    rupee_impact = (
                        abs(current_rev - previous_rev)
                        if current_rev is not None and previous_rev is not None
                        else None
                    )
                    signals.append((
                        "Marketing", r.get("channel", "Unknown"),
                        r.get("attributed_revenue_change_pct"), r["signal_strength"],
                        rupee_impact,
                    ))
        elif tool == "compare_inventory_performance" and result.get("status") == "ok":
            if result.get("signal_strength") in ("STRONG", "MODERATE"):
                # No revenue-equivalent figure exists for inventory — never
                # invented. rupee_impact stays None so this ranks by its own
                # relative magnitude, after any dimension with a real rupee
                # figure of matching or greater strength.
                signals.append((
                    "Inventory", "Availability / stockouts",
                    result.get("availability_change_pct_points"), result["signal_strength"],
                    None,
                ))

    strength_rank = {"STRONG": 2, "MODERATE": 1}
    signals.sort(
        key=lambda s: (
            -strength_rank.get(s[3], 0),
            s[4] is None,
            -(s[4] if s[4] is not None else 0),
            -(abs(s[2]) if s[2] is not None else 0),
            DIMENSION_PRIORITY.get(s[0], 99),
        )
    )
    return signals


def get_strongest_signal(trace: list, allowed_tools: set[str] | None = None) -> dict | None:
    """The single deterministic answer to "what's the strongest signal" —
    the UI should display exactly this, not recompute its own version
    (a prior version of app.py maintained a separate copy that
    inconsistently included WEAK signals). Ranked by rupee revenue impact
    where computable, see _collect_signals for why. allowed_tools scopes
    this to the question's dimension(s) — see _collect_signals."""
    signals = _collect_signals(trace, allowed_tools=allowed_tools)
    if not signals:
        return None
    dimension, label, change, strength, rupee_impact = signals[0]
    return {
        "dimension": dimension,
        "label": label,
        "change": change,
        "strength": strength,
        "rupee_impact": rupee_impact,
    }


def _any_dimension_tool_succeeded(trace: list, allowed_tools: set[str] | None = None) -> bool:
    dimension_tools = set(DIMENSION_TOOL.values()) if allowed_tools is None else set(allowed_tools)
    return any(step["tool"] in dimension_tools and step["result"].get("status") == "ok" for step in trace)


def _deterministic_confidence(
    trace: list, allowed_tools: set[str] | None = None
) -> tuple[str | None, str | None]:
    """allowed_tools scopes confidence to the same dimension(s) the answer
    itself is scoped to (see _collect_signals) — so a narrow category
    question's confidence reflects only category evidence, never an
    unrelated STRONG marketing signal the user didn't ask about. This
    keeps the Confidence badge, the Strongest Signal card, and the
    model's own Root Cause Ranking describing the same evidence."""
    if not trace:
        return None, None
    signals = _collect_signals(trace, allowed_tools=allowed_tools)
    strong_count = sum(1 for s in signals if s[3] == "STRONG")
    moderate_count = sum(1 for s in signals if s[3] == "MODERATE")

    if strong_count >= 2:
        return "HIGH", f"{strong_count} independent STRONG-signal dimensions were found."
    if strong_count == 1 or moderate_count >= 2:
        return "MEDIUM", "At least one meaningful signal was found, but causality is not established."
    if _any_dimension_tool_succeeded(trace, allowed_tools=allowed_tools):
        return "LOW", "All required dimensions were investigated but none showed a STRONG or MODERATE signal."
    return "LOW", "Evidence gathered was weak or insufficient to support a confident conclusion."


# ============================================================
# HEADLINE / PERIOD EXTRACTION FOR OUTPUT
# ============================================================

def _extract_headline(trace: list):
    for step in reversed(trace):
        if step["tool"] == "compare_sales_kpis" and step["result"].get("status") == "ok":
            result = step["result"]
            revenue = result.get("kpis", {}).get("revenue", {})
            return revenue.get("pct_change"), result.get("current_period"), result.get("previous_period")
        if step["tool"] == "compare_periods" and step["input"].get("metric") == "revenue" and step["result"].get("status") == "ok":
            result = step["result"]
            return result.get("pct_change"), result.get("period_a"), result.get("period_b")
    return None, None, None


def _period_label(period: dict | None) -> str:
    """Renders a period as 'label (start–end)' when a label is available,
    else 'start–end'. Uses an en dash INSIDE a period's own date range so
    it is never confused with the "to" that separates two periods being
    compared — a prior version rendered both with the word "to", producing
    ambiguous strings like "from 2026-06-01 to 2026-06-30 to 2026-07-01 to
    2026-07-31" in the deterministic fallback text."""
    if not period:
        return "an unknown period"
    start = period.get("start") or period.get("start_date")
    end = period.get("end") or period.get("end_date")
    label = period.get("label")
    if not (start and end):
        return "an unknown period"
    date_range = f"{start}\u2013{end}"
    return f"{label} ({date_range})" if label else date_range


# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================

def _signed_rupees(value: float) -> str:
    """Formats a signed rupee amount with the sign BEFORE the symbol
    (-₹2,155,113.10), matching standard currency convention — Python's
    default f"₹{value:,.2f}" would render a negative value as the
    confusing "₹-2,155,113.10"."""
    sign = "-" if value < 0 else ""
    return f"{sign}\u20b9{abs(value):,.2f}"


def _format_single_dimension_fallback(trace: list, dim: str) -> str | None:
    """Dedicated, question-aware fallback formatter for a question scoped
    to exactly ONE dimension (see _classify_scope). Uses the richest
    tool-provided figures for that specific dimension — e.g. category's
    own contribution_to_total_change_pct, which is the correct "how much
    did each contribute" number — rather than the generic cross-dimension
    "strongest observed contributing factors" list, which was designed
    for general/multi-dimension questions and previously became
    misleading for narrow ones (it could include an unrelated dimension's
    signal that scored higher on the old ranking, or simply omit the
    contribution-share figure the user actually asked for).

    Returns None if the relevant tool didn't run or returned no usable
    data, so the caller can fall through to the general fallback.
    """
    tool_name = DIMENSION_TOOL.get(dim)
    step = next(
        (s for s in trace if s["tool"] == tool_name and s["result"].get("status") == "ok"),
        None,
    )
    if step is None:
        return None
    result = step["result"]

    if dim == "category":
        records = [r for r in result.get("categories", []) if r.get("contribution_to_total_change_pct") is not None]
        if not records:
            return None
        records.sort(key=lambda r: abs(r["contribution_to_total_change_pct"]), reverse=True)
        lead = records[0]
        bullets = [
            f"• {r['category']}: {r['revenue_change_pct']:+.2f}% revenue change "
            f"({_signed_rupees(r['absolute_revenue_change'])}), "
            f"{abs(r['contribution_to_total_change_pct']):.2f}% of the total revenue change, "
            f"{r.get('signal_strength', '—')} signal"
            for r in records
        ]
        return (
            f"{lead['category']} contributed the most to the revenue change, accounting for "
            f"{abs(lead['contribution_to_total_change_pct']):.2f}% of the total change "
            f"({_signed_rupees(lead['absolute_revenue_change'])}).\n\n"
            "Category contributions:\n" + "\n".join(bullets) + "\n\n"
            "These are measured revenue contributions to the total change; they indicate "
            "association and relative impact, not proof of causation."
        )

    if dim == "channel":
        records = [r for r in result.get("channels", []) if r.get("revenue_change_pct") is not None]
        if not records:
            return None
        for r in records:
            current_rev = r.get("current_revenue")
            previous_rev = r.get("previous_revenue")
            r["_rupee_impact"] = (
                abs(current_rev - previous_rev)
                if current_rev is not None and previous_rev is not None
                else None
            )
        records.sort(key=lambda r: (r["_rupee_impact"] is None, -(r["_rupee_impact"] or 0)))
        lead = records[0]
        lead_impact_text = f" (\u20b9{lead['_rupee_impact']:,.2f} impact)" if lead["_rupee_impact"] is not None else ""
        bullets = [
            f"• {r['channel']}: {r['revenue_change_pct']:+.2f}% revenue change"
            + (f" (\u20b9{r['_rupee_impact']:,.2f} impact)" if r["_rupee_impact"] is not None else "")
            + f", {r.get('current_revenue_share_pct', 0):.1f}% of current revenue, "
            f"{r.get('signal_strength', '—')} signal"
            for r in records
        ]
        return (
            f"{lead['channel']} showed the largest measured revenue movement among sales "
            f"channels ({lead['revenue_change_pct']:+.2f}%{lead_impact_text}).\n\n"
            "Channel performance:\n" + "\n".join(bullets) + "\n\n"
            "These are measured revenue movements; they indicate association and relative "
            "impact, not proof of causation."
        )

    if dim == "marketing":
        records = [r for r in result.get("channels", []) if r.get("roas_change_pct") is not None]
        if not records:
            return None
        records.sort(key=lambda r: abs(r["roas_change_pct"]), reverse=True)
        lead = records[0]
        bullets = [
            f"• {r['channel']}: ROAS {r.get('previous_roas', '—')} \u2192 {r.get('current_roas', '—')} "
            f"({r['roas_change_pct']:+.2f}%), attributed revenue "
            f"{r.get('attributed_revenue_change_pct', 0):+.2f}%, {r.get('signal_strength', '—')} signal"
            for r in records
        ]
        return (
            f"{lead['channel']} showed the largest measured marketing efficiency deterioration "
            f"(ROAS change {lead['roas_change_pct']:+.2f}%).\n\n"
            "Marketing channel performance:\n" + "\n".join(bullets) + "\n\n"
            "These are measured marketing efficiency changes; they indicate association and "
            "relative impact, not proof of causation."
        )

    if dim == "inventory":
        availability_change = result.get("availability_change_pct_points")
        stockout_change = result.get("stockout_days_change")
        strength = result.get("signal_strength", "—")
        note = result.get("evidence_note") or ""
        pieces = []
        if availability_change is not None:
            pieces.append(f"availability changed {availability_change:+.2f} percentage points")
        if stockout_change is not None:
            pieces.append(f"stockout days changed {stockout_change:+d}")
        summary = ", ".join(pieces) if pieces else "no material inventory change was observed"
        text = f"Inventory shows a {strength} observed signal: {summary}."
        if note:
            text += f" {note}"
        text += "\n\nThis indicates an association with the revenue change, not proof of causation."
        return text

    return None


def _deterministic_fallback(trace: list, question: str = "") -> str:
    """Evidence-only fallback — never calls the LLM. Correctly distinguishes
    "a dimension was investigated but showed nothing STRONG/MODERATE" from
    "nothing was investigated at all", even for narrow questions that never
    called compare_sales_kpis.

    question: when provided, scopes the fallback to the question's actual
    dimension(s) via _classify_scope — a narrow single-dimension question
    (e.g. "Which categories contributed most to the revenue decline?")
    gets the dedicated _format_single_dimension_fallback narrative instead
    of the generic cross-dimension "strongest observed contributing
    factors" list, and never surfaces an unrelated dimension's evidence
    even if it happens to be present in the trace. Omitting question
    preserves the previous unscoped (all-dimension) behavior."""
    if not trace:
        return "AI investigation could not be completed and no business data was gathered. Please try again shortly."

    scope_dims = _classify_scope(question) if question else set(DIMENSION_ORDER)
    allowed_tools = {DIMENSION_TOOL[d] for d in scope_dims}

    if len(scope_dims) == 1:
        dedicated = _format_single_dimension_fallback(trace, next(iter(scope_dims)))
        if dedicated:
            return dedicated

    pct_change, current_period, previous_period = _extract_headline(trace)
    signals = _collect_signals(trace, allowed_tools=allowed_tools)
    any_dimension_evidence = _any_dimension_tool_succeeded(trace, allowed_tools=allowed_tools)

    if pct_change is None and not any_dimension_evidence:
        return (
            "The available tools did not return enough evidence to answer this question. "
            "The available data does not establish direct causation."
        )

    lines = []
    if pct_change is not None:
        direction = "declined" if pct_change < 0 else "increased" if pct_change > 0 else "was flat"
        lines.append(
            f"Revenue {direction} {abs(pct_change):.2f}% "
            f"from {_period_label(previous_period)} to {_period_label(current_period)}."
        )
        lines.append("")

    # Percentage-based dimensions (category revenue, channel revenue,
    # marketing attributed revenue) get a "%" suffix; inventory's figure is
    # percentage POINTS, a different unit, so it never gets "%" appended —
    # that distinction matters for numeric-integrity/grounding purposes.
    _PERCENT_DIMENSIONS = {"Category", "Channel", "Marketing"}

    if signals:
        lines.append("The strongest observed contributing factors were:")
        for dimension, label, change, strength, rupee_impact in signals[:5]:
            if change is None:
                change_text = "—"
            elif dimension in _PERCENT_DIMENSIONS:
                change_text = f"{change:+.2f}%"
            else:
                change_text = f"{change:+.2f} pp"
            impact_suffix = f", \u20b9{rupee_impact:,.0f} revenue impact" if rupee_impact is not None else ""
            lines.append(f"• {dimension} — {label}: {change_text} ({strength} signal{impact_suffix})")
        lines.append("")
    elif any_dimension_evidence:
        lines.append(
            "All required dimensions were investigated. None showed a STRONG or MODERATE "
            "signal — the available evidence does not point to a dominant contributing factor."
        )
        lines.append("")

    lines.append(
        "This is a deterministic summary of the evidence collected by the analytics tools. "
        "The available data indicates contributing signals but does not establish direct causation."
    )
    return "\n".join(lines)


# ============================================================
# DETERMINISTIC DIRECT ANALYTICS
# ============================================================

_MONTH_NAME_PATTERN = (
    r"(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
)


def _direct_result(
    question: str,
    tool_name: str,
    tool_input: dict,
    analytics_result: dict,
) -> InvestigationResult:
    """Builds an InvestigationResult for a deterministic analytics branch."""
    result = InvestigationResult(question=question)
    trace_status = (
        "success"
        if analytics_result.get("status") == "ok"
        else "no_data"
        if analytics_result.get("status") == "no_data"
        else "failed"
    )
    result.trace = [{
        "tool": tool_name,
        "input": tool_input,
        "result": analytics_result,
        "trace_status": trace_status,
    }]
    result.final_answer = _format_direct_analytics_answer(
        question, tool_name, analytics_result
    )
    result.status = "ok" if trace_status == "success" else "api_error"
    return result


def _extract_single_period(question: str) -> str | None:
    q = re.sub(r"\s+", " ", question.lower()).strip()

    iso_match = re.search(r"\b\d{4}-\d{1,2}\b", q)
    if iso_match:
        return iso_match.group(0)

    month_year_match = re.search(
        rf"\b{_MONTH_NAME_PATTERN}\s+\d{{4}}\b",
        q,
    )
    if month_year_match:
        return month_year_match.group(0)

    month_match = re.search(rf"\b{_MONTH_NAME_PATTERN}\b", q)
    if month_match:
        return month_match.group(0)

    if "latest month" in q:
        return "latest_month"
    if "previous month" in q:
        return "previous_month"

    return None


def _extract_trend_periods(question: str) -> tuple[str | None, str | None]:
    q = re.sub(r"\s+", " ", question.lower()).strip()

    range_match = re.search(
        rf"\bfrom\s+({_MONTH_NAME_PATTERN}(?:\s+\d{{4}})?|\d{{4}}-\d{{1,2}})"
        rf"\s+to\s+({_MONTH_NAME_PATTERN}(?:\s+\d{{4}})?|\d{{4}}-\d{{1,2}})\b",
        q,
    )
    if range_match:
        return range_match.group(1), range_match.group(2)

    between_match = re.search(
        rf"\bbetween\s+({_MONTH_NAME_PATTERN}(?:\s+\d{{4}})?|\d{{4}}-\d{{1,2}})"
        rf"\s+and\s+({_MONTH_NAME_PATTERN}(?:\s+\d{{4}})?|\d{{4}}-\d{{1,2}})\b",
        q,
    )
    if between_match:
        return between_match.group(1), between_match.group(2)

    return None, None


def _format_direct_analytics_answer(
    question: str,
    tool_name: str,
    analytics_result: dict,
) -> str:
    """Formats only values returned by deterministic analytics."""
    if analytics_result.get("status") == "no_data":
        return analytics_result.get(
            "reason",
            "No data is available for the requested period.",
        )

    if analytics_result.get("status") != "ok":
        return analytics_result.get(
            "reason",
            "The requested analytics could not be completed.",
        )

    if tool_name == "rank_months_by_revenue":
        best = analytics_result["best_month"]
        worst = analytics_result["worst_month"]
        q = question.lower()

        if "worst" in q and "best" not in q:
            return f'{worst["label"]} had the lowest revenue at ${worst["revenue"]:,.2f}.'

        answer = f'{best["label"]} had the highest revenue at ${best["revenue"]:,.2f}.'
        ranked = analytics_result.get("months", [])
        if len(ranked) > 1:
            details = "; ".join(
                f'{item["label"]}: ${item["revenue"]:,.2f}'
                for item in ranked[:3]
            )
            answer += f" Top months by revenue: {details}."
        return answer

    if tool_name == "get_sales_metrics":
        return (
            f'Revenue in {analytics_result["start"]} to {analytics_result["end"]} '
            f'was ${analytics_result["revenue"]:,.2f}.'
        )

    if tool_name == "get_revenue_trend":
        points = analytics_result.get("points", [])
        if not points:
            return "No monthly revenue data is available for the requested range."
        lines = [
            f'{point["label"]}: ${point["revenue"]:,.2f}'
            for point in points
        ]
        return "Revenue trend:\n" + "\n".join(lines)

    return "The requested analytics were completed."


def _run_direct_analytics(
    question: str,
    intent: Intent,
) -> InvestigationResult:
    """Executes the deterministic branch without invoking Sarvam/LangGraph."""
    sales, _, _, _ = metrics.load_all()

    if intent == Intent.CROSS_PERIOD_RANKING:
        year_match = re.search(r"\b(20\d{2})\b", question)
        year = year_match.group(1) if year_match else None
        analytics_result = metrics.rank_months_by_revenue(sales, year=year)
        return _direct_result(
            question,
            "rank_months_by_revenue",
            {"year": year},
            analytics_result,
        )

    if intent == Intent.SINGLE_PERIOD_LOOKUP:
        period = _extract_single_period(question)
        if period is None:
            return InvestigationResult(
                question=question,
                final_answer=(
                    "The requested period could not be identified, "
                    "so no period was invented."
                ),
                status="api_error",
            )

        resolved = metrics.resolve_named_period(sales, period)
        if resolved.get("status") != "ok":
            return InvestigationResult(
                question=question,
                final_answer=resolved.get(
                    "reason",
                    f"The requested period '{period}' is unavailable.",
                ),
                status="api_error",
            )

        analytics_result = metrics.get_sales_metrics(
            sales,
            resolved["start_date"],
            resolved["end_date"],
        )
        result = _direct_result(
            question,
            "get_sales_metrics",
            {
                "start": resolved["start_date"],
                "end": resolved["end_date"],
                "period": resolved["label"],
            },
            analytics_result,
        )
        result.current_period = resolved
        return result

    if intent == Intent.TIME_SERIES_TREND:
        start_period, end_period = _extract_trend_periods(question)
        if start_period is None or end_period is None:
            return InvestigationResult(
                question=question,
                final_answer=(
                    "The requested trend range could not be resolved from "
                    "the question, so no date range was invented."
                ),
                status="api_error",
            )

        start_resolved = metrics.resolve_named_period(sales, start_period)
        end_resolved = metrics.resolve_named_period(sales, end_period)

        if start_resolved.get("status") != "ok":
            return InvestigationResult(
                question=question,
                final_answer=start_resolved.get(
                    "reason",
                    f"The period '{start_period}' is unavailable.",
                ),
                status="api_error",
            )

        if end_resolved.get("status") != "ok":
            return InvestigationResult(
                question=question,
                final_answer=end_resolved.get(
                    "reason",
                    f"The period '{end_period}' is unavailable.",
                ),
                status="api_error",
            )

        analytics_result = metrics.get_revenue_trend(
            sales,
            start_resolved["start_date"],
            end_resolved["end_date"],
        )
        result = _direct_result(
            question,
            "get_revenue_trend",
            {
                "start": start_resolved["start_date"],
                "end": end_resolved["end_date"],
            },
            analytics_result,
        )
        result.current_period = start_resolved
        result.previous_period = end_resolved
        return result

    return InvestigationResult(
        question=question,
        final_answer=(
            f"No direct analytics handler is registered for intent "
            f"{intent.value}."
        ),
        status="api_error",
    )


# ============================================================
# MAIN INVESTIGATION


def run_investigation(
    question: str,
    *,
    current_period: dict | None = None,
    comparison_period: dict | None = None,
) -> InvestigationResult:
    """
    current_period / comparison_period: pass the exact dicts
    metrics.resolve_named_period() returns (`{"start_date", "end_date",
    "label"}`) for the UI's active selection — the model will use these
    directly unless the question explicitly names different periods. Omit
    both to fall back to a pure free-text investigation where the model
    must resolve periods itself.

    Retry safety: the whole graph is retried at most once, and ONLY if no
    tool has executed yet. Once `trace` is non-empty, a failure goes
    straight to the deterministic fallback — the graph is never restarted,
    so analytics work and trace entries are never duplicated.
    """
    intent = classify_intent(question)

    if intent in DIRECT_ANALYTICS_INTENTS:
        return _run_direct_analytics(question, intent)

    # SUPPORTED_INTENTS intentionally preserve the existing LangGraph +
    # Sarvam/tool-calling investigation path below, unchanged.
    if intent not in SUPPORTED_INTENTS:
        return InvestigationResult(
            question=question,
            final_answer=(
                f"This question was classified as {intent.value}, but no handler "
                "is registered for it."
            ),
            status="api_error",
        )

    result = InvestigationResult(question=question)

    try:
        llm = _get_llm(with_tools=True)
        llm_no_tools = _get_llm(with_tools=False)
    except RuntimeError as exc:
        result.final_answer = (
            "No SARVAM_API_KEY configured.\n\nSet SARVAM_API_KEY in your .env file to enable live AI investigations."
        )
        result.grounding_warning = str(exc)
        result.status = "no_api_key"
        return result

    system_prompt = _build_system_prompt(current_period, comparison_period)
    trace: list = []
    app = _build_graph(llm, llm_no_tools, trace, question)

    # Guarantee the revenue headline + required dimension evidence exist
    # before the model's first turn, for the application-supplied periods —
    # see _bootstrap_period_evidence for why this can no longer depend on
    # the model choosing to call compare_sales_kpis itself.
    bootstrap_note = _bootstrap_period_evidence(trace, question, current_period, comparison_period)

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    if bootstrap_note:
        messages.append(HumanMessage(content=bootstrap_note))

    initial_state: AgentState = {
        "messages": messages,
        # Tools already executed by the bootstrap count against the same
        # per-investigation budget as model-triggered tool calls.
        "tool_call_count": len(trace),
        "mandatory_evidence_injected": False,
    }

    try:
        try:
            final_state = app.invoke(initial_state)
        except Exception as first_error:
            if _is_rate_limit_error(first_error) or trace:
                raise
            time.sleep(1.5)
            try:
                final_state = app.invoke(initial_state)
            except Exception:
                raise first_error

        result.trace = trace
        _pct_change, current_period_used, previous_period_used = _extract_headline(trace)
        result.current_period = current_period_used
        result.previous_period = previous_period_used

        # Every dimension/scope-sensitive computation below uses the SAME
        # allowed_tools filter, so the Strongest Signal card, the
        # Confidence badge, and the fallback text (if used) all describe
        # exactly the evidence relevant to what was actually asked — never
        # a mix of the question's dimension plus something else the model
        # happened to also fetch.
        allowed_tools = _relevant_dimension_tools(question)

        raw_final_answer = _extract_text(final_state["messages"][-1].content).strip() if final_state.get("messages") else ""

        # Bounded, evidence-reusing recovery — see _synthesize_final_answer.
        # Never re-runs analytics tools; retries only the synthesis call
        # itself, up to MAX_SYNTHESIS_RETRIES times, against the SAME
        # trace collected above. The original question is passed through
        # explicitly so a retry can never lose track of what was actually
        # asked (see _attempt_synthesis_recovery's docstring for the bug
        # this fixes).
        recovered_answer, synthesis_outcome = _synthesize_final_answer(
            llm_no_tools, system_prompt, trace, question, raw_final_answer
        )

        if recovered_answer:
            result.final_answer = recovered_answer
            # A successfully recovered answer (verified against the same
            # evidence as any first-attempt answer) is NOT surfaced to the
            # user as a warning — retry mechanics are an internal
            # implementation detail, not something a business user needs
            # to see once the answer is confirmed correct. status stays
            # "ok" and grounding_warning stays unset either way.
        else:
            result.final_answer = _deterministic_fallback(trace, question)
            result.status = "synthesis_fallback" if not raw_final_answer else "grounding_failed"
            result.grounding_warning = (
                "The AI-generated explanation could not be produced or verified against the "
                f"collected evidence after {MAX_SYNTHESIS_RETRIES} retries. A deterministic "
                "evidence-only summary is shown instead."
            )

        result.confidence, result.confidence_reason = _deterministic_confidence(trace, allowed_tools=allowed_tools)
        result.strongest_signal = get_strongest_signal(trace, allowed_tools=allowed_tools)
        return result

    except Exception as exc:
        result.trace = trace
        _pct_change, current_period_used, previous_period_used = _extract_headline(trace)
        result.current_period = current_period_used
        result.previous_period = previous_period_used
        allowed_tools = _relevant_dimension_tools(question)

        if _is_rate_limit_error(exc):
            result.status = "model_rate_limited"
            if trace:
                result.final_answer = _deterministic_fallback(trace, question)
                result.grounding_warning = (
                    "The business-data investigation completed, but AI explanation generation is "
                    "temporarily unavailable because the model usage limit was reached."
                )
            else:
                result.final_answer = None
                result.grounding_warning = (
                    "AI investigation could not start because the model is temporarily unavailable "
                    "due to a usage limit."
                )

        elif _is_auth_error(exc):
            result.status = "auth_error"
            result.final_answer = _deterministic_fallback(trace, question) if trace else None
            result.grounding_warning = (
                "AI explanation generation failed due to an authentication error with the AI "
                "provider. Check that SARVAM_API_KEY is set correctly."
            )

        else:
            result.status = "api_error"
            result.final_answer = _deterministic_fallback(trace, question)
            result.grounding_warning = f"Agent execution failed after {len(trace)} tool call(s): {type(exc).__name__}: {exc}"

        if trace:
            result.confidence, result.confidence_reason = _deterministic_confidence(trace, allowed_tools=allowed_tools)
            result.strongest_signal = get_strongest_signal(trace, allowed_tools=allowed_tools)
        return result
