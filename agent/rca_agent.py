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

WHAT CHANGED IN THIS VERSION
------------------------------------------------------------------------
Three regressions from a previous hardening pass had crept back in, found
by actually re-reading the code rather than trusting prior comments:

1. FIXED (again): `_collect_signals` now breaks ties by absolute magnitude
   of change, then a fixed dimension priority — not by trace insertion
   order. Two runs with identical evidence always rank identically.
2. FIXED (again): `_deterministic_fallback` now checks whether ANY
   dimension tool succeeded (`_any_dimension_tool_succeeded`), independent
   of whether compare_sales_kpis ran. A narrow question (e.g. "why did
   category performance change?") whose only evidence is WEAK no longer
   gets misreported as "not enough evidence".
3. ADDED: `get_strongest_signal(trace)` — a single deterministic function,
   exposed as `result.strongest_signal`. Previously app.py maintained its
   own separate copy of this logic that (incorrectly) included WEAK
   signals, while confidence only counts STRONG/MODERATE — a real,
   user-visible inconsistency (the "Strongest Signal" card could show a
   WEAK result while the confidence badge said "no strong signal found").
   app.py should now read `result.strongest_signal` instead of
   recomputing it.

NEWLY INTEGRATED: the period-context system prompt (previously an
unwired draft) is now the actual SYSTEM_PROMPT, built by
`_build_system_prompt()`. `run_investigation()` accepts optional
`current_period` / `comparison_period` dicts — the exact shape
`metrics.resolve_named_period()` already returns
(`{"start_date", "end_date", "label"}`) — so the UI's sidebar selection
can be passed straight through with no reshaping. When omitted, the
prompt falls back to telling the model to resolve periods itself from the
question text, preserving the old free-text-only behavior.

Because the prompt now promises "Confidence is determined by the
application... never infer it yourself", confidence has to reach the
model BEFORE it writes its final answer, not just be computed afterward
for the UI. `_mandatory_evidence_node` now computes deterministic
confidence immediately after dimension evidence is collected and tells
the model to use that exact value — this covers every path that can lead
to a final answer (the model's own natural completion AND the forced
synthesis fallback), not just the synthesis-only path.

Design guarantees (unchanged):
- Business metrics and signal strength come only from analytics tools.
- Required dimension evidence is enforced deterministically after a
  successful revenue KPI comparison.
- The same resolved periods are reused for all mandatory evidence.
- Numeric claims are grounded against current-investigation evidence.
- AI output is rejected/falls back if numeric grounding fails.
- An identical (tool, args) request is never executed twice.
- The whole graph is retried at most once, and ONLY if no tool has
  executed yet — once any tool has run, a failure goes straight to the
  deterministic fallback rather than duplicating analytics work.
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


def _classify_scope(question: str) -> set[str]:
    """Deterministically decides mandatory dimension scope. General
    questions get all four dimensions; a genuinely narrow question gets
    only that one; anything ambiguous defaults to all four rather than
    risking under-investigation."""
    q = re.sub(r"\s+", " ", question.lower()).strip()

    if any(marker in q for marker in _GENERAL_MARKERS):
        return set(DIMENSION_ORDER)

    q_for_matching = q.replace("marketing channel", "marketing")
    hits = {dim for dim, kws in _DIMENSION_KEYWORDS.items() if any(kw in q_for_matching for kw in kws)}

    if len(hits) == 1:
        return hits
    return set(DIMENSION_ORDER)


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
        "numbers and deterministic evidence.\n\n"
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
    "4. time-series/trend\n\n"
    "Use the corresponding final response format exactly.\n\n"
    "Prioritize the direct answer over background explanation.\n"
    "Do not add unrelated dimensions.\n"
    "Do not invent numbers, periods, rankings, confidence, or causes.\n"
    "Use signal_strength values exactly as returned by the tools.\n"
    "Do not convert correlation or signal strength into proven causation.\n"
    "If the question is narrow, keep the answer narrow.\n"
    "If evidence is insufficient, say so explicitly.\n"
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

    confidence, confidence_reason = _deterministic_confidence(trace)

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


def _retry_with_strict_grounding(llm_no_tools, system_prompt: str, trace: list, suspicious: list[str]) -> str | None:
    evidence_text = json.dumps(
        [{"tool": step["tool"], "input": step["input"], "result": step["result"]} for step in trace],
        default=str,
    )
    strict_prompt = (
        "Your previous answer contained numeric figures that could not be "
        f"verified against the tool evidence: {', '.join(suspicious)}.\n\n"
        "Rewrite the answer using ONLY the evidence below. Every numeric claim must "
        "match a value appearing in that evidence. Do not invent calculations, "
        "numbers, causes, or assumptions. Do not use causal certainty. Prefer "
        "'contributing factor', 'strongest observed signal', or 'consistent with'. "
        "If causation is not established, explicitly say: 'The available data does "
        "not establish direct causation.'\n\n"
        "Use the exact ROOT CAUSE ANALYSIS format from the system instructions.\n\n"
        f"EVIDENCE:\n{evidence_text}"
    )
    try:
        response = llm_no_tools.invoke([SystemMessage(content=system_prompt), HumanMessage(content=strict_prompt)])
        return _extract_text(response.content).strip() or None
    except Exception:
        return None


# ============================================================
# DETERMINISTIC SIGNALS / STRONGEST SIGNAL / CONFIDENCE
# ============================================================

def _collect_signals(trace: list) -> list[tuple]:
    """(dimension, label, change, signal_strength) for every STRONG/MODERATE
    signal, deterministically ranked by:
      1. signal strength (STRONG > MODERATE)
      2. absolute magnitude of the relevant change (larger first)
      3. fixed dimension priority (Category > Channel > Inventory > Marketing)
    NEVER by trace insertion order."""
    signals = []
    for step in trace:
        tool, result = step["tool"], step["result"]
        if tool == "compare_category_performance":
            for r in result.get("categories", []):
                if r.get("signal_strength") in ("STRONG", "MODERATE"):
                    signals.append(("Category", r.get("category", "Unknown"), r.get("revenue_change_pct"), r["signal_strength"]))
        elif tool == "compare_channel_performance":
            for r in result.get("channels", []):
                if r.get("signal_strength") in ("STRONG", "MODERATE"):
                    signals.append(("Channel", r.get("channel", "Unknown"), r.get("revenue_change_pct"), r["signal_strength"]))
        elif tool == "compare_marketing_performance":
            for r in result.get("channels", []):
                if r.get("signal_strength") in ("STRONG", "MODERATE"):
                    signals.append(("Marketing", r.get("channel", "Unknown"), r.get("attributed_revenue_change_pct"), r["signal_strength"]))
        elif tool == "compare_inventory_performance" and result.get("status") == "ok":
            if result.get("signal_strength") in ("STRONG", "MODERATE"):
                signals.append(("Inventory", "Availability / stockouts", result.get("availability_change_pct_points"), result["signal_strength"]))

    strength_rank = {"STRONG": 2, "MODERATE": 1}
    signals.sort(
        key=lambda s: (
            -strength_rank.get(s[3], 0),
            -(abs(s[2]) if s[2] is not None else 0),
            DIMENSION_PRIORITY.get(s[0], 99),
        )
    )
    return signals


def get_strongest_signal(trace: list) -> dict | None:
    """The single deterministic answer to "what's the strongest signal" —
    the UI should display exactly this, not recompute its own version
    (a prior version of app.py maintained a separate copy that
    inconsistently included WEAK signals)."""
    signals = _collect_signals(trace)
    if not signals:
        return None
    dimension, label, change, strength = signals[0]
    return {"dimension": dimension, "label": label, "change": change, "strength": strength}


def _any_dimension_tool_succeeded(trace: list) -> bool:
    dimension_tools = set(DIMENSION_TOOL.values())
    return any(step["tool"] in dimension_tools and step["result"].get("status") == "ok" for step in trace)


def _deterministic_confidence(trace: list) -> tuple[str | None, str | None]:
    if not trace:
        return None, None
    signals = _collect_signals(trace)
    strong_count = sum(1 for s in signals if s[3] == "STRONG")
    moderate_count = sum(1 for s in signals if s[3] == "MODERATE")

    if strong_count >= 2:
        return "HIGH", f"{strong_count} independent STRONG-signal dimensions were found."
    if strong_count == 1 or moderate_count >= 2:
        return "MEDIUM", "At least one meaningful signal was found, but causality is not established."
    if _any_dimension_tool_succeeded(trace):
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
    if not period:
        return "unknown period"
    start = period.get("start") or period.get("start_date")
    end = period.get("end") or period.get("end_date")
    return f"{start} to {end}" if start and end else "unknown period"


# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================

def _deterministic_fallback(trace: list) -> str:
    """Evidence-only fallback — never calls the LLM. Correctly distinguishes
    "a dimension was investigated but showed nothing STRONG/MODERATE" from
    "nothing was investigated at all", even for narrow questions that never
    called compare_sales_kpis."""
    if not trace:
        return "AI investigation could not be completed and no business data was gathered. Please try again shortly."

    pct_change, current_period, previous_period = _extract_headline(trace)
    signals = _collect_signals(trace)
    any_dimension_evidence = _any_dimension_tool_succeeded(trace)

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

    if signals:
        lines.append("The strongest observed contributing factors were:")
        for dimension, label, change, strength in signals[:5]:
            change_text = f"{change:+.2f}" if change is not None else "—"
            lines.append(f"• {dimension} — {label}: {change_text} ({strength} signal)")
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
    initial_state: AgentState = {
        "messages": [SystemMessage(content=system_prompt), HumanMessage(content=question)],
        "tool_call_count": 0,
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

        final_answer = _extract_text(final_state["messages"][-1].content).strip() if final_state.get("messages") else ""

        if not final_answer:
            result.final_answer = _deterministic_fallback(trace)
            result.status = "synthesis_fallback"
        else:
            suspicious = _check_grounding(final_answer, trace)
            if suspicious:
                retried = _retry_with_strict_grounding(llm_no_tools, system_prompt, trace, suspicious)
                retried_suspicious = _check_grounding(retried, trace) if retried else suspicious
                if retried and not retried_suspicious:
                    result.final_answer = retried
                else:
                    result.final_answer = _deterministic_fallback(trace)
                    result.status = "grounding_failed"
                    result.grounding_warning = (
                        "The AI-generated explanation contained figures that could not be verified "
                        "against the collected evidence. A deterministic evidence-only summary is shown instead."
                    )
            else:
                result.final_answer = final_answer

        result.confidence, result.confidence_reason = _deterministic_confidence(trace)
        result.strongest_signal = get_strongest_signal(trace)
        return result

    except Exception as exc:
        result.trace = trace
        _pct_change, current_period_used, previous_period_used = _extract_headline(trace)
        result.current_period = current_period_used
        result.previous_period = previous_period_used

        if _is_rate_limit_error(exc):
            result.status = "model_rate_limited"
            if trace:
                result.final_answer = _deterministic_fallback(trace)
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
            result.final_answer = _deterministic_fallback(trace) if trace else None
            result.grounding_warning = (
                "AI explanation generation failed due to an authentication error with the AI "
                "provider. Check that SARVAM_API_KEY is set correctly."
            )

        else:
            result.status = "api_error"
            result.final_answer = _deterministic_fallback(trace)
            result.grounding_warning = f"Agent execution failed after {len(trace)} tool call(s): {type(exc).__name__}: {exc}"

        if trace:
            result.confidence, result.confidence_reason = _deterministic_confidence(trace)
            result.strongest_signal = get_strongest_signal(trace)
        return result
