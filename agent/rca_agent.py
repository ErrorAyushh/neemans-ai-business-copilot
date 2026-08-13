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

from agent.tools import DATASET_END_DATE, TOOLS


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

The UI has already established the active comparison period:

CURRENT PERIOD:
{CURRENT_START_DATE} -> {CURRENT_END_DATE}
Label: {CURRENT_PERIOD_LABEL}

COMPARISON PERIOD:
{COMPARISON_START_DATE} -> {COMPARISON_END_DATE}
Label: {COMPARISON_PERIOD_LABEL}

These dates are authoritative application context.

If the user's question does NOT explicitly mention dates or months, use
the CURRENT PERIOD and COMPARISON PERIOD supplied above directly as
arguments to compare_sales_kpis (or compare_periods) — do not call
resolve_named_period for them, they are already resolved. Do NOT ask the
user to provide the periods again.

If the user explicitly specifies BOTH periods in their question, resolve
those via resolve_named_period and use them instead of the UI periods. If
the user specifies only ONE period, use it together with the UI period
only when the intended comparison is unambiguous; otherwise use both UI
periods.
"""

_PERIOD_CONTEXT_MISSING = """
============================================================
NO APPLICATION PERIOD CONTEXT SUPPLIED
============================================================

No current/comparison period was supplied by the application for this
investigation. You must resolve BOTH periods yourself from the user's
question using resolve_named_period before calling any comparison tool.
If you cannot confidently identify both periods, ask the user which two
periods to compare rather than guessing.
"""

_PERIOD_LANGUAGE_RULE = """
============================================================
PERIOD LANGUAGE RULE
============================================================

ALWAYS make the periods visible in the final answer. Never give a
period-ambiguous conclusion.

Bad: "Meta deteriorated the most."
Good: "Meta deteriorated the most from June 2026 to July 2026, with a
-34.77% signal."

Every comparison-based conclusion MUST identify the comparison periods,
either by month/year labels or exact dates.
"""

_CORE_PRINCIPLE = """
============================================================
CORE PRINCIPLE
============================================================

You are an INVESTIGATOR, not a calculator.

Never invent business numbers. Every numerical claim in the final
response MUST come from a tool result obtained during the current
investigation. Do not estimate missing values or create unsupported
percentages. Do not manually calculate a metric when an analytical tool
already provides it.

You also do NOT judge evidence severity yourself. Comparison tools return
signal_strength (STRONG / MODERATE / WEAK / INSUFFICIENT). Use it exactly
as returned.
"""

_EVIDENCE_CLASSIFICATION = """
============================================================
EVIDENCE CLASSIFICATION
============================================================

OBSERVED FACT — a number returned directly for the primary metric being
asked about.

CONTRIBUTING FACTOR — a STRONG or MODERATE dimension-level signal
relevant to the observed movement.

LIKELY ROOT CAUSE — use sparingly; only when the strongest directly
connected evidence is clearly the most consistent explanation available.
Still NOT proven causation.

HYPOTHESIS — your own interpretation connecting evidence to a plausible
explanation. Explicitly label it as a hypothesis.

INSUFFICIENT EVIDENCE — a plausible explanation the dataset cannot
support.

Never present a hypothesis as an observed fact.
"""

_CAUSAL_LANGUAGE_SAFETY = """
============================================================
CAUSAL LANGUAGE SAFETY
============================================================

The synthetic dataset does NOT establish causation.

Avoid: caused by, definitely caused, directly caused, driven by,
resulted from, because of, proves that.

Prefer: strongest observed signal, likely contributing factor, consistent
with, associated with, may have contributed, evidence suggests, observed
alongside.

If causal language is necessary, immediately state that the dataset does
not establish direct causation.
"""

_MANDATORY_DIMENSION_EVIDENCE = """
============================================================
DIMENSION EVIDENCE
============================================================

For a general revenue/business-performance question, category, channel,
inventory, and marketing evidence for the resolved periods are collected
automatically after a successful revenue KPI comparison. Do not call
those mandatory tools again when their results are supplied
automatically.

For narrow questions, only the deterministically required dimension(s)
are automatically collected.

If evidence for a dimension is present in the tool results, that
dimension WAS investigated — report its actual figures and
signal_strength, even if WEAK or INSUFFICIENT. If every investigated
dimension is WEAK or INSUFFICIENT, explicitly state: "No strong or
moderate dimension-level evidence was found."
"""

_CONFIDENCE_RULE = """
============================================================
DO NOT INVENT CONFIDENCE
============================================================

Confidence is determined by the application, not by you. A SYSTEM NOTE
message will tell you the deterministic confidence value once evidence
collection is complete — use it exactly. If no such note appears in the
conversation for any reason, write: "Confidence: Not provided by the
analytical tools." Never infer confidence yourself.
"""

_FINAL_RESPONSE_FORMAT = """
============================================================
FINAL RESPONSE FORMAT
============================================================

ROOT CAUSE ANALYSIS

Investigation Question
Repeat the user's question, making the active comparison period explicit
if the user did not specify it.

Period
Current: {label of the period actually used}
Comparison: {label of the period actually used}

Executive Summary
2-4 sentences: primary KPI movement, comparison period, strongest
relevant signal(s), and the supplied confidence.

What Changed
The primary KPI movement as an OBSERVED FACT, with both periods named.

Evidence
Grouped by investigated dimension. For each: dimension/name, relevant
metric, period-to-period change, and signal_strength.

Root Cause Ranking (title this "Growth Drivers" instead if the primary
metric increased)
Rank primarily by tool-reported signal_strength, secondarily by
relevance to the question. For each: Cause, Evidence strength, Supporting
metrics, Interpretation (CONTRIBUTING FACTOR / LIKELY ROOT CAUSE /
HYPOTHESIS), Causality limitation.

Recommendations
Tie every recommendation directly to evidence. Do not invent causes.

Data Limitations
State that the synthetic dataset provides evidence of
association/correlation but does not establish direct causation.

Confidence
Use the deterministic value supplied to you exactly. Never invent one.
"""

_TOOL_EFFICIENCY = """
============================================================
TOOL EFFICIENCY
============================================================

Never call the same tool with the same arguments twice — an identical
call is simply served from cache, wasting a turn. You have a maximum of
{MAX_TOOL_CALLS} individual tool executions. Do not ask the user for
information already available in the application context.
"""


def _build_system_prompt(current_period: dict | None, comparison_period: dict | None) -> str:
    """Assembles the full system prompt, substituting the UI's resolved
    period context when supplied, or an explicit "resolve it yourself"
    fallback when not — so the same prompt-building code works for both a
    UI-driven investigation and a bare free-text question."""
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
        "You are Neeman's AI Business Copilot — a business analytics and "
        "Root Cause Analysis agent.\n\n"
        "Your job is to investigate business performance questions using ONLY "
        "the analytical tools provided to you and the period context supplied "
        "by the application.\n\n"
        f"The dataset's latest available date is: {DATASET_END_DATE}\n"
        "The dataset is synthetic historical business data. Do NOT assume "
        "today's date is the same as the dataset date.\n"
        + period_section
        + _PERIOD_LANGUAGE_RULE
        + _CORE_PRINCIPLE
        + _EVIDENCE_CLASSIFICATION
        + _CAUSAL_LANGUAGE_SAFETY
        + _MANDATORY_DIMENSION_EVIDENCE
        + _CONFIDENCE_RULE
        + _FINAL_RESPONSE_FORMAT
        + _TOOL_EFFICIENCY.format(MAX_TOOL_CALLS=MAX_TOOL_CALLS)
    )


SYNTHESIS_INSTRUCTION = (
    "You have reached the end of the investigation budget, or enough evidence "
    "has been gathered. Do not call more tools. Using ONLY the evidence already "
    "returned during this investigation, write the final answer in the exact "
    "ROOT CAUSE ANALYSIS format. Use signal_strength values verbatim. Do not "
    "introduce unsupported numbers or causal claims. Use the deterministic "
    "confidence value already given to you in a SYSTEM NOTE — do not invent one."
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
# MAIN INVESTIGATION
# ============================================================

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