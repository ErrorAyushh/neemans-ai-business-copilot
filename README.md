# Neeman's AI Business Copilot

An AI-powered business intelligence dashboard and Root Cause Analysis (RCA) agent built for Neeman's AI Intern assignment — **Part 1 (Business Analytics Copilot)** + **Part 2 (RCA Agent)**, delivered as a single coherent product. **Part 3 (AI Opportunity Roadmap)** is a separate document; see [Assignment Mapping](#assignment-mapping) below.

**Live app:** https://neemans-ai-business-copilot-dk3sbk24rnuy9hbzr8wgmp.streamlit.app/
**Repo:** _[ADD YOUR GITHUB REPO URL HERE]_

---

## Problem Statement

Neeman's is scaling fast — 46+ stores today, targeting 100 by FY2027-28, selling across its own website, marketplaces (Amazon, Myntra, Flipkart), and physical retail. That growth means performance questions get harder to answer quickly: *why did revenue move this month, which category or channel drove it, is inventory or marketing the reason, and what should the team do about it?*

Today, answering that well means someone manually pulling numbers from several sources, reconciling them, and writing up an explanation — by the time it's done, the window to act has often passed. Neeman's own AI Intern JD asks for exactly this: internal AI copilots that "eliminate manual work, improve decision-making, and unlock new business capabilities."

## Solution Overview

**Nexus (Neeman's AI Business Copilot)** is a two-part product built as one system:

1. **Part 1 — Business Analytics Copilot** — an Executive Overview and a Business Performance dashboard (Category / Channel / Inventory / Marketing) that answer *"what's happening"*, built entirely on deterministic pandas calculations.
2. **Part 2 — Root Cause Analysis Agent** — answers *"why is it happening, and what should we do"*, using an LLM (Sarvam AI) orchestrated with LangGraph to investigate, but never to calculate. Every number the agent states is pulled from the same analytics layer the dashboards use and is checked against that evidence before being shown to the user.

The two halves share one source of truth, so "what happened" and "why" can never disagree with each other.

## Key Capabilities

**Business Analytics Copilot (Part 1)**
- Executive Overview: revenue, orders, units, and average order value, with period-over-period % change — computed entirely by pandas, nothing estimated.
- Business Performance dashboard: category, channel, inventory, and marketing performance, each with period comparisons, revenue share, and ROAS/availability trends.
- Every number on these pages traces back to a single function in `analytics/metrics.py`.

**Root Cause Analysis Agent (Part 2)**
- Accepts free-text investigation questions and classifies their intent *before* any LLM call: two-period RCA, single-dimension investigation, dimension ranking, cross-period ranking, time-series trend, or single-period lookup.
- For a revenue-change question, automatically collects category/channel/inventory/marketing evidence the moment the headline comparison succeeds — no dimension can be silently skipped by the model.
- Ranks contributing signals and reports a confidence level (HIGH/MEDIUM/LOW) computed from how many STRONG/MODERATE signals were found — never estimated by the model.
- Generates recommendations tied directly to observed evidence, not generic advice.

**Evidence / Reliability Mechanisms**
- Numeric and period grounding: every figure and named period in the AI's answer is checked against the actual tool evidence from that investigation.
- Bounded recovery: if grounding fails, the model gets exactly one retry against the same evidence; if that also fails, a deterministic, evidence-only summary is shown instead of an unverified answer.
- Signal ranking, confidence, and the "strongest signal" are pure functions over the evidence trace — the UI reads these directly rather than re-deriving its own version.

## Architecture

```
data/*.csv
     │
     ▼
analytics/metrics.py         ← single source of truth for all business math
     │                          (revenue, category/channel/inventory/marketing
     │                           comparisons, signal_strength, period resolution,
     │                           monthly ranking, revenue trend)
     ▼
agent/tools.py                ← thin LangChain @tool wrappers around metrics.py
     │                          (no calculation logic of its own)
     ▼
agent/intent_router.py        ← deterministic, regex-based question classifier
     │                          (runs before any LLM call; routes each question to
     │                           either direct analytics or the RCA graph)
     ▼
agent/rca_agent.py            ← LangGraph orchestration + Sarvam AI (sarvam-105b):
     │                          agent → tools → mandatory-evidence → synthesize
     │                          graph, followed by numeric/period grounding
     │                          validation → one bounded correction retry →
     │                          deterministic fallback if still unverified
     ▼
app.py                        ← Streamlit UI: Executive Overview, Business
     │                          Performance (tabbed), Root Cause Analysis
     ▼
charts.py                     ← Plotly visualizations only — reads already-
                                 computed values, never recalculates them
```

**Why this shape:** this is a strict one-way pipeline, not a single LLM agent with open-ended tool access, on purpose. Every layer can only pass data downstream, and only `analytics/metrics.py` is allowed to touch a dataframe or compute a number. That means any business figure can be traced back to exactly one function, an intent-classification bug can't corrupt the analytics layer, and the LLM's role is structurally limited to orchestration and prose — it can't end up "helpfully" recomputing a percentage or filling in a missing month on its own, even if a prompt fails to stop it.

**Data flow guarantee:** a number can only reach the screen by originating in `analytics/metrics.py`. Nothing upstream of that file is allowed to compute a business figure.

## RCA Reliability / Grounding

- **Business numbers come from deterministic analytics.** All revenue, percentage-change, ROAS, inventory-availability, monthly-ranking, and trend math lives in `analytics/metrics.py`; the agent only reads and narrates results pandas already produced.
- **The LLM does not calculate source metrics.** It orchestrates tool calls and writes the final explanation — it never computes a percentage, a ranking, or a signal strength itself.
- **Evidence is collected before synthesis.** For a general revenue question, category/channel/inventory/marketing evidence is gathered automatically once the headline comparison succeeds, before the model is asked to write a final answer.
- **Intent routing determines which analytics are investigated.** A deterministic regex classifier decides the shape of the question before any LLM call, so the model is never left to guess which tool applies, or to invent a second time period just to force-fit the wrong tool.
- **Generated claims are checked against evidence where implemented.** Every number and named period in the AI's final answer is checked against the tool evidence collected during that investigation; unverifiable figures are flagged rather than shown.
- **Causality is not claimed without evidence.** The agent's language is deliberately hedged ("contributing factor," "consistent with," "evidence suggests"), and its "Data Limitations" section states plainly that association is not causation.
- **Deterministic fallback exists if AI synthesis cannot be safely verified.** If a generated answer fails grounding, the system attempts bounded recovery — one retry against the already-collected evidence — and if that also fails, replaces the answer with a deterministic, evidence-only summary rather than showing anything unverified.

### Causality

The RCA agent identifies **contributing signals and associations** in the data (e.g. "category X declined alongside an inventory stockout, both STRONG signals in the same window"). It does **not** prove causation, and its language is deliberately hedged rather than asserting a definitive cause. This distinction is enforced in the agent's system prompt and reflected in the "Data Limitations" section of every RCA response.

## AI Models / Tools Used

| Component | Choice | Why |
|---|---|---|
| LLM | **Sarvam AI — sarvam-105b** (via `langchain-sarvam`) | Reasoning-capable model with native tool-calling through the standard LangChain `BaseChatModel` interface |
| Orchestration | **LangGraph** | Explicit state machine for the investigation loop (agent → tools → mandatory evidence → synthesis), rather than an implicit agent loop that's harder to make deterministic |
| Intent routing | **Deterministic regex classifier** (`agent/intent_router.py`) | Decides the shape of the question — two-period RCA, dimension investigation, ranking, trend, single-period lookup — before any model call, so the LLM is never forced to guess a period it wasn't given |
| Business logic | **pandas** | All calculations — kept entirely separate from and outside the LLM |
| UI | **Streamlit + Plotly** | Fast to build a genuinely interactive internal tool with charts, tabs, and a chat-style investigation flow |
| Grounding/validation | Custom numeric + period matching layer in `rca_agent.py` | Checks every number and named period in an AI answer against the actual evidence the tools returned before it's shown |

## Setup Instructions

**1. Clone and install**
```bash
git clone <your-repo-url>
cd neemans-ai-copilot
pip install -r requirements.txt
```

**2. Configure your API key**

Create a `.env` file in the project root:
```
SARVAM_API_KEY=your-sarvam-api-key
```
Get a key from [dashboard.sarvam.ai](https://dashboard.sarvam.ai/). The dashboards work without a key; only the RCA Agent page needs it.

**3. Data**

Place `sales.csv`, `products.csv`, `inventory.csv`, and `marketing.csv` in a `data/` folder at the project root. This submission uses synthetic data with deliberately embedded patterns (a marketing spend cut, an inventory stockout event, a category slowdown) so the RCA agent has real signal to discover — see **Assumptions** below.

**4. Run**
```bash
streamlit run app.py
```

## Demo / Example Investigation

**Question:** *"Why did revenue decline in July compared with June 2026?"*

1. The intent router classifies this as a two-period revenue RCA and the agent resolves June 2026 and July 2026 via `resolve_named_period`.
2. `compare_sales_kpis` returns the headline comparison — in this project's synthetic dataset, June revenue of ₹15,448,942.10 against July revenue of ₹12,079,246.10, a **-21.81%** change.
3. Once that headline comparison succeeds, category, channel, inventory, and marketing evidence for the same two periods is collected automatically — for example, the Running category at **-30.88%**, inventory availability falling from **99.3% to 74.95%** with stockout days rising from **0 to 56**, and Meta-attributed revenue at **-34.77%**.
4. The model writes an explanation using only those figures, tagging each contributing factor with its tool-reported `signal_strength` and a confidence level computed from how many STRONG/MODERATE signals were found.
5. Before the answer is shown, every number and period it mentions is checked against the evidence from steps 2–3. If anything doesn't match, the system retries once against that same evidence; if it still doesn't match, a deterministic summary built directly from the tool evidence is shown instead.

The result names the inventory stockout and the Meta ROAS decline as the strongest observed contributing signals alongside the category-level drop — and states explicitly that this is association, not proven causation.

## Assumptions

- **All data is synthetic/dummy data, generated for this assignment.** Every number the dashboard or RCA agent shows is a demonstration of the system working correctly on that synthetic dataset — none of it reflects Neeman's actual sales, inventory, or marketing performance.
- Patterns (e.g. a category decline coinciding with an inventory stockout) were deliberately embedded so the RCA agent's findings are verifiable against known ground truth, not just plausible-sounding.
- The current dataset is not attributed to a specific store or region for orders (`sales.csv` has no `store_id`), so store-level RCA is out of scope for this submission — the AI Opportunity Roadmap (Part 3) addresses this as a natural next step.
- `signal_strength` thresholds (±10% = MODERATE, ±25% = STRONG) are fixed, documented magnitude tiers, **not statistical significance tests** — the app and the agent are both worded to reflect that distinction.
- Confidence levels (HIGH/MEDIUM/LOW) are computed from the number and strength of dimension-level signals found, not from any model-estimated certainty.

## Productionization

What's here today is a working prototype on synthetic data, built to demonstrate the architecture and grounding discipline — not a production system. Being explicit about that line matters more than pretending otherwise.

**Implemented in this submission:**
- Deterministic analytics layer (pandas) as the single source of truth for every business number
- LLM orchestration (LangGraph + Sarvam AI) that investigates and narrates but never calculates
- Deterministic intent routing before any LLM call
- Numeric/period grounding validation with a bounded retry and deterministic fallback
- A working Streamlit UI over static CSV files

**Would be required before this could run on real Neeman's data:**
- **Real data integrations** — this currently reads static CSVs; production would need live connections to Neeman's actual systems (POS/ERP, warehouse/inventory management, marketplace seller APIs for Amazon/Myntra/Flipkart, and marketing platform APIs for Meta/Google Ads), not a one-time file drop
- **A centralized data warehouse** — reconciling sales, inventory, and marketing data across sources reliably needs a proper warehouse/ELT layer upstream of `analytics/metrics.py`, not four independent CSVs assumed to already agree with each other
- **Authentication and role-based access control** — today anyone with the app URL sees everything; a production internal tool needs login and role-scoped access (e.g. a store manager shouldn't see company-wide financials)
- **Data quality checks** — missing dates, duplicate orders, schema drift, and late-arriving data all need validation before they reach the analytics layer; none of that exists here
- **Monitoring and observability** — tracking tool-call failures, grounding-failure rates, LLM latency/cost, and data-freshness in production, not just returning a degraded response in the UI
- **Evaluation** — a real test set of investigation questions with expected answers, run regularly against the agent to catch regressions when prompts, models, or analytics logic change
- **Auditability** — a persisted log of every investigation (question, tool calls, evidence, final answer) for compliance and post-hoc review, especially before this touches anything investor- or leadership-facing
- **Security** — secrets management beyond a local `.env` file, network/access controls around the data warehouse, and a review of what business data an LLM API call is allowed to see

None of the above is implemented in this submission. It's scoped out deliberately so the assignment stays focused on the analytics/RCA architecture itself — see the AI Opportunity Roadmap (Part 3) for how a couple of these gaps map to concrete next opportunities.

## Assignment Mapping

| Part | Deliverable | Where |
|---|---|---|
| Part 1 | Business Analytics Copilot | This app — Executive Overview + Business Performance pages |
| Part 2 | Root Cause Analysis Agent | This app — Root Cause Analysis page, `agent/rca_agent.py` |
| Part 3 | AI Opportunity Roadmap | Separate document: `AI_Opportunity_Roadmap.pdf` |
