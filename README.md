<div align="center">

# 🧠 Neeman's AI Business Copilot

**An AI analytics dashboard that never lies about its numbers — and an RCA agent that explains *why* they moved, without pretending to prove causation it can't prove.**

[

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)

](https://www.python.org/)
[

![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

](https://streamlit.io/)
[

![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-1C3C3C?style=flat-square)

](https://langchain-ai.github.io/langgraph/)
[

![Sarvam AI](https://img.shields.io/badge/Sarvam_AI-sarvam--105b-6C5CE7?style=flat-square)

](https://sarvam.ai/)
[

![pandas](https://img.shields.io/badge/pandas-Business_Logic-150458?style=flat-square&logo=pandas&logoColor=white)

](https://pandas.pydata.org/)

**Live app:** _[ADD YOUR DEPLOYED STREAMLIT URL HERE]_ · **Repo:** _[ADD YOUR GITHUB REPO URL HERE]_

</div>

---

> 📸 *Add a screenshot or short GIF of the Executive Overview and an RCA answer here before publishing — nothing sells "grounded evidence" like seeing it work.*

## Why I built this

Neeman's is scaling — 46+ stores today, targeting 100 by FY2027-28, selling across its own website, marketplaces, and physical retail. At that scale, "why did revenue move this month?" stops being a five-minute question. Someone has to pull numbers from four different places, reconcile them by hand, and write up an explanation — and by the time it's done, the window to act on it has usually closed.

So I built two things that work as one product: a dashboard that answers **what's happening**, and an AI agent that investigates **why** — one that's structurally *not allowed* to make a number up, because I don't think "trust me, the AI checked" is good enough for something a business actually runs on.

## Table of Contents

- [Key Capabilities](#-key-capabilities)
- [Architecture](#-architecture)
- [RCA Reliability & Grounding](#-rca-reliability--grounding)
- [Demo Walkthrough](#-demo-walkthrough)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Assumptions & Scope](#-assumptions--scope)
- [From Prototype to Production](#-from-prototype-to-production)
- [Assignment Mapping](#-assignment-mapping)

---

## 🎯 Key Capabilities

### 📊 Business Analytics Copilot
The "what's happening" half — an Executive Overview and a tabbed Business Performance dashboard (Category / Channel / Inventory / Marketing), every number computed by pandas:

- Revenue, orders, units, and average order value, with period-over-period % change
- Category and channel performance, with revenue share and comparison breakdowns
- Inventory availability, stockout days, and low-stock SKU tracking
- Marketing spend, attributed revenue, and ROAS by channel

### 🔎 Root Cause Analysis Agent
The "why, and what should we do" half — ask it a question in plain English:

- Classifies what *kind* of question you asked — two-period comparison, single-dimension deep-dive, ranking, trend, or single-period lookup — **before** the LLM ever sees it
- For a general revenue question, automatically pulls category/channel/inventory/marketing evidence the moment the headline number is confirmed — it can't quietly skip a dimension
- Ranks contributing signals and reports a confidence level computed from how many strong signals actually turned up — not a vibe
- Turns evidence into specific, evidence-linked recommendations instead of generic advice

### 🛡️ Evidence & Reliability Mechanisms
The part that makes the other two trustworthy:

- Every number and named time period in the AI's answer is checked against the real tool evidence before it's shown
- One bounded retry if that check fails — then a deterministic, evidence-only fallback if it *still* fails. The user never sees an unverified answer.
- Confidence and "strongest signal" are plain functions over the evidence, not model guesses — computed once, reused everywhere

---

## 🏗️ Architecture

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

<details>
<summary><b>Why a strict one-way pipeline instead of a single LLM agent with open tool access?</b></summary>
<br>

Because I wanted a business figure to be traceable back to *exactly one function*, every time. Each layer here can only pass data downstream — only `analytics/metrics.py` is allowed to touch a dataframe or compute a number. That means an intent-classification bug can't corrupt the analytics, and the LLM's job is structurally limited to orchestration and prose. It can't "helpfully" recompute a percentage or fill in a missing month on its own — not because the prompt tells it not to, but because the code never gives it the chance.

</details>

---

## 🛡️ RCA Reliability & Grounding

| Principle | How it's enforced |
|---|---|
| Business numbers come from deterministic analytics | Every calculation lives in `analytics/metrics.py`; the agent only narrates results pandas already produced |
| The LLM never calculates a source metric | It orchestrates tool calls and writes prose — it never computes a %, a ranking, or a signal strength |
| Evidence is collected *before* synthesis | Dimension evidence is gathered automatically once the headline comparison succeeds, before the model writes anything |
| The question decides the investigation | A deterministic regex classifier picks the right analytical workflow before any LLM call — no guessing, no invented periods |
| Claims are checked against evidence | Every number and named period in the final answer is verified against that investigation's actual tool output |
| Causality is never overclaimed | Hedged language throughout ("contributing factor," "consistent with," "evidence suggests") plus an explicit Data Limitations note |
| Bounded recovery, not silent failure | One retry against the already-collected evidence if grounding fails; a deterministic, evidence-only summary if it fails again |

> **On causality:** the agent identifies contributing signals and associations (e.g. *"category X declined alongside an inventory stockout, both STRONG signals in the same window"*) — it does not, and does not claim to, prove causation.

---

## 🎬 Demo Walkthrough

**Ask it:** *"Why did revenue decline in July compared with June 2026?"*

```
1. Intent router → classifies as a two-period revenue RCA
2. resolve_named_period → locks June 2026 and July 2026 to exact dates
3. compare_sales_kpis → ₹15,448,942.10 (Jun) → ₹12,079,246.10 (Jul) = -21.81%
4. Mandatory evidence collection fires automatically:
     • Category   → Running: -30.88%
     • Inventory  → Availability 99.3% → 74.95%, stockouts 0 → 56 days
     • Marketing  → Meta attributed revenue: -34.77%
5. LLM synthesizes an answer using ONLY those numbers, tagged with
   tool-reported signal_strength and a computed confidence level
6. Every number + period in that answer is checked against steps 3-4
   before it's ever shown to you
```

The result names the inventory stockout and the Meta ROAS drop as the strongest observed signals, alongside the category-level decline — and says plainly that this is association, not proven causation.

---

## 🧰 Tech Stack

| Layer | Choice | Why |
|---|---|---|
| **LLM** | Sarvam AI — `sarvam-105b` (via `langchain-sarvam`) | Reasoning-capable, native tool-calling through the standard LangChain interface |
| **Orchestration** | LangGraph | An explicit state machine for the investigation loop, not an implicit agent loop that's hard to make deterministic |
| **Intent Routing** | Deterministic regex classifier | Decides question shape before any model call — no LLM-guessed periods |
| **Business Logic** | pandas | All calculations, kept entirely outside the LLM |
| **UI** | Streamlit + Plotly | Fast to build genuinely interactive charts, tabs, and a chat-style investigation flow |
| **Grounding** | Custom numeric + period validator | Checks every AI-stated figure against real tool evidence pre-display |

---

## 🚀 Getting Started

```bash
# 1. Clone and install
git clone <your-repo-url>
cd neemans-ai-copilot
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
SARVAM_API_KEY=your-sarvam-api-key
```

> Get a key at [dashboard.sarvam.ai](https://dashboard.sarvam.ai/). The dashboards work with **no key at all** — only the Root Cause Analysis page needs one.

Drop `sales.csv`, `products.csv`, `inventory.csv`, and `marketing.csv` into a `data/` folder at the project root, then:

```bash
streamlit run app.py
```

That's it — Executive Overview loads first, Business Performance and Root Cause Analysis are a click away in the sidebar.

---

## 📋 Assumptions & Scope

- **All data is synthetic**, purpose-built for this assignment — nothing here reflects Neeman's actual sales, inventory, or marketing performance.
- Patterns (a marketing spend cut, an inventory stockout, a category slowdown) were deliberately embedded so the RCA agent's findings are checkable against known ground truth, not just plausible-sounding.
- `sales.csv` has no `store_id`, so store-level RCA is out of scope here — it's a natural next step (see the AI Opportunity Roadmap).
- `signal_strength` thresholds (±10% = MODERATE, ±25% = STRONG) are fixed magnitude tiers, **not statistical significance tests**.
- Confidence (HIGH/MEDIUM/LOW) is computed from the count and strength of dimension-level signals — never model-estimated.

---

## 🏭 From Prototype to Production

This is a working prototype on synthetic data, built to demonstrate the architecture and grounding discipline — not a production system. I'd rather say that plainly than have you find out the hard way.

**✅ Already here:**
- Deterministic analytics as the single source of truth for every number
- LLM orchestration that investigates and narrates, never calculates
- Deterministic intent routing ahead of every LLM call
- Numeric/period grounding with bounded retry + deterministic fallback
- A working Streamlit UI over static CSVs

**🔜 Needed for real Neeman's data:**
- Live integrations — POS/ERP, warehouse systems, marketplace seller APIs (Amazon/Myntra/Flipkart), marketing platform APIs (Meta/Google Ads) — not a one-time CSV drop
- A centralized data warehouse/ELT layer upstream of `metrics.py`, so sources are actually reconciled, not just assumed to agree
- Authentication + role-based access control (a store manager shouldn't see company-wide financials)
- Data quality checks — missing dates, duplicate orders, schema drift, late-arriving data
- Monitoring/observability — tool-call failures, grounding-failure rates, latency, cost, data freshness
- A real evaluation set of investigation questions, run regularly to catch regressions
- Persisted investigation logs for auditability, especially before anything touches investor- or leadership-facing numbers
- Production-grade secrets management and access controls around the data layer

Scoped out deliberately so this assignment stays focused on the analytics/RCA architecture — a few of these map directly to opportunities in Part 3.

---

## 🗺️ Assignment Mapping

| Part | Deliverable | Where to find it |
|---|---|---|
| **Part 1** | Business Analytics Copilot | This app — Executive Overview + Business Performance pages |
| **Part 2** | Root Cause Analysis Agent | This app — Root Cause Analysis page, `agent/rca_agent.py` |
| **Part 3** | AI Opportunity Roadmap | Separate document — `AI_Opportunity_Roadmap.pdf` |

---

<div align="center">

*Built for Neeman's AI Intern assignment. Every number you see either came from pandas, or was checked against something that did.*

</div>
