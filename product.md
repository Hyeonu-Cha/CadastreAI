# 🏠 CadastreAI — Product Document

> **The property register, intelligent.** An AI research agent for the Australian housing market — ask anything from "should I buy in Parramatta?" to "what does AHURI say about build-to-rent?" and get answers grounded in authoritative reports and live data, with citations you can audit.

*Cadastre (n.): the official register of property ownership, boundaries, and value — the foundational record of a nation's land.*

**Document version**: 1.0
**Status**: Pre-development (engineering starts Week 1 of 4-week build)
**Owner**: [Your name]

---

## 1. Executive Summary

### 1.1 What is CadastreAI?

CadastreAI is an AI-powered research assistant specialized in the Australian residential property market. It combines:

- **Retrieval-Augmented Generation (RAG)** over a curated corpus of ~300 authoritative reports (RBA, AHURI, CoreLogic/Cotality, Grattan Institute, SQM Research, Domain, PropTrack, Treasury, APRA)
- **Live-data tool-use** against public APIs (RBA statistics, ABS property indexes, SQM vacancy data, AIHW dashboards)
- **A fine-tuned embedding model** trained specifically on Australian housing terminology, so retrieval actually works for jargon like LVR, DTI, SAAR, hedonic index, NHFIC, HILDA

### 1.2 Why does it exist?

Australian housing decisions involve enormous sums of money and life-shaping consequences. Yet the information needed to make them is **fragmented across dozens of sources**, locked in dense PDFs, or gated behind paywalls. General-purpose tools (ChatGPT, Google) fail in three specific ways:

1. **No domain grounding** — they hallucinate numbers and invent sources
2. **No live data** — they can't tell you this week's cash rate or a specific postcode's vacancy rate
3. **No jargon fluency** — they confuse "days on market" with "days to list," miss that "LVR" and "loan-to-value" are the same thing, don't know AHURI is a research institute

CadastreAI fixes all three.

### 1.3 Who is it for?

Three personas, one product — with persona-aware UX:

| Persona | Primary need | Example query |
|---|---|---|
| 🏡 **Homebuyer** | Practical, local, decision-oriented | "Is Parramatta good for a family on $1.2M budget?" |
| 💼 **Investor** | Numeric, comparative, yield-focused | "Sydney suburbs with highest 3yr capital growth and sub-3% vacancy" |
| 🎓 **Researcher / Analyst** | Methodological, cross-source, rigorous | "How do CoreLogic's and ABS's price indexes differ methodologically?" |

### 1.4 Elevator pitch

> "Perplexity meets an AU housing economist. CadastreAI turns the fragmented landscape of RBA, AHURI, CoreLogic and ABS data into a single conversational research surface — every answer cited, every number traceable."

### 1.5 Name & positioning

**Cadastre** is the authoritative term for a nation's property register — the foundational record from which all property data flows. The name signals three things simultaneously: (1) deep domain expertise, (2) authoritative, research-grade intent, and (3) an Australia-first identity (the NSW cadastre, the Victorian cadastre, etc. are active public records). It's a name that analysts, policy researchers, and property professionals will immediately recognise, while remaining curious and memorable to everyone else.

---

## 2. Problem Statement

### 2.1 The user problem

A prospective homebuyer, investor, or analyst researching Australian property currently has to:

1. Visit 5–10 different websites (RBA, ABS, CoreLogic, Domain, SQM, AHURI, Grattan, state revenue offices)
2. Download multi-hundred-page PDFs and skim manually
3. Export and reconcile CSVs with inconsistent geographic definitions (LGA vs SA4 vs postcode)
4. Remember which source uses hedonic indexes vs median prices vs repeat-sales methodology
5. Synthesize everything into a decision — without expert help

This typically takes hours per question. Most people skip the research entirely and rely on agent opinion, social media, or gut feel — then make the largest financial decision of their life.

### 2.2 Why general AI doesn't solve this

| Problem | Generic LLM (ChatGPT, Gemini, etc.) | CadastreAI |
|---|---|---|
| Grounding | Hallucinates numbers | Every numeric claim cites a source |
| Freshness | Training cutoff months/years ago | Live RBA + ABS tool calls |
| AU-specific | US-biased data and vocabulary | AU-native corpus and terminology |
| Jargon | Misinterprets domain terms | Fine-tuned embeddings understand LVR = loan-to-value etc. |
| Multi-step | One-shot answer | Decomposes complex queries, reflects on gaps |

### 2.3 Why this niche matters

- AU housing is worth **>$10 trillion AUD**, the country's largest asset class
- **~600,000 property transactions per year**
- Housing is the #1 topic in Australian public policy, finance journalism, and household finance decisions
- No current tool combines the authoritative research corpus with live data in a conversational interface

---

## 3. Product Scope

### 3.1 In scope (v1)

- Residential property focus (houses, units, townhouses)
- Australia-wide with strongest coverage in capital cities
- Research-report-grade answers to questions about: market trends, affordability, rental market, policy, methodology, specific suburb/postcode data
- Citation-grounded responses with clickable source links
- Persona-aware UX (homebuyer, investor, researcher)
- Text + embedded chart outputs

### 3.2 Explicitly out of scope (v1)

These are valid v2+ directions but we are **cutting them** to make the 4-week build achievable:

- Commercial and industrial property
- Property-level valuations (would require licensed data from CoreLogic/PropTrack)
- Live MLS / for-sale listings
- User accounts, saved searches, watchlists
- Multi-turn conversational memory across sessions
- Mobile native apps (web-responsive only)
- Financial product recommendations (mortgages, insurance) — regulatory minefield
- Non-English support
- Voice interface

### 3.3 Non-goals

- Replacing a buyer's agent or mortgage broker
- Giving personalized financial advice (we'll be explicit about this in UX)
- Real-time alerting on market changes

---

## 4. User Personas (Detailed)

### 4.1 Persona A — "Searching Sarah" (Homebuyer)

**Demographics**: 32, Sydney, combined household income $180K, first-home buyer, renting in Newtown, looking to buy within 6 months.

**Goals**
- Understand if now is a good time to buy
- Compare 3–5 candidate suburbs
- Understand first-home-buyer schemes (NSW First Home Buyer Assistance, First Home Guarantee)
- Make sense of stamp duty and upfront costs

**Pain points**
- Overwhelmed by conflicting opinions online
- Doesn't know how to interpret data (auction clearance rates, days on market)
- Scared of making a half-million-dollar mistake

**Success looks like**
- Gets clear, plain-language answers with references she can verify
- Feels confident explaining her decision to her partner/parents
- Spends 20 minutes on CadastreAI instead of 5 hours of Google searching

### 4.2 Persona B — "Investor Ian" (Investor)

**Demographics**: 45, Melbourne, already owns PPR + 2 IPs, looking for next investment, values yield + growth balance.

**Goals**
- Screen suburbs by multiple criteria (yield, vacancy, growth, infrastructure)
- Understand current lending environment (rates, APRA policies)
- Track how different capital cities compare right now

**Pain points**
- Existing tools (CoreLogic RP Data, etc.) are expensive and clunky
- Hard to synthesize macro trends with micro suburb data
- Needs speed — wants answers in minutes, not hours

**Success looks like**
- Shortlists 5 suburbs from 40+ candidates in a single session
- Has backing data to justify decisions to spouse/accountant
- Catches market shifts faster than peers relying on property podcasts

### 4.3 Persona C — "Analyst Alex" (Researcher / Journalist / Student)

**Demographics**: 28, policy think-tank researcher, or property journalist, or finance Honours/PhD student.

**Goals**
- Write well-sourced articles or papers
- Trace a claim back to primary sources quickly
- Compare methodologies and findings across reports
- Build datasets combining multiple official sources

**Pain points**
- Too much time finding and reading PDFs
- Hard to know what RBA has said on topic X over the years
- Needs rigor — every figure must be traceable

**Success looks like**
- Uses CadastreAI as a first-pass literature review tool
- Gets pointed to the exact page and paragraph in relevant PDFs
- Saves 2–3 hours per article/chapter

---

## 5. Core Features

### 5.1 Feature map

| # | Feature | Persona(s) | Priority |
|---|---|---|---|
| F1 | Persona-aware chat interface | All | P0 |
| F2 | Grounded answers with inline citations | All | P0 |
| F3 | Multi-source research report retrieval | All | P0 |
| F4 | Live data tool-use (RBA, ABS, SQM) | Investor, Researcher | P0 |
| F5 | Query decomposition for complex questions | All | P0 |
| F6 | Reasoning trace / "show your work" panel | Researcher, Investor | P0 |
| F7 | Inline chart generation | Investor, Researcher | P1 |
| F8 | Suggested follow-up questions | Homebuyer | P1 |
| F9 | Suburb comparison mode (side-by-side) | Homebuyer, Investor | P1 |
| F10 | Source quality indicators | Researcher | P2 |
| F11 | Export answer as PDF/Markdown | Researcher | P2 |
| F12 | Glossary tooltips on jargon | Homebuyer | P2 |

### 5.2 Feature details

#### F1. Persona-aware chat interface
User selects persona on first visit (or system infers from first query). Affects:
- **Tone**: conversational for homebuyer, technical for researcher
- **Retrieval weighting**: boost practical docs for homebuyers, methodology papers for researchers
- **Default visuals**: simpler for homebuyer, chart-heavy for investor
- **Disclaimers**: stronger "not financial advice" language for homebuyers/investors

#### F2. Grounded answers with inline citations
Every factual claim carries a citation chip: `[RBA FSR 2024 • p.14]` or `[ABS 6432.0 • Q4 2025]`. Clicking opens source panel with the exact quote and a link to the PDF/page. Non-grounded claims are prefixed with "general knowledge:" and flagged visually.

#### F3. Multi-source research report retrieval
Corpus of ~300 curated AU housing reports, chunked and indexed. Retrieval is hybrid (BM25 + fine-tuned dense embeddings) with cross-encoder reranking. Fine-tuned on AU housing jargon for meaningful improvement over generic embeddings.

#### F4. Live data tool-use
Agent calls real APIs when data freshness matters:
- `rba_cash_rate`, `rba_mortgage_rates`, `rba_credit_aggregates`
- `abs_property_price_index`, `abs_building_approvals`, `abs_lending_indicators`
- `sqm_rental_vacancy`, `sqm_asking_prices_rents`
- `compute_rental_yield`, `compute_mortgage_repayment`, `compute_stamp_duty_nsw` (etc.)

Tool results carry `retrieved_at` timestamp shown to user.

#### F5. Query decomposition
LLM-based classifier decides if a query is atomic or complex. Complex queries are split into 2–4 sub-questions, answered independently, then synthesized. Example:

> "Is Parramatta good for a family on $1.2M budget?"

decomposes into:
1. What's the current price range for family homes in Parramatta?
2. How has the Parramatta market trended recently?
3. What's the rental market and vacancy rate there?
4. What schools/infrastructure considerations apply? *(out of scope — flagged to user)*

#### F6. Reasoning trace / "show your work"
Collapsible panel below each answer:
- Sub-questions the agent generated
- Tools called and their results
- Top 5 retrieved chunks with relevance scores
- Reflection step output ("answer draft assessed for completeness")

This transforms the product from a black box into an auditable research tool — critical for researcher persona, and a strong differentiator against ChatGPT.

#### F7. Inline chart generation
For time-series or comparison queries, agent calls `render_chart` tool that produces matplotlib → inline PNG. Always cites data source beneath the chart.

#### F8. Suggested follow-up questions
After each answer, 3 contextual follow-ups shown as clickable chips. Powered by a small LLM call over the answer.

#### F9. Suburb comparison mode
Persona-triggered UI state: pick 2–4 suburbs, get a structured comparison table (median price, 12m change, yield, vacancy, demographics) with a narrative summary.

---

## 6. User Flows

### 6.1 First-time visitor flow

```
Landing page
  ↓
"What kind of help do you need?"
  ↓
[Homebuyer] [Investor] [Researcher/Analyst] [Just exploring]
  ↓
Persona confirmed → Chat interface with persona-tailored placeholder examples
  ↓
User enters query
  ↓
Answer streams in with citations + reasoning trace collapsed by default
  ↓
3 follow-up suggestions appear
```

### 6.2 Homebuyer flow — "Is Parramatta right for me?"

```
User: "I've got $1.2M to spend in Sydney for a family home, is Parramatta a good bet?"
  ↓
Agent classifies: complex, needs-docs=true, needs-data=true, persona=homebuyer
  ↓
Decomposes into 4 sub-questions
  ↓
Parallel execution:
  ├─ Retrieve CoreLogic/Domain reports on Parramatta
  ├─ Tool: ABS property price index (Sydney/Greater West)
  ├─ Tool: SQM vacancy rate for 2150
  └─ Retrieve Grattan/AHURI on Sydney affordability context
  ↓
Reflect: "do I have price range, trend, yield/rental info, and context?"
  ↓
Synthesize with homebuyer tone + clear "not financial advice" disclaimer
  ↓
Answer shown with:
  • Plain-language summary at top
  • Price range [CoreLogic Oct 2025 • ...]
  • Trend context [RBA FSR 2024 • ...]
  • Follow-ups: "Compare with Blacktown?" / "What about stamp duty?" / "School zones data?"
```

### 6.3 Investor flow — "Screen Sydney suburbs"

```
User: "Sydney suburbs with highest 3yr capital growth and vacancy under 3%"
  ↓
Agent classifies: analytical, persona=investor, needs-data=true, needs-docs=maybe
  ↓
Plans:
  • Tool: CoreLogic capital growth data (top 20 suburbs, 3yr)
  • Tool: SQM vacancy rates for those suburbs
  • Filter: vacancy < 3%
  • Retrieve: recent investor-focused commentary for context
  ↓
Chart tool: bar chart of top 10 survivors
  ↓
Answer: ranked table + chart + contextual narrative + caveats about methodology
  ↓
Follow-ups: "Narrow by budget?" / "Rental yield for these?" / "Compare with Brisbane?"
```

### 6.4 Researcher flow — "Compare price index methodologies"

```
User: "How do CoreLogic hedonic index and ABS 6432.0 differ methodologically?"
  ↓
Agent classifies: definitional/methodological, persona=researcher, docs-heavy
  ↓
Retrieve:
  • CoreLogic methodology paper
  • ABS 6432.0 explanatory notes
  • Relevant RBA discussion paper on price measurement
  ↓
Synthesize structured comparison with side-by-side table
  ↓
Answer with page-level citations for every claim + links to original PDFs
  ↓
Follow-ups: "How does repeat-sales differ?" / "Implications for policy?" / "Which is used by RBA?"
```

### 6.5 Error / ambiguity flow

```
User: "What about Newtown?"
  ↓
Agent: "Newtown has several locations in Australia. Did you mean:
        • Newtown NSW 2042 (inner-west Sydney)
        • Newtown VIC 3220 (Geelong)
        • Newtown QLD 4350 (Toowoomba)"
  ↓
User clicks disambiguation → normal query flow resumes
```

---

## 7. UX Principles

1. **Citations aren't optional** — every number and factual claim is sourced or flagged as ungrounded.
2. **Show your work** — the reasoning trace is a feature, not a debug tool. It builds trust.
3. **Freshness is visible** — every data point shows its retrieval date; stale data is marked.
4. **Not financial advice** — persistent, proportional disclaimers. Stronger for homebuyer/investor personas.
5. **Accessible to the homebuyer, precise for the researcher** — persona-aware without forking the product.
6. **Fast to first token** — stream answers; show retrieval progress; don't make users wait in silence.

---

## 8. Success Metrics

### 8.1 Product / user metrics

| Metric | Target (end of 4-week build) | How measured |
|---|---|---|
| **Query success rate** | ≥70% of queries get a usable answer | Self-label 50 queries + user feedback |
| **Median time to first token** | ≤3 seconds | Frontend timing |
| **Median full-answer latency** | ≤12 seconds | Frontend timing |
| **Citations per answer** | ≥2 for research-mode answers | Automatic count |
| **Trace open rate** | ≥30% of power users open the reasoning trace | Frontend event |
| **Cost per query** | ≤AUD $0.02 | Backend logging of LLM + embedding + tool spend |

### 8.2 Engineering quality metrics

| Metric | Target | How measured |
|---|---|---|
| **Recall@10 (retrieval)** | Fine-tuned ≥0.80, base BGE baseline for comparison | Custom 100-query gold eval set |
| **MRR** | Fine-tuned ≥+15% over base BGE | Same eval set |
| **Answer faithfulness** | ≥0.85 | RAGAS or custom LLM-judge |
| **Groundedness** | ≥95% of numeric claims carry a citation | Regex + LLM audit on sample |
| **Tool-call accuracy** | ≥85% of tool calls are appropriate for the query | Manual review on 30-query agent eval |
| **Hallucination rate** | ≤5% of answers contain unsupported numeric claims | Manual review |

### 8.3 Portfolio / launch metrics (Weeks 4+)

| Metric | Target | Why |
|---|---|---|
| Blog post reads | 1,000+ within 30 days | Portfolio signal |
| Demo URL unique visits | 200+ | Product validation |
| GitHub stars | 50+ | Community interest |
| LinkedIn post impressions | 5,000+ | Professional reach |
| Positive recruiter mentions | ≥3 in first month | Primary ROI for this build |

---

## 9. Technical Architecture (Summary)

Full details in `project_plan.md`. At a glance:

```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit UI                          │
│  (Persona selector · Chat · Trace panel · Charts)        │
└───────────────────────────┬──────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────┐
│                   LangGraph Agent                        │
│  classify → decompose → retrieve/tool → reflect → answer │
└──────┬────────────────────────┬───────────────────┬──────┘
       │                        │                   │
┌──────▼──────┐       ┌─────────▼────────┐   ┌──────▼──────┐
│   Hybrid    │       │   Live Tools     │   │   Claude    │
│  Retrieval  │       │  RBA·ABS·SQM     │   │  Sonnet 4.5 │
│  BM25+Dense │       │  +Chart+Calc     │   │             │
│  +Reranker  │       └──────────────────┘   └─────────────┘
└──────┬──────┘
       │
┌──────▼───────────┐
│ Qdrant (~50k     │
│ chunks, fine-    │
│ tuned BGE embed) │
└──────────────────┘
```

---

## 10. Risks & Mitigations (Product Level)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Users interpret answers as financial advice | High | High | Proportional persistent disclaimers; never recommend specific actions; ASIC-informed guardrails in system prompt |
| Licensing issues with reproduced content | Medium | High | Only index publicly posted reports; short quoted spans; clear attribution; don't redistribute the corpus publicly |
| Agent hallucinates a suburb statistic | Medium | High | Numeric claims must come from tool calls OR retrieved chunks; LLM-judge flags unsupported numbers pre-render |
| Data goes stale (e.g., post-RBA decision) | High | Medium | Every tool call shows retrieval timestamp; cache TTL ≤24h for market-moving data |
| Fine-tuning doesn't outperform base embeddings | Low-Medium | Medium | Honest negative-result reporting is still a portfolio win; have hybrid+rerank as safety-net gains |
| Streamlit UI isn't impressive enough for portfolio | Medium | Medium | Budget optional Day 29 for Next.js polish pass if time allows |

---

## 11. Open Questions

Things to revisit as the build progresses:

1. **Geographic scope** — start national or Sydney-first? Current plan: national, Sydney-strongest.
2. **Corpus refresh** — manual re-ingestion cadence for v1 (quarterly?), or build an automated updater as stretch goal?
3. **Disambiguation UX** — dropdown vs inline re-prompt? Test both in Week 4.
4. **Researcher export format** — Markdown vs PDF vs Notion-style? Defer until user feedback.
5. **Monetization** — explicitly not pursued in v1 (portfolio project), but worth thinking about for v2 direction: freemium with usage-based paid tier? B2B licensing to buyer's agents?

---

## 12. Future Roadmap (Post-v1)

Not part of the 4-week build, but shows thinking for interviews and blog:

**v1.5 (Month 2)**
- Suburb comparison mode (F9)
- Stamp duty calculators for all states
- User accounts for saved queries

**v2 (Months 3–4)**
- Multi-turn conversational memory
- Email digest of market updates
- API access for researchers

**v3+ (6+ months)**
- Property-level valuation integration (pending licensed data)
- Commercial property expansion
- NZ market coverage

---

## 13. Appendix

### 13.1 Glossary (Product)

- **Agent**: a system that uses an LLM to decide which tools to call and how to compose the result, iterating until the task is done.
- **RAG**: Retrieval-Augmented Generation — retrieve relevant documents, feed them to an LLM, generate grounded answers.
- **Fine-tuned embedding**: a vector-encoder model adapted to a domain via contrastive training.
- **Hybrid retrieval**: combining keyword search (BM25) with semantic search (embeddings), merged via Reciprocal Rank Fusion.
- **Persona**: a pre-configured user profile that shapes tone, retrieval priorities, and UX.

### 13.2 Glossary (AU Housing Domain)

- **ABS**: Australian Bureau of Statistics
- **AHURI**: Australian Housing and Urban Research Institute
- **APRA**: Australian Prudential Regulation Authority
- **CoreLogic / Cotality**: primary commercial provider of AU property data
- **DTI**: Debt-to-Income ratio
- **LVR**: Loan-to-Value Ratio
- **NHFIC / Housing Australia**: National Housing Finance and Investment Corporation
- **PPR**: Principal Place of Residence
- **RBA**: Reserve Bank of Australia
- **SA4 / LGA**: geographic boundaries used by ABS
- **SQM Research**: independent AU property research firm

---

*This product document defines the what and why. For the how and when, see `project_plan.md`.*
