
# Regime-Change Gap Analysis — CadastreAI vs. the 2026 Housing Tax Reform

> **Status:** open finding, blocks `v1.0.0`
> **Repo state analysed:** `685d517` (28 Jun 2026, `main`)
> **Analysis date:** 27 Jul 2026
> **Severity:** High — the agent confidently states repealed tax law to the persona (`investor`) most likely to act on it, and the eval suite scores that behaviour as correct.

---

## 1. Executive summary

On 26 June 2026 the *Treasury Laws Amendment (Tax Reform No. 1) Act 2026* was enacted, materially changing the two tax settings that underpin the entire investor-persona value proposition: negative gearing and the CGT discount.

CadastreAI's last commit is **two days later**. It contains no reference to the reform, and — more importantly — **no mechanism that could ever surface it**:

| Layer | Can it detect a legal regime change? | Evidence |
|---|---|---|
| Ingestion | No | No ATO / legislation / EM scraper; Treasury scraper known-incomplete (`progress.md:485`) |
| Corpus | No | 3 Australian Treasury chunks; tax content concentrated in AHURI 2018 |
| Retrieval | No | `date` stored (`upsert.py:47`) but never read by any ranker |
| Fine-tune | No — actively harmful | Pair generation samples the stale corpus (`results/finetuned_vs_baseline.md`) |
| Prompts | No | No current date injected; no chunk age shown; reflector defaults to ship |
| Guardrails | No | 4 categories, all product-pick; no temporal/currency class |
| Tools | No | RBA/ABS/SQM live; zero tax tools; stamp duty hard-coded and stale |
| Eval | No — inverts the signal | Gold labels point at pre-reform chunks |

The failure is **architectural, not a data-refresh lag**. Re-scraping alone fixes none of rows 3–8.

---

## 2. What the law actually says now

*Treasury Laws Amendment (Tax Reform No. 1) Bill 2026* — passed Senate 25 Jun 2026, enacted 26 Jun 2026.

### 2.1 Negative gearing

- From the **2027–28 income year**, losses on **established** residential investment property **acquired after 7:30pm AEST 12 May 2026** are quarantined: deductible only against other residential property income, including future capital gains from rental property — **not** against salary or other personal income.
- **New builds** and key government housing priorities remain fully negatively geared.
- **Grandfathered:** properties held at, or under contract before, 7:30pm AEST 12 May 2026.
- **Carved out:** widely held trusts (incl. most MITs) and superannuation funds (incl. SMSFs).

### 2.2 Capital gains tax

- The **50% CGT discount** for individuals, trusts and partnerships is replaced by **cost-base indexation**, plus a reported 30% minimum rate on gains, for gains accruing **from 1 July 2027**.
- Prospective only: the 50% discount continues to apply to gains accrued up to 1 July 2027.
- **Retained:** the up-to-60% affordable-housing CGT discount; the four small-business CGT concessions.

### 2.3 Why the design matters for us

The enacted design is **neither** of the scenarios modelled in our corpus. AHURI 2018 (`the-income-tax-treatment-of-housing-assets…`) and AHURI 2025 (`modelling-landlord-behaviour…`) simulate (a) *complete abolition* of negative gearing and (b) *reduced CGT discount rates*. The Act delivers instead:

- **quarantining**, not abolition — the deduction is *deferred*, not destroyed;
- **indexation**, not a lower discount — value now scales with CPI and hold length, with no 12-month cliff;
- **grandfathering + new-build carve-out** — a permanent two-tier market.

Elasticities estimated for "abolish it" do not transfer to "quarantine it, with a ~14-month announcement lead and a standing new-build exemption."

---

## 3. Findings

### F-1 — Corpus has no post-reform primary source, and the scrapers cannot reach one
**Severity: High · Files: `src/ingest/scrapers/`, `scripts/collect_sources.py`**

Publisher/year distribution across `data/eval/candidates.jsonl` chunk IDs:

```
  68  ahuri 2023        42  ahuri 2017        21  ahuri 2024
  67  ahuri 2018        39  ahuri 2022        19  ahuri 2020
  48  proptrack (undated)                     18  ahuri 2019
  46  housing-australia-nhfic (undated)        9  proptrack 2026
  38  ahuri 2025                               5  ahuri 2026
  32  productivity-commission (undated)        3  australian-treasury (undated)
```

- Registered scrapers (`collect_sources.py:39-48`): `rba`, `ahuri`, `grattan`, `corelogic`, `sqm`, `domain_proptrack`, `treasury_pc`, `nhfic_apra`. **No ATO. No Federal Register of Legislation. No explanatory memoranda. No Budget papers.**
- `progress.md:485` already records: *"Treasury `/policy-topics/housing` is a topic hub, not a full paginated list — the hub links out to search-result pages at `/publications?topic=...` which this scraper does NOT follow."* That is precisely where the reform material sits.
- Net effect: the single most consequential document set for the investor persona is structurally unreachable.

### F-2 — Retrieval ignores document dates entirely
**Severity: High · Files: `src/index/upsert.py:47`, `src/index/hybrid.py`, `src/retrieval/retriever.py`, `src/retrieval/rerank.py`**

`date` is persisted in the Qdrant payload (`_PAYLOAD_FIELDS`, `upsert.py:47`) and documented in `retriever.py:6`. It is read by **no ranking, filtering or tie-break path** anywhere in the codebase. There is no recency prior, no `as_of` filter, and no supersession relation.

Consequence: `ahuri/2018-00_pathways-to-housing-tax-reform__0029` — *"the 50 per cent CGT discount for individuals for assets held for more than 12 months"* — competes on equal terms with any 2026 source. Under BM25 it does better than equal: it is the densest keyword match for "CGT discount" in the entire corpus (score 33.698 on the negative-gearing eval query, rank 1).

### F-3 — The eval suite scores the obsolete answer as correct
**Severity: Critical · Files: `data/eval/queries_all.jsonl`, `queries.jsonl`, `candidates.jsonl`, `agent_queries.jsonl`**

This is the finding that makes every other metric in `README.md` and `docs/report_v2.md` unfalsifiable.

| Location | Query | Gold chunks | Problem |
|---|---|---|---|
| `queries_all.jsonl:23` | *"effect of negative gearing on an investor's after-tax cash flow for a $700,000 investment property"* | `ahuri/2018-00_pathways-to-housing-tax-reform__0029`, `ahuri/2018-00_the-income-tax-treatment…__0009`, `__0010` | Retrieving these is now the **failure** mode; scored 1.0 |
| `queries_all.jsonl:29` | *"impact of abolishing negative gearing on rental supply and rents"* | AHURI 2018/2025, PC `public-housing` | Models a scenario that did not occur (abolition ≠ quarantining) |
| `candidates.jsonl:24` | *"CGT within 12 months versus held longer than a year"* | AHURI 2018 CGT-discount sections | **Question itself is malformed** post-reform — indexation has no 12-month cliff |
| `agent_queries.jsonl:10` | *"What is negative gearing in Australia and how does it work?"* | `expected_doc_publishers: [AHURI, Treasury, PC]` | Publisher expectation is satisfiable only by pre-reform text |
| `agent_queries.jsonl:26` | *"…what reforms have been proposed?"* | 3-leg decomposition | "Proposed" is now factually wrong framing — it is enacted |

Headline v6 faithfulness of **0.793** is therefore measuring fidelity to a repealed statute, not answer quality.

### F-4 — Fine-tuning bakes the pre-reform prior into the encoder
**Severity: Medium · Files: `src/training/generate_pairs.py`, `train_embeddings.py`, `results/finetuned_vs_baseline.md`**

`results/finetuned_vs_baseline.md` already diagnoses publisher collapse (R@5 0.383 → 0.177; FT model pulls everything toward AHURI register). The under-noted second-order effect: because anchors are LLM-generated *from the corpus*, the training signal encodes **"a correct answer looks like 2017–2018 AHURI policy analysis."** Fixing the sampling skew (remedy A in that doc) does not fix the temporal skew — it just rebalances *which* stale publishers dominate.

Sequencing implication: **do not re-run pair generation until F-1 is closed**, or the reform-era corpus will be diluted below the noise floor a second time.

### F-5 — Prompt stack cannot express "my evidence is out of date"
**Severity: High · File: `src/agent/nodes.py`**

- `_SYNTHESIZER_SYSTEM` (`nodes.py:1035-1049`): *"Answer the user's question concisely and ONLY from the supplied evidence"* and *"Never fall back to 'I cannot provide'…when the answer is in the tool results above."* Tuned to assert, not hedge.
- `_REFLECTOR_SYSTEM` (`nodes.py:804-819`): *"Default is to ship… Only mark incomplete when you can name a specific missing piece."* A reflector holding only pre-2026 evidence has no basis on which to name the gap.
- **No current date is injected** into any of the classifier, decomposer, router, reflector or synthesizer prompts.
- `_summarise_chunks_for_synth` (`nodes.py:1052+`) passes title / section / text — **chunk publication date is not shown to the synthesizer**, so it cannot reason about vintage even if prompted to.
- Compounding risk: *"The tool's data fields are the authoritative current value as of `retrieved_at` — treat them as fresh"* (`nodes.py:1044-1046`). Live July-2026 RBA figures get welded to 2018 tax framing inside a single cited paragraph. That reads **more** authoritative than a wholly stale answer, and is therefore more dangerous.

### F-6 — Investor persona is pointed directly at the stale material
**Severity: High · File: `src/agent/persona.py:73-78`**

> *"Lead with investment metrics (rental yield, vacancy, capital growth, cash-flow). Distinguish gross vs net where the evidence supports it; **flag tax-treatment items only when explicitly covered in retrieved policy text**."*

The only tax-treatment text in the corpus is pre-reform. The instruction that was designed as a hallucination guard now functions as a guarantee that the agent surfaces repealed rules.

`_DISCLAIMER_BASELINE` (`nodes.py:1025-1033`) covers licensing ("not a licensed financial, legal, tax… professional") but says nothing about **currency**. A user reads it as "get a professional to check my numbers," not "these rules changed ten weeks ago."

### F-7 — Guardrails have no temporal/currency class
**Severity: Medium · File: `src/agent/guardrails.py`**

`GuardrailCategory` = `mortgage_product | insurance_product | super_or_managed_fund | specific_security_pick`. All four are product-pick refusals. There is no category for "this answer is contingent on tax settings that changed on 12 May 2026," and no mechanism to inject a mandatory preamble rather than refuse outright.

Note the module's own design principle cuts in our favour here: *"guardrails on the hot path must be fast and deterministic"* — a regex class for `negative gear|capital gains|CGT|deduct|quarantin|stamp duty` fits the existing architecture exactly.

### F-8 — `compute.py` bracket vintage is stale and internally inconsistent
**Severity: Medium · File: `src/tools/compute.py`**

Four different claims about the same constant:

| Line | Claim |
|---|---|
| `compute.py:10` | *"NSW residential transfer duty using the **2025-26** brackets"* |
| `compute.py:141` | *"transfer duty rates effective **1 July 2024**"* |
| `compute.py:174` | *"Uses the **2024-25** indexed brackets published by Revenue NSW"* |
| `compute.py:209` | `source="Revenue NSW — Transfer duty (**2024-25** brackets, FHBAS thresholds)"` |

Revenue NSW indexes thresholds annually, so as at FY2026-27 all four are wrong and the emitted `citation` string misrepresents its own vintage to the user.

Separately: `compute.py` exposes yield, mortgage repayment and stamp duty. **There is no CGT or gearing calculator at all** — the one domain that changed has no tool, so the agent must fall back on retrieved prose, i.e. F-2 + F-5.

### F-9 — Structural break in every live series the tools return
**Severity: Medium · Files: `src/tools/rba_stats.py`, `abs_stats.py`, `sqm.py`**

The live tools correctly fetch current data (`f1.1-data.csv`, `f6-data.csv`, ABS 6432.0 / 8731.0 / 5601.0 `latest-release` URLs). But investor lending, established-vs-new-build price spreads, and rental yields all carry a **policy discontinuity at 12 May 2026**. The investor persona is prompted to lead with "capital growth, cash-flow" — i.e. to extrapolate across that break — and nothing in `_resolve_period` or the synthesizer flags it.

### F-10 — No representation of acquisition date or build status
**Severity: Medium · Files: `src/agent/graph.py` (`AgentState`), `src/tools/schemas.py`**

Post-reform, *any* correct answer about investor after-tax position is conditional on two variables: **acquisition date** (pre/post 7:30pm 12 May 2026) and **build status** (new vs established). Neither exists in `AgentState`, in any tool schema, or in the classifier's `Classification` output. The system cannot ask the disambiguating question because it has nowhere to store the answer.

---

## 4. Remediation plan

Ordered by dependency, not by cost.

```
F-3 (quarantine eval)  ──┐
F-1 (ingest primaries) ──┼──> F-2 (date-aware retrieval) ──> F-4 (re-finetune)
                         └──> F-5/F-6/F-7 (prompt + guardrail) ──> F-10 (state)
F-8, F-9 independent
```

1. **Quarantine the eval set first.** Until gold labels are corrected, every remediation looks like a regression. Tag tax-dependent queries with `as_of` / `regime`, split into `legacy_regime` and `current_regime` buckets, exclude the former from headline metrics.
2. **Ingest the primary sources.** Enacted Act, explanatory memorandum, ATO *Boosting home ownership* guidance, 2026-27 Budget papers. Tag `regime: post_2026_reform`.
3. **Make `date` a first-class retrieval signal** + hard supersession rule.
4. **Temporal guardrail class** with mandatory preamble injection (not refusal).
5. **Inject current date + chunk vintage** into the prompt stack so the reflector can name the gap.
6. **Re-run pair generation and fine-tune only after (2)**, with per-publisher *and* per-era caps.
7. **Fix `compute.py`** vintage; move NSW brackets to a dated config; add regime-aware CGT/gearing calculators or refuse.

---

## 5. Tickets

Drop-in for `tickets.md`, following the existing `Task W.NN` convention. Proposed new week block plus two cross-cutting entries.

```markdown
## WEEK 5 — Regime-change remediation (2026 housing tax reform)

> Trigger: Treasury Laws Amendment (Tax Reform No. 1) Act 2026, enacted 26 Jun 2026.
> Blocks `v1.0.0`. See `docs/regime_change_gap_analysis.md` for findings F-1..F-10.

### Day 31 — Stop the bleeding (eval + disclosure)

- [ ] **Task 5.01** — Add `as_of` (ISO date) and `regime` (`pre_2026_reform` | `post_2026_reform` | `regime_neutral`) fields to the eval schema; backfill across `data/eval/queries*.jsonl` and `agent_queries.jsonl` (F-3)
- [ ] **Task 5.02** — Audit all 100 queries + 30 agent queries for tax dependence; move regime-dependent items to `data/eval/queries_legacy_regime.jsonl` and exclude from headline metrics in `src/eval/retrieval_eval.py` and `agent_eval.py` (F-3)
- [ ] **Task 5.03** — Re-word the three malformed queries whose *premise* is now false (`queries_all.jsonl:24` 12-month CGT cliff; `:29` "abolishing"; `agent_queries.jsonl:26` "proposed reforms") and re-annotate gold chunks (F-3)
- [ ] **Task 5.04** — Add an interim banner to `src/app/streamlit_app.py` + README §Disclaimer stating the corpus predates the 2026 reform, pending Task 5.06 (F-6)

### Day 32 — Primary-source ingestion

- [ ] **Task 5.05** — Implement `src/ingest/scrapers/ato.py` — ATO new-legislation guidance under `/about-ato/new-legislation/`, filtered to housing/CGT/rental topics (F-1)
- [ ] **Task 5.06** — Implement `src/ingest/scrapers/legislation.py` — Federal Register of Legislation + APH bills pages; capture Act text and explanatory memoranda for housing-tax instruments (F-1)
- [ ] **Task 5.07** — Close the known Treasury scraper gap from `progress.md:485` — follow `/publications?topic=...` search-result pages out of the `/policy-topics/housing` hub (F-1)
- [ ] **Task 5.08** — Add Budget-paper ingestion (2026-27 BP1/BP2 housing + revenue measures) to `treasury_pc.py` (F-1)
- [ ] **Task 5.09** — Extend chunk metadata with `regime` and `supersedes` / `superseded_by` fields; backfill `regime` for all tax-topic chunks in `data/processed/chunks.jsonl`; re-upsert (F-1, F-2)

### Day 33 — Date-aware retrieval

- [ ] **Task 5.10** — Normalise the free-form `date` payload field to a sortable ISO value at index time (currently "YYYY" / "YYYY-mon" / free text per `collect_sources.py` schema) (F-2)
- [ ] **Task 5.11** — Add a configurable recency prior to `src/index/hybrid.py` RRF scoring; default off for `regime_neutral` topics, on for policy/tax (F-2)
- [ ] **Task 5.12** — Implement supersession handling in `src/retrieval/retriever.py`: when a retrieved chunk is `superseded_by`, force the successor into the same context window and demote the predecessor (F-2)
- [ ] **Task 5.13** — Add `as_of` / `regime` filter parameters to `retrieve()` and thread through `retrieve_or_tool` (F-2, F-10)
- [ ] **Task 5.14** — Regression-test date-aware retrieval against the `current_regime` eval split; target: post-reform gold chunk in top-3 for all tax queries (F-2, F-3)

### Day 34 — Prompt stack + guardrails

- [ ] **Task 5.15** — Inject current date into classifier / decomposer / router / reflector / synthesizer system prompts in `src/agent/nodes.py` (F-5)
- [ ] **Task 5.16** — Surface chunk publication date and `regime` in `_summarise_chunks_for_synth` and `_summarise_chunks` evidence blocks (F-5)
- [ ] **Task 5.17** — Amend `_REFLECTOR_SYSTEM` to admit "all retrieved evidence predates a known regime change" as a concrete, nameable gap that justifies one more pass (F-5)
- [ ] **Task 5.18** — Add `temporal_currency` guardrail category to `src/agent/guardrails.py` — regex on `negative gear|capital gains|CGT|quarantin|deduct|stamp duty|land tax`; action is **preamble injection**, not refusal; requires extending `GuardrailAction` with `"annotate"` (F-7)
- [ ] **Task 5.19** — Rewrite the investor addendum in `src/agent/persona.py:73-78` — replace "flag tax-treatment items only when explicitly covered in retrieved policy text" with regime-aware language; extend `_DISCLAIMER_BASELINE` with a currency caveat alongside the licensing caveat (F-6)
- [ ] **Task 5.20** — Add `acquisition_date` and `is_new_build` to `AgentState` and to the classifier's `Classification` schema; wire a disambiguation prompt in `src/app/disambiguation.py` when an investor tax query arrives without them (F-10)

### Day 35 — Tools

- [ ] **Task 5.21** — Fix the four conflicting NSW bracket vintages in `src/tools/compute.py` (lines 10, 141, 174, 209); move brackets to a dated config keyed by financial year; emit the actual vintage in the citation string (F-8)
- [ ] **Task 5.22** — Implement `compute_cgt_indexed()` — cost-base indexation for gains accruing from 1 Jul 2027, with pre-1-Jul-2027 gains on the 50% discount; expose in `src/tools/schemas.py` (F-8)
- [ ] **Task 5.23** — Implement `compute_gearing_position()` — returns deductible-against-salary vs quarantined split, keyed on `acquisition_date` and `is_new_build`; must refuse rather than guess when either is unknown (F-8, F-10)
- [ ] **Task 5.24** — Add a structural-break annotation to `rba_stats` / `abs_stats` / `sqm` envelopes when the requested period spans 12 May 2026; surface it in the citation string (F-9)

### Day 36 — Re-train + re-baseline

- [ ] **Task 5.25** — Re-run `generate_pairs.py` post-ingestion with per-publisher **and** per-era caps (remedy A of `results/finetuned_vs_baseline.md`, extended for temporal skew) (F-4)
- [ ] **Task 5.26** — Re-run the full retrieval + agent eval on the corrected split; publish `results/regime_change_v1.md` and refresh README / `docs/report_v2.md` headline tables (F-3, F-4)
- [ ] **Task 5.27** — Add a CI check to `.github/workflows/tests.yml` asserting no eval query tagged `pre_2026_reform` contributes to headline metrics (F-3)
```

Cross-cutting additions:

```markdown
- [ ] **Task X.06** — Regime-change watch: quarterly re-run of `scripts/collect_sources.py` against ATO + legislation scrapers, with a diff report on any chunk whose `regime` tag would change
- [ ] **Task X.07** — Document the corpus `as_of` date prominently in README, `product.md`, and the Streamlit footer; treat "corpus vintage" as a first-class released artifact alongside `final_metrics.json`
```

---

## 6. Sources

- Baker McKenzie — *Australia: Major Changes to CGT and Negative Gearing* — https://www.bakermckenzie.com/en/insight/publications/2026/07/australia-major-changes-to-cgt-and-negative-gearing
- ATO — *Tax reform – Boosting home ownership – Reforming negative gearing and capital gains tax* — https://www.ato.gov.au/about-ato/new-legislation/in-detail/individuals/tax-reform-boosting-home-ownership-reforming-negative-gearing-and-capital-gains-tax
- Corrs Chambers Westgarth — *Capital gains tax and negative gearing amendments: key changes and implications* — https://www.corrs.com.au/insights/capital-gains-tax-and-negative-gearing-amendments-key-changes-and-implications
- PwC Australia — *2026-27 Federal Budget – CGT and housing tax reform* — https://www.pwc.com.au/tax/tax-alerts/cgt-and-housing-tax-reform.html
- Treasury Ministers — *Second reading speech, Treasury Laws Amendment (Tax Reform No. 1) Bill 2026* — https://ministers.treasury.gov.au/ministers/jim-chalmers-2022/speeches/second-reading-speech-treasury-laws-amendment-tax-reform-no-1