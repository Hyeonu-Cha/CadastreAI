# Baseline retrieval error analysis

Source: `results/baseline.json` (100-query eval, BGE-base-en-v1.5, top-10).
40/100 queries returned Recall@10 = 0. This file reviews 20 of those
failures and groups them into a taxonomy that informs Phase 2 fixes.

## Summary

| Category                              | Count | Share |
|---------------------------------------|-------|-------|
| Near-miss within target document      | 7     | 35%   |
| Topic-cluster miss (adjacent doc)     | 5     | 25%   |
| Abstraction / genre mismatch          | 3     | 15%   |
| Temporal mismatch                     | 2     | 10%   |
| Multi-concept decomposition needed    | 2     | 10%   |
| Abbreviation / program disambiguation | 1     |  5%   |

The dataset's standout pattern is **near-miss inside the right document** —
BGE lands on the correct AHURI / RBA / NHFIC report but picks a
neighbouring chunk instead of the annotated gold chunk. That kind of
failure is cheap to attack with a cross-encoder reranker over the top-20,
because the reranker sees the full chunk text and can disambiguate
which paragraph answers the question.

## Taxonomy

### 1. Near-miss within target document (~35%)

BGE retrieves the right document but a different chunk. Classic
sub-document ranking problem.

- **Query:** How does AHURI define housing affordability stress and what income-to-rent threshold is typically used?
  - Gold: `ahuri/2014 housing-affordability-dynamics__0016`, `__0022`
  - Retrieved: `ahuri/2020 demand-side-assistance__0014`, `ahuri/2024 measuring-housing-affordability__0013`, `ahuri/2019 mortgage-stress__0007`
  - Right topic (affordability measurement), wrong documents; no overlap with the 2014 paper that is the gold. Sits near the boundary of "topic-cluster miss."

- **Query:** What methodology do AHURI researchers use to model landlord behaviour in Australian rental markets?
  - Gold: `ahuri/2025 modelling-landlord__0021`, `__0012`
  - Retrieved (rank 1): `ahuri/2025 modelling-landlord__0002`
  - Same paper, wrong section (intro rather than methods).

- **Query:** What are the methodological differences between HILDA and the Census for measuring housing tenure transitions?
  - Gold: `ahuri/2021 estimating-population-at-risk__0088`, `ahuri/2019 moving-downsizing__0044`
  - Retrieved: gendered-housing, social-inclusion, wellbeing-renters chunks — all AHURI, all wrong docs.

- **Query:** How operationally do Australian researchers distinguish market rental supply vs affordable rental supply?
  - Gold: `ahuri/2018 matching-markets__0068`, `ahuri/2018 inquiry-affordable-housing-supply__0036`
  - Retrieved: same inquiry doc `__0012`, plus urban productivity and strategic planning chunks. Right doc, wrong chunk.

- **Query:** Are there any shared equity schemes available for first home buyers in Victoria?
  - Gold: `ahuri/2023 financing-first-home-ownership__0048`, `__0074`
  - Retrieved: `ahuri/2023 inquiry-financing-first-home-ownership__0020` (sibling inquiry doc) and `ahuri/2022 assisting-first-homebuyers__0077`.

- **Query:** Which Australian capital cities currently have the highest gross rental yields for investment properties?
  - Gold: `proptrack/2025 hottest-investor-suburbs__0001`
  - Retrieved: `proptrack/2026 westpac-investor-report__0015`, `__0006` — same publisher, adjacent report.

- **Query:** What share of Australian residential property investors own two or more investment properties?
  - Gold: `ahuri/2022 regulation-of-residential-tenancies__0060`
  - Retrieved (rank 1): `ahuri/2022 regulation-of-residential-tenancies__0062` — same doc, neighbour chunk.

**Fix direction:** cross-encoder reranker over top-20 BGE hits should
recover most of these without any index changes.

### 2. Topic-cluster miss — adjacent document (~25%)

Retrieved chunks are in the same topic cluster but none from the annotated
gold documents.

- **Query:** How does depreciation on a newly built investment property reduce an investor's taxable income each year?
  - Gold: `ahuri/2018 changing-institutions-private-rental__0105`, `ahuri/2024 incentivising-small-scale-investors__0035`
  - Retrieved: three `ahuri/2018 income-tax-treatment-of-housing-assets` chunks. Tax cluster but wrong papers.

- **Query:** What does recent modelling say about the impact of abolishing negative gearing on rental supply and rents?
  - Gold: `ahuri/2018 navigating-a-changing-private-rental-sector__0031`, `ahuri/2025 modelling-landlord-behaviour__0088`
  - Retrieved: `ahuri/2018 income-tax-treatment` chunks and `pathways-to-housing-tax-reform`. Close but off-target.

- **Query:** How do NDIS specialist disability accommodation yields compare to traditional residential rental investments?
  - Gold: `ahuri/2022 accommodating-adults-with-intellectual-disabilities__0017`, `__0086`
  - Retrieved: `ahuri/2019 understanding-sda-funding`, `ahuri/2024 sda-in-social-housing`. SDA cluster but wrong doc.

- **Query:** What are the tax implications when an investor converts a former principal residence into a rental property?
  - Gold: `productivity-commission/housing-decisions-older-australians__0118`, `__0143`
  - Retrieved: three `ahuri/2018 income-tax-treatment` chunks. BGE defaults to the AHURI tax cluster; the gold lives in PC's older-Australian housing decisions report which also discusses this but under a different heading.

- **Query:** How do long-term capital growth rates compare between detached houses, units and townhouses?
  - Gold: `productivity-commission/housing-construction__0014`, `ahuri/2025 insights-into-short-term-rental__0052`
  - Retrieved: `rba/2015 long-run-trends-in-housing-price-growth__0013` (very good-looking match, arguably a better gold) plus AHURI changing-institutions. **Possibly a gold-annotation miss.**

**Fix direction:** reranker helps, but harder cases need query-aware
filtering (publisher and document type) or query rewriting that names
the specific regulatory concept (e.g. "Division 43 depreciation"
expansion).

### 3. Abstraction / genre mismatch (~15%)

Consumer-facing homebuyer/investor questions hit academic policy chunks,
or vice versa.

- **Query:** What upfront and ongoing costs beyond the purchase price should a first home buyer budget for?
  - Gold: `nhfic stamp-duty-reform-benefits__0003`, `proptrack/2025 budget-bonuses__0003`
  - Retrieved: AHURI international policy review + community-land-trusts chunk. Academic-flavour retrieval for a practical question.

- **Query:** What key metrics do investors use to compare investment property performance across different suburbs?
  - Gold: `proptrack/2026 westpac-investor-report__0015`, `ahuri/2017 social-impact-investment__0076`
  - Retrieved: AHURI medium-density + unlisted-wholesale-residential-property-funds. Abstract-investor flavour, not practical-investor flavour.

- **Query:** How does choosing a fixed versus variable rate mortgage affect the long-term affordability of a first home?
  - Gold: `rba/2023 fixed-rate-housing-loans__0010`, `__0008`
  - Retrieved: three AHURI financing-first-home-ownership chunks. Surface "first home" match dominates over the specific RBA paper on fixed-rate transmission.

**Fix direction:** persona-aware retrieval — for `homebuyer` queries,
boost PropTrack/NHFIC/Treasury chunks; for `investor`, boost
PropTrack/SQM; for `researcher`, boost AHURI/RBA/PC. Or train a
retriever that learns these priors from the gold set.

### 4. Temporal mismatch (~10%)

Queries mention "current", "recent", or a decade window; BGE ignores
the time signal and surfaces stale chunks.

- **Query:** Is it financially better to keep renting or buy a first home in Sydney given current prices and rates?
  - Gold: `rba/2017 housing-accessibility-for-first-home-buyers__0001`, `ahuri/2023 pathways-to-home-ownership__0084`
  - Retrieved: three 2008 SQM Research blog posts (bear market commentary from 17 years ago).

- **Query:** How do Sydney capital growth rates compare to Brisbane and Perth over the past decade?
  - Gold: `proptrack/2024-10 home-price-index__0003`, `proptrack/2025-04 home-price-index__0002`
  - Retrieved: RBA 2009, RBA 2023 rent inflation, SQM 2007. No recent PropTrack HPI.

**Fix direction:** date-aware reranking — parse temporal phrases
("current", "past decade", "over the last five years") and either
filter the candidate set by publication year or apply a recency boost.

### 5. Multi-concept / decomposition (~10%)

Query has multiple conjuncts; BGE matches only one.

- **Query:** Are there grants or stamp duty concessions specifically for buying a newly built home versus an existing dwelling?
  - Concepts: grants + stamp duty + new-build-vs-existing.
  - Retrieved matches grants/first-home cluster but misses the
    "newly built vs existing" contrast.

- **Query:** How does private investor activity in the housing market correlate with changes in the RBA cash rate?
  - Concepts: investor activity (a market-behaviour measurement) + cash-rate movement.
  - Gold is PropTrack investor reports that chart activity against rates;
    retrieved is RBA pass-through papers about investor **interest cost**,
    a different mechanism.

**Fix direction:** query decomposition (sub-queries per concept) with
union-then-rerank; or ColBERT-style late interaction to preserve
per-term scoring.

### 6. Abbreviation / program name disambiguation (~5%)

Specific Australian scheme acronyms collide.

- **Query:** How does the Family Home Guarantee help single parent first home buyers with a small deposit?
  - Gold: `nhfic hgs-trends-2023-24`, `hgs-trends-2022-23`
  - Retrieved: NHFIC FHLDS trends 2020-21, FHLDS trends 2021-22, PropTrack first-home-guarantee. BGE conflates FHG (Family Home Guarantee) with FHLDS (First Home Loan Deposit Scheme) — they are separate programs under the HGS umbrella.

**Fix direction:** a small glossary of scheme acronyms piped into query
rewriting; or a per-scheme BM25 score added to the BGE score.

## Not observed (relative to ticket's suggested taxonomy)

- **Numeric-needs-tools**: only one query in this sample ("How much CGT
  applies to property sold within 12 months vs held longer") has a
  numeric flavour, and the gold does exist in corpus — it's a
  near-miss, not a needs-calculator. We'll likely see this category
  emerge when users start asking "what's the break-even price?" style
  questions, but none of the current 100 do.

## Priorities for Phase 2

1. **Cross-encoder reranker over top-20** — attacks categories 1 and 2
   (60% of failures).
2. **Persona-aware source boosting** — category 3.
3. **Temporal phrase parser + recency boost** — category 4.
4. **Scheme-acronym glossary for query expansion** — category 6.
5. **Query decomposition** — category 5, likely deferred to later
   phase (bigger lift).

Also surface: the gold annotation for "houses vs units vs townhouses
capital growth" is probably weak — `rba/2015 long-run-trends-in-housing-price-growth`
(BGE's rank 1) looks like a better answer than the PC housing-construction
chunk currently annotated. Worth an annotation spot-check before Phase 2.
