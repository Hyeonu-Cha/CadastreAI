# Task 2.15 — Per-persona breakdown: fine-tuned vs baseline

Eval set: `data\eval\queries_all.jsonl` (n=100 queries shared across both runs).

## TL;DR

Fine-tuned regresses **on every persona** (R@10/MRR@10), confirming the corpus-level finding from Task 2.14. There is no persona slice where the current `bge-au-housing-v1` checkpoint should ship.

| metric | baseline | fine-tuned | Δ (abs) | Δ (rel) |
| --- | ---: | ---: | ---: | ---: |
| recall@5 | 0.383 | 0.177 | −0.207 | −53.9% |
| recall@10 | 0.450 | 0.210 | −0.240 | −53.3% |
| mrr@10 | 0.352 | 0.128 | −0.224 | −63.7% |
| ndcg@10 | 0.349 | 0.146 | −0.204 | −58.3% |

## Per-persona aggregates

| persona | n | R@5 BL → FT | R@10 BL → FT | MRR@10 BL → FT | nDCG@10 BL → FT |
| --- | ---: | --- | --- | --- | --- |
| homebuyer | 24 | 0.29 → 0.11 (−61.9%) | 0.40 → 0.11 (−72.4%) | 0.33 → 0.08 (−75.7%) | 0.30 → 0.08 (−73.0%) |
| investor | 25 | 0.17 → 0.08 (−53.8%) | 0.28 → 0.09 (−66.7%) | 0.21 → 0.06 (−70.8%) | 0.19 → 0.07 (−65.5%) |
| researcher | 51 | 0.53 → 0.25 (−51.9%) | 0.56 → 0.31 (−43.5%) | 0.43 → 0.18 (−57.7%) | 0.45 → 0.21 (−52.2%) |

## Win / tie / loss at the query level

Counts queries where FT's MRR@10 strictly beats / ties / falls short of BL's. A 'win' just means the rank of the first gold hit improved — even if both still missed top-1.

| persona | n | FT wins | ties | FT losses | mean ΔMRR | mean ΔnDCG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| homebuyer | 24 | 2 | 9 | 13 | −0.249 | −0.218 |
| investor | 25 | 0 | 13 | 12 | −0.146 | −0.127 |
| researcher | 51 | 4 | 22 | 25 | −0.251 | −0.235 |

## Top FT wins (where FT might still hold useful signal)

Per persona, the up-to-3 queries with the largest positive ΔMRR. If FT is purely random noise, wins should be tiny and far from the gold. Where the win margin is large *and* a gold hit moved into top-3, the FT model arguably learned something — those are the queries to study before deciding what to keep / discard from the training recipe.

### homebuyer

- **ΔMRR=+0.33** — Is it financially better to keep renting or buy a first home in Sydney given current prices and rates?
  - BL retrieved[:3]: `['sqm-research/2008-04_christopher-publicly-acknowledges-becoming-increasingly-bearish-on-residential-p__0003', 'housing-australia-nhfic/unknown_state-nations-housing-2021-22__0106', 'sqm-research/2008-11_credit-crisis-good-news-for-property-buyers-christopher-comments-on-the-affordab__0003']` (MRR=0.00)
  - FT retrieved[:3]: `['sqm-research/2008-11_credit-crisis-good-news-for-property-buyers-christopher-comments-on-the-affordab__0003', 'ahuri/2023-00_pathways-to-home-ownership-in-an-age-of-uncertainty__0042', 'ahuri/2023-00_pathways-to-home-ownership-in-an-age-of-uncertainty__0084']` (MRR=0.33)
  - gold: `['rba/2017-00_housing-accessibility-for-first-home-buyers__0001', 'ahuri/2023-00_pathways-to-home-ownership-in-an-age-of-uncertainty__0084', 'proptrack/unknown_regional-renaissance-why-more-city-dwellers-are-embracing-life-outside-the-capit__0002']`
- **ΔMRR=+0.13** — Is Australia on track to become a nation of renters rather than home-owners?
  - BL retrieved[:3]: `['ahuri/2024-00_planning-for-a-two-tenure-future__0018', 'ahuri/2024-00_planning-for-a-two-tenure-future__0029', 'ahuri/2024-00_planning-for-a-two-tenure-future__0012']` (MRR=0.20)
  - FT retrieved[:3]: `['ahuri/2023-00_inquiry-financing-first-home-ownership-opportunities-and-challenges__0039', 'ahuri/2018-00_social-housing-as-infrastructure-an-investment-pathway__0144', 'proptrack/unknown_is-australia-on-track-to-become-a-nation-of-renters__0000']` (MRR=0.33)
  - gold: `['proptrack/unknown_is-australia-on-track-to-become-a-nation-of-renters__0000']`

### investor

_No queries where FT MRR > BL MRR._

### researcher

- **ΔMRR=+1.00** — Should public housing agencies compete with the private sector in upper-market residential land development?
  - BL retrieved[:3]: `['ahuri/2016-00_individualised-housing-assistance-findings-and-policy-options__0013', 'productivity-commission/unknown_public-housing__0016', 'ahuri/2016-00_transforming-public-housing-in-a-federal-context__0018']` (MRR=0.00)
  - FT retrieved[:3]: `['productivity-commission/unknown_public-housing__0238', 'ahuri/2025-00_fine-tuning-the-machine-evaluating-machinery-of-government-for-housing-policy-ad__0048', 'ahuri/2020-00_the-uneven-distribution-of-housing-supply-20062016__0011']` (MRR=1.00)
  - gold: `['productivity-commission/unknown_public-housing__0238']`
- **ΔMRR=+0.39** — How do dwelling density and social housing concentration affect safety satisfaction for social renters versus private renters?
  - BL retrieved[:3]: `['ahuri/2014-00_wellbeing-outcomes-of-lower-income-renters-a-multilevel-analysis-of-area-effects__0078', 'ahuri/2014-00_wellbeing-outcomes-of-lower-income-renters-a-multilevel-analysis-of-area-effects__0050', 'ahuri/2014-00_wellbeing-outcomes-of-lower-income-renters-a-multilevel-analysis-of-area-effects__0077']` (MRR=0.11)
  - FT retrieved[:3]: `['ahuri/2014-00_wellbeing-outcomes-of-lower-income-renters-a-multilevel-analysis-of-area-effects__0134', 'ahuri/2014-00_wellbeing-outcomes-of-lower-income-renters-a-multilevel-analysis-of-area-effects__0079', 'ahuri/2014-00_wellbeing-outcomes-of-lower-income-renters-a-multilevel-analysis-of-area-effects__0136']` (MRR=0.50)
  - gold: `['ahuri/2014-00_wellbeing-outcomes-of-lower-income-renters-a-multilevel-analysis-of-area-effects__0079']`
- **ΔMRR=+0.17** — Why should state housing authorities set public housing market rents comparable to equivalent private rental accommodation?
  - BL retrieved[:3]: `['productivity-commission/unknown_public-housing__0416', 'productivity-commission/unknown_public-housing__0015', 'productivity-commission/unknown_public-housing__0411']` (MRR=0.33)
  - FT retrieved[:3]: `['productivity-commission/unknown_public-housing__0344', 'productivity-commission/unknown_public-housing__0411', 'productivity-commission/unknown_public-housing__0442']` (MRR=0.50)
  - gold: `['productivity-commission/unknown_public-housing__0411']`

## Top FT losses (where the regression is worst)

Same shape, opposite end of the distribution — useful when reasoning about *how* the model is failing (publisher collapse, vocabulary drift, etc.).

### homebuyer

- **ΔMRR=-1.00** — Can I use the First Home Super Saver Scheme to boost my deposit and how much can I withdraw?
  - BL retrieved[:3]: `['australian-treasury/unknown_home-ownership-support__0000', 'ahuri/2022-00_assisting-first-homebuyers-an-international-policy-review__0060', 'ahuri/2019-00_young-australians-and-the-housing-aspirations-gap__0116']` (MRR=1.00)
  - FT retrieved[:3]: `['ahuri/2022-00_predicting-risk-to-inform-housing-policy-and-practice__0083', 'ahuri/2022-00_assisting-first-homebuyers-an-international-policy-review__0060', 'ahuri/2022-00_assisting-first-homebuyers-an-international-policy-review__0124']` (MRR=0.00)
  - gold: `['australian-treasury/unknown_home-ownership-support__0000', 'ahuri/2023-00_inquiry-financing-first-home-ownership-opportunities-and-challenges__0020', 'housing-australia-nhfic/unknown_state-nations-housing-2021-22__0117']`
- **ΔMRR=-1.00** — What serviceability buffer do banks currently apply when assessing a first home buyer's mortgage application?
  - BL retrieved[:3]: `['apra/unknown_residential-mortgage-lending__0009', 'apra/unknown_housing-lending-standards-reinforcing-guidance-on-exceptions__0000', 'rba/2015-00_download_cb443d6a__0003']` (MRR=1.00)
  - FT retrieved[:3]: `['apra/unknown_a-prudential-approach-to-mortgage-lending__0004', 'rba/2016-00_download_e58c4720__0004', 'rba/2016-00_review_52e05e7b__0030']` (MRR=0.00)
  - gold: `['apra/unknown_prudential-practice-guide-apg-223-residential-mortgage-lending__0002', 'apra/unknown_residential-mortgage-lending__0009', 'apra/unknown_housing-lending-standards-reinforcing-guidance-on-exceptions__0000']`
- **ΔMRR=-1.00** — Are Australian buyers increasingly enquiring on interstate properties in search of affordability?
  - BL retrieved[:3]: `['proptrack/2024-00_buyers-look-further-afield-in-hunt-for-affordability__0000', 'proptrack/2026-00_proptrack-westpac-investor-report-2026__0007', 'proptrack/2024-00_buyers-look-further-afield-in-hunt-for-affordability__0001']` (MRR=1.00)
  - FT retrieved[:3]: `['proptrack/2026-00_proptrack-westpac-investor-report-2026__0007', 'proptrack/unknown_great-divide-how-state-borders-are-splitting-australias-property-market__0002', 'ahuri/2014-00_generational-change-in-home-purchase-opportunity-in-australia__0023']` (MRR=0.00)
  - gold: `['proptrack/2024-00_buyers-look-further-afield-in-hunt-for-affordability__0000']`

### investor

- **ΔMRR=-0.83** — How does land tax differ between NSW, Victoria and Queensland for investors owning multiple properties?
  - BL retrieved[:3]: `['ahuri/2022-00_regulation-of-residential-tenancies-and-impacts-on-investment__0037', 'proptrack/2024-00_how-many-victorian-investors-have-sold-due-to-higher-land-taxes__0001', 'proptrack/2024-00_how-many-victorian-investors-have-sold-due-to-higher-land-taxes__0000']` (MRR=1.00)
  - FT retrieved[:3]: `['ahuri/2018-00_social-housing-as-infrastructure-an-investment-pathway__0178', 'productivity-commission/unknown_defence-housing__0005', 'ahuri/2019-00_older-australians-and-the-housing-aspirations-gap__0138']` (MRR=0.17)
  - gold: `['proptrack/unknown_shares-vs-property-which-asset-came-out-on-top-in-the-past-five-years__0003', 'ahuri/2022-00_regulation-of-residential-tenancies-and-impacts-on-investment__0037', 'ahuri/2022-00_regulation-of-residential-tenancies-and-impacts-on-investment__0008']`
- **ΔMRR=-0.67** — What happened to Australian residential property listings in January 2016, and how did Sydney prices compare year on year?
  - BL retrieved[:3]: `['sqm-research/2016-00_national-listings-down-for-january-despite-mixed-capital-city-results-sqm-resear__0002', 'sqm-research/2016-00_national-listings-down-for-january-despite-mixed-capital-city-results-sqm-resear__0000', 'sqm-research/2024-02_normal-quiet-january-for-listings-5-february-2024__0000']` (MRR=1.00)
  - FT retrieved[:3]: `['rba/2016-00_download_d427f8f7__0009', 'rba/2016-00_review_52e05e7b__0012', 'sqm-research/2016-00_national-listings-down-for-january-despite-mixed-capital-city-results-sqm-resear__0002']` (MRR=0.33)
  - gold: `['sqm-research/2016-00_national-listings-down-for-january-despite-mixed-capital-city-results-sqm-resear__0002']`
- **ΔMRR=-0.50** — What drove strong 5-year returns in the global listed property market through 2007 and how did regional REIT markets differ?
  - BL retrieved[:3]: `['sqm-research/2008-00_the-adviser-edge-national-property-sector-review-january-2008__0054', 'sqm-research/2008-00_the-adviser-edge-national-property-sector-review-january-2008__0053', 'ahuri/2015-00_the-opportunity-of-unlisted-wholesale-residential-property-funds-in-enhancing-af__0048']` (MRR=0.50)
  - FT retrieved[:3]: `['sqm-research/2008-00_the-adviser-edge-national-property-sector-review-january-2008__0054', 'sqm-research/2008-00_the-adviser-edge-national-property-sector-review-january-2008__0066', 'sqm-research/2008-00_the-adviser-edge-national-property-sector-review-january-2008__0028']` (MRR=0.00)
  - gold: `['sqm-research/2008-00_the-adviser-edge-national-property-sector-review-january-2008__0053']`

### researcher

- **ΔMRR=-1.00** — How has the price elasticity of housing supply been estimated in recent Australian housing research?
  - BL retrieved[:3]: `['ahuri/2017-00_housing-supply-responsiveness-in-australia-distribution-drivers-and-institutiona__0007', 'ahuri/2017-00_housing-supply-responsiveness-in-australia-distribution-drivers-and-institutiona__0079', 'ahuri/2020-00_the-uneven-distribution-of-housing-supply-20062016__0019']` (MRR=1.00)
  - FT retrieved[:3]: `['ahuri/2017-00_housing-supply-responsiveness-in-australia-distribution-drivers-and-institutiona__0021', 'ahuri/2015-00_housing-markets-economic-productivity-and-risk-international-evidence-and-policy_2557ac76__0007', 'ahuri/2017-00_housing-supply-responsiveness-in-australia-distribution-drivers-and-institutiona__0079']` (MRR=0.00)
  - gold: `['ahuri/2017-00_housing-supply-responsiveness-in-australia-distribution-drivers-and-institutiona__0006', 'ahuri/2017-00_housing-supply-responsiveness-in-australia-distribution-drivers-and-institutiona__0007', 'ahuri/2017-00_housing-supply-responsiveness-in-australia-distribution-drivers-and-institutiona__0048']`
- **ΔMRR=-1.00** — What gaps exist in transdisciplinary frameworks for assistive technology in ageing and disability housing?
  - BL retrieved[:3]: `['ahuri/2021-00_impacts-of-new-and-emerging-assistive-technologies-for-ageing-and-disabled-housi__0038', 'ahuri/2021-00_impacts-of-new-and-emerging-assistive-technologies-for-ageing-and-disabled-housi__0109', 'ahuri/2021-00_impacts-of-new-and-emerging-assistive-technologies-for-ageing-and-disabled-housi__0024']` (MRR=1.00)
  - FT retrieved[:3]: `['ahuri/2021-00_impacts-of-new-and-emerging-assistive-technologies-for-ageing-and-disabled-housi__0096', 'ahuri/2021-00_impacts-of-new-and-emerging-assistive-technologies-for-ageing-and-disabled-housi__0093', 'ahuri/2021-00_impacts-of-new-and-emerging-assistive-technologies-for-ageing-and-disabled-housi__0097']` (MRR=0.00)
  - gold: `['ahuri/2021-00_impacts-of-new-and-emerging-assistive-technologies-for-ageing-and-disabled-housi__0038']`
- **ΔMRR=-1.00** — Why has private rental subsidy become the preferred housing assistance policy for vulnerable Australian families?
  - BL retrieved[:3]: `['ahuri/2020-00_inquiry-into-integrated-housing-support-for-vulnerable-families__0038', 'productivity-commission/unknown_public-housing__0075', 'ahuri/2020-00_inquiry-into-integrated-housing-support-for-vulnerable-families__0019']` (MRR=1.00)
  - FT retrieved[:3]: `['ahuri/2016-00_housing-assistance-need-and-provision-in-australia-a-household-based-policy-anal__0083', 'productivity-commission/unknown_public-housing__0225', 'productivity-commission/unknown_public-housing__0224']` (MRR=0.00)
  - gold: `['ahuri/2020-00_inquiry-into-integrated-housing-support-for-vulnerable-families__0038']`

## What this slice changes about Task 2.16+ planning

- Ablation table (Task 2.16) should still include `ft+hybrid+rerank` so we can see whether downstream BM25 fusion or a cross-encoder reranker rescues the FT signal — but the realistic expectation is that they don't, given the publisher-collapse failure mode is upstream of both.
- Blog draft (Task 2.18) should be honest about this: the lesson is *which of the candidate fixes (stratified sampling, anchor-style alignment, lower lr, eval-query mix-in, or moving to a cross-encoder) we'd try next*, not a victory lap.
