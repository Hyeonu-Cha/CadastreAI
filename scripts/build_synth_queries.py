"""Author synthetic queries, one per chunk sampled by sample_chunks_for_synth.

Each `(query, persona)` below was written by reading the sampled chunk's
title, section, and leading text, then drafting a realistic question that
the chunk actually answers. Personas follow the same homebuyer / investor
/ researcher scheme as the human-picked eval (Task 1.26–1.27):
 - academic/government/regulator publishers → researcher
 - PropTrack/CoreLogic consumer-facing content → homebuyer or investor
 - SQM listings/price commentary → investor

The chunk itself is the sole gold for each synthetic query. That's a
tighter bar than the BM25-derived gold (which allows up to 3 per query),
but it reflects how the chunk was sampled (random, not query-driven):
the query was written *for* the chunk, so the chunk is by construction
the right answer.

Emits `data/eval/queries_synth.jsonl` with the same schema as
`queries.jsonl`. With `--merge`, also writes a concatenated
`queries_all.jsonl` for convenience.

    python -m scripts.build_synth_queries --merge
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

# Manually-authored queries. Each key is a chunk_id from
# data/eval/synth_source_chunks.jsonl; each value is (query, persona).
SYNTH_QUERIES: dict[str, tuple[str, str]] = {
    "ahuri/2015-00_entries-and-exits-from-homelessness-a-dynamic-analysis-of-the-relationship-betwe__0025": (
        "What structural factors does AHURI use to explain regional differences in homelessness entries and exits?",
        "researcher",
    ),
    "ahuri/2023-00_crisis-accommodation-in-australia-now-and-for-the-future__0015": (
        "What supports do crisis accommodation services need to help clients with mental health and substance use needs?",
        "researcher",
    ),
    "ahuri/2013-00_understanding-and-addressing-community-opposition-to-affordable-housing-developm__0011": (
        "Does community opposition to affordable housing developments fade after projects are completed?",
        "researcher",
    ),
    "ahuri/2017-00_pathways-to-state-property-tax-reform__0099": (
        "Does retaining transfer duty alongside a broad-based land tax help dampen speculation on investment properties?",
        "researcher",
    ),
    "ahuri/2014-00_wellbeing-outcomes-of-lower-income-renters-a-multilevel-analysis-of-area-effects__0079": (
        "How do dwelling density and social housing concentration affect safety satisfaction for social renters versus private renters?",
        "researcher",
    ),
    "ahuri/2021-00_urban-productivity-and-affordable-rental-housing-supply-in-australian-cities-and__0082": (
        "Where should new affordable rental housing be targeted in Sydney and Melbourne to support access to employment?",
        "researcher",
    ),
    "ahuri/2021-00_impacts-of-new-and-emerging-assistive-technologies-for-ageing-and-disabled-housi__0038": (
        "What gaps exist in transdisciplinary frameworks for assistive technology in ageing and disability housing?",
        "researcher",
    ),
    "ahuri/2021-00_population-growth-and-mobility-in-australia-implications-for-housing-and-urban-d__0061": (
        "How does AHURI structure HILDA person-wave data in its econometric model of residential mobility decisions?",
        "researcher",
    ),
    "ahuri/2024-00_lived-experience-participation-and-influence-in-homelessness-and-housing-policy__0039": (
        "How does structural violence manifest in the emotional experience of people with lived experience of homelessness?",
        "researcher",
    ),
    "ahuri/2020-00_inquiry-into-integrated-housing-support-for-vulnerable-families__0038": (
        "Why has private rental subsidy become the preferred housing assistance policy for vulnerable Australian families?",
        "researcher",
    ),
    "ahuri/2016-00_the-role-of-private-rental-brokerage-in-housing-outcomes-for-vulnerable-australi__0008": (
        "What makes private rental brokerage programs successful at placing 'rental ready' clients into tenancies?",
        "researcher",
    ),
    "ahuri/2014-00_preventing-first-time-homelessness-amongst-older-australians__0034": (
        "Why do ABS Census counts underestimate the number of insecurely housed older Australians?",
        "researcher",
    ),
    "rba/2018-00_download_7401382b__0011": (
        "What cultural failures did APRA's Prudential Inquiry into Commonwealth Bank identify in the bank's risk management?",
        "researcher",
    ),
    "rba/2005-00_review_539b9dff__0049": (
        "How did the Australian low-doc loan market grow and what lenders entered it in the early 2000s?",
        "researcher",
    ),
    "rba/2015-00_review_ec5db784__0037": (
        "In the RBA's 2015 financial stability review, how do risks from commercial property compare with those from other non-financial businesses?",
        "researcher",
    ),
    "rba/2016-00_review_52e05e7b__0044": (
        "How did Australian lenders change pricing and non-price conditions on investor and interest-only mortgages through 2015?",
        "researcher",
    ),
    "rba/2023-00_financial-stability-risks-from-commercial-real-estate__0006": (
        "How did arrears and charge-off rates on US bank commercial real estate loans evolve in the first half of 2023?",
        "researcher",
    ),
    "rba/2004-00_download_81fbd111__0006": (
        "Who absorbs losses on prime loans in Australian residential mortgage-backed securities (RMBS)?",
        "researcher",
    ),
    "productivity-commission/unknown_public-housing__0185": (
        "Which research on rent-setting systems and poverty traps did the 1993 Industry Commission public housing inquiry cite?",
        "researcher",
    ),
    "productivity-commission/unknown_housing_f0804343__0007": (
        "How does client mix affect the equity, effectiveness and efficiency of social housing services in the Report on Government Services?",
        "researcher",
    ),
    "productivity-commission/unknown_housing-construction__0056": (
        "What benefits have past reviews attributed to Australia's single national construction code (NCC)?",
        "researcher",
    ),
    "productivity-commission/unknown_public-housing__0238": (
        "Should public housing agencies compete with the private sector in upper-market residential land development?",
        "researcher",
    ),
    "productivity-commission/unknown_public-housing__0411": (
        "Why should state housing authorities set public housing market rents comparable to equivalent private rental accommodation?",
        "researcher",
    ),
    "proptrack/unknown_boost-in-investor-activity-offers-relief-to-renters__0001": (
        "Is investor activity returning to the Australian rental market, and is that slowing rent growth?",
        "investor",
    ),
    "proptrack/2025-00_proptrack-home-price-index-april-2025__0002": (
        "Why are Australian unit prices diverging from house prices as affordability constraints worsen?",
        "homebuyer",
    ),
    "proptrack/unknown_is-australia-on-track-to-become-a-nation-of-renters__0000": (
        "Is Australia on track to become a nation of renters rather than home-owners?",
        "homebuyer",
    ),
    "proptrack/2024-00_new-housing-supply-falls-short-by-62000-homes-in-2024__0001": (
        "By how much did Australia's new housing supply fall short in FY2024, and which states had the largest gaps?",
        "researcher",
    ),
    "proptrack/2024-00_buyers-look-further-afield-in-hunt-for-affordability__0000": (
        "Are Australian buyers increasingly enquiring on interstate properties in search of affordability?",
        "homebuyer",
    ),
    "housing-australia-nhfic/unknown_environmental-social-and-governance-esg-reporting-standard__0000": (
        "What ESG reporting standard did CHIA and NHFIC develop for the Australian community housing sector?",
        "researcher",
    ),
    "housing-australia-nhfic/unknown_state-nations-housing-report-2022-23__0025": (
        "How did investor lending commitments fall across Australian states during 2022?",
        "researcher",
    ),
    "housing-australia-nhfic/unknown_state-nations-housing-2021-22__0113": (
        "How did rental affordability change across Australian capital cities from mid-2020 to September 2021 by income quintile?",
        "researcher",
    ),
    "housing-australia-nhfic/unknown_analysis-australias-rental-markets__0002": (
        "How has the gap between social housing rent and lower-end private rent changed over the past decade in NSW and Victoria?",
        "researcher",
    ),
    "sqm-research/2016-00_national-listings-down-for-january-despite-mixed-capital-city-results-sqm-resear__0002": (
        "What happened to Australian residential property listings in January 2016, and how did Sydney prices compare year on year?",
        "investor",
    ),
    "sqm-research/unknown_asking-prices-hit-record-high-distressed-listings-jump-52__0000": (
        "How did national residential listings and distressed listings change in October according to SQM Research?",
        "investor",
    ),
    "sqm-research/2008-00_the-adviser-edge-national-property-sector-review-january-2008__0053": (
        "What drove strong 5-year returns in the global listed property market through 2007 and how did regional REIT markets differ?",
        "investor",
    ),
    "apra/unknown_opening-statement-to-house-of-representatives-standing-committee-on-4__0001": (
        "What did APRA tell the House Economics Committee in October 2022 about the Financial Accountability Regime?",
        "researcher",
    ),
    "apra/unknown_apra-agrees-to-court-enforceable-undertaking-from-bank-of-queensland__0003": (
        "What capital add-on and remedial actions did APRA require of Bank of Queensland under its court-enforceable undertaking?",
        "researcher",
    ),
    "corelogic-cotality/unknown_podcast29-minhome-insurance-shortfalls-may-turn-cheap-homes-into-costly-futuresl__0005": (
        "How does the US National Flood Insurance Program (NFIP) fill gaps in residential flood insurance, and is it sustainable?",
        "homebuyer",
    ),
    "corelogic-cotality/unknown_insight-geospatial-mappinghow-caple-properties-remax-delivered-a-multi-touchpoin__0001": (
        "How did Caple Properties RE/MAX combine automated prospecting (RiTA) with human outreach to convert leads?",
        "investor",
    ),
    "australian-treasury/unknown_social-and-affordable-housing__0001": (
        "What do Australia's Social Housing Accelerator and the National Agreement on Social Housing and Homelessness (NASHH) fund?",
        "researcher",
    ),
}


def run(source: Path, out: Path, existing: Path | None, merged_out: Path | None) -> None:
    log.info("Loading sampled chunks from %s", source)
    sampled_ids: list[str] = []
    with source.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sampled_ids.append(rec["chunk_id"])

    missing = [cid for cid in sampled_ids if cid not in SYNTH_QUERIES]
    extra = [cid for cid in SYNTH_QUERIES if cid not in sampled_ids]
    if missing:
        raise SystemExit(f"Missing authored queries for {len(missing)} chunk(s): {missing[:3]}")
    if extra:
        log.warning("Authored queries for %d chunk_ids not in sample: %s", len(extra), extra[:3])

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as g:
        for cid in sampled_ids:
            query, persona = SYNTH_QUERIES[cid]
            g.write(
                json.dumps(
                    {"query": query, "persona": persona, "gold_chunk_ids": [cid]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    log.info("Wrote %d synthetic queries to %s", len(sampled_ids), out)
    print(f"synth={len(sampled_ids)} out={out}")

    if merged_out is not None:
        if existing is None or not existing.exists():
            raise SystemExit(f"--merge requires --existing pointing to an existing queries file (got {existing})")
        merged_out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with merged_out.open("w", encoding="utf-8") as g:
            for p in (existing, out):
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip("\n")
                        if not line.strip():
                            continue
                        g.write(line + "\n")
                        n += 1
        log.info("Merged %d queries into %s", n, merged_out)
        print(f"merged={n} out={merged_out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("data/eval/synth_source_chunks.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/eval/queries_synth.jsonl"))
    p.add_argument("--existing", type=Path, default=Path("data/eval/queries.jsonl"))
    p.add_argument("--merge", action="store_true", help="also write queries_all.jsonl (existing + synth)")
    p.add_argument("--merged-out", type=Path, default=Path("data/eval/queries_all.jsonl"))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    run(
        source=args.source,
        out=args.out,
        existing=args.existing if args.merge else None,
        merged_out=args.merged_out if args.merge else None,
    )


if __name__ == "__main__":
    main()
