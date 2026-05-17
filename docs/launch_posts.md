# Launch posts — drafts (Task 4.22)

Copy-paste-ready drafts for the four channels in the launch checklist.
Each is sized to the platform's conventions: LinkedIn rewards a
hook + scan-friendly bullets, Reddit rewards an interesting finding
up front and a comment-able question at the end, Medium wants the
full piece (we can either repost in full or run a TL;DR + canonical
link).

Replace these placeholders before posting:

- `__DEMO_URL__` — the public Streamlit URL (post-Task 4.15)
- `__LOOM_URL__` — the 3-min walkthrough Loom (post-Task 4.23)
- `#hashtag` lists are starting points, not gospel — strip ones that
  feel forced for your audience.

Order to post in: **Reddit first** (lowest stakes, fastest feedback;
fixes typos and tightens the hook before anything more visible),
**Medium next** (canonical written form, anchors the LinkedIn link),
**LinkedIn last** (highest visibility — go live once the others are
clean).

---

## 1. LinkedIn post

> ~1,800 chars. Hook in the first 2 lines (LinkedIn truncates around
> 200 chars before "see more"). Single image: the architecture diagram
> from the repo (`architecture.svg`).

```
I built a domain-tuned RAG agent for the Australian housing market
over four weeks. The interesting part wasn't the agent — it was
what broke when I tried to be clever, and what fixed itself when
I stopped.

CadastreAI answers questions about Australian property — first-home
buyer, investor, researcher, journalist personas — over a corpus
of ~42k chunks from RBA, ABS, AHURI, NHFIC, CoreLogic, Grattan,
and others, with live data tools for cash rates, vacancy, and
yields. Every claim is cited.

A few things I didn't expect:

→ The fine-tune was supposed to be the headline win. As a
  standalone retriever it collapsed: R@10 dropped from 0.450
  (base) to 0.210. It was useless alone — but in the
  hybrid + cross-encoder stack it beat every other configuration
  (R@10 = 0.770). A signal can be valuable in a stack and dead
  on its own. I almost killed the model for the wrong reason.

→ "Always use hybrid retrieval" is overfit advice. On the synthetic
  query split (no retriever bias), hybrid + rerank wins
  monotonically. On the held-out 100-query set, BM25 alone beats
  every variant (R@10 = 0.907). When users phrase questions the way
  documents do, lexical retrieval is hard to beat.

→ A worked example in a routing prompt is itself a routing change.
  I added a single GOOD/BAD example to disambiguate "compute yield"
  from "lookup yield." The example fixed the target query (tool
  accuracy 0.25 → 1.00) — and quietly nudged 12 unrelated queries
  into a different regime. Faithfulness on that switched cohort
  jumped +0.110. Prompt edits ripple.

Five-version agent eval table, fine-tune collapse-and-rescue ablation,
six-category retrieval failure taxonomy, and full reproducibility
scripts in the repo:

→ Code:  github.com/Hyeonu-Cha/CadastreAI
→ Demo:  __DEMO_URL__
→ Walkthrough (3 min):  __LOOM_URL__
→ Technical write-up:  github.com/Hyeonu-Cha/CadastreAI/blob/main/docs/blog.md

#RAG #LLM #MachineLearning #AppliedML #InformationRetrieval
#Anthropic #Claude #LangGraph #AustralianProperty
```

---

## 2. Medium / personal blog repost

The canonical write-up already lives at `docs/blog.md` (~800 lines,
~6,000 words). On Medium you have two options:

**Option A — repost in full.** Copy `docs/blog.md` into a Medium
draft, add a one-line "Originally at github.com/Hyeonu-Cha/CadastreAI"
note at the top, and tag with `Machine Learning`, `RAG`, `LLM`,
`Information Retrieval`, `Anthropic`. Medium's import tool handles
the markdown cleanly; the only tweak needed is replacing the relative
links (`[product.md](./product.md)`) with absolute GitHub URLs.

**Option B — TL;DR + canonical link.** Use the LinkedIn body
(slightly expanded) on Medium, with the GitHub blog as the
"continue reading" target. Faster to publish, lower SEO weight on
Medium itself.

Recommended: **Option A.** The full write-up is the thing — Medium
is a syndication channel for it, not a rewrite. Repost with
`canonical_url` set to the GitHub URL so search consolidates rank.

### Medium subtitle (used in the listing card)

```
Four weeks, ~42k chunks, five agent versions. What broke when I
tried to be clever — and what surprised me when I stopped.
```

### Medium tags (5 max)

```
Machine Learning, RAG, LLM, Information Retrieval, Anthropic
```

---

## 3. r/MachineLearning  —  [Project] post

> Anti-self-promotion rules apply. Lead with results, not framing.
> Use `[P]` flair (Project). Don't link the demo URL above the fold;
> reserve it for a comment if asked.

**Title**

```
[P] Domain-tuned RAG agent for Australian housing — fine-tune
collapse-and-rescue, "always use hybrid" was wrong on this corpus,
and a five-version agent eval
```

**Body**

```
Four-week portfolio project on a corpus of ~42k chunks from ten
Australian housing publishers (RBA, ABS, AHURI, NHFIC, CoreLogic,
Grattan, ...). Repo: github.com/Hyeonu-Cha/CadastreAI.

Three findings I didn't see in any of the RAG advice I read going in:

**1. Fine-tune collapse-and-rescue.** Fine-tuned `bge-base-en-v1.5`
on ~4k synthetic (anchor, positive) pairs from Claude Haiku +
BM25-mined hard negatives, MultipleNegativesRankingLoss, 3 epochs.
Standalone, the FT model regressed hard: R@10 0.450 → 0.210, with
the failure mode being publisher collapse (every query → AHURI
papers). Dev metrics during training were fine — held-out 10% of the
synthetic triplets hit MRR@10 = 0.79. The signal disappeared on real
queries.

But: stacked into hybrid + cross-encoder, FT vectors gave
R@10 = 0.770 on the full 100-query set, beating both base hybrid
(0.747) and base hybrid + rerank (0.673). The reranker absorbs the
publisher-collapse and reorders cleanly. Useful in a stack, dead
alone.

**2. "Always use hybrid" was wrong here.** On the synthetic-only
query split (gold authored *for* the chunk, retriever-independent),
hybrid + rerank wins everything. On the full 100-query set (which
folds in 60 BM25-pooled queries with their own selection bias),
BM25 alone scored R@10 = 0.907, hybrid 0.747, hybrid + rerank
0.673. Two splits, two stories — both honest, neither sufficient
on its own. I report both.

**3. A worked example in a routing prompt is itself a routing
change.** Added one GOOD/BAD example to a Haiku router prompt to
fix `compute_rental_yield` vs market-lookup confusion. Fixed the
target query (tool_acc 0.25 → 1.00) — and quietly migrated 12
unrelated queries from tool-only to doc-using. Faithfulness on
that switched cohort jumped +0.110, but a regex-based grounded-
ness metric slipped because more chunks → harder citation
adjacency. Prompt-engineering side effects are real and don't
show up in the metric you targeted.

Architecture is unsurprising — LangGraph 5-node agent (classify →
decompose → retrieve_or_tool → synthesize → reflect) with a
2-iteration cap, BGE base + BM25 RRF + cross-encoder rerank,
Anthropic prompt caching for ~90% input-token discount on warm
calls. Sonnet 4.6 for synth, Haiku 4.5 for routing/reflection.
Six-category retrieval failure taxonomy from the Phase 1 baseline
in the write-up.

Full technical write-up (eval tables, ablations, agent
evolution v1→v5):
github.com/Hyeonu-Cha/CadastreAI/blob/main/docs/blog.md

Happy to talk about any of this — particularly interested in
whether anyone has seen the publisher-collapse failure mode in
their own fine-tunes, and how you handled it.
```

---

## 4. r/LocalLLaMA  —  technical share

> Friendlier than r/MachineLearning; community is OK with personal
> tone if the content is technical. Use a discussion-anchored title.

**Title**

```
Built a RAG agent on a 42k-chunk Australian housing corpus over
4 weeks — three things that surprised me (fine-tune collapse,
hybrid wasn't always better, prompt examples ripple)
```

**Body**

```
Repo: github.com/Hyeonu-Cha/CadastreAI

Stack:
- BGE base-en-v1.5 + BM25 with RRF for retrieval
- cross-encoder/ms-marco-MiniLM-L-6-v2 for rerank
- LangGraph 5-node agent (classify → decompose → retrieve_or_tool
  → synthesize → reflect), max 2 iterations
- Claude Haiku 4.5 for routing/reflection, Sonnet 4.6 for synth
- Anthropic prompt caching on every system prompt (~90% input-
  token discount on warm calls)
- Qdrant for dense, in-process pickle for BM25 (the BM25 pickle
  bundles ~42k chunks + the model in 213 MB)
- Streamlit UI with persona switcher, inline citations, 5-step
  reasoning trace

Three findings that aren't in the "build a RAG" tutorials I read:

**The fine-tune wasn't dead, it was lonely.** I trained
`bge-base-en-v1.5` on ~4k synthetic pairs (Claude Haiku + BM25 hard
negatives). As a standalone retriever it collapsed onto AHURI
publishers — R@10 dropped 0.450 → 0.210. Dev metrics during
training looked great (MRR@10 = 0.79 on held-out triplets); on
real queries the signal vanished.

I left the fine-tune dormant for weeks. The deferred ablation —
ft + hybrid + cross-encoder — finally measured at R@10 = 0.770 on
the full 100-query set, beating both base hybrid (0.747) and
base hybrid+rerank (0.673). The reranker absorbs the publisher
collapse and reorders cleanly. The vectors carry useful signal
even when the retriever they're in can't use it.

**Two retrieval splits, two completely different stories.** My
synthetic split (40 queries, gold authored for each chunk) said
hybrid + rerank monotonically improves over base BGE. The full
100-query set (which includes 60 BM25-pooled queries) said BM25
alone beats every variant — R@10 = 0.907 vs hybrid's 0.747. That's
not a contradiction, it's a sampling artefact: when users phrase
questions the way the documents do, lexical retrieval is very
hard to beat. The "always use hybrid" advice is right on average
but can be wrong on a specific corpus + query distribution. I
report both splits in the README so the result is honest.

**A worked example in a routing prompt is a routing change.**
Added one GOOD/BAD example to the Haiku router to disambiguate
`compute_rental_yield` (needs explicit numbers) from market-yield
lookup (route to SQM + ABS). Fixed the target query (tool
accuracy 0.25 → 1.00). The same example also nudged 12 unrelated
queries from tool-only to doc-using — faithfulness on that
switched cohort went +0.110, but a regex grounded-ness metric
slipped because adjacent-citation discipline gets harder under
more chunks. Prompt-engineering side effects show up in metrics
you weren't watching.

Full write-up with the agent v1→v5 eval table, fine-tune collapse
analysis, and the six-category retrieval failure taxonomy:
github.com/Hyeonu-Cha/CadastreAI/blob/main/docs/blog.md

Demo: __DEMO_URL__

Curious if anyone else has run the same single-component vs.
in-stack ablation on a fine-tune that "regressed" — wondering how
common the lonely-vector failure mode actually is.
```

---

## 5. r/AusFinance  —  domain angle

> Domain subreddit; users won't care about RAG architecture but will
> care about *what the agent can answer about housing*. Lead with
> the use case, not the tech. Mods are strict about self-promotion;
> frame as a build log + ask for feedback.

**Title**

```
Built a research tool that answers Aus property questions with
cited evidence from RBA / ABS / AHURI / NHFIC — would love
feedback on the answers
```

**Body**

```
Hi all — spent the last four weeks building a research agent
over the Australian housing corpus (RBA, ABS, AHURI, NHFIC,
CoreLogic, SQM, Productivity Commission, Grattan, plus live
cash-rate / vacancy / yield tools). It's a portfolio project,
not a product, but the answers are cited and auditable so it
might actually be useful.

Personas: first-home buyer, investor, researcher, journalist.
Tone and source priority change per persona. Every numeric claim
has a source link or a "[tool: rba_cash_rate, retrieved_at: ...]"
marker that points to the live API call.

Some example queries it handles:

- "Is Parramatta a good bet for a family on $1.2M? What's the
   trend and yield?" (homebuyer)
- "Sydney suburbs with highest 3yr capital growth and vacancy
   under 3%" (investor)
- "How does CoreLogic's hedonic index differ from ABS 6432.0
   methodologically?" (researcher)
- "What's the 5-year story on housing supply in NSW?" (journalist)

It's not financial advice and it's not trying to be a real
estate platform — it's a way to ask a question and see the
underlying source you'd otherwise have to dig out of a PDF.

Demo (web, free, no signup): __DEMO_URL__
Repo (open source, MIT): github.com/Hyeonu-Cha/CadastreAI

What I'd genuinely value: try a question you'd actually ask
about your own situation, and tell me where the answer was
wrong, evasive, or missed an obvious source. The eval set has
100 queries but a handful of fresh ones from a different angle
would be the most useful feedback I could get before I call this
done.

Disclaimer: this is a research tool. Not financial advice.
Property decisions need a buyer's agent / mortgage broker /
solicitor / financial adviser.
```

---

## Posting checklist

- [ ] Replace `__DEMO_URL__` everywhere
- [ ] Replace `__LOOM_URL__` in LinkedIn (or strip the line if no Loom yet)
- [ ] **Re-verify eval numbers against the README** — every R@K, MRR,
      faithfulness, and tool-accuracy figure quoted in these drafts is
      a snapshot. If a final re-run (Task 4.17/4.18) shifts a number,
      update the post body before publishing. The split sizes
      (synthetic-only n, pooled n) in particular have drifted between
      docs in the past — spot-check both `README.md` (Results section)
      and `data/eval/queries_synth.jsonl`'s line count before posting.
- [ ] Sanity-check the GitHub URL renders + demo URL loads from
      a private window before posting
- [ ] Post Reddit first (lowest stakes), Medium next, LinkedIn last
- [ ] If r/MachineLearning auto-mods, swap in `[Research]` flair
      and re-post with the eval table front-and-centre
- [ ] r/AusFinance: read the latest pinned mod post for self-promo
      rules; frame as feedback request, not announcement
