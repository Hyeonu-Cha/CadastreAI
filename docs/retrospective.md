# CadastreAI — private retrospective (Task 4.25)

> Personal post-mortem on the four-week build. Not for the README.
> Be specific, be honest, name the surprises. The point of this
> document is the next project, not this one — write the version of
> this you wish you'd read before starting.

**Time-boxed:** 60-90 minutes. Past that, the marginal insight
isn't worth the editing time.

**Voice:** First person. Past tense. Not a sales document; if
something didn't work, say it didn't work.

**Audience:** Future-you, six months from now, picking up a
similar project. Write what you'd want to know.

---

## What I built (one paragraph, for context)

> Replace this paragraph with two-three sentences on what
> CadastreAI ended up being. Anchor it in the *outcome* a stranger
> reading this six months from now needs — not how it works, but
> what it does and how well.
>
> Prompt: if I had to pitch this in 30 seconds at the kitchen
> table, what would I actually say?

---

## What worked (and would do again)

> 3-5 specific things. "Domain corpus + cited evidence" is too
> abstract. "BM25 pickle bundling all chunks meant zero ingestion
> latency at startup, made the deploy a single COPY" is the kind
> of detail to capture.
>
> For each: what was it, why did it work, what's the *transferable*
> lesson?

- **{Thing 1}** — {what} — {why it worked} — {transfer}
- **{Thing 2}** —
- **{Thing 3}** —
- **{Thing 4}** —
- **{Thing 5}** —

### Prompts to dig deeper

- Which decisions, in retrospect, were forced by a constraint I
  didn't see at the time? (e.g., "I picked Qdrant because docker-
  compose was easy" — what would I have picked given infinite
  setup time?)
- Which choice felt risky going in but paid off?
- What did I get for free from the stack (LangGraph, Anthropic
  prompt caching, Streamlit) that I underestimated?

---

## What I cut, and why

> The work that *didn't* ship is often the highest-signal part of
> a retrospective. List what was on the original plan that you
> dropped, and why.

- **{Cut item 1}** — {what} — {when did I cut it} — {why}
- **{Cut item 2}** —
- **{Cut item 3}** —

### Prompts to dig deeper

- Did I cut the right things? Which cut do I most regret?
- Did I cut anything because I was tired, vs. because the cut was
  correct?
- What would I have cut earlier if I'd known what I know now?

---

## What surprised me

> The shape of "expected vs. actual." Findings that contradicted
> my prior. Failure modes I hadn't seen before. These are the bits
> a retrospective is most uniquely able to surface, because in six
> months I'll have flattened them back into "of course, that's
> obvious."

- **{Surprise 1}** —
- **{Surprise 2}** —
- **{Surprise 3}** —

### Prompts I should answer specifically

- What did the data say that contradicted what I expected?
  (Anchor on numbers if I can — "I expected fine-tune R@10 to
  improve over base 0.45; it dropped to 0.21.")
- What broke in a way that wouldn't have shown up in a tutorial?
- What did a code-review or eval-pass surface that I'd missed
  during initial development?

---

## What I'd change about how I worked

> Process, not artifacts. How did I plan, prioritise, sequence,
> debug, write commits, design experiments, write tests? What
> habit produced the most value, and what habit ate the most time
> for the least gain?

- **Sequencing:** {what worked / didn't}
- **Eval discipline:** {how often did I run the eval; was it
  enough?}
- **Commit/PR cadence:** {one PR per ticket — was that overhead or
  scaffolding?}
- **Tooling I leaned on:** {Claude Code, Anthropic prompt caching,
  ...}
- **What I avoided that I shouldn't have:** {profiling? a
  notebook? a particular library?}

---

## The single most expensive mistake

> One thing. Be specific. What did it cost (hours, dollars, eval
> drift)? How did I notice it? What would have caught it earlier?
> If I have to write this in three sentences, what are they?

> *{fill in}*

---

## The single most valuable habit

> Same shape, opposite sign. What did I do this round that I'd
> port verbatim into the next project? Why?

> *{fill in}*

---

## What I'd build differently if I started today

> Not a wishlist for this project. The architecture for v2 of *the
> kind of thing this is*, given everything I now know.

- {bullet}
- {bullet}
- {bullet}

### Concrete v2 questions to answer (don't skip — this section
is the most valuable for the next project and the easiest to skip)

- Would I still pick BGE-base + BM25 + cross-encoder rerank, or
  has a small reranker-free dense model overtaken it for my corpus
  shape?
- Would I keep LangGraph, switch to a managed-agent surface, or
  hand-roll the agent loop with a thinner controller?
- Streamlit vs. Next.js + a tiny FastAPI: did the demo benefit from
  the Streamlit DX enough to justify it, or did I burn time fighting
  layout?
- Hybrid eval split design: would I still mix BM25-pooled + synthetic
  queries, or commit to a single methodology even if it costs me
  query count?
- Would I run the fine-tune at all? — given that the dormant
  ft+hybrid+rerank ablation was what made it worthwhile, and the
  default-on retriever path didn't use it.

---

## Things I learned that aren't about this project

> Skill, taste, judgement. Things I picked up that will outlast
> this codebase.

- **About retrieval / RAG:** {what}
- **About agents:** {what}
- **About evaluation:** {what}
- **About shipping:** {what}
- **About working with AI coding assistants:** {what} — this build
  was done with Claude Code in the driver's seat for ~most of the
  PRs. Where did delegating work well? Where did it produce code
  I had to throw out? What did I learn about the boundary between
  "describe outcome" and "specify the diff"?
- **About my own working style:** {what}

---

## Open threads I'd want to follow

> Ideas the project surfaced that I won't pursue here, but want to
> remember. The "I should write a paper about this" bucket.

- **{Thread 1}** — {one sentence on what's interesting}
- **{Thread 2}** —
- **{Thread 3}** —

---

## What I'd tell my four-weeks-ago self

> Three sentences. No more. The advice has to fit on a sticky note.

> *{fill in}*

---

*Written {YYYY-MM-DD}. Locked once written — don't edit later. The
value is the snapshot, not the polish.*
