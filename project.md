# 🔧 SmolAlign — Train, Align & Serve a Small LLM (for $0)
### A Solo AI-Engineering Project: QLoRA SFT + DPO Alignment + Inference Optimization, end-to-end on free compute

---

## 📌 TL;DR

**SmolAlign** takes a small open base model (Qwen2.5-1.5B) and owns its
*entire* lifecycle — **on free hardware, with no paid APIs**:

1. **Data** — curate an SFT instruction set + a binarized preference set from open datasets
2. **SFT** — supervised fine-tune with **QLoRA** (4-bit) on a free Kaggle/Colab T4
3. **Alignment** — **DPO** (Direct Preference Optimization) from the SFT checkpoint, proving a preference metric moved
4. **Evaluation** — held-out preference accuracy + **LLM-judge pairwise win-rate** using a *free* judge API (Groq / Gemini free tier), position-swapped to control bias
5. **Inference optimization** — merge adapters, **quantize to GGUF** (llama.cpp), benchmark tokens/sec on CPU and **vLLM throughput/latency** on a free GPU
6. **Serving + demo** — OpenAI-compatible endpoint + a Gradio chat UI deployed free on HF Spaces

**End state**: a trained+aligned model on HF Hub (with model card), a public demo, a results write-up with before/after numbers, and a technical blog post.

> **Why this project:** it demonstrates the two skills that most cleanly
> upgrade an "applied-RAG/agent engineer" profile into a "full-stack ML
> engineer" one — *training and aligning a model's weights* (not just
> calling an API) and *making inference fast and cheap* — and it costs
> nothing to build.

---

## 🎯 Skills This Demonstrates (the point of the project)

| Skill | Where it shows up | New vs. a typical RAG/agent portfolio |
|---|---|---|
| PEFT / QLoRA training | Phase 2 | ✅ Owns model weights, not just prompts |
| Preference optimization (DPO/ORPO) | Phase 3 | ✅ Alignment, reward margins, the Zephyr recipe |
| Rigorous LLM eval | Phase 4 | ✅ Win-rate, judge-bias control, honest deltas |
| Quantization & serving | Phase 5–6 | ✅ GGUF, vLLM, throughput/latency profiling |
| MLOps on a budget | Phase 0, 6 | ✅ HF Hub, free GPU orchestration, reproducible runs |

---

## 💸 The $0 Constraint — Free Stack

Everything below has a free tier sufficient for this project. **No credit card required for the core path.**

| Need | Free option | Limits to plan around |
|---|---|---|
| **GPU for training** | **Kaggle Notebooks** (2× T4, **30 hrs/week**) — primary; Colab free (1× T4) — backup | Sessions time out (~9–12h Kaggle, ~3–4h Colab). **Checkpoint to HF Hub every N steps.** |
| **LLM judge API** | **Groq** (free, Llama-3.3-70B) or **Google Gemini** Flash free tier | Rate-limited, not throughput-limited — fine for batch eval with backoff. |
| **Base model weights** | HF Hub — **Qwen2.5-1.5B**, Gemma-2-2B, Phi-3-mini, SmolLM2 | Use **0.5–3B** so it fits a single free T4 with headroom. |
| **CPU inference / GGUF** | `llama.cpp` + `llama-cpp-python` on your own machine | Slower than GPU, but truly free + offline. |
| **Model & artifact hosting** | **HF Hub** (free, unlimited public model repos) | Use for checkpoints, datasets, the final model. |
| **Experiment tracking** | **Weights & Biases** free tier, or TensorBoard | — |
| **Demo hosting** | **HF Spaces** (free CPU) via Gradio | CPU only → serve a small GGUF quant. |
| **CI** | **GitHub Actions** free minutes | Run lint + unit tests, not GPU jobs. |

**The one rule:** keep the model small (1.5B). A 7B QLoRA *barely* fits a free T4; a 1.5B trains comfortably with a real batch size, and preference deltas show up faster.

---

## 🛠️ Tech Stack & Key Choices

| Layer | Choice | Why |
|---|---|---|
| **Base model** | `Qwen2.5-1.5B` (base, not instruct) | Strong small base; starting from *base* makes the SFT→DPO story honest. 0.5B for fast iteration, 3B as a stretch. |
| **PEFT** | `peft` LoRA (4-bit QLoRA via `bitsandbytes`) | Fits free T4; ref-model-free DPO by toggling the adapter. |
| **Trainers** | `trl` `SFTTrainer` + `DPOTrainer` | Canonical, well-documented; mirrors the Zephyr recipe. |
| **SFT data** | Subset of `HuggingFaceTB/smoltalk` or `HuggingFaceH4/ultrachat_200k` | High-quality, permissively licensed, chat-formatted. |
| **DPO data** | `HuggingFaceH4/ultrafeedback_binarized` (subset) | The standard binarized preference set; clean chosen/rejected. |
| **Eval judge** | Groq Llama-3.3-70B **or** Gemini Flash (free) | Replaces the usual paid GPT-4 judge at $0. |
| **Quantization** | `llama.cpp` GGUF (Q4_K_M / Q5_K_M / Q8_0) | CPU-servable, the de-facto local format. |
| **GPU serving** | `vLLM` (on free T4) | Continuous batching; for throughput/latency numbers. |
| **Demo** | `gradio` on HF Spaces (free CPU) | Simplest streaming chat UI; serves the GGUF. |
| **Tracking** | W&B free tier | Loss/reward curves for the blog. |

---

## 📁 Repository Structure

```
smolalign/
├── README.md
├── project.md                      # this file
├── progress.md                     # per-task log (see workflow below)
├── pyproject.toml                  # uv
├── .env.example                    # HF_TOKEN, JUDGE_API_KEY, WANDB_API_KEY
│
├── configs/
│   ├── sft_qlora.yaml              # base model, LoRA rank, lr, batch, seq len
│   ├── dpo.yaml                    # beta, lr, ref-free toggle
│   └── eval.yaml                   # judge model, n prompts, swap policy
│
├── data/
│   ├── prepare_sft.py              # curate + chat-format + dedupe + length filter
│   ├── prepare_dpo.py              # binarized chosen/rejected formatting
│   └── eval_prompts.jsonl          # held-out instruction prompts for win-rate (committed)
│
├── src/
│   ├── train/
│   │   ├── sft.py                  # QLoRA SFT (trl SFTTrainer)
│   │   ├── dpo.py                  # DPO from SFT checkpoint
│   │   └── merge.py                # merge adapter → fp16
│   ├── eval/
│   │   ├── pref_accuracy.py        # held-out preference accuracy (no API)
│   │   ├── judge.py                # free-API judge client (retry + position swap)
│   │   ├── winrate.py              # pairwise win-rate harness
│   │   └── regression.py           # lm-eval-harness slice (forgetting check)
│   ├── serve/
│   │   ├── quantize.sh             # convert → GGUF → quantize variants
│   │   ├── bench_cpu.py            # llama.cpp tokens/sec + RAM per quant
│   │   ├── bench_gpu.py            # vLLM throughput vs batch, p50/p99
│   │   └── server.py              # OpenAI-compatible endpoint wrapper
│   └── app/
│       └── gradio_app.py           # streaming chat UI
│
├── notebooks/
│   ├── kaggle_sft.ipynb            # Kaggle-ready SFT (2× T4)
│   └── kaggle_dpo.ipynb            # Kaggle-ready DPO
├── results/                        # eval JSON + benchmark dumps (committed)
└── docs/
    ├── blog.md                     # grows through the project
    └── model_card.md               # pushed to HF Hub
```

---

# 🔁 Task-Based Workflow (branch → PR → merge → next branch)

This is the operating model for the whole project. **One task = one branch = one PR.** Never work two tasks on one branch.

### Conventions

- **Task IDs**: `Task P.NN` — `P` = phase number, `NN` = zero-padded sequence (e.g. `Task 2.03`).
- **Branch name = the task ID + a short slug**, one-to-one with the task:
  `task/<P.NN>-<slug>` → e.g. `task/2.03-sft-full-run`.
- **`main`** is always green and always the merge target.

### The per-task loop

```bash
# 1. Start from an up-to-date main
git checkout main && git pull origin main

# 2. Create the branch for THIS task (name mirrors the task ID)
git checkout -b task/2.03-sft-full-run

# 3. Do the work for exactly that task
#    ... implement ...

# 4. Tick the box in project.md and add a note in progress.md
#    - [x] Task 2.03 ...   (in this file)
#    - progress.md: short note + any metric/PR link

# 5. Commit + push the branch
git add -A
git commit -m "Task 2.03 — full QLoRA SFT run on Kaggle T4 + push adapter to Hub"
git push -u origin task/2.03-sft-full-run

# 6. Open a PR, let CI (lint + unit tests) pass, then MERGE it (squash)

# 7. Delete the merged branch, return to step 1 for the next task
git checkout main && git pull origin main
git branch -d task/2.03-sft-full-run
```

### Rules

1. **Sequential by default** — finish, merge, and only then branch the next task. Tasks within a phase are mostly ordered; cross-phase tasks have hard dependencies (you can't DPO before SFT merges).
2. **Each PR is self-contained and reviewable** — code + config + the `project.md` checkbox + the `progress.md` note all in one PR.
3. **GPU runs happen in the notebook, artifacts land on HF Hub** — the PR commits the *config, code, curves, and results JSON*, not the multi-GB weights. The PR body links the HF Hub model/adapter revision.
4. **Squash-merge** so `main` history reads one line per task: `git log --oneline main` is your project ledger.
5. **Keep `main` green** — CI runs lint + CPU unit tests on every PR. GPU/training is never in CI.

---

# 📅 Phases & Tasks

Pace is yours — there are no fixed dates. The **checkpoints** at the end of each phase are what matter. Each `Task` is one branch + one PR per the workflow above.

---

## **PHASE 0 — Setup & Free-Compute Infrastructure**

### Goal
A reproducible skeleton where a free T4 can load the base model and a free API can act as judge — before any training.

- [ ] **Task 0.01** — `uv init`; create `pyproject.toml` (Python 3.11+), the directory skeleton above, `.gitignore` (ignore `data/raw`, `*.gguf`, `wandb/`, checkpoints), and `.env.example` (`HF_TOKEN`, `JUDGE_API_KEY`, `WANDB_API_KEY`)
- [ ] **Task 0.02** — Author `notebooks/kaggle_sft.ipynb` scaffold: install pinned deps, HF Hub login from a Kaggle secret, mount W&B, **checkpoint-to-Hub every N steps** (so a session timeout never loses a run). Document the Kaggle GPU-quota workflow in `docs/blog.md`.
- [ ] **Task 0.03** — Smoke-load `Qwen2.5-1.5B` in 4-bit (`bitsandbytes`) on a free T4; generate from one prompt; **record peak VRAM** and confirm headroom for a real batch size. Save the number for the blog.
- [ ] **Task 0.04** — Implement `src/eval/judge.py`: a free-API judge client (Groq **or** Gemini, selectable) with exponential backoff, a strict JSON verdict schema, and a **position-swap** helper (run each pair A-vs-B *and* B-vs-A). Unit-test the parsing + swap logic with a stubbed client.

### Phase 0 Checkpoint
✅ Repo skeleton + reproducible deps · ✅ Base model loads in 4-bit on free T4 with VRAM recorded · ✅ Free-API judge client with swap + backoff, unit-tested

---

## **PHASE 1 — Data**

### Goal
A clean SFT set, a clean binarized preference set, and a held-out prompt set for win-rate — all committed (or pushed to Hub) and documented.

- [ ] **Task 1.01** — `data/prepare_sft.py`: take a ~15–20k subset of `smoltalk`/`ultrachat_200k`, apply the model's chat template, dedupe (exact + near-dup by hash), drop over-length samples; emit `data/sft/{train,val}.jsonl` (or push as an HF dataset).
- [ ] **Task 1.02** — SFT data card: token-length histogram, source mix, sample count, example rows → `docs/blog.md` data section + `results/sft_data_stats.json`.
- [ ] **Task 1.03** — `data/prepare_dpo.py`: take a ~10k subset of `ultrafeedback_binarized`, format `{prompt, chosen, rejected}` against the chat template, drop pairs where chosen==rejected or either is empty; emit `data/dpo/{train,test}.jsonl` (keep a **held-out test split** for preference accuracy).
- [ ] **Task 1.04** — Author/curate **300 held-out instruction prompts** (diverse: reasoning, coding, summarization, refusal/safety, open-ended) → `data/eval_prompts.jsonl`. These are *never* trained on; they drive win-rate in Phase 4.

### Phase 1 Checkpoint
✅ SFT train/val sets formatted + deduped + documented · ✅ DPO train/test sets binarized · ✅ 300-prompt held-out eval set committed

---

## **PHASE 2 — Supervised Fine-Tuning (QLoRA)**

### Goal
A base model turned into a competent instruction-follower via 4-bit QLoRA, with training curves and a merged fp16 checkpoint on HF Hub.

- [ ] **Task 2.01** — `src/train/sft.py` + `configs/sft_qlora.yaml`: `trl` `SFTTrainer` with 4-bit base, LoRA (r=16, alpha=32, dropout=0.05, target all linear), packing on, cosine schedule, gradient checkpointing. Config-driven, seeded.
- [ ] **Task 2.02** — **Overfit sanity run** (50 steps on 50 samples): confirm loss drops toward ~0 and the model parrots a held example. Catches data/template bugs before burning GPU hours.
- [ ] **Task 2.03** — Full SFT run on Kaggle T4 (~1 epoch over the subset): log loss to W&B, **push the adapter to HF Hub** with a revision tag. Commit the curve screenshot + run config.
- [ ] **Task 2.04** — `src/train/merge.py`: merge adapter → fp16, push merged model to Hub. Generate + commit a fixed set of qualitative samples (`results/sft_samples.md`) for later before/after comparison.

### Phase 2 Checkpoint
✅ QLoRA SFT trains on free T4 · ✅ Loss curve + adapter on Hub · ✅ Merged fp16 model + qualitative samples committed

---

## **PHASE 3 — Alignment (DPO)**

### Goal
The SFT model preference-tuned with DPO, with a **measurable** improvement in reward margin / preference accuracy — the core "I can align a model" result.

- [ ] **Task 3.01** — `src/train/dpo.py` + `configs/dpo.yaml`: `trl` `DPOTrainer` from the SFT checkpoint, **reference-free via the peft adapter toggle** (no second model in memory → fits T4). β≈0.1, low lr (~5e-6), 1 epoch.
- [ ] **Task 3.02** — DPO sanity run: confirm `rewards/margins` trends **up** and `rewards/accuracies` (chosen logp > rejected logp) climbs above 0.5 and keeps rising. If flat → data/format bug, fix before the full run.
- [ ] **Task 3.03** — Full DPO run: log reward margin + accuracy curves, push the DPO adapter (and merged model) to Hub with a revision tag. Commit curves + config.
- [ ] **Task 3.04** *(stretch)* — Single-stage **ORPO** variant from base for comparison (one less training stage); note the tradeoff vs SFT→DPO in the blog.

### Phase 3 Checkpoint
✅ DPO runs ref-free on free T4 · ✅ Reward margin/accuracy curves trending up · ✅ DPO model on Hub

---

## **PHASE 4 — Evaluation (the honest part)**

### Goal
Prove the alignment *helped*, with the same skepticism CadastreAI's eval used: a no-API metric, a judged win-rate with bias control, and a forgetting check.

- [ ] **Task 4.01** — `src/eval/pref_accuracy.py`: on the held-out DPO **test split**, compute preference accuracy (model assigns higher logp to chosen) for **SFT vs DPO**. No API, fully reproducible. → `results/pref_accuracy.json`.
- [ ] **Task 4.02** — `src/eval/winrate.py`: generate SFT and DPO answers for the 300 held-out prompts; judge pairwise with the free-API judge, **position-swapped both ways**; count win/tie/loss. Report win-rate + a 95% bootstrap CI.
- [ ] **Task 4.03** — Run the win-rate eval; quantify **judge bias** via swap-agreement (how often A-vs-B and B-vs-A agree); discard or down-weight non-agreeing pairs. → `results/winrate.json` + transcript samples.
- [ ] **Task 4.04** *(optional)* — `src/eval/regression.py`: a small `lm-eval-harness` slice (ARC-easy, HellaSwag subset, TruthfulQA) on base/SFT/DPO to check DPO didn't cause **catastrophic forgetting**.
- [ ] **Task 4.05** — Results write-up: before/after table (pref-accuracy, win-rate vs SFT, win-rate vs the *official instruct* model as a sanity ceiling), with the n-sensitivity / judge-bias caveats stated plainly. → `docs/blog.md` eval section.

### Phase 4 Checkpoint
✅ Preference accuracy (no-API) SFT vs DPO · ✅ Position-swapped win-rate with CI + bias check · ✅ Honest before/after table

---

## **PHASE 5 — Inference Optimization**

### Goal
Make the model fast and cheap, and *measure* it: a quality-vs-speed-vs-memory table across quantizations, on free hardware.

- [ ] **Task 5.01** — `src/serve/quantize.sh`: convert merged model → GGUF (f16), then quantize **Q8_0 / Q5_K_M / Q4_K_M**; push GGUFs to a Hub model repo.
- [ ] **Task 5.02** — `src/serve/bench_cpu.py`: with `llama-cpp-python`, benchmark **tokens/sec, prompt-eval time, and RAM** per quant on your own CPU; spot-check answer quality per quant on 20 prompts. → `results/bench_cpu.json`.
- [ ] **Task 5.03** — `src/serve/bench_gpu.py`: serve the fp16 model with **vLLM** on a free T4; measure **throughput vs batch size** and **p50/p99 latency**; chart it. → `results/bench_gpu.json` + figure.
- [ ] **Task 5.04** *(stretch)* — Compare 4-bit GPU serving (AWQ or GPTQ) and/or speculative decoding (a 0.5B draft model) against the fp16 baseline; report the speedup at equal quality.
- [ ] **Task 5.05** — Optimization write-up: the quality (win-rate delta) vs speed (tok/s) vs memory (RAM/VRAM) tradeoff table — the "I can serve, not just train" section. → `docs/blog.md`.

### Phase 5 Checkpoint
✅ GGUF quants on Hub · ✅ CPU tok/s + RAM per quant · ✅ vLLM throughput/latency curve · ✅ Tradeoff table

---

## **PHASE 6 — Serving & Demo**

### Goal
A public, clickable demo running the aligned model — for free.

- [ ] **Task 6.01** — `src/serve/server.py`: an **OpenAI-compatible** endpoint (llama.cpp server or vLLM) so the model is drop-in for any OpenAI client; `curl` smoke test in the README.
- [ ] **Task 6.02** — `src/app/gradio_app.py`: a streaming chat UI (system prompt, temperature, the model's chat template) backed by the GGUF via `llama-cpp-python`.
- [ ] **Task 6.03** — Deploy the Gradio app to **HF Spaces (free CPU)** serving a small quant (Q4_K_M); wire `HF_TOKEN` as a Space secret; confirm a public URL responds.
- [ ] **Task 6.04** — `scripts/deploy_check.sh`: health check + 5 smoke prompts (reasoning / coding / summarize / refusal / open-ended) with pass/fail on basic content + latency budget. Exit codes mirror CadastreAI's convention (0 healthy, 1 smoke fail, 2 env misconfig).

### Phase 6 Checkpoint
✅ OpenAI-compatible endpoint · ✅ Streaming Gradio UI · ✅ Public HF Spaces demo · ✅ Deploy-check script

---

## **PHASE 7 — Write-up & Polish**

### Goal
Turn the working system into portfolio-quality artifacts.

- [ ] **Task 7.01** — README: hero one-liner, the recipe diagram (base → SFT → DPO → quantize → serve), quickstart (`uv sync` → run demo locally), results tables, "what I'd do differently."
- [ ] **Task 7.02** — Blog post in `docs/blog.md`: the SFT→DPO recipe, training curves, the **eval methodology** (win-rate + judge-bias control), the serving benchmarks, and lessons learned. Cross-post (personal blog, LinkedIn, r/LocalLLaMA, r/MachineLearning).
- [ ] **Task 7.03** — HF Hub **model card** (`docs/model_card.md`): intended use, training data + recipe, eval results, quantizations, **limitations + safety notes**.
- [ ] **Task 7.04** — Demo GIF/short video in the README; tag `v1.0.0`; flip the repo public.

### Phase 7 Checkpoint
✅ README + blog + model card · ✅ Demo media · ✅ `v1.0.0` tagged, public

---

## ♻️ Cross-Cutting / Ongoing

- [ ] **Task X.01** — Maintain `progress.md`: one short note per merged task (metric, Hub revision, PR link).
- [ ] **Task X.02** — Pin **seeds + dep versions** for every training run so results are reproducible; record GPU type (Kaggle vs Colab) per run.
- [ ] **Task X.03** — Keep `.env.example` synced with any new secret (`HF_TOKEN`, `JUDGE_API_KEY`, `WANDB_API_KEY`).
- [ ] **Task X.04** — Keep the demo's served quant in sync with the latest DPO revision on Hub.

---

## 🎯 Success Criteria

You've nailed this project if all of these are true at the end:

1. ✅ **Measurable alignment win** — DPO beats SFT on held-out preference accuracy **and** on position-swapped judge win-rate (with CI), stated honestly.
2. ✅ **You trained the weights** — QLoRA SFT + DPO adapters and merged models live on HF Hub with a model card.
3. ✅ **You optimized serving** — a quality-vs-speed-vs-memory table across ≥3 quantizations, plus a vLLM throughput/latency curve.
4. ✅ **Public & reproducible** — anyone can open the Kaggle notebook, retrain, and run the demo; total spend **$0**.
5. ✅ **Portfolio artifacts** — blog + model card + README + live demo, all linked, all walkable for a recruiter.

---

## ⚠️ Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **Free GPU session times out mid-run** | Checkpoint to HF Hub every N steps (Task 0.02); resume from the last revision. Keep runs ≤ a few hours by using 1.5B + a data subset. |
| **DPO shows no/negative gain** | Most common cause is a chat-template mismatch between SFT and DPO — assert identical formatting. If genuinely flat, *document the negative result* with the reward curves: a rigorous "DPO helped on X but not Y" is itself a strong portfolio story. |
| **Judge bias inflates win-rate** | Always position-swap (Task 0.04/4.03); report swap-agreement; sanity-ceiling against the official instruct model so a suspiciously high win-rate is obvious. |
| **Catastrophic forgetting from DPO** | Low lr + β≈0.1 + 1 epoch; verify with the lm-eval slice (Task 4.04). |
| **OOM on free T4** | Drop to 0.5B, shorten seq len, enable gradient checkpointing + packing, reduce LoRA target modules. |
| **Scope creep** | Explicitly out of scope for v1: RLHF/PPO, multi-GPU, models >3B, RAG, tool-use. Those are v2. |

---

## 📚 References to Read Before Starting

- TRL docs — `SFTTrainer` and `DPOTrainer` (the reference-free / peft path)
- QLoRA paper (Dettmers et al.) — 4-bit NF4 + LoRA
- DPO paper (Rafailov et al.) — the loss and what β does
- The **Zephyr** report (HuggingFaceH4) — the SFT→DPO recipe this mirrors
- AlpacaEval / MT-Bench — for win-rate methodology and judge-bias pitfalls
- `llama.cpp` quantization docs + the **vLLM** docs on continuous batching

---

<div align="center">
  A solo AI-engineering project — train, align, quantize, and serve a small LLM end-to-end, on free compute.<br/>
  <sub>One task · one branch · one PR.</sub>
</div>
