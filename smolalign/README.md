# 🔧 SmolAlign

**Train, align, quantize, and serve a small instruction-following LLM — end to end, on free compute.**

A solo AI-engineering portfolio project: take a small open base model
(Qwen2.5-1.5B), turn it into a competent assistant with **QLoRA SFT**, align
it with **DPO**, prove the gain with a **position-swapped LLM-judge win-rate**,
then **quantize and benchmark** it for serving — all for **$0** on Kaggle's
free GPU, free judge APIs, and the Hugging Face Hub.

> 📋 Full plan, task breakdown, and the one-task-one-branch-one-PR workflow:
> **[`project.md`](./project.md)**

---

## Pipeline

```
Qwen2.5-1.5B (base)
   └─ QLoRA SFT  ──▶  DPO alignment  ──▶  eval (win-rate + pref-acc)
                                              └─ quantize (GGUF) ──▶ serve (vLLM / llama.cpp) ──▶ Gradio demo
```

## Quickstart (local, CPU demo)

```bash
# Python env
uv sync                       # or: pip install -e ".[serve,app]"

# Copy env template and fill in tokens (all free tiers)
cp .env.example .env          # HF_TOKEN, JUDGE_API_KEY, WANDB_API_KEY

# Run the chat demo against a quantized GGUF (Phase 6)
python -m src.app.gradio_app
```

Training runs in the free Kaggle notebooks under [`notebooks/`](./notebooks/);
artifacts (adapters, merged models, GGUFs) live on the HF Hub.

## Status

This is a fresh scaffold. Work proceeds task-by-task per
[`project.md`](./project.md) — see [`progress.md`](./progress.md) for the log.

## License

MIT.
