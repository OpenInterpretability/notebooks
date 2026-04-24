# OpenInterpretability — Training & Interpretability Notebooks

> Everything you need to go from zero to a published SAE, understand it, share it, edit it, replicate research, and monitor production.

[![openinterp.org/train](https://img.shields.io/badge/site-openinterp.org%2Ftrain-8b5cf6)](https://openinterp.org/train)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

---

## The core ladder — train your first SAE

| Tier | Notebook | Platform | VRAM | Cost | Model | Time |
|---|---|---|---|---|---|---|
| **🚀 Hobbyist** | [`01_hobbyist_gemma2_2b_colab.ipynb`](./notebooks/01_hobbyist_gemma2_2b_colab.ipynb) | Colab Free T4 | 15 GB | **$0** | Gemma-2-2B | 30–40 min |
| **⚡ Explorer** | [`02_explorer_qwen35_4b_kaggle.ipynb`](./notebooks/02_explorer_qwen35_4b_kaggle.ipynb) | Kaggle 2× T4 | 32 GB | **$0** | Qwen3.5-4B (hybrid GDN) | 4–5 h |
| **👑 Paper-grade** | [`03_papergrade_qwen36_27b_cloud.ipynb`](./notebooks/03_papergrade_qwen36_27b_cloud.ipynb) | Cloud RTX 6000 Pro | 96 GB | ~$30–60 | Qwen3.6-27B | 20–24 h |

## After you train — close the loop

| Notebook | What it does |
|---|---|
| [`04_discover_features.ipynb`](./notebooks/04_discover_features.ipynb) | Auto-label your SAE's features with Claude or GPT-4, emit `feature_catalog.json` |
| [`05_build_shareable_trace.ipynb`](./notebooks/05_build_shareable_trace.ipynb) | Your SAE + your prompt → `trace.json` in the Trace Theater format |
| [`06_steer_your_model.ipynb`](./notebooks/06_steer_your_model.ipynb) | Live feature intervention: baseline vs α ∈ {-3, 0, 1, 3}. Q1 preview of the Q2 Sandbox. |

## Before you train — reduce friction

| Notebook | What it does |
|---|---|
| [`07_pick_your_tier.ipynb`](./notebooks/07_pick_your_tier.ipynb) | VRAM calculator + layer recommender. Zero GPU needed. |

## More models — same recipe, different architectures

| Notebook | Model | Platform |
|---|---|---|
| [`08_explorer_llama3_8b_kaggle.ipynb`](./notebooks/08_explorer_llama3_8b_kaggle.ipynb) | Llama-3.1-8B (Meta license) | Kaggle 2× T4 |
| [`09_explorer_mistral_7b_kaggle.ipynb`](./notebooks/09_explorer_mistral_7b_kaggle.ipynb) | Mistral-7B-v0.3 | Kaggle 2× T4 |
| [`10_hobbyist_phi3_mini_colab.ipynb`](./notebooks/10_hobbyist_phi3_mini_colab.ipynb) | Phi-3-mini-4k (Microsoft) | Colab Free T4 |

## Research-grade — replicate published results

| Notebook | Paper / protocol |
|---|---|
| [`11_stage_gate_g1.ipynb`](./notebooks/11_stage_gate_g1.ipynb) | Stage Gate 1 correlation pre-test (mechreward protocol) — ρ ≥ 0.30 on held-out GSM8K |
| [`12_batchtopk_vs_topk.ipynb`](./notebooks/12_batchtopk_vs_topk.ipynb) | BatchTopK vs TopK (Bussmann et al., [arxiv:2412.06410](https://arxiv.org/abs/2412.06410)) |

## Safety + production preview

| Notebook | What it does |
|---|---|
| [`13_watchtower_preview.ipynb`](./notebooks/13_watchtower_preview.ipynb) | Monitor input prompts for anomalous feature activations. Q1 preview of Q4 Watchtower Enterprise. Forward-only, no generation. |

---

## Quick start — open any notebook directly

**Colab Free T4** notebooks open via the badge at the top of the file, or directly:
`https://colab.research.google.com/github/OpenInterpretability/notebooks/blob/main/notebooks/<filename>.ipynb`

**Kaggle** notebooks open via "Create → New Notebook → File → Import from URL":
`https://raw.githubusercontent.com/OpenInterpretability/notebooks/main/notebooks/<filename>.ipynb`

---

## Shared recipe (every training tier)

All tiers use the same research-grade protocol; hyperparameters scale:

- **TopK activation** (Gao et al. 2024) — hard top-k, no L1 penalty
- **AuxK auxiliary loss** — dead-feature revival (α=1/32, k_aux=d/2)
- **Geometric-median b_dec init** (Weiszfeld) — robust to heavy-tailed residuals
- **Decoder column renorm** every step — keeps features interpretable
- **Cosine LR + warmup** — non-zero floor for continued dead-feature revival
- **HuggingFace streaming checkpoints** — crash-safe, never lose more than 5–10 min
- **sae_lens-compatible export** — safetensors + cfg.json

---

## After you train

Your SAE is an asset. Put it to work:

- **[Trace it](https://openinterp.org/observatory/trace)** — Trace Theater (10 scenarios across math, code, medical, safety, and more)
- **[Discover features](./notebooks/04_discover_features.ipynb)** — auto-label with LLM-judge
- **[Publish to Atlas](https://openinterp.org/observatory/atlas)** (Q2 2026) — cross-model feature graph
- **[Edit with Sandbox](https://openinterp.org/laboratory/sandbox)** (Q2 2026) — drag-and-drop steering

---

## Contributing

Unusual architectures (Mamba, RWKV, diffusion-LM), alternative platforms (TPU, ROCm, MPS), or novel objectives especially welcome.

- [Open an issue](https://github.com/OpenInterpretability/notebooks/issues/new)
- [Read the manifesto](https://openinterp.org/manifesto)
- [See the roadmap](https://openinterp.org/roadmap)
- [Browse the org](https://github.com/OpenInterpretability)

---

## Standing on the shoulders of

- [SAELens](https://github.com/jbloomAus/SAELens) · checkpoint format standard
- [Gemma Scope](https://huggingface.co/google/gemma-scope) · reference SAE suite (Lieberum et al. 2024)
- [Gao et al. 2024](https://arxiv.org/abs/2406.04093) · TopK + AuxK recipe
- [Bussmann et al. 2024](https://arxiv.org/abs/2412.06410) · BatchTopK
- [Neuronpedia](https://neuronpedia.org) · the SAE encyclopedia

Built by the [OpenInterpretability](https://github.com/OpenInterpretability) collective. MIT License. 2026.
