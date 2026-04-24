# OpenInterpretability — Training Notebooks

> Train your first SAE in 30 min on free Colab. Train a hybrid-architecture SAE in 4 h on free Kaggle. Train paper-grade on cloud. One ladder, zero gatekeeping.

[![openinterp.org/train](https://img.shields.io/badge/site-openinterp.org%2Ftrain-8b5cf6)](https://openinterp.org/train)
[![License MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

---

## The ladder

| Tier | Notebook | Platform | VRAM | Cost | Model | Time |
|---|---|---|---|---|---|---|
| **🚀 Hobbyist** | [`01_hobbyist_gemma2_2b_colab.ipynb`](./notebooks/01_hobbyist_gemma2_2b_colab.ipynb) | Colab Free T4 | 15 GB | **$0** | Gemma-2-2B | 30–40 min |
| **⚡ Explorer** | [`02_explorer_qwen35_4b_kaggle.ipynb`](./notebooks/02_explorer_qwen35_4b_kaggle.ipynb) | Kaggle 2× T4 | 32 GB | **$0** (30 h/wk) | Qwen3.5-4B (hybrid GDN) | 4–5 h |
| **👑 Paper-grade** | [`03_papergrade_qwen36_27b_cloud.ipynb`](./notebooks/03_papergrade_qwen36_27b_cloud.ipynb) | Vast.ai RTX 6000 Pro | 96 GB | ~$30–60 | Qwen3.6-27B | 20–24 h |

## Open in your platform

### Tier 1 — Hobbyist (Colab Free)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OpenInterpretability/notebooks/blob/main/notebooks/01_hobbyist_gemma2_2b_colab.ipynb)

### Tier 2 — Explorer (Kaggle Free)

Upload `notebooks/02_explorer_qwen35_4b_kaggle.ipynb` via **Kaggle → Create → New Notebook → File → Import Notebook**. Free tier 2× T4 or 1× P100. Add `HF_TOKEN` in Secrets before running.

### Tier 3 — Paper-grade (cloud)

Run on Vast.ai / Lambda / RunPod with RTX 6000 Pro (96 GB) or equivalent. `git clone` this repo, `jupyter notebook`, open Tier 3, set `HF_TOKEN` env var. ~$30–60 per complete training run.

---

## Shared recipe (every tier)

All three use the same research-grade protocol; only hyperparameters scale:

- **TopK activation** (Gao et al. 2024) — hard top-k, no L1 penalty
- **AuxK auxiliary loss** — dead-feature revival (α=1/32, k_aux=d/2)
- **Geometric-median b_dec init** (Weiszfeld iteration) — robust to heavy-tailed residuals
- **Decoder column renorm** every step — keeps features interpretable
- **Cosine LR + warmup** — non-zero floor for continued dead-feature revival
- **HuggingFace streaming checkpoints** — crash-safe, never lose more than 5–10 min
- **sae_lens-compatible export** — safetensors + cfg.json, drop-in compatible with [SAELens](https://github.com/jbloomAus/SAELens) and [Neuronpedia](https://neuronpedia.org)

---

## What you get when you finish

1. **Your own SAE on HuggingFace** — citable, reusable, shareable
2. **val_report.json** — var_expl, L0, dead fraction on held-out data
3. **cfg.json** — architecture + hyperparameters for reproducibility
4. **A publishable artifact** — every tier produces an SAE usable in papers or deployments

---

## Prerequisites

- **Tier 1**: Google account + HuggingFace account. Edit one line (`HF_USERNAME`). That's it.
- **Tier 2**: Completed Tier 1 or equivalent background. Kaggle account + HF account.
- **Tier 3**: Cloud GPU access. HF account. Comfort with terminal + SSH.

---

## After you train

Your SAE is an asset. Put it to work:

- **[Trace it](https://openinterp.org/observatory/trace)** — watch features ignite token-by-token
- **[Publish to Atlas](https://openinterp.org/observatory/atlas)** — Q2 2026, cross-model feature graph
- **[Edit with Sandbox](https://openinterp.org/laboratory/sandbox)** — Q2 2026, drag-and-drop steering
- **[Contribute an Expedition](https://openinterp.org/academy/expeditions)** — Q3 2026, turn your run into a tutorial

---

## Contributing

Unusual architectures (Mamba, RWKV, diffusion-LM), alternative platforms (TPU, ROCm, MPS), or novel objectives (JumpReLU, BatchTopK, Gated SAE) especially welcome. Open an issue or a PR.

- [Open an issue](https://github.com/OpenInterpretability/notebooks/issues/new)
- [Read the manifesto](https://openinterp.org/manifesto)
- [See the roadmap](https://openinterp.org/roadmap)

---

## Standing on the shoulders of

- [SAELens](https://github.com/jbloomAus/SAELens) · checkpoint format standard
- [Gemma Scope](https://huggingface.co/google/gemma-scope) · reference at-scale SAE suite
- Gao et al. 2024 ([arxiv:2406.04093](https://arxiv.org/abs/2406.04093)) · TopK + AuxK recipe
- [Neuronpedia](https://neuronpedia.org) · the SAE encyclopedia

Built by [Caio Vicentino](https://huggingface.co/caiovicentino1) + OpenInterpretability. MIT License. 2026.
