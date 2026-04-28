<div align="center">

# OpenInterpretability · `notebooks`

### From `pip install transformers` to your own paper-grade SAE — 31 Colab / Kaggle / cloud notebooks covering training, hallucination research, crosscoders, and product reproducers.

[![openinterp.org/train](https://img.shields.io/badge/site-openinterp.org%2Ftrain-8b5cf6)](https://openinterp.org/train)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE)
[![Notebooks](https://img.shields.io/badge/notebooks-31-f97316)](./notebooks)
[![Discussions](https://img.shields.io/github/discussions/OpenInterpretability/notebooks)](https://github.com/OpenInterpretability/notebooks/discussions)
[![Good first issues](https://img.shields.io/github/issues/OpenInterpretability/notebooks/good%20first%20issue)](https://github.com/OpenInterpretability/notebooks/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)

</div>

---

## Part of a 5-repo ecosystem

| Repo | What's in it |
|---|---|
| [`.github`](https://github.com/OpenInterpretability/.github) | Org profile + shared CoC + SECURITY |
| [`web`](https://github.com/OpenInterpretability/web) | Next.js site behind openinterp.org |
| **`notebooks`** (you are here) | 31 training + interpretability + product-reproducer notebooks |
| [`cli`](https://github.com/OpenInterpretability/cli) | `pip install openinterp` — Python SDK |
| [`mechreward`](https://github.com/OpenInterpretability/mechreward) | SAE features as dense RL reward |

---

## 🚀 The core ladder — train your first SAE

| Tier | Notebook | Platform | VRAM | Cost | Model | Time |
|---|---|---|---|---|---|---|
| **Hobbyist** | [`01_hobbyist_gemma2_2b_colab.ipynb`](./notebooks/01_hobbyist_gemma2_2b_colab.ipynb) | Colab Free T4 | 15 GB | **$0** | Gemma-2-2B | 30–40 min |
| **Explorer** | [`02_explorer_qwen35_4b_kaggle.ipynb`](./notebooks/02_explorer_qwen35_4b_kaggle.ipynb) | Kaggle 2× T4 | 32 GB | **$0** | Qwen3.5-4B (hybrid GDN) | 4–5 h |
| **Paper-grade** | [`03_papergrade_qwen36_27b_cloud.ipynb`](./notebooks/03_papergrade_qwen36_27b_cloud.ipynb) | Cloud RTX 6000 Pro | 96 GB | ~$30–60 | Qwen3.6-27B | 20–24 h |

## 🔍 After you train — close the loop

| Notebook | What it does |
|---|---|
| [`04_discover_features.ipynb`](./notebooks/04_discover_features.ipynb) | Auto-label your SAE's features with Claude or GPT-4, emit `feature_catalog.json` |
| [`05_build_shareable_trace.ipynb`](./notebooks/05_build_shareable_trace.ipynb) | Your SAE + your prompt → `trace.json` in the Trace Theater format |
| [`06_steer_your_model.ipynb`](./notebooks/06_steer_your_model.ipynb) | Live feature intervention: baseline vs α ∈ {−3, 0, 1, 3}. Q1 preview of the Q2 Sandbox. |

## 🧭 Before you train — reduce friction

| Notebook | What it does |
|---|---|
| [`07_pick_your_tier.ipynb`](./notebooks/07_pick_your_tier.ipynb) | VRAM calculator + layer recommender. Zero GPU needed. |

## 🧪 More models — same recipe, different architectures

| Notebook | Model | Platform |
|---|---|---|
| [`08_explorer_llama3_8b_kaggle.ipynb`](./notebooks/08_explorer_llama3_8b_kaggle.ipynb) | Llama-3.1-8B (Meta license) | Kaggle 2× T4 |
| [`09_explorer_mistral_7b_kaggle.ipynb`](./notebooks/09_explorer_mistral_7b_kaggle.ipynb) | Mistral-7B-v0.3 | Kaggle 2× T4 |
| [`10_hobbyist_phi3_mini_colab.ipynb`](./notebooks/10_hobbyist_phi3_mini_colab.ipynb) | Phi-3-mini-4k (Microsoft) | Colab Free T4 |

## 🎓 Research-grade — replicate published results

| Notebook | Paper / protocol |
|---|---|
| [`11_stage_gate_g1.ipynb`](./notebooks/11_stage_gate_g1.ipynb) | Stage Gate 1 correlation pre-test ([mechreward protocol](https://github.com/OpenInterpretability/mechreward)) — ρ ≥ 0.30 on held-out GSM8K |
| [`12_batchtopk_vs_topk.ipynb`](./notebooks/12_batchtopk_vs_topk.ipynb) | BatchTopK vs TopK (Bussmann et al., [arxiv:2412.06410](https://arxiv.org/abs/2412.06410)) |

## 🛡️ Safety + production preview

| Notebook | What it does |
|---|---|
| [`13_watchtower_preview.ipynb`](./notebooks/13_watchtower_preview.ipynb) | Monitor input prompts for anomalous feature activations. Q1 preview of Q4 Watchtower Enterprise. Forward-only, no generation. |

## 🔗 Circuits — attribution graphs between SAE features

| Notebook | What it does |
|---|---|
| [`14_attribution_patching.ipynb`](./notebooks/14_attribution_patching.ipynb) | **AtP\*** (Kramár et al. 2024, [arxiv:2403.00745](https://arxiv.org/abs/2403.00745)) — QK-fix + GradDrop node attribution |
| [`15_sparse_feature_circuits.ipynb`](./notebooks/15_sparse_feature_circuits.ipynb) | Marks et al. 2024 ([arxiv:2403.19647](https://arxiv.org/abs/2403.19647)) replication — node + edge + error-term DAG |
| [`16_autocircuit_acdc.ipynb`](./notebooks/16_autocircuit_acdc.ipynb) | ACDC slow-mode via [AutoCircuit](https://github.com/UFO-101/auto-circuit) |
| [`17_train_crosscoder.ipynb`](./notebooks/17_train_crosscoder.ipynb) | Sparse Crosscoder (Lindsey et al. 2024) — shared dictionary across L11/L31/L55 |

All circuit notebooks emit JSON consumed directly by the [**Circuit Canvas**](https://openinterp.org/observatory/circuits) on openinterp.org.

## 📊 Leaderboard — InterpScore v0.0.1

| Notebook | What it does |
|---|---|
| [`18_interpscore_eval.ipynb`](./notebooks/18_interpscore_eval.ipynb) | Composite SAE ranking — loss_recovered + alive + L0 + sparse probing + TPP. Emits `interpscore.json` → PR to [`web/lib/leaderboard.ts`](https://github.com/OpenInterpretability/web/blob/main/lib/leaderboard.ts). |

## 🔭 Lenses — classic layer-wise prediction tools

| Notebook | Method |
|---|---|
| [`19_logit_lens.ipynb`](./notebooks/19_logit_lens.ipynb) | Logit Lens (nostalgebraist 2020). 5 lines of PyTorch, ~5 min on T4. |
| [`20_tuned_lens.ipynb`](./notebooks/20_tuned_lens.ipynb) | Tuned Lens (Belrose et al. 2023, [arxiv:2303.08112](https://arxiv.org/abs/2303.08112)). Pretrained or fresh-fit. |

## 📏 Probing — the supervised baselines SAE features must beat

| Notebook | Method |
|---|---|
| [`21_linear_probe.ipynb`](./notebooks/21_linear_probe.ipynb) | sklearn LogisticRegression on residuals + **diff-of-means baseline** (Farquhar 2023 requires it) |
| [`22_ccs_probe.ipynb`](./notebooks/22_ccs_probe.ipynb) | Contrast Consistent Search (Burns 2022) with honest critique baselines |
| [`23_repe_reading_vector.ipynb`](./notebooks/23_repe_reading_vector.ipynb) | Representation Engineering LAT (Zou 2023) — extract + monitor + steer |

## 🌀 Hallucination — detection & steering arc

The full research arc behind the [2026-04-25 blog post](https://openinterp.org/blog) on hallucination
in 27B reasoning models. Notebooks 24 → 28b shipped 2026-04-25 → 26.

| Notebook | What it does |
|---|---|
| [`24_hallucination_entity_separation_qwen36_27b.ipynb`](./notebooks/24_hallucination_entity_separation_qwen36_27b.ipynb) | v0.0.1 — fake AUROC=1.0 from a 2× tokenization confound. The honest negative result. |
| [`24b_hallucination_v002_ferrando_proper.ipynb`](./notebooks/24b_hallucination_v002_ferrando_proper.ipynb) | Ferrando 2024 replication on Qwen3.6-27B. **AUROC 0.84 on 226 real Wikidata entities.** |
| [`25_steering_f61723_calibration.ipynb`](./notebooks/25_steering_f61723_calibration.ipynb) | Single-feature steering null result. Detection ≠ control. |
| [`26_multi_feature_steering.ipynb`](./notebooks/26_multi_feature_steering.ipynb) | Multi-feature top-K (no controls). The version we almost shipped overclaimed. |
| [`27_multi_feature_steering_with_controls.ipynb`](./notebooks/27_multi_feature_steering_with_controls.ipynb) | The walk-back. 6 controls (random-K + Claude judge + permutation). **It induces hallucination, not calibration.** |
| [`28_paper_baselines_qwen36_27b.ipynb`](./notebooks/28_paper_baselines_qwen36_27b.ipynb) | ICML MI Workshop 2026 paper-1 baselines. **L31/f34957 0.81 vs LR ceiling 0.887 vs diff-of-means 0.859.** Per-layer scan, bootstrap CI. |
| [`28b_sensitivity_refusal_only.ipynb`](./notebooks/28b_sensitivity_refusal_only.ipynb) | Sensitivity ablation — same residual capture, two labelling rules. Reviewer-defence. |

## 🔀 Crosscoders — cross-model + cross-stage

The methodology behind paper-1's Pearson causal-equivalence (`Pearson_CE`) finding.
First per-feature causal-equivalence test in the crosscoder literature.

| Notebook | What it does | Pair |
|---|---|---|
| [`17_train_crosscoder.ipynb`](./notebooks/17_train_crosscoder.ipynb) | Cross-LAYER crosscoder (Lindsey 2024). Single model, multi-layer. | Gemma-2-2B L6/L12/L18 |
| [`17b_crosscoder_model_diff_papergrade.ipynb`](./notebooks/17b_crosscoder_model_diff_papergrade.ipynb) | Cross-MODEL crosscoder + Pearson_CE. **Median cosine 0.965 vs CE 0.616 — 38% gap.** | Gemma-2-2B base/IT |
| [`17c_crosscoder_rl_diffing_papergrade.ipynb`](./notebooks/17c_crosscoder_rl_diffing_papergrade.ipynb) | Cross-STAGE crosscoder. LoRA toggle pattern (single base + PEFT.disable_adapter). | Qwen3.5-4B base vs mechreward-G3 |

## 🛡️ Guards — product reproducers

Each notebook reproduces an exact metric behind a shipped openinterp Guard
(SDK on PyPI, demo on HF, landing on openinterp.org/products/X).
**Drop-in `pip install openinterp` and you have these probes.**

| Notebook | Product | Headline number | Reproducer |
|---|---|---|---|
| [`30_hallucinationguard_proof_qwen36_27b.ipynb`](./notebooks/30_hallucinationguard_proof_qwen36_27b.ipynb) | FabricationGuard PoC v1 | Single-feature failed cross-bench (0.50–0.60) | [Open in Colab](https://colab.research.google.com/github/OpenInterpretability/notebooks/blob/main/notebooks/30_hallucinationguard_proof_qwen36_27b.ipynb) |
| [`31_hallucinationguard_v2_linear_probe.ipynb`](./notebooks/31_hallucinationguard_v2_linear_probe.ipynb) | **FabricationGuard v2** (production) | **AUROC 0.88 cross-task · −88% confident-wrong** | [Open in Colab](https://colab.research.google.com/github/OpenInterpretability/notebooks/blob/main/notebooks/31_hallucinationguard_v2_linear_probe.ipynb) |
| [`32_reasoningguard_proof_qwen36_27b.ipynb`](./notebooks/32_reasoningguard_proof_qwen36_27b.ipynb) | ReasoningGuard PoC | TBD — passes 3/3 ships v0.3 | [Open in Colab](https://colab.research.google.com/github/OpenInterpretability/notebooks/blob/main/notebooks/32_reasoningguard_proof_qwen36_27b.ipynb) |

Each reproducer ships:
- `probe.joblib` + `meta.json` to HF dataset (drop-in for the SDK)
- `verdict.json` with raw numbers
- `headline.png` for landing pages / posts
- All artifacts pushed to `caiovicentino1/<ProductName>-linearprobe-qwen36-27b` (HF dataset)

---

## 🛠️ Shared recipe (every training tier)

All tiers use the same research-grade protocol; hyperparameters scale:

- **TopK activation** ([Gao et al. 2024](https://arxiv.org/abs/2406.04093)) — hard top-k, no L1 penalty
- **AuxK auxiliary loss** — dead-feature revival (α=1/32, k_aux=d/2, dead_threshold=10M tokens)
- **Geometric-median `b_dec` init** (Weiszfeld) — robust to heavy-tailed residuals
- **Decoder column renorm every step** — keeps features interpretable
- **Cosine LR + warmup** — non-zero floor for continued dead-feature revival
- **HuggingFace streaming checkpoints** — crash-safe, never lose more than 5-10 min
- **sae_lens-compatible export** — `safetensors` + `cfg.json`

---

## 🚦 Hard constraints on every notebook

If you port an existing notebook or write a new one, honor these — CI and review will check:

| ✅ DO | ❌ DON'T |
|---|---|
| `dtype=torch.bfloat16` | `torch_dtype=` (deprecated in transformers 5.x) |
| `attn_implementation='sdpa'` | `flash-attn` (reproducibility + install pain) |
| HF_TOKEN via Colab/Kaggle **secret** | Hard-coded tokens |
| HF streaming checkpoints every 5-10M tokens | Drive-only checkpoints (kernel dies = data loss) |
| Per-layer `model.language_model.layers[N]` fallback | Hard-coded `.layers[N]` (breaks on multimodal) |
| Honest var_expl + L0 + dead% | Cherry-picked seeds |

---

## 📓 How to contribute a new notebook

> Full rules in [CONTRIBUTING.md](./CONTRIBUTING.md). The 3 most common PR patterns:

### 1. Port a notebook to a new model

The most valuable contribution. Pick an existing notebook that matches your tier (01 for hobbyist, 02 for Kaggle-scale, 03 for paper-grade) and swap:

```python
MODEL_ID   = 'meta-llama/Llama-3.2-3B'   # was: 'google/gemma-2-2b'
LAYER      = 14                           # was: 15 — middle-stack heuristic
D_MODEL    = 3072                         # was: 2304
```

Name the new file `NN_<tier>_<model-slug>_<platform>.ipynb` where `NN` is the next free number.

**PR title**: `Add Hobbyist tier for Llama-3.2-3B (notebook 24)` — include a screenshot of the final eval cell output.

### 2. Replicate a 2024-2026 paper

Add a notebook under `notebooks/` that reproduces the main result. Structure:

1. Title markdown cell with **full citation + arxiv link**
2. Install cell with **pinned versions**
3. Config cell with **all hyperparameters from the paper**
4. Implementation of the method (inline, not a separate repo — notebooks are self-contained)
5. Validation cell that outputs **the paper's headline metric**

**PR title**: `Replicate: <paper short title> (notebook NN)` — match the paper's exact numbers within tolerance.

### 3. Add a platform (TPU, ROCm, MPS)

Right now every notebook assumes CUDA. Adding a platform is a multi-notebook effort, usually via a common helper:

- Write `notebooks/_platform_<name>.py` with `pick_device()`, `get_dtype()`, etc.
- Patch one existing notebook to use it as proof-of-concept
- Open a **draft PR** and tag @caiovicentino for design review before the full port

---

## ✔️ Before opening a PR — validate locally

```bash
python3 -c "import json; json.load(open('notebooks/YOUR_NOTEBOOK.ipynb'))"
```

This catches the most common breakage (bad JSON, unclosed cell). CI also runs `nbformat.validate` on every PR.

If you have a GPU and want to dry-run the first ~10 cells:

```bash
jupyter nbconvert --to notebook --execute notebooks/YOUR_NOTEBOOK.ipynb --ExecutePreprocessor.timeout=300
```

(Expect the heavy training cells to fail under 300s — that's fine; the goal is to catch import errors + dtype bugs early.)

---

## Output schemas other tools consume

If your notebook emits a JSON that the website consumes, match the schema:

| Tool | Schema (TypeScript source) |
|---|---|
| Trace Theater | [`web/lib/trace-data.ts` · `TraceScenario`](https://github.com/OpenInterpretability/web/blob/main/lib/trace-data.ts) |
| Circuit Canvas | [`web/lib/circuit-data.ts` · `CircuitData`](https://github.com/OpenInterpretability/web/blob/main/lib/circuit-data.ts) |
| InterpScore leaderboard | [`web/lib/leaderboard.ts` · `LeaderboardEntry`](https://github.com/OpenInterpretability/web/blob/main/lib/leaderboard.ts) |

---

## 🎯 After you run a notebook

Your SAE is an asset. Put it to work:

- **[Trace it](https://openinterp.org/observatory/trace)** — Trace Theater (10 scenarios) — view + share
- **[Submit to InterpScore](https://openinterp.org/interpscore)** — public leaderboard
- **[Edit with Sandbox](https://openinterp.org/laboratory/sandbox)** (Q2 2026) — drag-and-drop steering
- **[Contribute an Expedition](https://openinterp.org/academy/expeditions)** (Q3 2026) — turn your run into a tutorial

---

## Community

- 💬 [Discussions](https://github.com/OpenInterpretability/notebooks/discussions) — "which notebook should I use for X?"
- 🟢 [Good-first-issues](https://github.com/OpenInterpretability/notebooks/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) — start here
- 📖 [Contributor guide](./CONTRIBUTING.md) — full workflow
- ✉️ hi@openinterp.org

---

## Standing on the shoulders of

- [SAELens](https://github.com/jbloomAus/SAELens) · our checkpoint format
- [Gemma Scope](https://huggingface.co/google/gemma-scope) · reference at-scale SAE suite
- [Gao et al. 2024](https://arxiv.org/abs/2406.04093) · TopK + AuxK recipe
- [Bussmann et al. 2024](https://arxiv.org/abs/2412.06410) · BatchTopK
- [Neuronpedia](https://neuronpedia.org) · the SAE encyclopedia

Apache-2.0 · [openinterp.org](https://openinterp.org) · 2026
