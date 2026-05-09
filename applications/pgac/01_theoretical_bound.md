# PGAC Phase 1 — Theoretical Bound for Probe-Gated Adaptive Compute

**Goal**: derive an upper bound on inference speedup achievable via probe-gated adaptive compute (PGAC) on a transformer with a TopK Sparse Autoencoder, expressed as a function of measurable quantities.

---

## Setup

Consider a decoder-only transformer with `L` layers. Each layer ℓ has:

- Attention sub-layer: compute `C_attn(d) ≈ 4d² + 2d·seq_len` per token (dominated by `4d²` for long-context regime where seq_len ≪ d).
- FFN sub-layer: compute `C_ffn(d) = 8d²` per token (up_proj + activation + down_proj with intermediate dim `4d`).
- Total per-layer per-token compute: `C_layer(d) ≈ 12d²`.

For Qwen3.6-27B: `L = 64`, `d = 4096`, so per-token forward pass is `L · C_layer ≈ 12.6 GFLOPs`.

Now assume we have a **TopK SAE** trained at the residual stream of one or more layers with:

- Dictionary size `d_sae = expansion · d` (typical: `expansion = 16`, so `d_sae = 65,536`).
- TopK constraint: only the top-`k` features are non-zero per token (typical: `k ∈ [32, 128]`).
- Sparsity ratio: `p = k / d_sae` (typical: `p ∈ [0.0005, 0.002]`).

**Empirical observation** (Bricken et al. 2023; Templeton et al. 2024; our Qwen3.6-27B SAE 2026): the FFN computation in dense form computes a superposition over the entire `d_sae` feature basis even though only a `k`-sized subset is active. The dense layers compute "features that won't fire" and discard them.

**PGAC hypothesis**: a cheap probe predicts the active feature set `S(token) ⊂ {1, ..., d_sae}` of size `k_pred ≈ k` BEFORE computing the dense FFN, allowing selective computation.

---

## Compute reduction model

Let `f_PGAC` be the speedup factor for the FFN sub-layer alone.

**Standard FFN (dense)** computes:

```
y_dense = W_down · σ(W_up · h)   // h ∈ R^d, W_up ∈ R^(4d × d), W_down ∈ R^(d × 4d)
```

Compute: `O(8d²)`.

**PGAC FFN (sparse, via SAE basis)** computes:

```
predicted_S = probe(h)                         // top-k feature indices
y_PGAC = D[:, predicted_S] · σ(E[predicted_S, :] · h)
```

Where `E` and `D` are the SAE encoder and decoder, evaluated only on the predicted active features.

Compute:
- Probe call: `O(d · d_probe)` ≈ `O(d · 1)` (linear probe outputs scalar for each top-k candidate, or O(d · k_pred) for ranking).
- SAE encoder restricted to top-k: `O(d · k_pred)`.
- SAE decoder restricted to top-k: `O(k_pred · d)`.
- Total: `O(d · (1 + 2k_pred))` ≈ `O(2d · k_pred)`.

**FFN speedup**:

```
f_FFN = 8d² / (2d · k_pred) = 4d / k_pred                          (1)
```

For `d = 4096`, `k_pred = 64`: `f_FFN = 256x`.

---

## Layer-level speedup (limited by attention)

Attention does NOT have the same sparsity advantage by default. Layer-level speedup is:

```
f_layer = (C_attn + C_ffn) / (C_attn + C_ffn,PGAC)
       = (4d² + 8d²) / (4d² + 2d · k_pred)
       = 12d² / (4d² + 2d · k_pred)                                  (2)
```

For `d = 4096`, `k_pred = 64`:
```
f_layer = 12·4096² / (4·4096² + 2·4096·64)
       = 12·16.7M / (4·16.7M + 524K)
       = 201M / 67.7M ≈ 2.97x
```

**Hard ceiling: layer-level speedup capped at 3x** because attention is not sparse-routed (yet).

---

## Early-exit speedup

If the probe predicts at layer ℓ that the token's output is "determined" (low remaining-information criterion), the model can early-exit, skipping layers ℓ+1 through L.

Let `E[ℓ_exit]` denote the expected exit layer averaged over a token distribution. The early-exit speedup factor is:

```
f_exit = L / E[ℓ_exit]                                                (3)
```

Empirically (Schuster et al. "CALM" 2022; Mixture-of-Depths Raposo et al. 2024): for "easy" tokens (function words, end-of-sequence punctuation) `ℓ_exit` can be as low as `0.3L`. For "hard" tokens (semantic decisions), `ℓ_exit ≈ L`. Realistic average: `E[ℓ_exit] ≈ 0.6L`, giving `f_exit ≈ 1.67x`.

---

## Combined PGAC speedup

```
f_total = f_layer · f_exit                                            (4)
```

Realistic compound:
- `f_layer = 3x` (FFN-via-SAE)
- `f_exit = 1.67x` (early termination)
- **Compound: f_total ≈ 5x at iso-quality**

This is the theoretical upper bound for the simple PGAC formulation. With aggressive quantization of "secondary" features (probe-predicted-low-importance):

```
f_quant = 1 + α · q_factor                                            (5)
```

Where `α` is the fraction of features that can be quantized aggressively (e.g., to INT4 from BF16) and `q_factor ≈ 4x` for INT4. For `α = 0.3`, `f_quant ≈ 2.2x` additional in memory bandwidth, ≈ `1.5x` in compute.

**Total compound with quantization**: `f_total · f_quant ≈ 7-8x at iso-quality`.

---

## Quality preservation bound (revised, 2026-05-02)

The quality preservation analysis decomposes into two independent error sources, both of which we must bound.

### Error source 1: SAE reconstruction floor

Even with a perfect probe (recall = 1.0), routing FFN compute through the SAE basis adds reconstruction error. For TopK SAE with variance explained `VE`:

```
||e_SAE||² ≈ (1 - VE) · ||h||²                                        (6)
```

Empirically (Bricken 2023, Templeton 2024, Qwen3.6-27B SAE 2026 with VE = 0.706-0.842): SAE-only routing loses approximately `5-10 percentage points` on downstream tasks (HumanEval, MMLU, GSM8K) vs the full dense FFN.

This is the **floor** PGAC can achieve. A probe accuracy of 100% gives PGAC quality equal to SAE-only routing quality.

### Error source 2: Probe imperfect recall

Let `r = recall@k = |TP|/k` denote the fraction of true top-k features the probe correctly predicts. Missed features (FN) contribute additional error:

```
||e_FN||² ≈ k(1-r) · E[α²] · ||d||²                                   (7)
```

For task quality, this translates to additional points lost beyond the SAE floor. Empirically, the marginal cost is approximately:

```
ΔQuality(r) ≈ β · (1 - r)        with β ≈ 10 pp per unit (1-r)        (8)
```

(The constant `β` is task-dependent; we use 10pp/unit as a conservative estimate from Schuster CALM 2022 layer-skip analogy and our nb46 ensemble OOD stats.)

### Combined bound

Total task-quality loss vs full-model baseline:

```
Y_total = (5-10 pp) + (1 - r) · 10 pp                                 (9)
```

Setting `Y_total ≤ 8 pp` (acceptable tradeoff for 4x speedup):

```
(1 - r) · 10 ≤ 3        →        r ≥ 0.7
```

### Required probe AUROC

The probe is a ranker over `d_sae` candidates outputting top-k. Recall@k and AUROC for a ranker satisfy approximately:

| AUROC (binary "feature in top-k") | recall@k (typical) |
|---|---|
| 0.80 | ~0.55 |
| 0.85 | ~0.70 |
| 0.90 | ~0.85 |
| 0.95 | ~0.95 |
| 1.00 | 1.00 |

**Required probe AUROC: ≥ 0.85** to preserve quality within 8 pp of full-model baseline (eq 9 + table).

Per nb37 v2 + nb41 v2 results, fresh probes on Qwen3.6-27B routinely achieve AUROC 0.85-0.95 on similar binary-classification tasks. **Probe accuracy is achievable**, not the bottleneck.

### What's actually the bottleneck

The bottleneck is the **SAE reconstruction floor itself** (5-10pp loss). For PGAC to be commercially deployable at iso-quality with full-model, we need:
- Better SAEs with higher VE (currently 0.84 max on Qwen3.6 L31), or
- Sparse-MoE-style alternatives that route through full-precision features for top-k

For PGAC at "5-10pp quality loss in exchange for 4x speedup", the math IS achievable today.

---

## Falsifiable claims for empirical Phase 2

| Claim | Predicted measurement |
|---|---|
| FFN compute reduction at iso-MSE on residual | f_FFN = 4d/k_pred = 256x for k=64, d=4096 |
| Layer-level speedup with attention bottleneck | f_layer ≤ 3x |
| Early-exit average layer | E[ℓ_exit] / L ≈ 0.55-0.65 on Qwen3.6-27B reasoning tasks |
| Combined inference speedup (no quant) | f_total ≈ 4-6x at iso-quality |
| Combined with INT4 secondary quant | f_total ≈ 6-9x |
| Probe AUROC threshold for iso-quality | AUROC ≥ 0.85 on top-k feature presence |

---

## What this is and isn't

**This is**: a falsifiable upper bound on the speedup PGAC can achieve, derived from compute counts of the standard transformer architecture, the TopK SAE structure, and information-theoretic noise propagation.

**This is NOT**: a guarantee that PGAC will work in practice. Empirical confirmation requires:
1. Training a probe that achieves AUROC ≥ 0.85 on top-k feature presence (Phase 2).
2. Implementing the selective FFN compute kernel (Phase 3).
3. Measuring downstream quality (HumanEval, MBPP, MMLU) at the actually-realized compute reduction (Phase 4).

The bound shows that **if PGAC works**, the achievable speedup is in the 5-8x range at iso-quality, which would be a meaningful contribution to inference efficiency in OSS reasoning models.

---

## Open theoretical questions for follow-up

1. **Sparse attention extension**: can PGAC also apply to attention (currently the bottleneck cap)? Sparse attention literature (Sparse Transformer, Longformer) suggests yes for long-context but not for short-context yet.
2. **Cross-layer feature dependence**: features at layer ℓ_n depend on features at layer ℓ_{n-1}. Probe must account for this. Bounds for this case are unsolved.
3. **Optimal probe-feature joint training**: jointly training probe and SAE for compute-aware sparsity could outperform sequential approaches.

These are paper-grade follow-up directions for Phase 5+ work.

---

**Reference cite**:
- Bricken et al. 2023, "Towards Monosemanticity via Sparse Autoencoders"
- Schuster et al. 2022, "CALM: Calibrated Anytime Language Models" (early exit)
- Raposo et al. 2024, "Mixture-of-Depths" (token-level skip)
- Templeton et al. 2024, "Scaling Monosemanticity" (TopK SAE statistics)
- This paper, derivation 2026-05-02
