"""
PGAC Phase 1 — simulation of theoretical bound.

Numerical realization of the math in 01_theoretical_bound.md, using
realistic Qwen3.6-27B parameters + published TopK SAE statistics.

Outputs:
- Speedup curves as function of (k_pred, sparsity_ratio, exit_layer)
- Quality preservation bound vs probe accuracy
- Compound speedup with quantization
- Saved figure: pgac_speedup_curves.png

No GPU needed — pure numerical simulation. Real activation statistics
on Qwen3.6-27B SAE would be Phase 2 (when Colab fires up).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('/Volumes/SSD Major/fish/openinterp-work/applications/pgac')

# ============================================================================
# Qwen3.6-27B parameters
# ============================================================================
d_model = 4096
n_layers = 64
expansion = 16
d_sae = expansion * d_model      # 65536
seq_len_typical = 2048

# ============================================================================
# Compute counts (per-token, per-layer)
# ============================================================================
def standard_layer_compute(d):
    """Standard transformer layer compute per token."""
    c_attn = 4 * d**2 + 2 * d * seq_len_typical  # qkv + softmax + proj + bias
    c_ffn = 8 * d**2                              # up + activation + down (4d intermediate)
    return c_attn + c_ffn

def pgac_layer_compute(d, k_pred, attention_speedup=1.0):
    """PGAC layer compute: same attention, sparse FFN via SAE basis."""
    c_attn = (4 * d**2 + 2 * d * seq_len_typical) / attention_speedup
    c_ffn_pgac = 2 * d * k_pred                   # encode top-k + decode top-k
    return c_attn + c_ffn_pgac

# ============================================================================
# Speedup curves
# ============================================================================
k_range = np.array([16, 32, 64, 128, 256, 512])

# (1) FFN-only speedup vs k_pred
ffn_speedup = (8 * d_model**2) / (2 * d_model * k_range)

# (2) Layer-level speedup (attention bottleneck) vs k_pred
layer_speedup = standard_layer_compute(d_model) / np.array([
    pgac_layer_compute(d_model, k) for k in k_range
])

# (3) Combined with early exit: speedup factor as fraction of layers used
exit_fractions = np.linspace(0.3, 1.0, 50)  # 30% of layers used (very aggressive) to 100%
exit_speedups = 1.0 / exit_fractions

# (4) Compound: layer_speedup × exit_speedup
k_pred_chosen = 64
layer_speedup_at_64 = standard_layer_compute(d_model) / pgac_layer_compute(d_model, k_pred_chosen)
combined_speedup = layer_speedup_at_64 * exit_speedups

# (5) With aggressive INT4 quantization on secondary features
def quant_speedup(quant_fraction, quant_factor=2.0):
    """Speedup from quantizing 'quant_fraction' of features more aggressively."""
    return 1.0 / ((1 - quant_fraction) + quant_fraction / quant_factor)

quant_fractions = np.linspace(0, 0.7, 50)
quant_speedups = quant_speedup(quant_fractions, quant_factor=2.0)

# ============================================================================
# Quality preservation bound (REVISED 2026-05-02 — see 01_theoretical_bound.md)
# Two error sources:
#   1. SAE reconstruction floor: ~5-10pp task quality loss at VE=0.84
#   2. Probe imperfect recall: (1-r) × 10pp marginal cost
# Total: Y_total = SAE_floor + (1-r) × β
# ============================================================================
SAE_FLOOR_PP = 7.5      # midpoint of 5-10pp range
BETA_PP = 10.0          # marginal cost per unit (1-recall)
ACCEPTABLE_TOTAL_PP = 8.0  # acceptable task quality loss vs full model

# Required recall to keep total loss ≤ ACCEPTABLE
required_recall = 1 - (ACCEPTABLE_TOTAL_PP - SAE_FLOOR_PP) / BETA_PP
required_recall = max(0.0, min(1.0, required_recall))  # clamp to [0,1]

# AUROC ↔ recall@k mapping (empirical, ranker-typical)
auroc_to_recall = {
    0.80: 0.55,
    0.85: 0.70,
    0.90: 0.85,
    0.95: 0.95,
    1.00: 1.00,
}

# Required AUROC: linear interpolation
auroc_arr = np.array(list(auroc_to_recall.keys()))
recall_arr = np.array(list(auroc_to_recall.values()))
required_auroc = np.interp(required_recall, recall_arr, auroc_arr)

# Quality loss curve: f(recall) = SAE_floor + (1-recall)*BETA
recall_range = np.linspace(0, 1, 100)
quality_loss_pp = SAE_FLOOR_PP + (1 - recall_range) * BETA_PP

# ============================================================================
# Print headline numbers
# ============================================================================
print('=== PGAC theoretical bound — Qwen3.6-27B parameters ===')
print(f'  d_model = {d_model}')
print(f'  n_layers = {n_layers}')
print(f'  d_sae = {d_sae} (expansion {expansion}x)')
print(f'  seq_len = {seq_len_typical}')
print()

print(f'=== Speedup at k_pred = {k_pred_chosen} ===')
print(f'  FFN-only:       {ffn_speedup[k_range == k_pred_chosen][0]:.1f}x')
print(f'  Layer-level:    {layer_speedup[k_range == k_pred_chosen][0]:.2f}x  (capped by attention)')
print()

print(f'=== Compound speedup (combined factors) ===')
exit_realistic = 0.6  # avg exit at 60% of layers
quant_realistic = 0.3  # 30% of features INT4
total = (layer_speedup[k_range == k_pred_chosen][0]
         * (1 / exit_realistic)
         * quant_speedup(quant_realistic, quant_factor=2.0))
print(f'  Layer × Exit × Quant = {layer_speedup[k_range == k_pred_chosen][0]:.2f} × {1/exit_realistic:.2f} × {quant_speedup(quant_realistic, 2.0):.2f}')
print(f'  Total: {total:.2f}x at iso-quality (theoretical upper bound)')
print()

print(f'=== Quality preservation requirement (REVISED) ===')
print(f'  SAE reconstruction floor (task quality loss): ~{SAE_FLOOR_PP:.1f} pp')
print(f'  Marginal cost per (1-recall) unit: ~{BETA_PP:.1f} pp')
print(f'  Acceptable total quality loss: ≤ {ACCEPTABLE_TOTAL_PP:.1f} pp')
print(f'  → Required probe recall@k: ≥ {required_recall:.2%}')
print(f'  → Required probe AUROC: ≥ {required_auroc:.3f}')
print(f'  Achievable per nb37/nb41 v2 fresh-probe results (typical 0.85-0.95 on similar tasks)')
print()

# ============================================================================
# Plot
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: FFN speedup vs k_pred
ax = axes[0, 0]
ax.plot(k_range, ffn_speedup, 'o-', linewidth=2.5, color='#10b981', markersize=8, label='FFN compute')
ax.plot(k_range, layer_speedup, 'o-', linewidth=2.5, color='#3b82f6', markersize=8, label='Layer-level')
ax.axhline(3.0, color='red', linestyle='--', alpha=0.5, label='Attention cap (3x)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('k_pred (top-k features predicted)', fontsize=11)
ax.set_ylabel('Speedup factor', fontsize=11)
ax.set_title('FFN-only vs Layer-level speedup', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, which='both')
ax.set_xticks(k_range)
ax.set_xticklabels(k_range)

# Panel 2: Compound with early exit
ax = axes[0, 1]
ax.plot(exit_fractions, combined_speedup, '-', linewidth=2.5, color='#10b981',
        label=f'k_pred={k_pred_chosen}')
ax.fill_between(exit_fractions, layer_speedup_at_64,
                combined_speedup, alpha=0.2, color='#10b981',
                label='Early-exit gain')
ax.axhline(layer_speedup_at_64, color='#3b82f6', linestyle='--', alpha=0.7,
           label=f'Layer-only ({layer_speedup_at_64:.2f}x)')
ax.axvline(0.6, color='gray', linestyle=':', alpha=0.5, label='Realistic exit (60%)')
ax.set_xlabel('Average exit-layer fraction', fontsize=11)
ax.set_ylabel('Combined speedup', fontsize=11)
ax.set_title('Layer + early-exit compound', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0.3, 1.0)

# Panel 3: Quality preservation — task quality loss vs probe recall
ax = axes[1, 0]
ax.plot(recall_range, quality_loss_pp, '-', linewidth=2.5, color='#f59e0b',
        label=f'Total task loss = SAE floor ({SAE_FLOOR_PP}pp) + (1-r)×{BETA_PP}pp')
ax.axhline(SAE_FLOOR_PP, color='#3b82f6', linestyle=':', alpha=0.7,
           label=f'SAE recon floor ({SAE_FLOOR_PP}pp, r=1.0)')
ax.axhline(ACCEPTABLE_TOTAL_PP, color='#ef4444', linestyle='--', linewidth=2,
           label=f'Acceptable threshold ({ACCEPTABLE_TOTAL_PP}pp)')
ax.fill_between(recall_range, 0, quality_loss_pp,
                where=quality_loss_pp <= ACCEPTABLE_TOTAL_PP,
                color='#10b981', alpha=0.2, label='Iso-quality region')
ax.axvline(required_recall, color='gray', linestyle=':', alpha=0.5)
ax.text(required_recall - 0.02, ACCEPTABLE_TOTAL_PP + 1,
        f'Required recall = {required_recall:.2%}\n→ AUROC ≥ {required_auroc:.2f}',
        rotation=0, fontsize=9, va='bottom', ha='right')
ax.set_xlabel('Probe recall@k on top-k feature presence', fontsize=11)
ax.set_ylabel('Total task quality loss (pp vs full model)', fontsize=11)
ax.set_title('Quality preservation: probe recall required', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 20)

# Panel 4: Compound with quantization
ax = axes[1, 1]
ax.plot(quant_fractions * 100, quant_speedups, '-', linewidth=2.5, color='#a855f7',
        label='Quant alone')
total_compound = (layer_speedup_at_64 * (1 / exit_realistic)) * quant_speedups
ax.plot(quant_fractions * 100, total_compound, '-', linewidth=3.5, color='#10b981',
        label='PGAC + early-exit + quant')
ax.axhline(layer_speedup_at_64 * (1 / exit_realistic), color='#3b82f6', linestyle='--',
           label=f'PGAC + exit alone ({layer_speedup_at_64 * (1 / exit_realistic):.2f}x)')
ax.set_xlabel('% of features quantized to INT4', fontsize=11)
ax.set_ylabel('Total speedup at iso-quality', fontsize=11)
ax.set_title('Quantization compound: full PGAC stack', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.axvline(30, color='gray', linestyle=':', alpha=0.5, label='Realistic 30% INT4')

plt.tight_layout()
plt.savefig(OUT / 'pgac_speedup_curves.png', dpi=170, bbox_inches='tight')
print(f'✓ Saved: {OUT / "pgac_speedup_curves.png"}')

# ============================================================================
# Sanity checks
# ============================================================================
print()
print('=== Sanity checks ===')
print(f'  Standard 64-layer Qwen3.6-27B per-token forward: {n_layers * standard_layer_compute(d_model) / 1e9:.2f} GFLOPs')
print(f'  PGAC 64-layer (k=64, attention-capped): {n_layers * pgac_layer_compute(d_model, 64) / 1e9:.2f} GFLOPs')
print(f'  PGAC w/ exit at 60% layers: {0.6 * n_layers * pgac_layer_compute(d_model, 64) / 1e9:.2f} GFLOPs')
print(f'  Compound speedup vs standard: {(n_layers * standard_layer_compute(d_model)) / (0.6 * n_layers * pgac_layer_compute(d_model, 64)):.2f}x')
print()

# ============================================================================
# Summary table for paper
# ============================================================================
print('=== Summary table for paper ===')
print(f'{"Configuration":<40} {"Speedup":>10}')
print('-' * 52)
print(f'{"FFN-only (k=64)":<40} {ffn_speedup[k_range == 64][0]:>9.1f}x')
print(f'{"Layer-level (k=64, attn cap)":<40} {layer_speedup[k_range == 64][0]:>9.2f}x')
print(f'{"+ Early exit (avg layer 60%)":<40} {layer_speedup_at_64 * (1/exit_realistic):>9.2f}x')
print(f'{"+ INT4 quant on 30% features":<40} {layer_speedup_at_64 * (1/exit_realistic) * quant_speedup(0.3, 2.0):>9.2f}x')
print(f'{"+ Sparse attention (3x attn)":<40} {(12*d_model**2 / (4*d_model**2/3 + 2*d_model*64)) * (1/exit_realistic) * quant_speedup(0.3, 2.0):>9.2f}x')

print('\n✓ PGAC Phase 1 simulation complete')
