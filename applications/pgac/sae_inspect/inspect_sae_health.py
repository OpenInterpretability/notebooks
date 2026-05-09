"""
Inspect SAE weights to estimate health WITHOUT running Qwen3.6-27B.

Compares L39 (worst-4 in current Phase 4 training, regressing post-bump)
vs L51 (not in worst-4, healthier baseline).

Computes proxy metrics for ve / alive% / capacity:
1. W_dec row norm distribution → "pseudo-dead" detection
2. W_enc row norm distribution → input sensitivity per feature
3. b_enc distribution → bias offset (negative means feature needs strong activation)
4. Effective rank via SVD → actual usable capacity vs nominal d_sae
5. Cross-layer comparison summary
"""

import torch
import numpy as np
from safetensors.torch import load_file
from pathlib import Path
import json

HERE = Path(__file__).parent

LAYERS_TO_INSPECT = [
    ('L39', HERE / 'sae_L39_latest.safetensors', 'WORST-4 (regressing)'),
    ('L51', HERE / 'sae_L51_latest.safetensors', 'NOT in worst-4 (healthier)'),
]

D_MODEL = 5120
N_FEATURES_EXPECTED = 40960
K_TOPK = 128


def inspect_sae(name, path, label):
    print(f'\n{"=" * 80}')
    print(f'Inspecting {name} — {label}')
    print(f'File: {path}')
    print(f'Size: {path.stat().st_size / 1e6:.1f} MB')
    print(f'{"=" * 80}')

    weights = load_file(str(path))
    print(f'\nKeys: {list(weights.keys())}')
    for k, v in weights.items():
        print(f'  {k}: shape={tuple(v.shape)}, dtype={v.dtype}')

    W_enc = weights['W_enc'].float()  # (d_in, n_features) = (5120, 40960)
    W_dec = weights['W_dec'].float()  # (n_features, d_in) = (40960, 5120)
    b_enc = weights['b_enc'].float()  # (n_features,)
    b_dec = weights['b_dec'].float()  # (d_in,)

    metrics = {'name': name, 'label': label}

    # === Metric 1: W_dec row norms (decoder direction strength) ===
    # In a TopK SAE, dead features have W_dec rows ~ random/small (no learning signal)
    # Healthy features have W_dec rows ~ unit-norm after renorm (we apply renorm_decoder)
    dec_norms = W_dec.norm(dim=-1)  # (n_features,)
    print(f'\n[W_dec row norms] (each row is decoder direction for one feature)')
    print(f'  mean: {dec_norms.mean():.4f} (expected ~1.0 after renorm)')
    print(f'  std:  {dec_norms.std():.4f}')
    print(f'  min:  {dec_norms.min():.4f}')
    print(f'  max:  {dec_norms.max():.4f}')
    print(f'  median: {dec_norms.median():.4f}')
    # Pseudo-dead: rows with norm < 0.5 (haven't been learned)
    pseudo_dead_dec = (dec_norms < 0.5).sum().item()
    near_unit = ((dec_norms > 0.95) & (dec_norms < 1.05)).sum().item()
    metrics['dec_norm_mean'] = dec_norms.mean().item()
    metrics['dec_norm_std'] = dec_norms.std().item()
    metrics['pseudo_dead_dec'] = pseudo_dead_dec
    metrics['near_unit_dec'] = near_unit
    print(f'  pseudo-dead (norm<0.5): {pseudo_dead_dec} / {N_FEATURES_EXPECTED} = {100*pseudo_dead_dec/N_FEATURES_EXPECTED:.2f}%')
    print(f'  near-unit (0.95-1.05): {near_unit} / {N_FEATURES_EXPECTED} = {100*near_unit/N_FEATURES_EXPECTED:.2f}%')

    # === Metric 2: W_enc column norms (encoder weight per feature) ===
    # Strong column norm → feature is "easy to activate"
    enc_norms = W_enc.norm(dim=0)  # (n_features,)
    print(f'\n[W_enc column norms] (each col is encoder weight for one feature)')
    print(f'  mean: {enc_norms.mean():.4f}')
    print(f'  std:  {enc_norms.std():.4f}')
    print(f'  min:  {enc_norms.min():.4f}')
    print(f'  max:  {enc_norms.max():.4f}')
    metrics['enc_norm_mean'] = enc_norms.mean().item()
    metrics['enc_norm_std'] = enc_norms.std().item()

    # === Metric 3: b_enc distribution ===
    # Negative b_enc means feature needs strong positive activation to fire
    # Very negative = feature rarely fires (proxy for dead)
    print(f'\n[b_enc distribution]')
    print(f'  mean: {b_enc.mean():.4f}')
    print(f'  std:  {b_enc.std():.4f}')
    print(f'  min:  {b_enc.min():.4f}')
    print(f'  max:  {b_enc.max():.4f}')
    print(f'  fraction < -1.0: {(b_enc < -1.0).float().mean():.4f}')
    print(f'  fraction > 0:    {(b_enc > 0).float().mean():.4f}')
    metrics['b_enc_mean'] = b_enc.mean().item()
    metrics['b_enc_min'] = b_enc.min().item()
    metrics['frac_b_enc_strongly_neg'] = (b_enc < -1.0).float().mean().item()

    # === Metric 4: Effective rank via singular values of W_dec ===
    # Effective rank: number of singular values needed to explain 99% variance
    # Lower than nominal n_features = SAE not using full capacity
    print(f'\n[Effective rank via W_dec SVD] (sample 4096 features for speed)')
    sample_idx = torch.randperm(N_FEATURES_EXPECTED)[:4096]
    W_dec_sample = W_dec[sample_idx]  # (4096, 5120)
    s = torch.linalg.svdvals(W_dec_sample)  # singular values, descending
    s_normalized = s / s.sum()
    cumsum = s_normalized.cumsum(0)
    rank_99 = (cumsum < 0.99).sum().item() + 1
    rank_95 = (cumsum < 0.95).sum().item() + 1
    rank_90 = (cumsum < 0.90).sum().item() + 1
    print(f'  Sample size: 4096 features')
    print(f'  Rank explaining 90% variance: {rank_90} (max possible: {min(4096, D_MODEL)})')
    print(f'  Rank explaining 95% variance: {rank_95}')
    print(f'  Rank explaining 99% variance: {rank_99}')
    print(f'  Top 10 singular values: {s[:10].tolist()}')
    metrics['effective_rank_99'] = rank_99
    metrics['effective_rank_95'] = rank_95

    # === Metric 5: Cosine similarity matrix (decoder feature uniqueness) ===
    # If many features have very similar W_dec, capacity is wasted
    print(f'\n[Decoder feature uniqueness] (sample 1024 features)')
    sample_idx2 = torch.randperm(N_FEATURES_EXPECTED)[:1024]
    W_dec_norm = torch.nn.functional.normalize(W_dec[sample_idx2], dim=-1)
    cos_sim = W_dec_norm @ W_dec_norm.T  # (1024, 1024)
    # Off-diagonal cosine similarities
    mask = ~torch.eye(1024, dtype=torch.bool)
    off_diag_cos = cos_sim[mask]
    print(f'  Off-diag cosine sim mean: {off_diag_cos.mean():.4f} (lower = more unique)')
    print(f'  Off-diag cosine sim std:  {off_diag_cos.std():.4f}')
    print(f'  Fraction |cos| > 0.5 (highly similar): {(off_diag_cos.abs() > 0.5).float().mean():.4f}')
    print(f'  Fraction |cos| > 0.9 (near-duplicate): {(off_diag_cos.abs() > 0.9).float().mean():.6f}')
    metrics['off_diag_cos_mean'] = off_diag_cos.mean().item()
    metrics['frac_high_cos'] = (off_diag_cos.abs() > 0.5).float().mean().item()
    metrics['frac_dup_cos'] = (off_diag_cos.abs() > 0.9).float().mean().item()

    return metrics


def main():
    all_metrics = []
    for name, path, label in LAYERS_TO_INSPECT:
        if not path.exists():
            print(f'⚠ {path} not found, skipping')
            continue
        m = inspect_sae(name, path, label)
        all_metrics.append(m)

    # === Summary comparison ===
    print(f'\n{"=" * 80}')
    print('CROSS-LAYER COMPARISON SUMMARY')
    print(f'{"=" * 80}')
    print(f'{"Metric":<35} {"L39 (worst)":<20} {"L51 (ok)":<20} {"Delta":<10}')
    print(f'{"-" * 85}')
    if len(all_metrics) >= 2:
        m1, m2 = all_metrics[0], all_metrics[1]
        for k in ['dec_norm_mean', 'dec_norm_std', 'pseudo_dead_dec', 'near_unit_dec',
                  'enc_norm_mean', 'b_enc_mean', 'b_enc_min',
                  'frac_b_enc_strongly_neg', 'effective_rank_99', 'effective_rank_95',
                  'off_diag_cos_mean', 'frac_high_cos', 'frac_dup_cos']:
            v1 = m1.get(k)
            v2 = m2.get(k)
            if isinstance(v1, float):
                print(f'{k:<35} {v1:<20.4f} {v2:<20.4f} {v1 - v2:<+10.4f}')
            else:
                print(f'{k:<35} {v1:<20} {v2:<20} {v1 - v2:<+10}')

    # Save JSON for record
    out_path = HERE / 'inspect_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f'\n✓ Saved to {out_path}')


if __name__ == '__main__':
    main()
