"""
Builder for nb_subjective_time_probe_v1.ipynb.

Subjective-Time Probe v1: does Qwen3.6-27B's residual stream encode
"how far through the thinking phase am I"?

Motivation: "Can LLMs Perceive Time?" (arXiv 2604.00010, ICLR 2026) shows
behavioral evidence that LLMs estimate their own task duration with 4-7×
error — they do NOT perceive their own time. That paper is purely diagnostic
(no internal probes, no architecture proposals). We test the question
mechanistically: is the information IN the residual stream and behaviorally
unused (epiphenomenal-time probe), or is it absent from the residual stream
entirely (confirms 2604.00010 mechanistically)?

Reuses the PSAE v1.5 cache (already on Drive at openinterp_runs/predictive_sae_v1/cache):
- 133 GSM8K prompts × 5 thinking-fractions × 3 layers (L11, L31, L55)
- That's 665 (residual, fraction) pairs per layer, scalar regression target.

Target: scalar (fraction ∈ {0.10, 0.25, 0.50, 0.75, 1.00}) — NOT sparse top-k.
Marginal-fit pathology does NOT apply (would-be marginal IS the target).

Approach: per-layer Ridge regression. Pool all 5 fractions × 133 prompts → 665 samples
in 5120-dim residual space. Train/test split 80/20 stratified by prompt
(no leakage of prompts across splits).

Three baselines (all evaluated under same metric):
- B0 — Random-feature baseline: random 5120×1 weight vector
- B1 — Shuffled-target baseline: shuffle fraction labels across samples
- B2 — Constant-mean baseline: predict mean fraction = 0.52 (R² = 0)

Metrics:
- R² on held-out prompts (primary)
- Spearman correlation (rank-based, robust to scale)
- MAE (interpretable error in "fraction units")

Decision rules:
- R² > 0.5 AND Spearman > 0.7 → time IS encoded → MI finding, opens steering
- 0.2 < R² < 0.5 → weak signal, caveat-rich paper
- R² < 0.2 → time NOT encoded in residual → confirms 2604.00010 mechanistically

Compute: ~5 min on Colab T4 (sklearn Ridge on 665×5120 is trivial).

Target: NeurIPS MI Workshop 2026 (Sep) as paper-8 candidate. Sits alongside
paper-6 (Two Forms of Epiphenomenal Probes) as methodology twin.
"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path("/Volumes/SSD Major/fish/openinterp-work/notebooks")


def code(lines, **meta):
    return {"cell_type": "code", "metadata": meta or {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}


def md(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def build():
    cells = []

    cells.append(md([
        "# Subjective-Time Probe v1 — Does Qwen3.6-27B Encode 'How Far Into Thinking I Am'?",
        "",
        "**Question**: when the model is partway through its thinking phase, does its residual",
        "stream carry information about HOW FAR ALONG it is — i.e., did this residual come from",
        "10%, 50%, or 75% of the way through the thinking span?",
        "",
        "**Why**: \"Can LLMs Perceive Time?\" (arXiv 2604.00010) showed behavioral evidence",
        "that LLMs estimate their own task duration with 4-7× error. They tested behavior, not",
        "internals. We probe the residual stream to see if the information is THERE but unused",
        "(epiphenomenal-time, paralleling paper-6 epiphenomenal-probes finding) or genuinely",
        "ABSENT (confirms behavioral finding mechanistically).",
        "",
        "**Data**: reuses PSAE v1.5 cache. 133 GSM8K prompts × 5 fractions × 3 layers (L11/L31/L55)",
        "= 665 (residual, fraction) pairs per layer. No re-running Qwen3.6-27B needed.",
        "",
        "**Target**: scalar (fraction ∈ {0.10, 0.25, 0.50, 0.75, 1.00}) — NOT sparse top-k.",
        "Marginal-fit pathology does NOT apply here.",
        "",
        "**Three baselines** (Phase 6c-class controls):",
        "- **B0 Random-feature**: 1000 random 5120-d weight vectors, take median R²",
        "- **B1 Shuffled-target**: shuffle fraction labels across samples, retrain",
        "- **B2 Constant-mean**: predict mean fraction = 0.52 → R² = 0 (definition)",
        "",
        "**Decision matrix**:",
        "| R² (per layer) | Spearman | Interpretation |",
        "|---|---|---|",
        "| > 0.5 | > 0.7 | 🟢 Time encoded → epiphenomenal-time experiments next |",
        "| 0.2–0.5 | 0.3–0.7 | 🟡 Weak signal, caveat-rich |",
        "| < 0.2 | < 0.3 | 🔴 Time NOT in residual → confirms 2604.00010 mech-level |",
        "",
        "**Compute**: ~5 min on any GPU; even CPU works (sklearn Ridge on 665×5120).",
    ]))

    cells.append(md(["## 1. Drive mount + paths"]))
    cells.append(code([
        "from pathlib import Path",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "",
        "DRIVE = Path('/content/drive/MyDrive')",
        "PSAE_CACHE = DRIVE / 'openinterp_runs' / 'predictive_sae_v1' / 'cache'",
        "OUT = DRIVE / 'openinterp_runs' / 'subjective_time_probe_v1'",
        "OUT.mkdir(parents=True, exist_ok=True)",
        "",
        "assert PSAE_CACHE.exists(), f'PSAE cache not found: {PSAE_CACHE}'",
        "print(f'PSAE cache: {sorted(p.name for p in PSAE_CACHE.iterdir())}')",
        "print(f'OUT: {OUT}')",
    ]))

    cells.append(md(["## 2. Install (sklearn + scipy — usually already in Colab)"]))
    cells.append(code([
        "import sys, subprocess",
        "def pip(*a): return subprocess.run([sys.executable, '-m', 'pip', *a], check=False)",
        "for mod in ('sklearn', 'scipy'):",
        "    try: __import__(mod)",
        "    except ImportError: pip('install', '-q', 'scikit-learn' if mod == 'sklearn' else mod)",
        "",
        "import torch, numpy as np",
        "from sklearn.linear_model import Ridge",
        "from sklearn.metrics import r2_score, mean_absolute_error",
        "from scipy.stats import spearmanr",
        "print(f'torch {torch.__version__}, np {np.__version__}')",
    ]))

    cells.append(md(["## 3. Config"]))
    cells.append(code([
        "LAYERS    = [11, 31, 55]",
        "FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]",
        "D_MODEL   = 5120",
        "TRAIN_FRAC = 0.80   # split by PROMPT, not by sample, to avoid leakage",
        "RIDGE_ALPHA = 1.0",
        "N_RANDOM_BASELINES = 100",
        "SEED = 42",
        "",
        "np.random.seed(SEED)",
        "torch.manual_seed(SEED)",
    ]))

    cells.append(md(["## 4. Load PSAE cache → build (residual, fraction, prompt_idx) tuples"]))
    cells.append(code([
        "residuals = torch.load(PSAE_CACHE / 'residuals_multilayer.pt', map_location='cpu', weights_only=False)",
        "N = residuals[LAYERS[0]][FRACTIONS[0]].shape[0]",
        "print(f'N prompts: {N}, layers: {LAYERS}, fractions: {FRACTIONS}')",
        "",
        "# Per-layer dataset construction.",
        "# For each (prompt p, fraction f) we have one residual vector. Target = f.",
        "datasets = {}",
        "for L in LAYERS:",
        "    X_list, y_list, prompt_idx_list = [], [], []",
        "    for f in FRACTIONS:",
        "        X = residuals[L][f].numpy().astype(np.float32)  # (N, D_MODEL)",
        "        X_list.append(X)",
        "        y_list.append(np.full(N, f, dtype=np.float32))",
        "        prompt_idx_list.append(np.arange(N))",
        "    X_all = np.concatenate(X_list, axis=0)         # (N * 5, D_MODEL)",
        "    y_all = np.concatenate(y_list, axis=0)         # (N * 5,)",
        "    p_all = np.concatenate(prompt_idx_list, axis=0)  # (N * 5,) prompt index",
        "    datasets[L] = (X_all, y_all, p_all)",
        "    print(f'L{L}: X {X_all.shape}, y mean={y_all.mean():.3f}, y std={y_all.std():.3f}')",
    ]))

    cells.append(md(["## 5. Train/test split by PROMPT (avoids leakage between fractions of same prompt)"]))
    cells.append(code([
        "rng = np.random.default_rng(SEED)",
        "prompt_perm = rng.permutation(N)",
        "n_train_prompts = int(N * TRAIN_FRAC)",
        "train_prompts = set(prompt_perm[:n_train_prompts].tolist())",
        "test_prompts  = set(prompt_perm[n_train_prompts:].tolist())",
        "print(f'split: {len(train_prompts)} train prompts, {len(test_prompts)} test prompts')",
        "",
        "splits = {}",
        "for L in LAYERS:",
        "    X, y, p = datasets[L]",
        "    train_mask = np.array([pi in train_prompts for pi in p])",
        "    test_mask  = np.array([pi in test_prompts  for pi in p])",
        "    splits[L] = (X[train_mask], y[train_mask], X[test_mask], y[test_mask])",
        "    print(f'L{L}: train {train_mask.sum()}, test {test_mask.sum()}')",
    ]))

    cells.append(md(["## 6. Real probe + B0 random-feature + B1 shuffled-target + B2 constant-mean"]))
    cells.append(code([
        "results = {f'L{L}': {} for L in LAYERS}",
        "",
        "for L in LAYERS:",
        "    X_tr, y_tr, X_te, y_te = splits[L]",
        "    print(f'\\n=== L{L} ===')",
        "",
        "    # --- REAL Ridge probe ---",
        "    probe = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)",
        "    probe.fit(X_tr, y_tr)",
        "    y_pred = probe.predict(X_te)",
        "    r2 = r2_score(y_te, y_pred)",
        "    sp = spearmanr(y_te, y_pred).statistic",
        "    mae = mean_absolute_error(y_te, y_pred)",
        "    print(f'[REAL]      R²={r2:.4f}  Spearman={sp:.4f}  MAE={mae:.4f}')",
        "    results[f'L{L}']['REAL'] = {'R2': float(r2), 'Spearman': float(sp), 'MAE': float(mae)}",
        "",
        "    # --- B0 random-feature: 100 random 5120-d directions, project residuals, fit scalar regression ---",
        "    r2s, sps, maes = [], [], []",
        "    for k in range(N_RANDOM_BASELINES):",
        "        w_rand = np.random.default_rng(SEED + k).standard_normal(D_MODEL).astype(np.float32)",
        "        w_rand /= np.linalg.norm(w_rand) + 1e-9  # unit norm",
        "        # 1-d projection then linear regression on the scalar",
        "        proj_tr = X_tr @ w_rand  # (N_tr,)",
        "        proj_te = X_te @ w_rand",
        "        # Closed-form linear fit on 1-d feature",
        "        a = np.cov(proj_tr, y_tr, bias=True)[0, 1] / (proj_tr.var() + 1e-9)",
        "        b = y_tr.mean() - a * proj_tr.mean()",
        "        y_pred_rand = a * proj_te + b",
        "        r2s.append(r2_score(y_te, y_pred_rand))",
        "        sps.append(spearmanr(y_te, y_pred_rand).statistic if proj_te.std() > 0 else 0.0)",
        "        maes.append(mean_absolute_error(y_te, y_pred_rand))",
        "    print(f'[B0 random] R²={np.median(r2s):.4f} (5-95%: {np.percentile(r2s, 5):.4f}–{np.percentile(r2s, 95):.4f})  '",
        "          f'Spearman={np.median(sps):.4f}  MAE={np.median(maes):.4f}')",
        "    results[f'L{L}']['B0_random'] = {",
        "        'R2_median': float(np.median(r2s)), 'R2_p5': float(np.percentile(r2s, 5)), 'R2_p95': float(np.percentile(r2s, 95)),",
        "        'Spearman_median': float(np.median(sps)), 'MAE_median': float(np.median(maes)),",
        "    }",
        "",
        "    # --- B1 shuffled-target: shuffle y_tr, retrain Ridge ---",
        "    rng_b1 = np.random.default_rng(SEED + 1)",
        "    y_tr_shuf = rng_b1.permutation(y_tr)",
        "    probe_b1 = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)",
        "    probe_b1.fit(X_tr, y_tr_shuf)",
        "    y_pred_b1 = probe_b1.predict(X_te)",
        "    r2_b1 = r2_score(y_te, y_pred_b1)",
        "    sp_b1 = spearmanr(y_te, y_pred_b1).statistic",
        "    mae_b1 = mean_absolute_error(y_te, y_pred_b1)",
        "    print(f'[B1 shuf]   R²={r2_b1:.4f}  Spearman={sp_b1:.4f}  MAE={mae_b1:.4f}')",
        "    results[f'L{L}']['B1_shuffled_target'] = {'R2': float(r2_b1), 'Spearman': float(sp_b1), 'MAE': float(mae_b1)}",
        "",
        "    # --- B2 constant-mean: predict y_tr.mean() always ---",
        "    y_pred_b2 = np.full_like(y_te, y_tr.mean())",
        "    r2_b2 = r2_score(y_te, y_pred_b2)",
        "    mae_b2 = mean_absolute_error(y_te, y_pred_b2)",
        "    print(f'[B2 const]  R²={r2_b2:.4f}  Spearman=undef    MAE={mae_b2:.4f}')",
        "    results[f'L{L}']['B2_constant_mean'] = {'R2': float(r2_b2), 'Spearman': None, 'MAE': float(mae_b2), 'pred_value': float(y_tr.mean())}",
        "",
        "import json",
        "with open(OUT / 'subjective_time_probe_v1_results.json', 'w') as f:",
        "    json.dump(results, f, indent=2)",
        "print(f'\\n✓ saved {OUT / \"subjective_time_probe_v1_results.json\"}')",
    ]))

    cells.append(md(["## 7. Side-by-side comparison + verdict per layer"]))
    cells.append(code([
        "print(f'{\"Layer\":<8}{\"REAL R²\":>10}{\"B0 R²\":>10}{\"B1 R²\":>10}{\"REAL ρ\":>10}{\"B0 ρ\":>10}{\"REAL MAE\":>12}{\"B2 MAE\":>10}')",
        "print('-' * 78)",
        "for L in LAYERS:",
        "    r = results[f'L{L}']",
        "    print(f'L{L:<7}{r[\"REAL\"][\"R2\"]:>10.4f}{r[\"B0_random\"][\"R2_median\"]:>10.4f}{r[\"B1_shuffled_target\"][\"R2\"]:>10.4f}'",
        "          f'{r[\"REAL\"][\"Spearman\"]:>10.4f}{r[\"B0_random\"][\"Spearman_median\"]:>10.4f}'",
        "          f'{r[\"REAL\"][\"MAE\"]:>12.4f}{r[\"B2_constant_mean\"][\"MAE\"]:>10.4f}')",
        "",
        "print('\\n=== PER-LAYER VERDICT ===')",
        "for L in LAYERS:",
        "    r = results[f'L{L}']",
        "    real_r2 = r['REAL']['R2']",
        "    real_sp = r['REAL']['Spearman']",
        "    b0_r2 = r['B0_random']['R2_median']",
        "",
        "    flags = []",
        "    if real_r2 > 0.5 and real_sp > 0.7:",
        "        flags.append('🟢 STRONG: time encoded')",
        "    elif real_r2 > 0.2 and real_sp > 0.3:",
        "        flags.append('🟡 WEAK: partial signal')",
        "    else:",
        "        flags.append('🔴 NULL: time absent from residual')",
        "",
        "    if real_r2 > b0_r2 + 0.1:",
        "        flags.append(f'✓ REAL beats B0 by {real_r2 - b0_r2:+.3f}')",
        "    else:",
        "        flags.append(f'✗ REAL within noise of B0 (gap {real_r2 - b0_r2:+.3f})')",
        "",
        "    print(f'L{L}: R²={real_r2:.3f}  ρ={real_sp:.3f}  B0={b0_r2:.3f}  →  {\" / \".join(flags)}')",
        "",
        "# Aggregate",
        "n_strong = sum(1 for L in LAYERS if results[f'L{L}']['REAL']['R2'] > 0.5 and results[f'L{L}']['REAL']['Spearman'] > 0.7)",
        "n_null   = sum(1 for L in LAYERS if results[f'L{L}']['REAL']['R2'] < 0.2)",
        "print(f'\\n=== AGGREGATE: {n_strong}/3 STRONG layers, {n_null}/3 NULL layers ===')",
    ]))

    cells.append(md(["## 8. Plot: predicted vs actual fraction per layer"]))
    cells.append(code([
        "import matplotlib.pyplot as plt",
        "",
        "fig, axes = plt.subplots(1, len(LAYERS), figsize=(5*len(LAYERS), 5), sharey=True)",
        "for ax, L in zip(axes, LAYERS):",
        "    X_tr, y_tr, X_te, y_te = splits[L]",
        "    probe = Ridge(alpha=RIDGE_ALPHA, random_state=SEED).fit(X_tr, y_tr)",
        "    y_pred = probe.predict(X_te)",
        "    # Slight jitter on x to avoid overplot at discrete y_te values",
        "    rng_jit = np.random.default_rng(0)",
        "    jit = rng_jit.normal(0, 0.005, size=y_te.shape)",
        "    ax.scatter(y_te + jit, y_pred, alpha=0.4, s=20, color='C0', label='REAL probe')",
        "    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, lw=1, label='ideal')",
        "    ax.axhline(y_tr.mean(), color='gray', ls=':', alpha=0.6, label=f'B2 const ({y_tr.mean():.2f})')",
        "    ax.set_xlim(0, 1.05)",
        "    ax.set_ylim(-0.2, 1.2)",
        "    r2 = results[f'L{L}']['REAL']['R2']",
        "    sp = results[f'L{L}']['REAL']['Spearman']",
        "    ax.set_title(f'L{L}  R²={r2:.3f}  ρ={sp:.3f}', fontsize=12)",
        "    ax.set_xlabel('actual fraction of thinking')",
        "    if L == LAYERS[0]:",
        "        ax.set_ylabel('predicted fraction')",
        "        ax.legend(loc='upper left', fontsize=9)",
        "    ax.grid(True, alpha=0.3)",
        "fig.suptitle('Subjective-Time Probe v1: predicted vs actual thinking-fraction\\n'",
        "             'If time IS encoded, points cluster along the dashed ideal line.', fontsize=11, y=1.02)",
        "fig.tight_layout()",
        "fig.savefig(OUT / 'subjective_time_probe_v1_scatter.png', dpi=130, bbox_inches='tight')",
        "print(f'✓ saved {OUT / \"subjective_time_probe_v1_scatter.png\"}')",
    ]))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = NOTEBOOKS_DIR / "nb_subjective_time_probe_v1.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {out_path} ({len(cells)} cells, {out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    build()
