"""
Builder for nb_predictive_sae_v15_baseline.ipynb.

PSAE v1.5 RANDOM-FEATURE BASELINES (Phase 6c gate).

Why this notebook exists:
- The 2026-05-04 PSAE v1.5 run shipped paper-grade recall numbers (L11 r@1024 ≈
  0.87, L31 ≈ 0.83, L55 ≈ 0.70 at N_test=27) but NO random-feature baseline.
- Phase 6c hard rule: any linear probe paper at N<100 (here N_test=27) MUST report
  random-feature baseline + capacity sweep before claiming signal.
- This notebook fills that gap. Reuses the existing Drive cache — does NOT re-run
  Qwen3.6-27B inference. Only re-runs the probe training step with two controls.

Two baselines (both reuse cache + same train/test split + same recall@m metric):
  B0 — SAE-init no-training
        probe.weight = W_enc.T, probe.bias = b_enc, ZERO training steps.
        Tests how much recall comes from the intrinsic SAE encoder mapping vs.
        the AdamW fine-tuning step.
  B1 — Shuffled-source training
        Same recipe (init from SAE encoder, AdamW 5 epochs, bs=256, lr=1e-3,
        top-k ranking loss) but X_train is shuffled across samples while y_train
        stays put — destroys (residual_at_frac, target_at_end) correspondence.
        Tests if the high recall is real predictive structure or N=106 noise fit.

Output: random_baseline_results.json next to predictive_sae_v15_results.json on
Drive, plus a stdout table comparing REAL vs B0 vs B1 side by side.

Compute: ~5 min on T4, ~1-2 min on A100. Trivial on any Colab GPU.

Decision rules:
  - B1 r@1024 < 0.05 (≈ chance level 1024/65536 = 1.6%)  →  signal is REAL
  - B0 r@1024 < REAL × 0.5                                →  training adds value
  - B1 r@1024 > 0.30                                       →  paper-3 reformulates
  - B0 r@1024 ≥ REAL × 0.9                                 →  signal is SAE-only,
                                                              paper-3 pivots to
                                                              "SAE feature persistence"

HARD RULES applied:
- Drive checkpoint not needed (B0 trivial, B1 ~30s/probe)
- No SAE retraining — uses existing caiovicentino1/qwen36-27b-sae-papergrade
- All hparams mirrored from original notebook (seed=42 included)
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
        "# Predictive SAE v1.5 — Random-Feature Baselines (Phase 6c gate)",
        "",
        "**Why**: PSAE v1.5 (run 2026-05-04) reported recall@1024 ≈ 0.83-0.87 on N_test=27.",
        "Phase 6c hard rule requires random-feature baseline before claiming signal at N<100.",
        "This notebook fills that gap WITHOUT re-running Qwen3.6-27B inference.",
        "",
        "**Two controls**:",
        "- **B0 SAE-init no-training**: probe weights = SAE encoder, eval directly.",
        "  Tests how much recall is intrinsic to the SAE mapping vs. AdamW training step.",
        "- **B1 Shuffled-source training**: same recipe but X_train shuffled across samples.",
        "  Tests if the signal is real predictive structure or N=106 noise fit.",
        "",
        "**Decision matrix**:",
        "| condition | interpretation |",
        "|---|---|",
        "| B1 r@1024 < 0.05 (≈chance) | signal real |",
        "| B0 r@1024 < REAL × 0.5 | training adds value |",
        "| B1 r@1024 > 0.30 | paper-3 must reformulate |",
        "| B0 r@1024 ≥ REAL × 0.9 | signal is SAE-only, reframe as 'feature persistence' |",
        "",
        "**Reads**: existing `predictive_sae_v1/cache/` on Drive (residuals+features captured 2026-05-04).",
        "**Writes**: `random_baseline_results.json` next to `predictive_sae_v15_results.json`.",
        "",
        "**Compute**: ~5 min T4 / ~1-2 min A100.",
    ]))

    # ============================================================
    # Cell 1 — Drive + paths
    # ============================================================
    cells.append(md(["## 1. Drive mount + paths"]))
    cells.append(code([
        "from pathlib import Path",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "",
        "DRIVE = Path('/content/drive/MyDrive')",
        "PSAE_DIR = DRIVE / 'openinterp_runs' / 'predictive_sae_v1'",
        "CACHE = PSAE_DIR / 'cache'",
        "OUT_JSON = PSAE_DIR / 'random_baseline_results.json'",
        "REAL_JSON = PSAE_DIR / 'predictive_sae_v15_results.json'",
        "",
        "assert CACHE.exists(), f'Cache dir not found: {CACHE}'",
        "assert REAL_JSON.exists(), f'Real results not found: {REAL_JSON}'",
        "print(f'CACHE contents: {sorted(p.name for p in CACHE.iterdir())}')",
        "print(f'REAL results: {REAL_JSON} ({REAL_JSON.stat().st_size/1024:.1f} KB)')",
    ]))

    # ============================================================
    # Cell 2 — Install minimal deps
    # ============================================================
    cells.append(md(["## 2. Install (huggingface_hub + safetensors — usually already in Colab)"]))
    cells.append(code([
        "import sys, subprocess",
        "def pip(*a): return subprocess.run([sys.executable, '-m', 'pip', *a], check=False)",
        "",
        "for mod in ('huggingface_hub', 'safetensors'):",
        "    try:",
        "        __import__(mod)",
        "    except ImportError:",
        "        pip('install', '-q', mod)",
        "",
        "import torch, numpy as np",
        "from huggingface_hub import hf_hub_download",
        "from safetensors.torch import load_file",
        "print(f'torch {torch.__version__}, cuda {torch.cuda.is_available()}, dev {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')",
    ]))

    # ============================================================
    # Cell 3 — Config (mirror original notebook EXACTLY)
    # ============================================================
    cells.append(md(["## 3. Config — mirrors original PSAE v1.5 notebook"]))
    cells.append(code([
        "SAE_REPO    = 'caiovicentino1/qwen36-27b-sae-papergrade'",
        "D_MODEL     = 5120",
        "D_SAE       = 65536",
        "K_SAE       = 128",
        "LAYERS      = [11, 31, 55]",
        "SOURCE_FRACS = [0.10, 0.25, 0.50, 0.75]",
        "TRAIN_FRAC  = 0.80",
        "N_EPOCHS    = 5",
        "BATCH_SIZE  = 256",
        "LR          = 1e-3",
        "WD          = 1e-5",
        "SEED        = 42",
        "M_VALUES    = [128, 256, 512, 1024, 2048, 4096]",
        "",
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "print(f'device={DEVICE}')",
    ]))

    # ============================================================
    # Cell 4 — Load cache
    # ============================================================
    cells.append(md(["## 4. Load cached residuals + features"]))
    cells.append(code([
        "residuals = torch.load(CACHE / 'residuals_multilayer.pt', map_location='cpu', weights_only=False)",
        "features = torch.load(CACHE / 'features_multilayer.pt', map_location='cpu', weights_only=False)",
        "",
        "N = residuals[LAYERS[0]][SOURCE_FRACS[0]].shape[0]",
        "print(f'N total prompts: {N}')",
        "print(f'residual sample shape: {residuals[11][0.10].shape}, dtype {residuals[11][0.10].dtype}')",
        "print(f'feature sample keys: {list(features[11][1.00].keys())}, indices shape {features[11][1.00][\"indices\"].shape}')",
    ]))

    # ============================================================
    # Cell 5 — Download SAE encoders
    # ============================================================
    cells.append(md(["## 5. Download SAE encoder weights (3 files, ~4.5GB total)"]))
    cells.append(code([
        "encoders = {}",
        "for L in LAYERS:",
        "    print(f'downloading SAE L{L}...')",
        "    path = hf_hub_download(SAE_REPO, f'sae_L{L}_latest.safetensors')",
        "    w = load_file(path)",
        "    encoders[L] = {",
        "        'W_enc': w['W_enc'].to(torch.float32),  # (D_MODEL, D_SAE)",
        "        'b_enc': w['b_enc'].to(torch.float32),  # (D_SAE,)",
        "    }",
        "    print(f'  L{L} W_enc {tuple(encoders[L][\"W_enc\"].shape)}, b_enc {tuple(encoders[L][\"b_enc\"].shape)}')",
        "",
        "for L in LAYERS:",
        "    assert encoders[L]['W_enc'].shape == (D_MODEL, D_SAE)",
        "    assert encoders[L]['b_enc'].shape == (D_SAE,)",
        "print('encoders loaded ✓')",
    ]))

    # ============================================================
    # Cell 6 — Probe + loss (mirrors notebook)
    # ============================================================
    cells.append(md(["## 6. Probe construction + top-k ranking loss (verbatim from PSAE v1.5)"]))
    cells.append(code([
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "from torch.optim.lr_scheduler import CosineAnnealingLR",
        "",
        "def make_multihot(top_i, n_features=D_SAE):",
        "    mh = torch.zeros(top_i.shape[0], n_features, dtype=torch.bool)",
        "    mh.scatter_(-1, top_i, True)",
        "    return mh",
        "",
        "def topk_ranking_loss(logits, target_active, k=K_SAE):",
        "    pos_mask = target_active.bool()",
        "    pos_logits = (logits * pos_mask.float()).sum(-1) / k",
        "    neg_logits = (logits * (~pos_mask).float()).sum(-1) / (logits.shape[-1] - k)",
        "    return F.softplus(-(pos_logits - neg_logits)).mean()",
        "",
        "def make_probe(W_enc, b_enc):",
        "    probe = nn.Linear(D_MODEL, D_SAE).to(DEVICE, torch.float32)",
        "    probe.weight.data = W_enc.T.contiguous().to(DEVICE)",
        "    probe.bias.data = b_enc.clone().to(DEVICE)",
        "    return probe",
        "",
        "def train_probe(probe, X_train, y_train, n_epochs=N_EPOCHS):",
        "    opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WD)",
        "    sched = CosineAnnealingLR(opt, T_max=n_epochs)",
        "    X = X_train.to(DEVICE); y = y_train.to(DEVICE)",
        "    probe.train()",
        "    for epoch in range(n_epochs):",
        "        idx = torch.randperm(X.shape[0], device=DEVICE)",
        "        for i in range(0, X.shape[0], BATCH_SIZE):",
        "            b = idx[i:i+BATCH_SIZE]",
        "            logits = probe(X[b])",
        "            loss = topk_ranking_loss(logits, y[b])",
        "            opt.zero_grad(); loss.backward(); opt.step()",
        "        sched.step()",
        "    probe.eval()",
        "    return probe",
        "",
        "@torch.no_grad()",
        "def eval_probe(probe, X_test, y_test_multi, m_values=M_VALUES):",
        "    probe.eval()",
        "    X = X_test.to(DEVICE); y = y_test_multi.to(DEVICE).bool()",
        "    logits = probe(X)",
        "    out = {}",
        "    for m in m_values:",
        "        _, top_m = logits.topk(m, dim=-1)",
        "        pred = torch.zeros_like(logits, dtype=torch.bool)",
        "        pred.scatter_(-1, top_m, True)",
        "        recall = (pred & y).sum(-1).float() / K_SAE",
        "        out[m] = (float(recall.mean().item()), float(recall.std().item()))",
        "    return out",
    ]))

    # ============================================================
    # Cell 7 — Run both baselines
    # ============================================================
    cells.append(md(["## 7. Run B0 (no-train) + B1 (shuffled) for all (L, frac)"]))
    cells.append(code([
        "import time",
        "",
        "target_multihots = {L: make_multihot(features[L][1.00]['indices']) for L in LAYERS}",
        "",
        "perm = torch.randperm(N, generator=torch.Generator().manual_seed(SEED))",
        "n_train = int(N * TRAIN_FRAC)",
        "train_idx, test_idx = perm[:n_train], perm[n_train:]",
        "print(f'split: train={len(train_idx)}, test={len(test_idx)}')",
        "",
        "results = {",
        "    '_meta': {",
        "        'notebook': 'nb_predictive_sae_v15_baseline.ipynb',",
        "        'device': str(DEVICE),",
        "        'n_train': int(n_train), 'n_test': int(N - n_train),",
        "        'seed': SEED, 'd_sae': D_SAE, 'k_sae': K_SAE, 'n_epochs': N_EPOCHS,",
        "        'm_values': M_VALUES,",
        "        'baselines': {",
        "            'B0': 'SAE-init no-training (probe = SAE encoder, zero training steps)',",
        "            'B1': 'Shuffled-source training (X_train shuffled, y_train kept)',",
        "        },",
        "    },",
        "    'B0_sae_init_no_train': {f'L{L}': {} for L in LAYERS},",
        "    'B1_shuffled_source':   {f'L{L}': {} for L in LAYERS},",
        "}",
        "",
        "shuf_gen = torch.Generator().manual_seed(SEED + 1)",
        "",
        "for L in LAYERS:",
        "    W_enc = encoders[L]['W_enc']",
        "    b_enc = encoders[L]['b_enc']",
        "    y_full = target_multihots[L]",
        "    for frac in SOURCE_FRACS:",
        "        tag = f'L{L} f={int(frac*100):02d}'",
        "        X = residuals[L][frac].to(torch.float32)",
        "        X_train = X[train_idx]; X_test = X[test_idx]",
        "        y_train = y_full[train_idx]; y_test = y_full[test_idx]",
        "",
        "        # B0 — SAE-init no-training",
        "        t = time.time()",
        "        probe_b0 = make_probe(W_enc, b_enc)",
        "        m_b0 = eval_probe(probe_b0, X_test, y_test)",
        "        print(f'[B0] {tag} r@128={m_b0[128][0]:.3f} r@1024={m_b0[1024][0]:.3f} r@4096={m_b0[4096][0]:.3f}  ({time.time()-t:.1f}s)')",
        "        results['B0_sae_init_no_train'][f'L{L}'][f'{frac:.2f}'] = {str(m): list(m_b0[m]) for m in M_VALUES}",
        "        del probe_b0",
        "",
        "        # B1 — shuffled-source training",
        "        t = time.time()",
        "        shuffle_idx = torch.randperm(X_train.shape[0], generator=shuf_gen)",
        "        X_train_sh = X_train[shuffle_idx]",
        "        probe_b1 = make_probe(W_enc, b_enc)",
        "        probe_b1 = train_probe(probe_b1, X_train_sh, y_train)",
        "        m_b1 = eval_probe(probe_b1, X_test, y_test)",
        "        print(f'[B1] {tag} r@128={m_b1[128][0]:.3f} r@1024={m_b1[1024][0]:.3f} r@4096={m_b1[4096][0]:.3f}  ({time.time()-t:.1f}s)')",
        "        results['B1_shuffled_source'][f'L{L}'][f'{frac:.2f}'] = {str(m): list(m_b1[m]) for m in M_VALUES}",
        "        del probe_b1, X_train_sh",
        "        if torch.cuda.is_available(): torch.cuda.empty_cache()",
        "",
        "import json",
        "with open(OUT_JSON, 'w') as f:",
        "    json.dump(results, f, indent=2)",
        "print(f'\\n✓ saved {OUT_JSON}')",
    ]))

    # ============================================================
    # Cell 8 — Side-by-side comparison table
    # ============================================================
    cells.append(md(["## 8. Comparison: REAL vs B0 vs B1"]))
    cells.append(code([
        "import json",
        "real = json.load(open(REAL_JSON))",
        "",
        "print(f'{\"tag\":<12}{\"REAL@1024\":>11}{\"B0@1024\":>11}{\"B1@1024\":>11}    {\"REAL@4096\":>11}{\"B0@4096\":>11}{\"B1@4096\":>11}')",
        "print('-' * 88)",
        "for L in LAYERS:",
        "    for frac in SOURCE_FRACS:",
        "        tag = f'L{L} f={int(frac*100):02d}'",
        "        r1024 = real['recall'][f'L{L}'][str(frac)]['1024'][0]",
        "        r4096 = real['recall'][f'L{L}'][str(frac)]['4096'][0]",
        "        b0_1024 = results['B0_sae_init_no_train'][f'L{L}'][f'{frac:.2f}']['1024'][0]",
        "        b0_4096 = results['B0_sae_init_no_train'][f'L{L}'][f'{frac:.2f}']['4096'][0]",
        "        b1_1024 = results['B1_shuffled_source'][f'L{L}'][f'{frac:.2f}']['1024'][0]",
        "        b1_4096 = results['B1_shuffled_source'][f'L{L}'][f'{frac:.2f}']['4096'][0]",
        "        print(f'{tag:<12}{r1024:>11.3f}{b0_1024:>11.3f}{b1_1024:>11.3f}    {r4096:>11.3f}{b0_4096:>11.3f}{b1_4096:>11.3f}')",
    ]))

    # ============================================================
    # Cell 9 — Decision verdict (auto-flag)
    # ============================================================
    cells.append(md(["## 9. Decision verdict"]))
    cells.append(code([
        "verdict_lines = []",
        "for L in LAYERS:",
        "    for frac in SOURCE_FRACS:",
        "        tag = f'L{L} f={int(frac*100):02d}'",
        "        r = real['recall'][f'L{L}'][str(frac)]['1024'][0]",
        "        b0 = results['B0_sae_init_no_train'][f'L{L}'][f'{frac:.2f}']['1024'][0]",
        "        b1 = results['B1_shuffled_source'][f'L{L}'][f'{frac:.2f}']['1024'][0]",
        "",
        "        flags = []",
        "        if b1 < 0.05: flags.append('🟢 signal-real')",
        "        elif b1 > 0.30: flags.append('🔴 N-artifact-risk')",
        "        else: flags.append('🟡 partial-signal')",
        "",
        "        if b0 >= r * 0.9: flags.append('🟡 SAE-only')",
        "        elif b0 < r * 0.5: flags.append('🟢 training-adds-value')",
        "        else: flags.append('⚪ training-marginal')",
        "",
        "        verdict_lines.append(f'{tag}: REAL={r:.3f} B0={b0:.3f} B1={b1:.3f}  {\" \".join(flags)}')",
        "",
        "print('\\n=== PER-SITE VERDICT (recall@1024) ===')",
        "for line in verdict_lines: print(line)",
        "",
        "# Aggregate",
        "n_real = sum(1 for L in LAYERS for f in SOURCE_FRACS",
        "             if results['B1_shuffled_source'][f'L{L}'][f'{f:.2f}']['1024'][0] < 0.05)",
        "n_total = len(LAYERS) * len(SOURCE_FRACS)",
        "print(f'\\n=== AGGREGATE: {n_real}/{n_total} sites pass B1<0.05 signal-real bar ===')",
        "",
        "n_train_adds = sum(1 for L in LAYERS for f in SOURCE_FRACS",
        "                   if results['B0_sae_init_no_train'][f'L{L}'][f'{f:.2f}']['1024'][0] <",
        "                      real['recall'][f'L{L}'][str(f)]['1024'][0] * 0.5)",
        "print(f'=== AGGREGATE: {n_train_adds}/{n_total} sites pass B0<REAL×0.5 training-adds-value bar ===')",
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

    out_path = NOTEBOOKS_DIR / "nb_predictive_sae_v15_baseline.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {out_path} ({len(cells)} cells, {out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    build()
