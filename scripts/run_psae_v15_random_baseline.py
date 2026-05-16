#!/usr/bin/env python3
"""
PSAE v1.5 — random-feature baselines (Phase 6c compliance).

Two baselines computed against the same train/test split + same recall@m metric:

  B0 (SAE-init no training):
      Use the SAE encoder weights as the probe (zero training steps).
      Tests how much of the recall comes from the intrinsic same-layer mapping.

  B1 (shuffled-source training):
      Same recipe as original (init from SAE encoder, AdamW, top-k ranking loss,
      5 epochs, bs=256, lr=1e-3) but X_train is shuffled across samples while
      y_train stays put — destroys the (residual_at_frac, target_at_end) correspondence.
      Tests if the high recall is from real predictive structure or from
      probe-on-N=106 fitting noise.

Outputs JSON next to predictive_sae_v15_results.json on Drive, plus a stdout
side-by-side table for fast visual comparison.

Compute: ~30-50 min on Mac CPU, no GPU.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------- Config (mirrors original notebook) ----------
DRIVE = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive")
PSAE_DIR = DRIVE / "openinterp_runs" / "predictive_sae_v1"
CACHE = PSAE_DIR / "cache"

SAE_REPO    = "caiovicentino1/qwen36-27b-sae-papergrade"
D_MODEL     = 5120
D_SAE       = 65536
K_SAE       = 128
LAYERS      = [11, 31, 55]
SOURCE_FRACS = [0.10, 0.25, 0.50, 0.75]
TRAIN_FRAC  = 0.80
N_EPOCHS    = 5
BATCH_SIZE  = 256
LR          = 1e-3
WD          = 1e-5
SEED        = 42
M_VALUES    = [128, 256, 512, 1024, 2048, 4096]

DEVICE = torch.device("cpu")  # explicit — this script is the CPU-local baseline

OUT_JSON = PSAE_DIR / "random_baseline_results.json"


# ---------- Loaders ----------
def load_cache():
    print("[load] residuals + features + traces from Drive cache")
    residuals = torch.load(CACHE / "residuals_multilayer.pt", map_location="cpu", weights_only=False)
    features = torch.load(CACHE / "features_multilayer.pt", map_location="cpu", weights_only=False)
    return residuals, features


def load_sae_encoders():
    """Return {L: (W_enc Tensor[D_MODEL, D_SAE], b_enc Tensor[D_SAE])}."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    encoders = {}
    for L in LAYERS:
        print(f"[load] SAE L{L} from {SAE_REPO}")
        path = hf_hub_download(SAE_REPO, f"sae_L{L}_latest.safetensors")
        weights = load_file(path)
        # SAE encoder convention: W_enc is (D_MODEL, D_SAE), b_enc is (D_SAE,)
        # Probe is nn.Linear(D_MODEL, D_SAE) → probe.weight is (D_SAE, D_MODEL) = W_enc.T
        W_enc = weights["W_enc"].to(torch.float32)  # (D_MODEL, D_SAE)
        b_enc = weights["b_enc"].to(torch.float32)  # (D_SAE,)
        assert W_enc.shape == (D_MODEL, D_SAE), f"L{L} W_enc shape {W_enc.shape}"
        assert b_enc.shape == (D_SAE,), f"L{L} b_enc shape {b_enc.shape}"
        encoders[L] = (W_enc, b_enc)
    return encoders


# ---------- Probe + loss (mirrors notebook) ----------
def make_multihot(top_i, n_features=D_SAE):
    mh = torch.zeros(top_i.shape[0], n_features, dtype=torch.bool)
    mh.scatter_(-1, top_i, True)
    return mh


def topk_ranking_loss(logits, target_active, k=K_SAE):
    pos_mask = target_active.bool()
    pos_logits = (logits * pos_mask.float()).sum(-1) / k
    neg_logits = (logits * (~pos_mask).float()).sum(-1) / (logits.shape[-1] - k)
    return F.softplus(-(pos_logits - neg_logits)).mean()


def make_probe(W_enc, b_enc):
    probe = nn.Linear(D_MODEL, D_SAE).to(DEVICE, torch.float32)
    # match notebook init: probe.weight = W_enc.T (so W_enc transposed → (D_SAE, D_MODEL))
    probe.weight.data = W_enc.T.contiguous()
    probe.bias.data = b_enc.clone()
    return probe


def train_probe(probe, X_train, y_train, n_epochs=N_EPOCHS):
    opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WD)
    sched = CosineAnnealingLR(opt, T_max=n_epochs)
    probe.train()
    for epoch in range(n_epochs):
        idx = torch.randperm(X_train.shape[0])
        ep_loss = 0.0
        nb = 0
        for i in range(0, X_train.shape[0], BATCH_SIZE):
            b = idx[i:i + BATCH_SIZE]
            logits = probe(X_train[b])
            loss = topk_ranking_loss(logits, y_train[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        sched.step()
    probe.eval()
    return probe


@torch.no_grad()
def eval_probe(probe, X_test, y_test_multi, m_values=M_VALUES):
    probe.eval()
    logits = probe(X_test)
    out = {}
    for m in m_values:
        _, top_m = logits.topk(m, dim=-1)
        pred = torch.zeros_like(logits, dtype=torch.bool)
        pred.scatter_(-1, top_m, True)
        recall = (pred & y_test_multi.bool()).sum(-1).float() / K_SAE
        out[m] = (float(recall.mean().item()), float(recall.std().item()))
    return out


# ---------- Main ----------
def main():
    t0 = time.time()
    residuals, features = load_cache()

    # Confirm structure
    sample = residuals[LAYERS[0]][SOURCE_FRACS[0]]
    N = sample.shape[0]
    print(f"[info] N={N}, d_model={sample.shape[1]}")

    encoders = load_sae_encoders()

    # Build target multihots (same as notebook)
    target_multihots = {}
    for L in LAYERS:
        feat_end = features[L][1.00]
        indices = feat_end["indices"]
        target_multihots[L] = make_multihot(indices)
        print(f"[info] L{L} target multihot shape {target_multihots[L].shape}")

    # Deterministic train/test split — match notebook seed
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(SEED))
    n_train = int(N * TRAIN_FRAC)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    print(f"[info] split: train={len(train_idx)}, test={len(test_idx)}")

    results = {
        "_meta": {
            "script": "run_psae_v15_random_baseline.py",
            "device": "cpu",
            "n_train": int(n_train),
            "n_test": int(N - n_train),
            "seed": SEED,
            "d_sae": D_SAE,
            "k_sae": K_SAE,
            "n_epochs": N_EPOCHS,
            "m_values": M_VALUES,
            "baselines": {
                "B0": "SAE-init, no training (probe = SAE encoder, eval directly)",
                "B1": "Shuffled-source training (X_train shuffled, y_train kept; same recipe)",
            },
        },
        "B0_sae_init_no_train": {f"L{L}": {} for L in LAYERS},
        "B1_shuffled_source":  {f"L{L}": {} for L in LAYERS},
    }

    # Shuffle generator (separate from split seed to avoid leakage)
    shuf_gen = torch.Generator().manual_seed(SEED + 1)

    for L in LAYERS:
        W_enc, b_enc = encoders[L]
        y_full = target_multihots[L]
        for frac in SOURCE_FRACS:
            tag = f"L{L} frac={frac:.2f}"
            X = residuals[L][frac].to(torch.float32)
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y_full[train_idx]
            y_test = y_full[test_idx]

            # ---------- B0: SAE-init no training ----------
            t = time.time()
            probe_b0 = make_probe(W_enc, b_enc)
            m_b0 = eval_probe(probe_b0, X_test, y_test)
            dt = time.time() - t
            print(f"[B0] {tag} no_train  r@128={m_b0[128][0]:.3f} r@1024={m_b0[1024][0]:.3f} r@4096={m_b0[4096][0]:.3f}  ({dt:.1f}s)")
            results["B0_sae_init_no_train"][f"L{L}"][f"{frac:.2f}"] = {
                str(m): list(m_b0[m]) for m in M_VALUES
            }

            # ---------- B1: shuffled-source training ----------
            t = time.time()
            shuffle_idx = torch.randperm(X_train.shape[0], generator=shuf_gen)
            X_train_shuffled = X_train[shuffle_idx]  # break (residual, target) correspondence
            probe_b1 = make_probe(W_enc, b_enc)  # same init as real probe
            probe_b1 = train_probe(probe_b1, X_train_shuffled, y_train)
            m_b1 = eval_probe(probe_b1, X_test, y_test)
            dt = time.time() - t
            print(f"[B1] {tag} shuffled  r@128={m_b1[128][0]:.3f} r@1024={m_b1[1024][0]:.3f} r@4096={m_b1[4096][0]:.3f}  ({dt:.1f}s)")
            results["B1_shuffled_source"][f"L{L}"][f"{frac:.2f}"] = {
                str(m): list(m_b1[m]) for m in M_VALUES
            }

            # free
            del probe_b0, probe_b1, X_train_shuffled
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---------- Persist ----------
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] {OUT_JSON}")

    # ---------- Side-by-side summary table ----------
    print("\n=== Comparison: REAL vs B0 (no-train) vs B1 (shuffled) ===")
    real_path = PSAE_DIR / "predictive_sae_v15_results.json"
    real = json.load(open(real_path)) if real_path.exists() else None
    print(f"{'tag':<14}{'real@1024':>11}{'B0@1024':>11}{'B1@1024':>11}{'real@4096':>11}{'B0@4096':>11}{'B1@4096':>11}")
    for L in LAYERS:
        for frac in SOURCE_FRACS:
            tag = f"L{L} f={int(frac*100):02d}"
            real_str = "—"
            b0_str = f"{results['B0_sae_init_no_train'][f'L{L}'][f'{frac:.2f}']['1024'][0]:.3f}"
            b1_str = f"{results['B1_shuffled_source'][f'L{L}'][f'{frac:.2f}']['1024'][0]:.3f}"
            real_4096_str = "—"
            b0_4096_str = f"{results['B0_sae_init_no_train'][f'L{L}'][f'{frac:.2f}']['4096'][0]:.3f}"
            b1_4096_str = f"{results['B1_shuffled_source'][f'L{L}'][f'{frac:.2f}']['4096'][0]:.3f}"
            if real is not None:
                rcell_1024 = real["recall"][f"L{L}"][str(frac)]["1024"][0]
                rcell_4096 = real["recall"][f"L{L}"][str(frac)]["4096"][0]
                real_str = f"{rcell_1024:.3f}"
                real_4096_str = f"{rcell_4096:.3f}"
            print(f"{tag:<14}{real_str:>11}{b0_str:>11}{b1_str:>11}{real_4096_str:>11}{b0_4096_str:>11}{b1_4096_str:>11}")

    print(f"\n[done] total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
