#!/usr/bin/env python3
"""
Upload LeJEPA v1 artifacts to HuggingFace as a model repo.

Scope: latest.pt checkpoint (309 MB) + history.json + curves.png + README.
Out of scope: per-epoch checkpoints (100 × 309MB = 30GB) — bulky, only
needed for resume which is local-only.

Target repo: caiovicentino1/lejepa-v1-tinyimagenet
License: Apache-2.0

Run:
    HF_TOKEN=hf_xxx python3 upload_lejepa_v1_to_hf.py
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

DRIVE = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive")
LEJEPA = DRIVE / "openinterp_runs" / "lejepa_v1"

REPO_ID = "caiovicentino1/lejepa-v1-tinyimagenet"
REPO_TYPE = "model"


UPLOADS = [
    (LEJEPA / "ckpt" / "latest.pt",       "latest.pt"),
    (LEJEPA / "logs" / "history.json",    "training_history.json"),
    (LEJEPA / "lejepa_v1_curves.png",     "training_curves.png"),
]


README = """---
license: apache-2.0
language:
- en
tags:
- jepa
- self-supervised
- representation-learning
- tiny-imagenet
- vit
library_name: pytorch
---

# LeJEPA v1 POC — Tiny-ImageNet

100-epoch reproduction of the LeJEPA self-supervised representation learning
recipe from [Balestriero & LeCun, "LeJEPA: Provable and Scalable
Self-Supervised Learning Without the Heuristics" (arXiv:2511.08544)](https://arxiv.org/abs/2511.08544),
on the [Tiny-ImageNet](https://huggingface.co/datasets/zh-plus/tiny-imagenet)
dataset (200 classes, 100K images).

**Status**: POC. 100-epoch reproduction (paper uses 800). Final linear-probe
val_acc 23.77% (47× chance on 200 classes). Random-encoder baseline pending
(see [`nb_lejepa_v1_random_encoder_baseline.ipynb`](https://github.com/OpenInterpretability/openinterp-work/blob/main/notebooks/nb_lejepa_v1_random_encoder_baseline.ipynb)).

## Final metrics (epoch 99)

| Metric | Value |
|---|---|
| SIGReg loss | 1.584 (down from 11.826 at epoch 0) |
| Invariance loss | 0.126 (down from 0.447) |
| Probe CE loss | 3.610 (down from 5.195) |
| Linear-probe val_acc | **0.2377** |

No representation collapse — SIGReg loss converges, val_acc rises monotonically.

## Architecture (matches paper MINIMAL.md)

- Encoder: timm `vit_small_patch8_224`, `img_size=128`, `num_classes=512`
- Projection: MLP 512 → 2048 → 2048 → 16 (proj_dim)
- 4 views per image (V=4)
- Loss: `λ·SIGReg(proj) + (1-λ)·invariance_loss`, λ=0.02
- Linear probe joint with `.detach()` for monitoring

## Hyperparameters

- AdamW lr=2e-3, wd=5e-2
- Batch 256, bf16 mixed precision
- 100 epochs, cosine LR with linear warmup

## Files

| File | Description |
|---|---|
| `latest.pt` | Final encoder + projection + probe state dict (309 MB) |
| `training_history.json` | Per-epoch metrics for all 100 epochs |
| `training_curves.png` | Loss + val_acc curves |

## Limitations

- POC at 100 epochs (paper uses 800). Linear-probe accuracy proportional.
- Random-encoder baseline still pending — until it's run, the contribution
  of JEPA training (vs random encoder + linear probe alone) is not isolated.
- Not SOTA. DINOv2/MAE on the same scale would beat this.

## Citation

```
@misc{openinterp-lejepa-v1-tinyimagenet-2026,
  author = {Vicentino, Caio},
  title  = {LeJEPA v1 POC: Tiny-ImageNet 100-epoch reproduction},
  year   = {2026},
  url    = {https://huggingface.co/caiovicentino1/lejepa-v1-tinyimagenet}
}
```
"""


def main():
    api = HfApi()
    print(f"creating model repo {REPO_ID}")
    create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True)

    api.upload_file(
        path_or_fileobj=README.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="add README",
    )
    print("  ✓ README.md")

    for src, dst in UPLOADS:
        if not src.exists():
            print(f"  ⚠ skip {src} (not found)")
            continue
        sz = src.stat().st_size / (1024 * 1024)
        print(f"  uploading {dst} ({sz:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dst,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"add {dst}",
        )
        print(f"  ✓ {dst}")

    print(f"\n✓ uploaded to https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
