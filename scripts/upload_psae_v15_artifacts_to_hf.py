#!/usr/bin/env python3
"""
Upload PSAE v1.5 artifacts to HuggingFace as a dataset repo.

Scope: cache files (residuals + features, ~43MB), result JSONs, figures.
Out of scope: trained probes (16GB) — these are large, reproducible from
cache + notebook, and don't add value as standalone artifacts.

Target repo: caiovicentino1/openinterp-psae-v15-marginal-fit-pathology
License: Apache-2.0

Run:
    HF_TOKEN=hf_xxx python3 upload_psae_v15_artifacts_to_hf.py
    or
    huggingface-cli login  (interactive) then python3 ...
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

DRIVE = Path("/Users/caiovicentino/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive")
PSAE = DRIVE / "openinterp_runs" / "predictive_sae_v1"

REPO_ID = "caiovicentino1/openinterp-psae-v15-marginal-fit-pathology"
REPO_TYPE = "dataset"


# Files to upload, with their target paths in the repo
UPLOADS = [
    # JSON results
    (PSAE / "predictive_sae_v15_results.json", "results/predictive_sae_v15_results.json"),
    (PSAE / "random_baseline_results.json",    "results/random_baseline_results.json"),
    (PSAE / "feature_support_analysis.json",   "results/feature_support_analysis.json"),

    # Cache (reusable inputs for any re-run)
    (PSAE / "cache" / "residuals_multilayer.pt", "cache/residuals_multilayer.pt"),
    (PSAE / "cache" / "features_multilayer.pt",  "cache/features_multilayer.pt"),
    (PSAE / "cache" / "thinking_traces.pt",      "cache/thinking_traces.pt"),

    # Figures
    (PSAE / "figures" / "recall_multilayer.png",                "figures/recall_multilayer.png"),
    (PSAE / "figures" / "recall_multilayer_with_baselines.png", "figures/recall_multilayer_with_baselines.png"),
]


README = """---
license: apache-2.0
language:
- en
tags:
- mechanistic-interpretability
- sparse-autoencoder
- linear-probes
- honest-negative
- qwen3.6-27b
size_categories:
- n<1K
---

# PSAE v1.5 — Marginal-Fit Pathology Honest-Negative

Reproducibility artifacts for the paper *The Marginal-Fit Pathology in
Predictive SAE Feature Trajectory Probes* (workshop submission, NeurIPS MI
Workshop 2026).

**TL;DR**: We trained linear probes on Qwen3.6-27B residuals to predict
end-of-thinking SAE features from earlier-thinking residuals across L11/L31/L55.
Naive recall@1024 = 0.83-0.87 looked paper-grade. The shuffled-source baseline
B1 reproduces this within ±0.03 at all 12 sites. The trivial constant baseline
B2 ("predict the top-M most-common features in train, ignore input") reaches
recall@1024 = 1.000 at L11/L31 — strictly exceeding the trained probe. The
predictive claim does not survive.

## Contents

| Path | Size | Description |
|---|---|---|
| `cache/residuals_multilayer.pt` | 39 MB | Qwen3.6-27B residuals, 133 GSM8K prompts × 4 thinking-fractions × 3 layers |
| `cache/features_multilayer.pt` | 2.9 MB | SAE TopK features (`indices`, `values`) at end-of-thinking, 3 layers |
| `cache/thinking_traces.pt` | 1.2 MB | Thinking-phase text traces |
| `results/predictive_sae_v15_results.json` | 7 KB | Original PSAE v1.5 recall@k numbers (REAL) |
| `results/random_baseline_results.json` | ~30 KB | B0 (SAE-init no-train) + B1 (shuffled-source) recall@k |
| `results/feature_support_analysis.json` | ~3 KB | B2 trivial constant baseline + concentration stats |
| `figures/recall_multilayer.png` | 200 KB | Original PSAE v1.5 figure |
| `figures/recall_multilayer_with_baselines.png` | 280 KB | REAL + B0 + B1 + B2 overlay |

## Reproduction

```python
import torch
residuals = torch.load("cache/residuals_multilayer.pt", weights_only=False)
features  = torch.load("cache/features_multilayer.pt",  weights_only=False)

# residuals[L][frac] = Tensor[N=133, d_model=5120]
# features[L][1.00]['indices'] = Tensor[N=133, k=128]  ← target
```

The SAE encoder weights live in
[`caiovicentino1/qwen36-27b-sae-papergrade`](https://huggingface.co/caiovicentino1/qwen36-27b-sae-papergrade).

The two notebooks that produced these results:
- `nb_predictive_sae_v1.ipynb` — original REAL training
- `nb_predictive_sae_v15_baseline.ipynb` — B0 + B1 baselines (this paper)

Both are in [OpenInterpretability/openinterp](https://github.com/OpenInterpretability/openinterp).

## Citation

```
@misc{openinterp-psae-v15-marginal-fit-pathology-2026,
  author = {Vicentino, Caio},
  title  = {The Marginal-Fit Pathology in Predictive SAE Feature
            Trajectory Probes},
  year   = {2026},
  url    = {https://huggingface.co/datasets/caiovicentino1/openinterp-psae-v15-marginal-fit-pathology}
}
```

Part of the OpenInterpretability honest-negative methodology series. Sibling
papers: *Two Forms of Epiphenomenal Probes in Code Agents*,
*Saturation-Direction Lever: A Five-Class Taxonomy of Probe Causality*.
"""


def main():
    api = HfApi()
    print(f"creating dataset repo {REPO_ID}")
    create_repo(REPO_ID, repo_type=REPO_TYPE, exist_ok=True)

    # Upload README first
    api.upload_file(
        path_or_fileobj=README.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="add README",
    )
    print("  ✓ README.md")

    # Upload each file
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

    print(f"\n✓ all uploaded to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
