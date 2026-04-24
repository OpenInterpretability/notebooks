<!-- Thanks for contributing a notebook! -->

## What this notebook does

<!-- 1-3 sentences. What does it train / evaluate / visualize? -->

## Cell count + runtime

- Cells: `XX` (Y markdown + Z code)
- Runtime: `~X min` on `PLATFORM` (e.g. Colab T4 / Kaggle 2×T4 / Vast.ai RTX 6000)
- Total training tokens: `XXM` (if applicable)

## Constraints checklist

- [ ] `dtype=torch.bfloat16` (never `torch_dtype=`)
- [ ] `attn_implementation='sdpa'` (never flash-attn)
- [ ] HF token via Colab secret / Kaggle secret / env var (never hard-coded)
- [ ] Saves artifacts to HuggingFace (not just Drive) for reproducibility
- [ ] Runs start-to-finish on the target platform without user-specific Drive paths
- [ ] `python3 -c "import json; json.load(open('<notebook>'))"` passes

## What the output looks like

<!-- Paste a brief log / markdown / screenshot from a representative run. -->

## Honest caveats

<!-- e.g. "var_exp plateaus at 0.7 — may need more tokens for robust results" -->
