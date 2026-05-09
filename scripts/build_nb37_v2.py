"""
Builder for nb37_v2_multiprobe_dpo_extended.ipynb

Re-train Multi-Probe DPO with finer granularity to resolve ambiguity in nb41
(ratio=1.74, between gradual <1.5 and phase-transition >2.0).

Changes from nb37 v1:
- num_train_epochs: 2 → 5 (250 effective steps vs 80)
- save_steps: 20 → 20 (same, but with save_total_limit=None to keep ALL)
- save_total_limit: None (keep all 12-13 checkpoints, vs default 3)
- LR: 5e-6 (UNCHANGED — isolated test of "more steps")
- Reuse existing pairs.json from nb37 (skip build pairs)

Output: NEW directory 37v2_multiprobe_dpo_extended/, 12-13 checkpoints across 250 steps

Compute: ~65 min DPO + ~5 min model load = ~70 min total

Then run nb41 v2 against checkpoints from this dir → 13-point trajectory should resolve
ambiguity (gradual vs grokking).
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("/Volumes/SSD Major/fish/openinterp-work/notebooks")


def code(lines, **meta):
    return {
        "cell_type": "code",
        "metadata": meta or {},
        "execution_count": None,
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


def md(lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in lines],
    }


def build():
    cells = []

    cells.append(md([
        "# Notebook 37 v2 — Multi-Probe DPO Extended",
        "",
        "Re-train DPO with finer checkpoint granularity to resolve nb41 ambiguity (ratio=1.74).",
        "",
        "**Changes vs nb37 v1**:",
        "- `num_train_epochs`: 2 → **5** (250 effective steps vs 80)",
        "- `save_total_limit`: default 3 → **None** (keep ALL ~13 checkpoints)",
        "- `save_steps`: 20 (same)",
        "- `learning_rate`: 5e-6 (UNCHANGED — isolated test of 'more steps' alone)",
        "- **Reuse existing pairs.json** from nb37 v1 (skip build pairs)",
        "",
        "**Drive**: `/content/drive/MyDrive/openinterp_runs/37v2_multiprobe_dpo_extended/`",
        "",
        "**Compute**: ~70 min DPO + ~5 min model load",
        "",
        "**Decision after**: re-run nb41 against new checkpoints. With 13 data points (vs 4), phase-transition ratio should clearly resolve as either >2.0 (grokking) or <1.5 (gradual).",
    ]))

    # Phase 1
    cells.append(md(["## Phase 1 — Drive mount + dirs"]))
    cells.append(code([
        "from pathlib import Path",
        "import os, sys, time",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "",
        "DRIVE_ROOT = Path('/content/drive/MyDrive')",
        "NB_NAME = '37v2_multiprobe_dpo_extended'",
        "OUT = DRIVE_ROOT / 'openinterp_runs' / NB_NAME",
        "OUT.mkdir(parents=True, exist_ok=True)",
        "",
        "# Source: nb37 v1 pairs.json",
        "NB37_V1 = DRIVE_ROOT / 'openinterp_runs' / '37_multiprobe_dpo_full'",
        "PAIRS_SRC = NB37_V1 / 'pairs.json'",
        "assert PAIRS_SRC.exists(), 'nb37 v1 pairs.json not found'",
        "import shutil",
        "shutil.copy(PAIRS_SRC, OUT / 'pairs.json')",
        "print(f'✓ Copied pairs.json from nb37 v1')",
        "print(f'✓ OUT: {OUT}')",
    ]))

    cells.append(md(["## Phase 1.5 — Deps"]))
    cells.append(code([
        "!pip install -q -U torchao",
        "!pip install -q -U transformers accelerate datasets",
        "!pip install -q -U trl peft huggingface_hub",
        "print('✓ deps')",
    ]))

    # Phase 2 — model
    cells.append(md(["## Phase 2 — HF login + Qwen3.6-27B + LoRA wrap"]))
    cells.append(code([
        "import torch, json, time",
        "import numpy as np",
        "from huggingface_hub import login, create_repo, HfApi",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "from peft import LoraConfig, get_peft_model",
        "from datasets import Dataset",
        "from trl import DPOTrainer, DPOConfig",
        "",
        "CFG = {",
        "    'model_id':            'Qwen/Qwen3.6-27B',",
        "    'num_train_epochs':    5,                    # was 2 in v1",
        "    'lora_r':              16,",
        "    'lora_alpha':          32,",
        "    'lora_dropout':        0.05,",
        "    'dpo_lr':              5e-6,                 # UNCHANGED vs v1",
        "    'dpo_beta':            0.1,",
        "    'save_steps':          20,",
        "    'save_total_limit':    None,                 # keep ALL — was default 3 in v1",
        "    'random_seed':         42,",
        "    'output_repo':         'caiovicentino1/openinterp-37v2-multiprobe-dpo-extended',",
        "}",
        "",
        "torch.manual_seed(CFG['random_seed']); np.random.seed(CFG['random_seed'])",
        "",
        "import getpass",
        "HF_TOKEN = os.environ.get('HF_TOKEN') or getpass.getpass('HF token: ')",
        "login(HF_TOKEN, add_to_git_credential=False)",
        "try: create_repo(CFG['output_repo'], repo_type='dataset', private=False, exist_ok=True, token=HF_TOKEN)",
        "except Exception as e: print(e)",
        "",
        "device = 'cuda'; assert torch.cuda.is_available()",
        "tok = AutoTokenizer.from_pretrained(CFG['model_id'])",
        "model = AutoModelForCausalLM.from_pretrained(",
        "    CFG['model_id'], torch_dtype=torch.bfloat16, device_map='auto',",
        ")",
        "print(f'✓ Base loaded — {torch.cuda.get_device_name(0)}')",
        "",
        "lora_cfg = LoraConfig(",
        "    r=CFG['lora_r'], lora_alpha=CFG['lora_alpha'], lora_dropout=CFG['lora_dropout'],",
        "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],",
        "    bias='none', task_type='CAUSAL_LM',",
        ")",
        "model = get_peft_model(model, lora_cfg)",
        "model.print_trainable_parameters()",
    ]))

    # Phase 3 — DPO config + train
    cells.append(md(["## Phase 3 — DPO training (extended, ~70min)"]))
    cells.append(code([
        "with open(OUT / 'pairs.json') as f:",
        "    pairs = json.load(f)",
        "print(f'Loaded {len(pairs)} pairs')",
        "",
        "ds = Dataset.from_list(pairs).train_test_split(test_size=0.2, seed=CFG['random_seed'])",
        "print(f\"  Train: {len(ds['train'])}, Eval: {len(ds['test'])}\")",
        "",
        "dpo_cfg = DPOConfig(",
        "    output_dir=str(OUT / 'dpo_run'),",
        "    num_train_epochs=CFG['num_train_epochs'],",
        "    per_device_train_batch_size=1,",
        "    gradient_accumulation_steps=4,",
        "    learning_rate=CFG['dpo_lr'],",
        "    beta=CFG['dpo_beta'],",
        "    save_steps=CFG['save_steps'],",
        "    save_total_limit=CFG['save_total_limit'],",
        "    save_strategy='steps',",
        "    logging_steps=2,",
        "    bf16=True,",
        "    remove_unused_columns=False,",
        "    report_to='none',",
        "    seed=CFG['random_seed'],",
        ")",
        "",
        "trainer = DPOTrainer(",
        "    model=model, args=dpo_cfg,",
        "    train_dataset=ds['train'], eval_dataset=ds['test'],",
        "    processing_class=tok,",
        ")",
        "",
        "n_steps_total = len(ds['train']) * CFG['num_train_epochs'] // (1 * 4)",
        "n_checkpoints_expected = n_steps_total // CFG['save_steps']",
        "print(f'Expected: {n_steps_total} effective steps, {n_checkpoints_expected} checkpoints saved')",
        "",
        "t0 = time.time()",
        "trainer.train()",
        "print(f'✓ DPO complete in {(time.time()-t0)/60:.1f} min')",
        "",
        "# Save final adapter",
        "model.save_pretrained(str(OUT / 'lora_final'))",
        "print(f'✓ Saved lora_final to {OUT / \"lora_final\"}')",
    ]))

    # Phase 4 — checkpoint inventory + HF push
    cells.append(md(["## Phase 4 — Inventory + HF push"]))
    cells.append(code([
        "dpo_run = OUT / 'dpo_run'",
        "checkpoints = sorted([d for d in dpo_run.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')],",
        "                     key=lambda d: int(d.name.split('-')[1]))",
        "print(f'Saved {len(checkpoints)} checkpoints:')",
        "for c in checkpoints:",
        "    adapter = c / 'adapter_model.safetensors'",
        "    size_mb = adapter.stat().st_size / 1e6 if adapter.exists() else 0",
        "    print(f'  {c.name}: {size_mb:.1f} MB')",
        "",
        "(OUT / 'CHECKPOINT_LIST.json').write_text(json.dumps([c.name for c in checkpoints], indent=2))",
        "",
        "# HF push (metadata only — checkpoints stay on Drive due to size)",
        "api = HfApi()",
        "(OUT / 'README.md').write_text(f'''---",
        "license: apache-2.0",
        "tags: [dpo, multi-probe, extended-training]",
        "---",
        "",
        "# nb37 v2 — Extended DPO training",
        "",
        "Re-trained nb37 with `num_train_epochs=5` (vs 2) and `save_total_limit=None` (vs 3) to provide finer checkpoint granularity for nb41 v2 phase-transition analysis.",
        "",
        "Total checkpoints: {len(checkpoints)}",
        "Drive location: openinterp_runs/37v2_multiprobe_dpo_extended/dpo_run/",
        "''')",
        "",
        "try:",
        "    api.upload_folder(folder_path=str(OUT), repo_id=CFG['output_repo'],",
        "                      repo_type='dataset', token=HF_TOKEN,",
        "                      commit_message='nb37 v2 complete',",
        "                      allow_patterns=['README.md', 'CHECKPOINT_LIST.json', 'pairs.json'])",
        "    print('✓ pushed to HF')",
        "except Exception as e:",
        "    print(f'HF push failed: {e}')",
        "",
        "(OUT / '_phase4_done.txt').write_text(f'ts={time.time()}, n_ckpts={len(checkpoints)}')",
    ]))

    cells.append(md([
        "## Done",
        "",
        "Run `nb41_v2_grokking_forward_only_extended.ipynb` next — same forward-only methodology applied to all checkpoints in `dpo_run/`.",
        "",
        "Expected: 12-13 data points (vs 4 in v1) → ratio resolves clearly:",
        "- ratio > 2.0 → 🔴 grokking signal (paper-3 angle)",
        "- ratio < 1.5 → 🟢 gradual (honest negative, blog post angle)",
    ]))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = NOTEBOOKS_DIR / "nb37_v2_multiprobe_dpo_extended.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    print(f"✓ wrote {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
