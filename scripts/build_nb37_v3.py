"""
Builder for nb37_v3_multiprobe_dpo_super_extended.ipynb

Extended DPO training to 400 effective steps with finer save_steps=10 → 40
checkpoints. Confirms whether the step-200 phase transition observed in nb41 v2
(ratio 2.60) was real grokking or measurement noise at training boundary.

Changes vs nb37 v2:
- num_train_epochs: 5 → 10 (200 → 400 effective steps)
- save_steps: 20 → 10 (10 → 40 checkpoints)
- save_total_limit: None (keep all 40)
- LR: 5e-6 (UNCHANGED — isolated test of "more steps")
- Reuse pairs.json from nb37 v2 (skip generation)

Output: NEW dir 37v3_multiprobe_dpo_super_extended/

Compute: ~95 min DPO + ~5 min model load = ~1.7h
Drive cost: 40 × 318MB = 12.7GB (within free tier 15GB if no other large checkpoints)

After this, run nb41 v3 forward-only against the 40 checkpoints to get a
high-resolution view of the trajectory. Phase transition signature should
either (a) continue at step 250-300 (real construct-then-compress), (b) saturate
(transition was at boundary), or (c) reverse (step-200 was noise).

Decision matrix after:
- 🔴 ratio > 2.0 with peak past step 200 → REAL grokking confirmed, paper-2 strengthens
- 🟡 ratio 1.5-2.0 → ambiguous still, methodology limit
- 🟢 ratio < 1.5 → step-200 was noise, paper-2 needs reframing

Drive: /content/drive/MyDrive/openinterp_runs/37v3_multiprobe_dpo_super_extended/
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
        "# Notebook 37 v3 — Super-Extended Multi-Probe DPO (400 steps)",
        "",
        "Resolves the phase-transition vs noise question from nb41 v2 (ratio 2.60 observed only at final checkpoint).",
        "",
        "**Changes vs nb37 v2**:",
        "- `num_train_epochs`: 5 → **10** (200 → **400** effective steps)",
        "- `save_steps`: 20 → **10** (10 → **40** checkpoints)",
        "- `save_total_limit`: None (keep all 40)",
        "- `learning_rate`: 5e-6 (UNCHANGED for fair scaling test)",
        "- Reuses pairs.json from nb37 v2 (no re-generation)",
        "",
        "**Drive**: `/content/drive/MyDrive/openinterp_runs/37v3_multiprobe_dpo_super_extended/`",
        "",
        "**Compute**: ~95 min DPO + ~5 min load = **~1.7h**",
        "",
        "**Drive cost**: 40 × 318MB ≈ **12.7GB** (close to free-tier 15GB; verify capacity before starting)",
        "",
        "**Decision matrix after nb41 v3 re-analysis**:",
        "- 🔴 ratio > 2.0 with peak past step 200 → REAL grokking, paper-2 strengthens",
        "- 🟡 ratio 1.5-2.0 → ambiguous, methodology limit acknowledged",
        "- 🟢 ratio < 1.5 → step-200 was noise; paper-2 reframes as gradual learning",
    ]))

    # Phase 1
    cells.append(md(["## Phase 1 — Drive mount + dirs"]))
    cells.append(code([
        "from pathlib import Path",
        "import os, sys, time, shutil",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "",
        "DRIVE_ROOT = Path('/content/drive/MyDrive')",
        "NB_NAME = '37v3_multiprobe_dpo_super_extended'",
        "OUT = DRIVE_ROOT / 'openinterp_runs' / NB_NAME",
        "OUT.mkdir(parents=True, exist_ok=True)",
        "",
        "# Source: nb37 v2 pairs.json (or v1 fallback)",
        "NB37_V2 = DRIVE_ROOT / 'openinterp_runs' / '37v2_multiprobe_dpo_extended'",
        "NB37_V1 = DRIVE_ROOT / 'openinterp_runs' / '37_multiprobe_dpo_full'",
        "PAIRS_SRC = NB37_V2 / 'pairs.json' if (NB37_V2 / 'pairs.json').exists() else NB37_V1 / 'pairs.json'",
        "assert PAIRS_SRC.exists(), 'pairs.json not found in nb37 v2 or v1'",
        "shutil.copy(PAIRS_SRC, OUT / 'pairs.json')",
        "print(f'✓ Copied pairs.json from {PAIRS_SRC.parent.name}')",
        "print(f'✓ OUT: {OUT}')",
        "",
        "# Verify Drive capacity (40 checkpoints × ~318MB ≈ 12.7GB)",
        "import shutil as sh",
        "free_gb = sh.disk_usage('/content/drive/MyDrive').free / 1e9",
        "print(f'Drive free space: {free_gb:.1f} GB')",
        "if free_gb < 15:",
        "    print(f'⚠️ WARNING: only {free_gb:.1f}GB free, need ~13GB for 40 checkpoints + buffer')",
        "    print('   Consider cleaning old runs or use save_total_limit=20 instead')",
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
        "import getpass",
        "",
        "CFG = {",
        "    'model_id':              'Qwen/Qwen3.6-27B',",
        "    'num_train_epochs':      10,                   # was 5 in v2 → 400 effective steps",
        "    'lora_r':                16,",
        "    'lora_alpha':            32,",
        "    'lora_dropout':          0.05,",
        "    'dpo_lr':                5e-6,                 # UNCHANGED for fair scaling",
        "    'dpo_beta':              0.1,",
        "    'save_steps':            10,                   # was 20 → 2× granularity",
        "    'save_total_limit':      None,                 # keep all 40",
        "    'random_seed':           42,",
        "    'output_repo':           'caiovicentino1/openinterp-37v3-multiprobe-dpo-super-extended',",
        "}",
        "torch.manual_seed(CFG['random_seed']); np.random.seed(CFG['random_seed'])",
        "",
        "HF_TOKEN = os.environ.get('HF_TOKEN') or getpass.getpass('HF token: ')",
        "login(HF_TOKEN, add_to_git_credential=False)",
        "try: create_repo(CFG['output_repo'], repo_type='dataset', private=False, exist_ok=True, token=HF_TOKEN)",
        "except Exception as e: print(e)",
        "",
        "device = 'cuda'",
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

    # Phase 3 — train
    cells.append(md(["## Phase 3 — DPO training (super-extended, ~95min)"]))
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
        "n_steps_total = len(ds['train']) * CFG['num_train_epochs'] // 4",
        "n_ckpts = n_steps_total // CFG['save_steps']",
        "print(f'Expected: {n_steps_total} effective steps, {n_ckpts} checkpoints')",
        "",
        "t0 = time.time()",
        "trainer.train()",
        "elapsed_min = (time.time() - t0) / 60",
        "print(f'\\n✓ DPO complete in {elapsed_min:.1f} min')",
        "",
        "model.save_pretrained(str(OUT / 'lora_final'))",
        "print(f'✓ saved lora_final to {OUT / \"lora_final\"}')",
    ]))

    # Phase 4 — inventory + push
    cells.append(md(["## Phase 4 — Inventory + HF push"]))
    cells.append(code([
        "dpo_run = OUT / 'dpo_run'",
        "checkpoints = sorted([d for d in dpo_run.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')],",
        "                     key=lambda d: int(d.name.split('-')[1]))",
        "print(f'Saved {len(checkpoints)} checkpoints:')",
        "for c in checkpoints[:5]:",
        "    adapter = c / 'adapter_model.safetensors'",
        "    print(f'  {c.name}: {adapter.stat().st_size / 1e6:.1f} MB')",
        "if len(checkpoints) > 10:",
        "    print(f'  ...')",
        "    for c in checkpoints[-3:]:",
        "        adapter = c / 'adapter_model.safetensors'",
        "        print(f'  {c.name}: {adapter.stat().st_size / 1e6:.1f} MB')",
        "",
        "(OUT / 'CHECKPOINT_LIST.json').write_text(json.dumps([c.name for c in checkpoints], indent=2))",
        "",
        "api = HfApi()",
        "(OUT / 'README.md').write_text(f'''---",
        "license: apache-2.0",
        "tags: [dpo, multi-probe, super-extended-training]",
        "---",
        "",
        "# nb37 v3 — Super-Extended DPO (400 steps)",
        "",
        "Re-trained nb37 with `num_train_epochs=10` (vs 5 in v2, vs 2 in v1), `save_steps=10` for high-resolution checkpoint trajectory across 400 effective steps.",
        "",
        "Tests whether the step-200 phase-transition signature observed in nb41 v2 (ratio 2.60) is real grokking or measurement boundary artifact. Run nb41 v3 forward-only after this to get the trajectory.",
        "",
        "Total checkpoints: {len(checkpoints)}",
        "Drive location: openinterp_runs/37v3_multiprobe_dpo_super_extended/dpo_run/",
        "''')",
        "",
        "try:",
        "    api.upload_folder(folder_path=str(OUT), repo_id=CFG['output_repo'],",
        "                      repo_type='dataset', token=HF_TOKEN,",
        "                      commit_message='nb37 v3 super-extended complete',",
        "                      allow_patterns=['README.md', 'CHECKPOINT_LIST.json', 'pairs.json'])",
        "    print('✓ pushed to HF (metadata)')",
        "except Exception as e:",
        "    print(f'HF push failed: {e}')",
        "",
        "(OUT / '_phase4_done.txt').write_text(f'ts={time.time()}, n_ckpts={len(checkpoints)}')",
    ]))

    cells.append(md([
        "## Done",
        "",
        "Run `nb41_v3_grokking_super_extended.ipynb` next:",
        "- 41 checkpoints (base + 40) × 20 hold-out forward passes = 820 forward passes",
        "- Same methodology as nb41 v2 with Qwen3.6 LoRA key fix",
        "- High-resolution trajectory across 400 steps",
        "- ~30-40min compute",
        "",
        "Decision after:",
        "- 🔴 ratio > 2.0 with peak past step 200 → REAL grokking, paper-2 strengthens",
        "- 🟡 ratio 1.5-2.0 → ambiguous, methodology limit acknowledged",
        "- 🟢 ratio < 1.5 → step-200 was noise, paper-2 reframes as gradual",
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

    out_path = NOTEBOOKS_DIR / "nb37_v3_multiprobe_dpo_super_extended.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    print(f"✓ wrote {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
