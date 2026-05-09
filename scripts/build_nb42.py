"""
Builder for nb42_multiprobe_grpo_pilot.ipynb

PILOT — Multi-Probe Orthogonal Reward GRPO on Qwen3.6-27B reasoning.

Hypothesis: GRPO with continuous multi-probe reward (FG + RG combined, with FG ⊥ RG
Pearson +0.014 from nb37) produces stronger preference learning than DPO with binary
preference pairs, because:
1. Continuous reward signal preserves full probe information (not collapsed to binary)
2. Online sampling explores beyond fixed dataset (DPO is bounded by chosen-rejected pairs)
3. Multi-axis orthogonality is anti-Goodhart by construction

Pilot scope (~4-5h compute):
- 40 unique prompts (20 gsm8k + 20 simpleqa, balanced subset of nb37 pairs.json)
- 4 candidates per prompt (group_size=4)
- 2 epochs → 80 prompt iterations / grad_accum 4 = 20 effective grad steps
- save_steps=5 → 4 checkpoints (steps 5, 10, 15, 20)
- LR=5e-6 (same as DPO for fair comparison)
- KL beta=0.1
- LoRA r=16 alpha=32

Decision after: if reward trends UP and probes shift, scale to full (~12h). If flat, pivot.

Drive: /content/drive/MyDrive/openinterp_runs/42_multiprobe_grpo_pilot/
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
        "# Notebook 42 — Multi-Probe Orthogonal Reward GRPO (PILOT)",
        "",
        "**Thesis**: GRPO with continuous multi-probe reward (FG + RG, Pearson +0.014 orthogonal) produces stronger preference learning than multi-probe DPO (nb37), because continuous reward + online sampling > binary preference pairs.",
        "",
        "**Pilot scope**: 40 prompts × 4 candidates × 2 epochs = 80 prompt iterations / accum 4 = 20 effective grad steps. ~4-5h compute. Decides whether to scale to full (250 steps, ~12h).",
        "",
        "**Reward**: r(c) = -(0.5 · FG(c) + 0.5 · RG(c)) — minimize fabrication and unfaithfulness simultaneously.",
        "",
        "**Anti-Goodhart by construction**: probes pre-validated orthogonal (Pearson +0.014), so single-axis Goodhart is impossible by definition.",
        "",
        "**Compute budget**: ~R$15 (~$3 USD).",
        "",
        "**Drive**: `/content/drive/MyDrive/openinterp_runs/42_multiprobe_grpo_pilot/`",
    ]))

    # Phase 1
    cells.append(md(["## Phase 1 — Setup + Drive"]))
    cells.append(code([
        "from pathlib import Path",
        "import os, sys, time, json, shutil",
        "import torch, numpy as np",
        "",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "DRIVE = Path('/content/drive/MyDrive')",
        "OUT = DRIVE / 'openinterp_runs' / '42_multiprobe_grpo_pilot'",
        "OUT.mkdir(parents=True, exist_ok=True)",
        "(OUT / '_dry_run.txt').write_text('drive ok')",
        "",
        "NB37 = DRIVE / 'openinterp_runs' / '37_multiprobe_dpo_full'",
        "PAIRS_SRC = NB37 / 'pairs.json'",
        "print(f'OUT: {OUT}')",
        "print(f'pairs source exists: {PAIRS_SRC.exists()}')",
    ]))
    cells.append(code([
        "!pip install -q -U torchao",
        "!pip install -q -U transformers accelerate datasets",
        "!pip install -q -U trl peft huggingface_hub",
        "!pip install -q joblib scikit-learn",
        "print('✓ deps')",
    ]))

    # Phase 2 — model + login
    cells.append(md(["## Phase 2 — HF login + Qwen3.6-27B + LoRA wrap"]))
    cells.append(code([
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "from huggingface_hub import login, create_repo, HfApi, hf_hub_download",
        "from peft import LoraConfig, get_peft_model",
        "import getpass, joblib",
        "",
        "CFG = {",
        "    'model_id':            'Qwen/Qwen3.6-27B',",
        "    'capture_layer_fg':    31,",
        "    'capture_layer_rg':    55,",
        "    'n_prompts':           40,",
        "    'num_generations':     4,",
        "    'num_train_epochs':    2,",
        "    'max_completion_length': 1024,",
        "    'temperature':         0.8,",
        "    'beta':                0.1,",
        "    'learning_rate':       5e-6,",
        "    'lora_r':              16,",
        "    'lora_alpha':          32,",
        "    'lora_dropout':        0.05,",
        "    'reward_alpha_fg':     0.5,",
        "    'reward_alpha_rg':     0.5,",
        "    'save_steps':          5,",
        "    'random_seed':         42,",
        "    'fg_probe_repo':       'caiovicentino1/FabricationGuard-linearprobe-qwen36-27b',",
        "    'rg_probe_repo':       'caiovicentino1/ReasoningGuard-linearprobe-qwen36-27b',",
        "    'output_repo':         'caiovicentino1/openinterp-42-multiprobe-grpo-pilot',",
        "}",
        "THINK_CLOSE_ID = 248069",
        "torch.manual_seed(CFG['random_seed']); np.random.seed(CFG['random_seed'])",
        "",
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
        "print(f'✓ Base loaded — {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')",
    ]))
    cells.append(code([
        "# LoRA wrap — same config as nb37 for fair comparison",
        "lora_cfg = LoraConfig(",
        "    r=CFG['lora_r'], lora_alpha=CFG['lora_alpha'], lora_dropout=CFG['lora_dropout'],",
        "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],",
        "    bias='none', task_type='CAUSAL_LM',",
        ")",
        "model = get_peft_model(model, lora_cfg)",
        "model.print_trainable_parameters()",
    ]))

    # Phase 3 — probes
    cells.append(md(["## Phase 3 — FG + RG probes (continuous reward)"]))
    cells.append(code([
        "fg_path = hf_hub_download(repo_id=CFG['fg_probe_repo'], filename='probe.joblib', repo_type='dataset')",
        "rg_path = hf_hub_download(repo_id=CFG['rg_probe_repo'], filename='probe.joblib', repo_type='dataset')",
        "fg_artifact = joblib.load(fg_path)",
        "rg_artifact = joblib.load(rg_path)",
        "fg_clf = fg_artifact['probe']; fg_scaler = fg_artifact['scaler']",
        "rg_clf = rg_artifact['probe']; rg_scaler = rg_artifact['scaler']",
        "# Monkey-patch for sklearn 1.5+ compat (probe trained on older sklearn)",
        "if not hasattr(fg_clf, 'multi_class'): fg_clf.multi_class = 'auto'",
        "if not hasattr(rg_clf, 'multi_class'): rg_clf.multi_class = 'auto'",
        "print(f'✓ FG: {type(fg_clf).__name__}, RG: {type(rg_clf).__name__}')",
        "",
        "def fg_score(activation):",
        "    x = activation.float().cpu().numpy().reshape(1, -1)",
        "    return float(fg_clf.predict_proba(fg_scaler.transform(x))[0, 1])",
        "def rg_score(activation):",
        "    x = activation.float().cpu().numpy().reshape(1, -1)",
        "    return float(rg_clf.predict_proba(rg_scaler.transform(x))[0, 1])",
    ]))

    # Phase 4 — hooks + reward function
    cells.append(md([
        "## Phase 4 — Activation hooks + reward function",
        "",
        "Reward: `r(c) = -(α · FG(c) + β · RG(c))`. Negative because GRPO MAXIMIZES — we want to MINIMIZE fabrication and unfaithfulness, so reward = -score.",
    ]))
    cells.append(code([
        "captured = {}",
        "_pos = {'pos': None}",
        "",
        "def make_hook(layer_idx):",
        "    def hook(module, input, output):",
        "        h = output[0] if isinstance(output, tuple) else output",
        "        pos = _pos['pos']",
        "        if pos is None or pos >= h.shape[1]:",
        "            return",
        "        captured[f'L{layer_idx}'] = h[0, pos, :].detach().cpu().to(torch.float16).clone()",
        "    return hook",
        "",
        "def get_layers(m):",
        "    candidates = [",
        "        lambda x: x.base_model.model.model.layers,",
        "        lambda x: x.model.layers,",
        "    ]",
        "    for fn in candidates:",
        "        try:",
        "            l = fn(m)",
        "            if hasattr(l, '__len__') and len(l) > 0: return l",
        "        except AttributeError: continue",
        "    raise RuntimeError('no layers')",
        "",
        "hook_handles = []",
        "for L in [CFG['capture_layer_fg'], CFG['capture_layer_rg']]:",
        "    h = get_layers(model)[L].register_forward_hook(make_hook(L))",
        "    hook_handles.append(h)",
        "print(f'✓ Hooks at L{CFG[\"capture_layer_fg\"]}, L{CFG[\"capture_layer_rg\"]}')",
    ]))
    cells.append(code([
        "def find_end_think_pos(token_ids):",
        "    ids = token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids)",
        "    for i in range(len(ids) - 1, -1, -1):",
        "        if ids[i] == THINK_CLOSE_ID:",
        "            return i",
        "    return None",
        "",
        "_reward_log = []  # track for analysis",
        "",
        "def multiprobe_reward(prompts, completions, **kwargs):",
        "    \"\"\"GRPO reward: -(α·FG + β·RG). Higher = better (less fabrication, more faithful).\"\"\"",
        "    rewards = []",
        "    for prompt, comp in zip(prompts, completions):",
        "        # GRPO passes completion as STR (not token ids) — re-tokenize prompt+completion",
        "        full_text = prompt + comp + '<|im_end|>'",
        "        try:",
        "            enc = tok(full_text, return_tensors='pt')",
        "            ids = enc['input_ids'].to(device)",
        "        except Exception:",
        "            rewards.append(0.0)",
        "            continue",
        "        end_pos = find_end_think_pos(ids[0])",
        "        if end_pos is None:",
        "            # No </think> in output — penalize (model didn't reason)",
        "            rewards.append(-1.0)",
        "            _reward_log.append({'fg': None, 'rg': None, 'r': -1.0, 'reason': 'no_think'})",
        "            continue",
        "        captured.clear()",
        "        _pos['pos'] = end_pos",
        "        with torch.no_grad():",
        "            _ = model(ids)",
        "        act_fg = captured.get(f'L{CFG[\"capture_layer_fg\"]}')",
        "        act_rg = captured.get(f'L{CFG[\"capture_layer_rg\"]}')",
        "        if act_fg is None or act_rg is None:",
        "            rewards.append(-1.0)",
        "            continue",
        "        fg = fg_score(act_fg)",
        "        rg = rg_score(act_rg)",
        "        # MINIMIZE fabrication AND unfaithfulness → reward = -weighted sum",
        "        r = -(CFG['reward_alpha_fg'] * fg + CFG['reward_alpha_rg'] * rg)",
        "        rewards.append(r)",
        "        _reward_log.append({'fg': fg, 'rg': rg, 'r': r, 'reason': 'ok'})",
        "    return rewards",
        "",
        "# Quick sanity test",
        "test_prompt = 'Q: What is 2+2?\\n\\nThink step by step.'",
        "test_comp = '<think>\\n2+2 = 4\\n</think>\\n\\nThe answer is 4.'",
        "test_r = multiprobe_reward([test_prompt], [test_comp])",
        "print(f'Sanity reward: {test_r}')",
    ]))

    # Phase 5 — GRPO setup + train
    cells.append(md(["## Phase 5 — GRPO trainer config + training"]))
    cells.append(code([
        "from trl import GRPOConfig, GRPOTrainer",
        "from datasets import Dataset",
        "",
        "# Load + subset prompts — PRE-TEMPLATE with <think> prefix for Qwen3.6 reasoning mode",
        "with open(PAIRS_SRC) as f:",
        "    all_pairs = json.load(f)",
        "rng = np.random.default_rng(CFG['random_seed'])",
        "indices = rng.choice(len(all_pairs), size=CFG['n_prompts'], replace=False)",
        "prompts_templated = []",
        "for i in indices:",
        "    p = all_pairs[i]",
        "    messages = [{'role': 'user', 'content': p['prompt']}]",
        "    text = tok.apply_chat_template(messages, tokenize=False,",
        "                                   add_generation_prompt=True, enable_thinking=True)",
        "    prompts_templated.append({'prompt': text})",
        "assert '<think>' in prompts_templated[0]['prompt'], 'Chat template missing <think>'",
        "ds = Dataset.from_list(prompts_templated)",
        "print(f'Train set: {len(ds)} prompts (chat-templated with <think> prefix)')",
        "",
        "# GRPO config",
        "grpo_cfg = GRPOConfig(",
        "    output_dir=str(OUT / 'grpo_run'),",
        "    num_train_epochs=CFG['num_train_epochs'],",
        "    per_device_train_batch_size=1,",
        "    gradient_accumulation_steps=4,",
        "    learning_rate=CFG['learning_rate'],",
        "    beta=CFG['beta'],",
        "    num_generations=CFG['num_generations'],",
        "    max_completion_length=CFG['max_completion_length'],",
        "    temperature=CFG['temperature'],",
        "    save_steps=CFG['save_steps'],",
        "    save_total_limit=None,",
        "    save_strategy='steps',",
        "    logging_steps=1,",
        "    bf16=True,",
        "    remove_unused_columns=False,",
        "    report_to='none',",
        "    seed=CFG['random_seed'],",
        ")",
        "",
        "trainer = GRPOTrainer(",
        "    model=model,",
        "    args=grpo_cfg,",
        "    train_dataset=ds,",
        "    reward_funcs=multiprobe_reward,",
        "    processing_class=tok,",
        ")",
        "",
        "n_total_iters = len(ds) * CFG['num_train_epochs']",
        "n_eff_steps = n_total_iters // 4",
        "print(f'Total prompt iterations: {n_total_iters}')",
        "print(f'Effective grad steps: {n_eff_steps}')",
        "print(f'Checkpoints expected: {n_eff_steps // CFG[\"save_steps\"]}')",
    ]))
    cells.append(code([
        "t0 = time.time()",
        "trainer.train()",
        "print(f'✓ GRPO complete in {(time.time()-t0)/60:.1f} min')",
        "",
        "model.save_pretrained(str(OUT / 'lora_final'))",
        "print(f'✓ saved lora_final to {OUT / \"lora_final\"}')",
        "",
        "# Save reward log",
        "(OUT / 'reward_log.json').write_text(json.dumps(_reward_log, indent=2))",
        "print(f'✓ reward_log: {len(_reward_log)} entries')",
    ]))

    # Phase 6 — analysis
    cells.append(md(["## Phase 6 — Analysis: reward trajectory + probe shift"]))
    cells.append(code([
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "",
        "with open(OUT / 'reward_log.json') as f:",
        "    log = json.load(f)",
        "df = pd.DataFrame([r for r in log if r['reason'] == 'ok'])",
        "print(f'Logged rewards: {len(df)}')",
        "print()",
        "print('=== Reward stats ===')",
        "print(df[['fg', 'rg', 'r']].describe().round(4))",
        "print()",
        "",
        "# Bin by chunks to see trajectory",
        "n_chunks = 10",
        "df['chunk'] = pd.cut(df.index, bins=n_chunks, labels=range(n_chunks))",
        "agg = df.groupby('chunk')[['fg', 'rg', 'r']].mean().round(4)",
        "print('=== Trajectory by chunk (early → late training) ===')",
        "print(agg)",
    ]))
    cells.append(code([
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))",
        "chunks = list(range(n_chunks))",
        "for ax, col, color, title in zip(axes,",
        "                                  ['fg', 'rg', 'r'],",
        "                                  ['C0', 'C1', 'C2'],",
        "                                  ['FabricationGuard score', 'ReasonGuard score', 'Combined reward']):",
        "    vals = agg[col].values",
        "    ax.plot(chunks, vals, marker='o', color=color)",
        "    ax.set_xlabel('Training chunk (early → late)'); ax.set_ylabel(col)",
        "    ax.set_title(f'{title} vs training progress')",
        "    ax.grid(alpha=0.3)",
        "    if col == 'r':",
        "        ax.axhline(vals[0], color='gray', linestyle='--', alpha=0.5, label='early baseline')",
        "        ax.legend()",
        "plt.tight_layout()",
        "plt.savefig(OUT / 'fig_reward_trajectory.png', dpi=150)",
        "plt.show()",
        "",
        "# Decision: did reward go UP (toward 0)? Probes go DOWN?",
        "early_r = agg['r'].values[0]",
        "late_r = agg['r'].values[-1]",
        "delta_r = late_r - early_r",
        "print(f'\\nEarly reward: {early_r:.4f}')",
        "print(f'Late reward:  {late_r:.4f}')",
        "print(f'Delta:        {delta_r:+.4f}')",
        "if delta_r > 0.02:",
        "    print('🟢 GRPO trained — reward improved')",
        "elif delta_r < -0.02:",
        "    print('🔴 GRPO regressed — reward got WORSE (over-optimization risk)')",
        "else:",
        "    print('🟡 Flat — no clear training signal in pilot')",
    ]))

    # Phase 7 — verdict + push
    cells.append(md(["## Phase 7 — FINAL_VERDICT + HF push"]))
    cells.append(code([
        "verdict = {",
        "    'experiment': 'nb42 multi-probe GRPO pilot',",
        "    'hypothesis': 'Continuous multi-probe reward in GRPO produces stronger preference learning than DPO with binary pairs',",
        "    'n_prompts': CFG['n_prompts'],",
        "    'num_generations': CFG['num_generations'],",
        "    'effective_steps': int(n_eff_steps),",
        "    'reward_early_chunk': float(early_r),",
        "    'reward_late_chunk':  float(late_r),",
        "    'reward_delta':       float(delta_r),",
        "    'fg_early': float(agg['fg'].values[0]),",
        "    'fg_late':  float(agg['fg'].values[-1]),",
        "    'rg_early': float(agg['rg'].values[0]),",
        "    'rg_late':  float(agg['rg'].values[-1]),",
        "    'training_signal': 'positive' if delta_r > 0.02 else ('negative' if delta_r < -0.02 else 'flat'),",
        "    'compared_to_dpo': 'TBD — need to run nb43 GRPO full + compare nb37 DPO eval',",
        "}",
        "(OUT / 'FINAL_VERDICT.json').write_text(json.dumps(verdict, indent=2))",
        "print(json.dumps(verdict, indent=2))",
    ]))
    cells.append(code([
        "api = HfApi()",
        "(OUT / 'README.md').write_text(f'''---",
        "license: apache-2.0",
        "tags: [grpo, multi-probe, mechanistic-interpretability, pilot]",
        "---",
        "",
        "# nb42 — Multi-Probe Orthogonal Reward GRPO (Pilot)",
        "",
        "First test of GRPO with continuous multi-probe reward (FabricationGuard + ReasonGuard, Pearson +0.014 orthogonal) on Qwen3.6-27B reasoning.",
        "",
        "Pilot scope: {CFG[\"n_prompts\"]} prompts × {CFG[\"num_generations\"]} candidates × {CFG[\"num_train_epochs\"]} epochs.",
        "",
        "Reward: r(c) = -(α · FG(c) + β · RG(c)).",
        "",
        "See FINAL_VERDICT.json for trajectory analysis.",
        "",
        "## Methodology lineage",
        "- Activation Reward Models (Jul 2025) — extends to multi-axis probe ensemble",
        "- Linear Probe Penalties (Dec 2024) — extends from sycophancy single-axis to FG+RG orthogonal",
        "- Multi-Objective GRPO Safe Alignment (Mar 2025) — uses interpretability-validated probes instead of trained RM",
        "- DeepSeek-R1 GRPO (Nature 2025) — methodology baseline",
        "''')",
        "",
        "try:",
        "    api.upload_folder(folder_path=str(OUT), repo_id=CFG['output_repo'],",
        "                      repo_type='dataset', token=HF_TOKEN,",
        "                      commit_message='nb42 multi-probe GRPO pilot complete',",
        "                      allow_patterns=['README.md', 'FINAL_VERDICT.json', 'fig_*.png',",
        "                                      'reward_log.json', '_*.txt'])",
        "    print('✓ pushed')",
        "except Exception as e:",
        "    print(f'HF push failed: {e}')",
    ]))

    cells.append(md([
        "## Done",
        "",
        "Decision matrix from FINAL_VERDICT:",
        "- 🟢 `training_signal: positive` (reward delta > 0.02) → scale to nb43 full (250 steps, ~12h)",
        "- 🔴 `training_signal: negative` (reward delta < -0.02) → over-optimization or reward hacking, debug before scaling",
        "- 🟡 `training_signal: flat` → pilot too short to see signal; either run more epochs or accept that 20 effective steps is below threshold",
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

    out_path = NOTEBOOKS_DIR / "nb42_multiprobe_grpo_pilot.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    print(f"✓ wrote {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
