"""
Builder for nb43_multiprobe_grpo_full.ipynb

FULL — Multi-Probe Orthogonal Reward GRPO on Qwen3.6-27B reasoning.

Scaling up nb42 pilot (which showed POSITIVE training signal in 20 steps):
- Prompts: 40 → 100 (more diversity, balanced 50/50 gsm8k/simpleqa)
- Epochs: 2 → 5 (more exposure to each prompt)
- Effective grad steps: 20 → 125 (5× more)
- max_completion_length: 1024 → 1536 (reduce no_think rate from 53% to ~30%)
- save_steps: 5 → 25 (5 checkpoints across run)
- Same LR=5e-6, beta=0.1, num_generations=4 (consistency w/ pilot)

Pilot baseline (nb42, 20 effective steps):
- FG: 0.508 → 0.485 (-0.023)
- RG: 0.353 → 0.316 (-0.037)
- Combined reward: -0.430 → -0.400 (+0.030)

Hypothesis: with 5× more steps, both probes shift further (FG ~0.42, RG ~0.27)
and reward stabilizes near -0.30. Phase transition signature in fresh-probe AUROC
expected at later checkpoints.

Output: paper-3 head-to-head comparison vs nb37 v2 DPO + nb42 pilot.

Drive: /content/drive/MyDrive/openinterp_runs/43_multiprobe_grpo_full/
Compute: ~12h on RTX 6000 Blackwell
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
        "# Notebook 43 — Multi-Probe Orthogonal Reward GRPO (FULL)",
        "",
        "**Scaling up nb42 pilot** (which showed POSITIVE training signal in 20 effective steps).",
        "",
        "**Pilot baseline (nb42)**:",
        "- FG: 0.508 → 0.485 (-0.023)",
        "- RG: 0.353 → 0.316 (-0.037)",
        "- Reward: -0.430 → -0.400 (+0.030)",
        "",
        "**Full config changes**:",
        "- Prompts: 40 → **100** (balanced 50/50 gsm8k/simpleqa)",
        "- Epochs: 2 → **5**",
        "- Effective grad steps: 20 → **125** (5× more)",
        "- max_completion_length: 1024 → **1536** (reduce no_think 53% → ~30%)",
        "- save_steps: 5 → **25** (5 checkpoints across run)",
        "- LR/beta/num_generations: UNCHANGED for consistency w/ pilot",
        "",
        "**Compute**: ~12h on RTX 6000 Blackwell",
        "",
        "**Drive**: `/content/drive/MyDrive/openinterp_runs/43_multiprobe_grpo_full/`",
        "",
        "**Anti-Goodhart by construction**: probes pre-validated orthogonal (Pearson +0.014). Single-axis Goodhart impossible because maximizing FG cannot reduce RG and vice versa.",
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
        "OUT = DRIVE / 'openinterp_runs' / '43_multiprobe_grpo_full'",
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
        "!pip install -q -U trl peft huggingface_hub joblib scikit-learn",
        "print('✓ deps')",
    ]))

    # Phase 2 — model
    cells.append(md(["## Phase 2 — Login + Qwen3.6-27B + LoRA wrap"]))
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
        "    'n_prompts':           100,",
        "    'num_generations':     4,",
        "    'num_train_epochs':    5,",
        "    'max_completion_length': 1536,",
        "    'temperature':         0.8,",
        "    'beta':                0.1,",
        "    'learning_rate':       5e-6,",
        "    'lora_r':              16,",
        "    'lora_alpha':          32,",
        "    'lora_dropout':        0.05,",
        "    'reward_alpha_fg':     0.5,",
        "    'reward_alpha_rg':     0.5,",
        "    'save_steps':          25,",
        "    'save_total_limit':    None,",
        "    'random_seed':         42,",
        "    'fg_probe_repo':       'caiovicentino1/FabricationGuard-linearprobe-qwen36-27b',",
        "    'rg_probe_repo':       'caiovicentino1/ReasoningGuard-linearprobe-qwen36-27b',",
        "    'output_repo':         'caiovicentino1/openinterp-43-multiprobe-grpo-full',",
        "}",
        "THINK_CLOSE_ID = 248069",
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

    # Phase 3 — probes
    cells.append(md(["## Phase 3 — FG + RG probes (continuous reward)"]))
    cells.append(code([
        "fg_path = hf_hub_download(repo_id=CFG['fg_probe_repo'], filename='probe.joblib', repo_type='dataset')",
        "rg_path = hf_hub_download(repo_id=CFG['rg_probe_repo'], filename='probe.joblib', repo_type='dataset')",
        "fg_artifact = joblib.load(fg_path)",
        "rg_artifact = joblib.load(rg_path)",
        "fg_clf = fg_artifact['probe']; fg_scaler = fg_artifact['scaler']",
        "rg_clf = rg_artifact['probe']; rg_scaler = rg_artifact['scaler']",
        "if not hasattr(fg_clf, 'multi_class'): fg_clf.multi_class = 'auto'",
        "if not hasattr(rg_clf, 'multi_class'): rg_clf.multi_class = 'auto'",
        "print(f'✓ probes loaded')",
        "",
        "def fg_score(activation):",
        "    x = activation.float().cpu().numpy().reshape(1, -1)",
        "    return float(fg_clf.predict_proba(fg_scaler.transform(x))[0, 1])",
        "def rg_score(activation):",
        "    x = activation.float().cpu().numpy().reshape(1, -1)",
        "    return float(rg_clf.predict_proba(rg_scaler.transform(x))[0, 1])",
    ]))

    # Phase 4 — hooks + reward
    cells.append(md(["## Phase 4 — Activation hooks + reward function"]))
    cells.append(code([
        "captured = {}",
        "_pos = {'pos': None}",
        "",
        "def make_hook(layer_idx):",
        "    def hook(module, input, output):",
        "        h = output[0] if isinstance(output, tuple) else output",
        "        pos = _pos['pos']",
        "        if pos is None or pos >= h.shape[1]: return",
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
        "_reward_log = []",
        "",
        "def multiprobe_reward(prompts, completions, **kwargs):",
        "    rewards = []",
        "    for prompt, comp in zip(prompts, completions):",
        "        full_text = prompt + comp + '<|im_end|>'",
        "        try:",
        "            enc = tok(full_text, return_tensors='pt')",
        "            ids = enc['input_ids'].to(device)",
        "        except Exception:",
        "            rewards.append(0.0); continue",
        "        end_pos = find_end_think_pos(ids[0])",
        "        if end_pos is None:",
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
        "            rewards.append(-1.0); continue",
        "        fg = fg_score(act_fg); rg = rg_score(act_rg)",
        "        r = -(CFG['reward_alpha_fg'] * fg + CFG['reward_alpha_rg'] * rg)",
        "        rewards.append(r)",
        "        _reward_log.append({'fg': fg, 'rg': rg, 'r': r, 'reason': 'ok'})",
        "    return rewards",
        "",
        "# Sanity",
        "test_p = '<|im_start|>user\\nWhat is 2+2?<|im_end|>\\n<|im_start|>assistant\\n<think>\\n'",
        "test_c = '2+2 = 4\\n</think>\\n\\nThe answer is 4.'",
        "print(f'Sanity reward: {multiprobe_reward([test_p], [test_c])}')",
    ]))

    # Phase 5 — GRPO setup + train
    cells.append(md(["## Phase 5 — GRPO trainer config + training"]))
    cells.append(code([
        "from trl import GRPOConfig, GRPOTrainer",
        "from datasets import Dataset",
        "",
        "with open(PAIRS_SRC) as f:",
        "    all_pairs = json.load(f)",
        "rng = np.random.default_rng(CFG['random_seed'])",
        "",
        "# Stratified sampling: 50 gsm8k + 50 simpleqa",
        "gsm = [p for p in all_pairs if p['src'] == 'gsm8k']",
        "sqa = [p for p in all_pairs if p['src'] == 'simpleqa']",
        "rng.shuffle(gsm); rng.shuffle(sqa)",
        "selected = gsm[:50] + sqa[:50]",
        "rng.shuffle(selected)",
        "print(f'Selected: {len(selected)} prompts (50 gsm8k + 50 simpleqa)')",
        "",
        "# PRE-TEMPLATE with <think> prefix (lesson from nb42)",
        "prompts_templated = []",
        "for p in selected:",
        "    messages = [{'role': 'user', 'content': p['prompt']}]",
        "    text = tok.apply_chat_template(messages, tokenize=False,",
        "                                   add_generation_prompt=True, enable_thinking=True)",
        "    prompts_templated.append({'prompt': text, 'src': p['src']})",
        "assert '<think>' in prompts_templated[0]['prompt'], 'Template missing <think>'",
        "ds = Dataset.from_list(prompts_templated)",
        "print(f'✓ {len(ds)} prompts chat-templated')",
    ]))
    cells.append(code([
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
        "    save_total_limit=CFG['save_total_limit'],",
        "    save_strategy='steps',",
        "    logging_steps=2,",
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
        "n_ckpts = n_eff_steps // CFG['save_steps']",
        "print(f'Total iters: {n_total_iters}')",
        "print(f'Effective grad steps: {n_eff_steps}')",
        "print(f'Expected checkpoints: {n_ckpts}')",
    ]))
    cells.append(code([
        "t0 = time.time()",
        "trainer.train()",
        "elapsed_min = (time.time() - t0) / 60",
        "print(f'\\n✓ GRPO complete in {elapsed_min:.1f} min')",
        "",
        "model.save_pretrained(str(OUT / 'lora_final'))",
        "(OUT / 'reward_log.json').write_text(json.dumps(_reward_log, indent=2))",
        "print(f'✓ saved lora_final + reward_log ({len(_reward_log)} entries)')",
    ]))

    # Phase 6 — analysis (binned by chunk)
    cells.append(md(["## Phase 6 — Analysis: trajectory + per-checkpoint validation"]))
    cells.append(code([
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "",
        "with open(OUT / 'reward_log.json') as f:",
        "    log = json.load(f)",
        "",
        "df_all = pd.DataFrame(log)",
        "df = pd.DataFrame([r for r in log if r['reason'] == 'ok'])",
        "n_ok = len(df); n_total = len(df_all); n_no_think = len(df_all) - n_ok",
        "print(f'Total reward calls: {n_total}')",
        "print(f'  ok: {n_ok} ({100*n_ok/n_total:.1f}%)')",
        "print(f'  no_think: {n_no_think} ({100*n_no_think/n_total:.1f}%)')",
        "print()",
        "print(df[['fg', 'rg', 'r']].describe().round(4))",
        "",
        "# Bin by 20 chunks for finer trajectory than pilot's 10",
        "n_chunks = 20",
        "df['chunk'] = pd.cut(df.index, bins=n_chunks, labels=range(n_chunks))",
        "agg = df.groupby('chunk', observed=True)[['fg', 'rg', 'r']].mean().round(4)",
        "print('\\n=== Trajectory by chunk (early → late) ===')",
        "print(agg)",
    ]))
    cells.append(code([
        "fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))",
        "chunks = list(range(n_chunks))",
        "for ax, col, color, title in zip(axes,",
        "                                  ['fg', 'rg', 'r'],",
        "                                  ['C0', 'C1', 'C2'],",
        "                                  ['FabricationGuard score', 'ReasonGuard score', 'Combined reward']):",
        "    vals = agg[col].values",
        "    ax.plot(chunks, vals, marker='o', color=color, markersize=6)",
        "    ax.set_xlabel('Training chunk (early → late, 20 chunks)'); ax.set_ylabel(col)",
        "    ax.set_title(f'{title} vs training progress')",
        "    ax.grid(alpha=0.3)",
        "    if col == 'r':",
        "        ax.axhline(vals[0], color='gray', linestyle='--', alpha=0.5, label='early baseline')",
        "        ax.legend()",
        "plt.tight_layout()",
        "plt.savefig(OUT / 'fig_reward_trajectory_full.png', dpi=150)",
        "plt.show()",
        "",
        "# Compare quartiles for cleaner reading",
        "df['quartile'] = pd.cut(df.index, bins=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])",
        "q_agg = df.groupby('quartile', observed=True)[['fg', 'rg', 'r']].agg(['mean', 'std']).round(4)",
        "print('\\n=== By quartile ===')",
        "print(q_agg)",
        "",
        "early_r = agg['r'].values[0]",
        "late_r = agg['r'].values[-1]",
        "delta_r = late_r - early_r",
        "print(f'\\nEarly reward (chunk 0): {early_r:.4f}')",
        "print(f'Late reward (chunk {n_chunks-1}): {late_r:.4f}')",
        "print(f'Delta:                  {delta_r:+.4f}')",
        "if delta_r > 0.04:  # tighter gate than pilot's 0.02 since N=125 vs 20",
        "    print('🟢 GRPO trained STRONGLY — reward improved beyond pilot baseline')",
        "elif delta_r > 0.02:",
        "    print('🟡 GRPO trained MARGINALLY — similar to pilot, may need more epochs')",
        "elif delta_r > 0:",
        "    print('🟡 GRPO weakly trained — direction correct but small effect')",
        "else:",
        "    print('🔴 GRPO regressed — possible reward hacking or training instability')",
    ]))

    # Phase 7 — verdict + push
    cells.append(md(["## Phase 7 — FINAL_VERDICT + push"]))
    cells.append(code([
        "verdict = {",
        "    'experiment': 'nb43 multi-probe GRPO FULL',",
        "    'pilot_baseline_nb42': {'fg_delta': -0.023, 'rg_delta': -0.037, 'reward_delta': 0.030, 'effective_steps': 20},",
        "    'n_prompts': CFG['n_prompts'],",
        "    'effective_steps': int(n_eff_steps),",
        "    'reward_early_chunk': float(early_r),",
        "    'reward_late_chunk':  float(late_r),",
        "    'reward_delta':       float(delta_r),",
        "    'fg_early': float(agg['fg'].values[0]),",
        "    'fg_late':  float(agg['fg'].values[-1]),",
        "    'rg_early': float(agg['rg'].values[0]),",
        "    'rg_late':  float(agg['rg'].values[-1]),",
        "    'no_think_rate': float(n_no_think / n_total),",
        "    'training_signal': (",
        "        'strong_positive' if delta_r > 0.04",
        "        else 'marginal_positive' if delta_r > 0.02",
        "        else 'weak_positive' if delta_r > 0",
        "        else 'negative_or_flat'",
        "    ),",
        "}",
        "(OUT / 'FINAL_VERDICT.json').write_text(json.dumps(verdict, indent=2))",
        "print(json.dumps(verdict, indent=2))",
    ]))
    cells.append(code([
        "api = HfApi()",
        "(OUT / 'README.md').write_text('''---",
        "license: apache-2.0",
        "tags: [grpo, multi-probe, mechanistic-interpretability, qwen36-27b]",
        "---",
        "",
        "# nb43 — Multi-Probe Orthogonal Reward GRPO (FULL)",
        "",
        "Scaled version of nb42 pilot. 100 prompts × 5 epochs × 4 candidates = 125 effective grad steps.",
        "",
        "Reward: `r(c) = -(0.5 · FG(c) + 0.5 · RG(c))` — multi-probe orthogonal continuous reward.",
        "",
        "Pilot (nb42, 20 steps): FG -0.023, RG -0.037, reward +0.030. This run scales 5×.",
        "",
        "## Methodology lineage",
        "- Activation Reward Models (Jul 2025) — extends to multi-axis probe ensemble",
        "- Linear Probe Penalties Reduce LLM Sycophancy (Dec 2024) — single-axis precursor",
        "- Multi-Objective GRPO Safe Alignment (Mar 2025) — uses interpretability-validated probes instead of trained RM",
        "- DeepSeek-R1 GRPO (Nature 2025) — methodology baseline",
        "- Qwen-Scope (Alibaba Apr 2026) — concurrent SAE infrastructure",
        "",
        "See FINAL_VERDICT.json for trajectory analysis.",
        "''')",
        "",
        "try:",
        "    api.upload_folder(folder_path=str(OUT), repo_id=CFG['output_repo'],",
        "                      repo_type='dataset', token=HF_TOKEN,",
        "                      commit_message='nb43 multi-probe GRPO full complete',",
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
        "- 🟢 `strong_positive` (delta > 0.04): clear signal — paper-3 head-to-head GRPO vs DPO submission",
        "- 🟡 `marginal_positive` (0.02-0.04): same as pilot — diminishing returns at this scale, methodology limit",
        "- 🟡 `weak_positive` (0-0.02): direction correct but tiny — needs more steps OR different reward weighting",
        "- 🔴 `negative_or_flat` or worse: reward hacking, KL too tight, or methodology issue",
        "",
        "Then run nb41 v3 forward-only with checkpoints from this run for cross-checkpoint phase transition analysis.",
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

    out_path = NOTEBOOKS_DIR / "nb43_multiprobe_grpo_full.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    print(f"✓ wrote {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
