"""
Builder for nb41_v2_grokking_forward_only_extended.ipynb

Forward-only grokking analysis on nb37 v2 EXTENDED checkpoints (10 checkpoints + base = 11 data points).

Differences from nb41 v1:
- Source: 37v2_multiprobe_dpo_extended/dpo_run/ (10 checkpoints across 200 steps)
  vs 37_multiprobe_dpo_full/dpo_run/ (only 3 checkpoints across 80 steps)
- Wider learning range: 0.69 → 0.46 (delta -0.23) vs v1's 0.69 → 0.66 (-0.04)
- Higher resolution: 11 data points should resolve phase-transition ratio CLEANLY
  (v1 ratio was 1.74 = ambiguous; v2 should land >2.0 if grokking, <1.5 if gradual)

Same methodology as nb41:
- Apply fix_qwen36_adapter_keys() to strip .language_model.
- Forward pass on (prompt + chosen) per checkpoint
- Capture L31/L55 at end-of-think
- Score with FabricationGuard + ReasonGuard probes
- Fresh-probe AUROC progression + ratio heuristic

Compute: 11 checkpoints × 20 forward × ~5s = ~18min total
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
        "# Notebook 41 v2 — Grokking Forward-Only EXTENDED",
        "",
        "Re-run nb41 with extended checkpoints from nb37 v2 (10 checkpoints across 200 steps vs 3 across 80 steps).",
        "",
        "**Resolves nb41 v1 ambiguity** (ratio=1.74, between gradual <1.5 and grokking >2.0).",
        "",
        "**Higher resolution + wider learning range**:",
        "- nb37 v1 trajectory: 0.69 → 0.66 (Δ=-0.04, mild)",
        "- nb37 v2 trajectory: 0.69 → 0.46 (Δ=-0.23, real DPO learning)",
        "- Checkpoints: steps 20/40/60/80/100/120/140/160/180/200 + base",
        "",
        "**Compute**: ~18min (11 ckpts × 20 forward passes × ~5s)",
        "",
        "**Drive**: `/content/drive/MyDrive/openinterp_runs/41v2_grokking_forward_extended/`",
    ]))

    # Phase 1
    cells.append(md(["## Phase 1 — Setup"]))
    cells.append(code([
        "from pathlib import Path",
        "import os, json, time, shutil",
        "import torch, numpy as np",
        "",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "DRIVE = Path('/content/drive/MyDrive')",
        "OUT = DRIVE / 'openinterp_runs' / '41v2_grokking_forward_extended'",
        "OUT.mkdir(parents=True, exist_ok=True)",
        "",
        "# Source: nb37 v2 extended checkpoints",
        "NB37_V2 = DRIVE / 'openinterp_runs' / '37v2_multiprobe_dpo_extended'",
        "DPO_RUN = NB37_V2 / 'dpo_run'",
        "PAIRS = NB37_V2 / 'pairs.json'",
        "print(f'OUT: {OUT}')",
        "print(f'nb37 v2 dpo_run exists: {DPO_RUN.exists()}')",
        "if DPO_RUN.exists():",
        "    ckpt_dirs = sorted([d for d in DPO_RUN.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')],",
        "                       key=lambda d: int(d.name.split('-')[1]))",
        "    print(f'  found {len(ckpt_dirs)} checkpoints')",
    ]))
    cells.append(code([
        "!pip install -q -U torchao",
        "!pip install -q -U transformers accelerate peft huggingface_hub safetensors scikit-learn joblib",
    ]))

    # Phase 2 — model + probes
    cells.append(md(["## Phase 2 — Qwen3.6-27B base + probes"]))
    cells.append(code([
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "from huggingface_hub import login, hf_hub_download, HfApi, create_repo",
        "import getpass, joblib",
        "",
        "CFG = {",
        "    'model_id':         'Qwen/Qwen3.6-27B',",
        "    'capture_layer_fg': 31,",
        "    'capture_layer_rg': 55,",
        "    'n_holdout':        20,",
        "    'random_seed':      42,",
        "    'fg_probe_repo':    'caiovicentino1/FabricationGuard-linearprobe-qwen36-27b',",
        "    'rg_probe_repo':    'caiovicentino1/ReasoningGuard-linearprobe-qwen36-27b',",
        "    'output_repo':      'caiovicentino1/openinterp-41v2-grokking-extended',",
        "    'gate_ratio':       2.0,",
        "}",
        "THINK_CLOSE_ID = 248069",
        "torch.manual_seed(CFG['random_seed']); np.random.seed(CFG['random_seed'])",
        "",
        "HF_TOKEN = os.environ.get('HF_TOKEN') or getpass.getpass('HF token: ')",
        "login(HF_TOKEN, add_to_git_credential=False)",
        "",
        "device = 'cuda'",
        "tok = AutoTokenizer.from_pretrained(CFG['model_id'])",
        "base_model = AutoModelForCausalLM.from_pretrained(",
        "    CFG['model_id'], torch_dtype=torch.bfloat16, device_map='auto',",
        ")",
        "base_model.eval()",
        "print(f'✓ Base loaded — {torch.cuda.get_device_name(0)}, layers={len(base_model.model.layers)}')",
        "",
        "fg_path = hf_hub_download(repo_id=CFG['fg_probe_repo'], filename='probe.joblib', repo_type='dataset')",
        "rg_path = hf_hub_download(repo_id=CFG['rg_probe_repo'], filename='probe.joblib', repo_type='dataset')",
        "fg_artifact = joblib.load(fg_path); rg_artifact = joblib.load(rg_path)",
        "fg_clf = fg_artifact['probe']; fg_scaler = fg_artifact['scaler']",
        "rg_clf = rg_artifact['probe']; rg_scaler = rg_artifact['scaler']",
        "if not hasattr(fg_clf, 'multi_class'): fg_clf.multi_class = 'auto'",
        "if not hasattr(rg_clf, 'multi_class'): rg_clf.multi_class = 'auto'",
        "",
        "def fg_score(activation):",
        "    x = activation.float().cpu().numpy().reshape(1, -1)",
        "    return float(fg_clf.predict_proba(fg_scaler.transform(x))[0, 1])",
        "def rg_score(activation):",
        "    x = activation.float().cpu().numpy().reshape(1, -1)",
        "    return float(rg_clf.predict_proba(rg_scaler.transform(x))[0, 1])",
        "print(f'✓ FG/RG probes loaded')",
    ]))

    # Phase 3 — fix keys + discover
    cells.append(md(["## Phase 3 — Fix keys + discover all checkpoints"]))
    cells.append(code([
        "from safetensors.torch import load_file, save_file",
        "from peft import PeftModel",
        "",
        "def fix_qwen36_adapter_keys(adapter_dir, fixed_dir):",
        "    src = adapter_dir / 'adapter_model.safetensors'",
        "    state = load_file(str(src))",
        "    fixed = {k.replace('.language_model.', '.'): v for k, v in state.items()}",
        "    fixed_dir.mkdir(parents=True, exist_ok=True)",
        "    save_file(fixed, str(fixed_dir / 'adapter_model.safetensors'))",
        "    shutil.copy(adapter_dir / 'adapter_config.json', fixed_dir / 'adapter_config.json')",
        "    return fixed_dir",
        "",
        "checkpoints = []",
        "for d in sorted(DPO_RUN.iterdir(), key=lambda x: int(x.name.split('-')[1]) if x.name.startswith('checkpoint-') else 0):",
        "    if d.is_dir() and d.name.startswith('checkpoint-') and (d / 'adapter_model.safetensors').exists():",
        "        step = int(d.name.split('-')[1])",
        "        checkpoints.append({'step': step, 'orig_dir': d})",
        "checkpoints.sort(key=lambda x: x['step'])",
        "",
        "fixed_root = OUT / 'fixed_adapters'",
        "for c in checkpoints:",
        "    fdir = fixed_root / f\"step-{c['step']}\"",
        "    fix_qwen36_adapter_keys(c['orig_dir'], fdir)",
        "    c['fixed_dir'] = fdir",
        "    print(f\"  step {c['step']}: fixed → {fdir}\")",
        "print(f'✓ {len(checkpoints)} checkpoints prepared')",
    ]))

    # Phase 4 — hooks + holdout
    cells.append(md(["## Phase 4 — Hooks + 20 hold-out prompts"]))
    cells.append(code([
        "captured = {}; _pos = {'pos': None}",
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
        "with open(PAIRS) as f:",
        "    all_pairs = json.load(f)",
        "rng = np.random.default_rng(CFG['random_seed'])",
        "indices = rng.choice(len(all_pairs), size=CFG['n_holdout'], replace=False)",
        "holdout = [all_pairs[i] for i in indices]",
        "src_dist = {}",
        "for p in holdout: src_dist[p['src']] = src_dist.get(p['src'], 0) + 1",
        "print(f'Hold-out: {len(holdout)}, sources: {src_dist}')",
    ]))

    # Phase 5 — main loop
    cells.append(md(["## Phase 5 — Forward-only score per checkpoint"]))
    cells.append(code([
        "from tqdm.auto import tqdm",
        "import gc",
        "",
        "def find_end_think_pos(token_ids):",
        "    ids = token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids)",
        "    for i in range(len(ids) - 1, -1, -1):",
        "        if ids[i] == THINK_CLOSE_ID: return i",
        "    return None",
        "",
        "def score_one(model, prompt, chosen):",
        "    messages = [{'role': 'user', 'content': prompt}]",
        "    user_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)",
        "    full_text = user_text + chosen + '<|im_end|>'",
        "    enc = tok(full_text, return_tensors='pt')",
        "    ids = enc['input_ids'].to(device)",
        "    end_pos = find_end_think_pos(ids[0])",
        "    if end_pos is None:",
        "        return {'fg': float('nan'), 'rg': float('nan'), 'end_pos': None, 'act_fg': None}",
        "    captured.clear(); _pos['pos'] = end_pos",
        "    with torch.no_grad():",
        "        _ = model(ids)",
        "    act_fg = captured.get(f'L{CFG[\"capture_layer_fg\"]}')",
        "    act_rg = captured.get(f'L{CFG[\"capture_layer_rg\"]}')",
        "    return {",
        "        'fg': fg_score(act_fg) if act_fg is not None else float('nan'),",
        "        'rg': rg_score(act_rg) if act_rg is not None else float('nan'),",
        "        'end_pos': end_pos,",
        "        'act_fg': act_fg.clone() if act_fg is not None else None,",
        "    }",
        "",
        "ckpt_seq = [{'step': 0, 'fixed_dir': None, 'name': 'base'}] + checkpoints",
        "print(f'Will score {len(ckpt_seq)} checkpoints × {CFG[\"n_holdout\"]} = {len(ckpt_seq) * CFG[\"n_holdout\"]} forward passes')",
    ]))
    cells.append(code([
        "results_path = OUT / 'forward_results.jsonl'",
        "acts_dir = OUT / 'acts'",
        "acts_dir.mkdir(exist_ok=True)",
        "",
        "done_keys = set()",
        "if results_path.exists():",
        "    with open(results_path) as f:",
        "        for line in f:",
        "            try: done_keys.add(f\"step{json.loads(line)['step']}_{json.loads(line)['pair_id']}\")",
        "            except: continue",
        "    print(f'Resume: {len(done_keys)} done')",
        "",
        "hook_handles = []",
        "for L in [CFG['capture_layer_fg'], CFG['capture_layer_rg']]:",
        "    h = get_layers(base_model)[L].register_forward_hook(make_hook(L))",
        "    hook_handles.append(h)",
        "",
        "for ck_i, ck in enumerate(ckpt_seq):",
        "    print(f'\\n=== Checkpoint {ck_i+1}/{len(ckpt_seq)}: step={ck[\"step\"]} ===')",
        "    if ck['step'] == 0:",
        "        active = base_model",
        "    else:",
        "        for h in hook_handles:",
        "            try: h.remove()",
        "            except: pass",
        "        hook_handles = []",
        "        try:",
        "            active = PeftModel.from_pretrained(base_model, str(ck['fixed_dir']), is_trainable=False)",
        "            active.eval()",
        "            print(f'  ✓ Loaded LoRA step {ck[\"step\"]}')",
        "        except Exception as e:",
        "            print(f'  ❌ Load failed: {e}'); continue",
        "        for L in [CFG['capture_layer_fg'], CFG['capture_layer_rg']]:",
        "            h = get_layers(active)[L].register_forward_hook(make_hook(L))",
        "            hook_handles.append(h)",
        "    ",
        "    for p in tqdm(holdout, desc=f'step{ck[\"step\"]}'):",
        "        key = f\"step{ck['step']}_{p['id']}\"",
        "        if key in done_keys: continue",
        "        try:",
        "            res = score_one(active, p['prompt'], p['chosen'])",
        "        except torch.cuda.OutOfMemoryError:",
        "            torch.cuda.empty_cache(); gc.collect()",
        "            print(f'OOM on {key}'); continue",
        "        if res.get('act_fg') is not None:",
        "            torch.save(res['act_fg'], acts_dir / f'{key}.pt')",
        "        record = {",
        "            'step': ck['step'], 'pair_id': p['id'], 'src': p['src'],",
        "            'fg': res['fg'], 'rg': res['rg'], 'end_pos': res['end_pos'],",
        "        }",
        "        with open(results_path, 'a') as f:",
        "            f.write(json.dumps(record) + '\\n')",
        "        done_keys.add(key)",
        "    ",
        "    if ck['step'] != 0 and isinstance(active, PeftModel):",
        "        for h in hook_handles:",
        "            try: h.remove()",
        "            except: pass",
        "        hook_handles = []",
        "        del active; torch.cuda.empty_cache(); gc.collect()",
        "        for L in [CFG['capture_layer_fg'], CFG['capture_layer_rg']]:",
        "            h = get_layers(base_model)[L].register_forward_hook(make_hook(L))",
        "            hook_handles.append(h)",
        "",
        "print('\\n✓ Phase 5 complete')",
    ]))

    # Phase 6 — analysis
    cells.append(md(["## Phase 6 — Analysis: 11-point trajectory"]))
    cells.append(code([
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.metrics import roc_auc_score",
        "",
        "with open(results_path) as f:",
        "    records = [json.loads(line) for line in f]",
        "df = pd.DataFrame(records)",
        "print(f'Records: {len(df)}, unique steps: {df[\"step\"].nunique()}')",
        "agg = df.groupby('step')[['fg', 'rg']].agg(['mean', 'std']).round(4)",
        "print(agg)",
        "",
        "fg_means = df.groupby('step')['fg'].mean().values",
        "fg_var = float(np.var(fg_means))",
        "print(f'\\nFG variance across steps: {fg_var:.6f}')",
        "if fg_var < 1e-8:",
        "    print('🔴 STILL identical — fix did not propagate')",
        "else:",
        "    print('🟢 Real differences — proceed')",
    ]))
    cells.append(code([
        "agg_r = df.groupby('step').agg({'fg': ['mean', 'std'], 'rg': ['mean', 'std']}).reset_index()",
        "steps = agg_r['step'].values",
        "fg_mean = agg_r[('fg', 'mean')].values; fg_std = agg_r[('fg', 'std')].values",
        "rg_mean = agg_r[('rg', 'mean')].values; rg_std = agg_r[('rg', 'std')].values",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))",
        "axes[0].errorbar(steps, fg_mean, yerr=fg_std, marker='o', capsize=4, label='FabricationGuard')",
        "axes[0].axhline(fg_mean[0], color='gray', linestyle='--', alpha=0.5, label='base')",
        "axes[0].set_xlabel('DPO step (extended training)'); axes[0].set_ylabel('Mean FG score')",
        "axes[0].set_title('FabricationGuard vs DPO step (11 checkpoints)')",
        "axes[0].legend(); axes[0].grid(alpha=0.3)",
        "axes[1].errorbar(steps, rg_mean, yerr=rg_std, marker='s', color='C1', capsize=4, label='ReasonGuard')",
        "axes[1].axhline(rg_mean[0], color='gray', linestyle='--', alpha=0.5)",
        "axes[1].set_xlabel('DPO step'); axes[1].set_ylabel('Mean RG score')",
        "axes[1].set_title('ReasonGuard vs DPO step (11 checkpoints)')",
        "axes[1].legend(); axes[1].grid(alpha=0.3)",
        "plt.tight_layout()",
        "plt.savefig(OUT / 'fig_probe_vs_step_extended.png', dpi=150)",
        "plt.show()",
    ]))
    cells.append(code([
        "def load_acts(step_filter):",
        "    acts = []",
        "    for r in records:",
        "        if r['step'] == step_filter:",
        "            p = acts_dir / f\"step{r['step']}_{r['pair_id']}.pt\"",
        "            if p.exists(): acts.append(torch.load(p).float().numpy())",
        "    return np.array(acts) if acts else None",
        "",
        "X_base = load_acts(0)",
        "final_step = max(df['step'].unique())",
        "X_final = load_acts(int(final_step))",
        "fresh_results = {}",
        "if X_base is not None and X_final is not None and len(X_base) >= 5 and len(X_final) >= 5:",
        "    X = np.vstack([X_base, X_final])",
        "    y = np.concatenate([np.zeros(len(X_base)), np.ones(len(X_final))])",
        "    clf = LogisticRegression(C=1.0, max_iter=2000)",
        "    clf.fit(X, y)",
        "    print(f'Fresh probe trained on {len(X_base)} base + {len(X_final)} final')",
        "    for s in sorted(df['step'].unique()):",
        "        Xs = load_acts(int(s))",
        "        if Xs is not None and len(Xs) > 0:",
        "            fresh_results[int(s)] = float(clf.predict_proba(Xs)[:, 1].mean())",
        "    print('\\nFresh probe P(final-like) per step:')",
        "    for s, v in fresh_results.items():",
        "        print(f'  step {s:3d}: {v:.3f}')",
    ]))
    cells.append(code([
        "if fresh_results and len(fresh_results) >= 5:",
        "    ks = sorted(fresh_results.keys())",
        "    vals = [fresh_results[k] for k in ks]",
        "    deltas = [abs(vals[i+1] - vals[i]) for i in range(len(vals)-1)]",
        "    max_d = max(deltas); avg_d = sum(deltas) / len(deltas)",
        "    ratio = max_d / max(avg_d, 1e-6)",
        "    transition_step = ks[deltas.index(max_d) + 1]",
        "    print(f'Max delta: {max_d:.4f} at step {transition_step}')",
        "    print(f'Avg delta: {avg_d:.4f}')",
        "    print(f'Max/avg ratio: {ratio:.2f}')",
        "    if ratio > CFG['gate_ratio']:",
        "        print('  🔴 PHASE TRANSITION SIGNAL — grokking-like')",
        "    elif ratio < 1.5:",
        "        print('  🟢 GRADUAL — no grokking')",
        "    else:",
        "        print('  🟡 AMBIGUOUS')",
        "    plt.figure(figsize=(8, 4.5))",
        "    plt.plot(ks, vals, marker='D', color='C2', markersize=8)",
        "    plt.axhline(0.5, color='gray', linestyle=':', label='chance')",
        "    plt.xlabel('DPO step'); plt.ylabel('Fresh-probe P(final-like)')",
        "    plt.title(f'Fresh probe progression (extended, ratio={ratio:.2f})')",
        "    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()",
        "    plt.savefig(OUT / 'fig_fresh_probe_extended.png', dpi=150)",
        "    plt.show()",
        "else:",
        "    ratio = float('nan')",
    ]))

    # Phase 7 — verdict + push
    cells.append(md(["## Phase 7 — FINAL_VERDICT + push"]))
    cells.append(code([
        "verdict = {",
        "    'experiment': 'nb41 v2 grokking forward-only EXTENDED',",
        "    'fix_applied': 'strip_language_model_from_lora_keys',",
        "    'n_checkpoints': int(df['step'].nunique()),",
        "    'training_loss_descent_v2': '-0.234 (step 0 → step 200)',",
        "    'fg_variance_across_steps': float(fg_var),",
        "    'fix_worked': bool(fg_var > 1e-8),",
        "    'probe_scores_per_step': {",
        "        str(int(s)): {",
        "            'fg_mean': float(df[df['step']==s]['fg'].mean()),",
        "            'rg_mean': float(df[df['step']==s]['rg'].mean()),",
        "            'n': int((df['step']==s).sum()),",
        "        } for s in sorted(df['step'].unique())",
        "    },",
        "    'fresh_probe_per_step': {str(s): v for s, v in (fresh_results or {}).items()},",
        "    'phase_transition_ratio': float(ratio) if ratio == ratio else None,",
        "    'grokking_signal': bool(ratio > CFG['gate_ratio']) if ratio == ratio else None,",
        "}",
        "(OUT / 'FINAL_VERDICT.json').write_text(json.dumps(verdict, indent=2))",
        "print(json.dumps(verdict, indent=2))",
    ]))
    cells.append(code([
        "api = HfApi()",
        "try: create_repo(CFG['output_repo'], repo_type='dataset', private=False, exist_ok=True, token=HF_TOKEN)",
        "except Exception as e: print(e)",
        "",
        "(OUT / 'README.md').write_text('''---",
        "license: apache-2.0",
        "tags: [grokking, dpo, probing, qwen36-27b, extended-training]",
        "---",
        "",
        "# nb41 v2 — Grokking forward-only on extended DPO checkpoints",
        "",
        "Resolves nb41 v1 ambiguity (ratio=1.74) using nb37 v2 extended training (10 checkpoints across 200 steps with -0.23 loss descent vs v1's 4 checkpoints across 80 steps with -0.04 descent).",
        "",
        "Methodology: forward-only on (prompt + chosen), capture L31/L55 at end-of-think, score with FG+RG probes, fresh-probe AUROC progression.",
        "",
        "Key fix: strip `.language_model.` from saved LoRA keys before `PeftModel.from_pretrained()` (Qwen3.6 PEFT-save bug).",
        "",
        "See FINAL_VERDICT.json.",
        "''')",
        "",
        "api.upload_folder(folder_path=str(OUT), repo_id=CFG['output_repo'],",
        "                  repo_type='dataset', token=HF_TOKEN,",
        "                  commit_message='nb41 v2 complete',",
        "                  allow_patterns=['README.md', 'FINAL_VERDICT.json', 'fig_*.png', 'forward_results.jsonl'])",
        "print(f'✓ pushed to https://huggingface.co/datasets/{CFG[\"output_repo\"]}')",
    ]))

    cells.append(md([
        "## Done",
        "",
        "11 data points spanning 0.69 → 0.46 loss range. **This should resolve grokking definitively**:",
        "- ratio > 2.0 → 🔴 phase transition signal — paper-3 'Grokking in DPO via probe-detected progress'",
        "- ratio < 1.5 → 🟢 gradual — paper writeup 'preference learning is non-grokking-like in this regime'",
        "- 1.5 < ratio < 2.0 → 🟡 still ambiguous (unlikely with 11 points; would need different methodology)",
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

    out_path = NOTEBOOKS_DIR / "nb41_v2_grokking_forward_extended.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    print(f"✓ wrote {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
