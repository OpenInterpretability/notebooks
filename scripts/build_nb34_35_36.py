#!/usr/bin/env python3
"""
Generate 3 portfolio notebooks: nb34, nb35, nb36.
All cell sources stored as list-of-lines to avoid quote escaping issues.
"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path('/Volumes/SSD Major/fish/openinterp-work/notebooks')
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def md(text):
    if isinstance(text, str):
        lines = text.rstrip('\n').split('\n')
    else:
        lines = list(text)
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [l + '\n' for l in lines],
    }


def code(text):
    if isinstance(text, str):
        lines = text.rstrip('\n').split('\n')
    else:
        lines = list(text)
    return {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': [l + '\n' for l in lines],
    }


def write_nb(cells, out):
    nb = {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11'},
            'colab': {'provenance': [], 'machine_shape': 'hm'},
            'accelerator': 'GPU',
        },
        'nbformat': 4,
        'nbformat_minor': 4,
    }
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f'Wrote {out.name}  ({out.stat().st_size/1024:.1f} KB, {len(cells)} cells)')


def drive_mount_block(nb_name):
    return [
        md(['## 0. Drive mount + checkpoint dir (non-negotiable)']),
        code([
            '# === DRIVE MOUNT — non-negotiable for any run >30min ===',
            'from pathlib import Path',
            'import os, sys',
            '',
            'try:',
            '    from google.colab import drive',
            '    drive.mount("/content/drive", force_remount=False)',
            'except Exception as e:',
            '    print(f"Drive mount FAILED: {e}"); raise',
            '',
            'DRIVE_ROOT = Path("/content/drive/MyDrive")',
            'assert DRIVE_ROOT.exists(), "Drive mount silently failed"',
            f'NB_NAME = "{nb_name}"',
            'OUT = DRIVE_ROOT / "openinterp_runs" / NB_NAME',
            'OUT.mkdir(parents=True, exist_ok=True)',
            '(OUT / "_dry_run.txt").write_text("drive mount OK")',
            'print(f"✓ Drive checkpoint dir: {OUT}")',
            'print(f"  Contents: {sorted(p.name for p in OUT.iterdir())}")',
        ]),
    ]


# ───────────────────────────────────────────────────────────────────
# NB 34 — ReasonGuard v0.2 multi-bench combined-train
# ───────────────────────────────────────────────────────────────────
def build_nb34():
    cells = []

    cells.append(md([
        '# ReasonGuard v0.2 — Multi-Bench Combined Training',
        '',
        '**Notebook 34 · OpenInterp · 2026-04-29**',
        '',
        'Trains the production ReasonGuard probe using **FabricationGuard methodology**: combined training set across all 3 reasoning benches (GSM8K + StrategyQA + MATH), single probe at L55/mid_think.',
        '',
        '## Why this exists',
        '',
        'ReasonGuard v0.1 (notebook 32) trained on GSM8K alone and **failed cross-bench** (AUROC 0.605 on StrategyQA vs 0.888 within). Hypothesis: domain-bound signal.',
        '',
        'FabricationGuard solved the same problem by training multi-bench from the start (TruthfulQA + HaluEval + MMLU + SimpleQA), achieving AUROC 0.882 cross-bench. v0.2 applies the same recipe to reasoning-faithfulness.',
        '',
        '## Inputs',
        '',
        '`rollouts.npz` from notebook 32 v2, persisted at:',
        '```',
        '/content/drive/MyDrive/openinterp_runs/32_reasoningguard_v2/rollouts.npz',
        '```',
        'Contains 650 rollouts × 12 (layer, position) combos of residuals + labels.',
        '',
        '## Output',
        '',
        '- Probe v0.2 (`probe.joblib` with combined-train weights)',
        '- Cross-bench AUROC table (per-bench held-out)',
        '- Comparison with v0.1 numbers',
        '- HF push to `caiovicentino1/ReasoningGuard-linearprobe-qwen36-27b` v0.2 branch',
        '',
        '## Compute',
        '',
        'CPU-only (sklearn). ~5 min on Colab free tier. No GPU needed.',
    ]))

    cells.extend(drive_mount_block('34_reasonguard_v0_2_multibench'))

    cells.append(md(['## 1. Setup']))
    cells.append(code([
        '%pip install -q -U scikit-learn numpy pandas matplotlib joblib huggingface_hub',
        'import numpy as np, json, joblib, time',
        'from pathlib import Path',
        'from sklearn.linear_model import LogisticRegressionCV',
        'from sklearn.preprocessing import StandardScaler',
        'from sklearn.model_selection import train_test_split, StratifiedKFold',
        'from sklearn.metrics import roc_auc_score, roc_curve',
        'import matplotlib.pyplot as plt',
        '',
        'CFG = {',
        '    "rollouts_path": str(DRIVE_ROOT / "openinterp_runs" / "32_reasoningguard_v2" / "rollouts.npz"),',
        '    "probe_layer":   55,',
        '    "probe_position":"mid_think",',
        '    "lr_C_sweep":    [0.001, 0.01, 0.1, 1.0, 10.0],',
        '    "random_seed":   42,',
        '    "train_size":    0.7,',
        '    "hf_results_repo": "caiovicentino1/ReasoningGuard-linearprobe-qwen36-27b",',
        '}',
        'print(json.dumps(CFG, indent=2, default=str))',
    ]))

    cells.append(md(['## 2. Load rollouts from notebook 32']))
    cells.append(code([
        'rollouts = np.load(CFG["rollouts_path"], allow_pickle=True)',
        'print("Available arrays:", sorted(rollouts.files)[:20], "...")',
        '',
        '# Reconstruct data dict — schema: y_<bench>, res_<bench>_<pos>_L<layer>',
        'benches = sorted({k.split("_",1)[1] for k in rollouts.files if k.startswith("y_")})',
        'print(f"Benches found: {benches}")',
        'for b in benches:',
        '    y = rollouts[f"y_{b}"]',
        '    print(f"  {b}: n={len(y)}, halu_rate={100*y.mean():.1f}%")',
        '',
        'L, POS = CFG["probe_layer"], CFG["probe_position"]',
        'data = {}',
        'for b in benches:',
        '    key = f"res_{b}_{POS}_L{L}"',
        '    if key not in rollouts.files:',
        '        print(f"  ⚠️  {key} missing — bench {b} skipped")',
        '        continue',
        '    data[b] = {"X": rollouts[key], "y": rollouts[f"y_{b}"]}',
    ]))

    cells.append(md(['## 3. Combined-train probe (FabricationGuard methodology)']))
    cells.append(code([
        '# 70/30 split per bench, stratified by label',
        'splits = {}',
        'for b, d in data.items():',
        '    if len(np.unique(d["y"])) < 2:',
        '        print(f"  {b}: only one class — skipping"); continue',
        '    idx_tr, idx_te = train_test_split(',
        '        np.arange(len(d["y"])), test_size=1-CFG["train_size"],',
        '        stratify=d["y"], random_state=CFG["random_seed"])',
        '    splits[b] = {"tr": idx_tr, "te": idx_te}',
        '    print(f"  {b}: {len(idx_tr)} train, {len(idx_te)} test")',
        '',
        'X_train_combined = np.concatenate([data[b]["X"][splits[b]["tr"]] for b in splits])',
        'y_train_combined = np.concatenate([data[b]["y"][splits[b]["tr"]] for b in splits])',
        'print(f"\\nCombined train: n={len(y_train_combined)}, halu_rate={100*y_train_combined.mean():.1f}%")',
        '',
        'scaler = StandardScaler().fit(X_train_combined)',
        'clf = LogisticRegressionCV(',
        '    Cs=CFG["lr_C_sweep"], cv=5, penalty="l2", solver="lbfgs",',
        '    max_iter=2000, scoring="roc_auc", n_jobs=-1, refit=True,',
        ').fit(scaler.transform(X_train_combined), y_train_combined)',
        'print(f"Best C: {float(clf.C_[0])}")',
    ]))

    cells.append(md(['## 4. Per-bench held-out evaluation']))
    cells.append(code([
        'results = {}',
        'for b, sp in splits.items():',
        '    X_te, y_te = data[b]["X"][sp["te"]], data[b]["y"][sp["te"]]',
        '    scores = clf.predict_proba(scaler.transform(X_te))[:, 1]',
        '    auc = roc_auc_score(y_te, scores)',
        '    results[b] = {"auroc": float(auc), "n_test": len(y_te)}',
        '    badge = "✅" if auc >= 0.70 else "🟡" if auc >= 0.60 else "❌"',
        '    print(f"  {b:12s}: AUROC = {auc:.3f} (n={len(y_te)})  {badge}")',
        '',
        '# Comparison with v0.1',
        'v01 = {"gsm8k_within": 0.888, "strategyqa_cross": 0.605}',
        'print("\\n=== v0.1 vs v0.2 ===")',
        'print(f"  v0.1 GSM8K within (single-bench train):     {v01[\'gsm8k_within\']:.3f}")',
        'print(f"  v0.1 StrategyQA cross (single-bench train): {v01[\'strategyqa_cross\']:.3f}")',
        'print(f"  v0.2 GSM8K held-out (combined-train):       {results.get(\'gsm8k\', {}).get(\'auroc\', float(\'nan\')):.3f}")',
        'print(f"  v0.2 StrategyQA held-out (combined-train):  {results.get(\'strategyqa\', {}).get(\'auroc\', float(\'nan\')):.3f}")',
        'if "math" in results:',
        '    print(f"  v0.2 MATH held-out (combined-train):        {results[\'math\'][\'auroc\']:.3f}")',
        '',
        '# Verdict',
        'all_aurocs = [r["auroc"] for r in results.values()]',
        'mean_auc = np.mean(all_aurocs)',
        'all_passed = all(a >= 0.70 for a in all_aurocs)',
        'print(f"\\nMean AUROC: {mean_auc:.3f}")',
        'print(f"All benches ≥ 0.70: {\'✅\' if all_passed else \'❌\'}")',
    ]))

    cells.append(md(['## 5. Save probe v0.2 + push to HF']))
    cells.append(code([
        'joblib.dump({',
        '    "probe": clf, "scaler": scaler,',
        '    "layer": CFG["probe_layer"], "position": CFG["probe_position"],',
        '    "C": float(clf.C_[0]),',
        '    "training": "combined-bench (gsm8k + strategyqa + math)",',
        '    "version": "v0.2",',
        '}, OUT / "probe_v02.joblib")',
        '',
        'verdict = {',
        '    "version": "v0.2",',
        '    "date": time.strftime("%Y-%m-%d"),',
        '    "methodology": "FabricationGuard-style combined-bench training",',
        '    "training_benches": list(splits.keys()),',
        '    "per_bench_auroc": results,',
        '    "mean_auroc": float(np.mean([r["auroc"] for r in results.values()])),',
        '    "all_passed_70": all(r["auroc"] >= 0.70 for r in results.values()),',
        '    "comparison_v01": v01,',
        '    "config": {k: v for k, v in CFG.items() if k != "rollouts_path"},',
        '}',
        '(OUT / "verdict_v02.json").write_text(json.dumps(verdict, indent=2, default=str))',
        'print(json.dumps(verdict, indent=2))',
        '',
        'HF_TOKEN = os.environ.get("HF_TOKEN")',
        'if HF_TOKEN:',
        '    from huggingface_hub import HfApi',
        '    api = HfApi()',
        '    api.upload_folder(',
        '        folder_path=str(OUT), repo_id=CFG["hf_results_repo"],',
        '        repo_type="dataset", token=HF_TOKEN,',
        '        commit_message=f"ReasonGuard v0.2 — combined-train probe ({time.strftime(\'%Y-%m-%d\')})",',
        '        path_in_repo="v0.2/",',
        '    )',
        '    print(f"\\n✅ Pushed to https://huggingface.co/datasets/{CFG[\'hf_results_repo\']}/tree/main/v0.2")',
    ]))

    cells.append(md([
        '## 6. Honest interpretation',
        '',
        '**If all benches ≥ 0.70**: ReasonGuard v0.2 generalizes across reasoning domains. Ship as `live` on ProbeBench, replace v0.1 entry. Methodology validated: multi-bench training is necessary for cross-domain transfer in reasoning probes.',
        '',
        '**If some benches < 0.70**: partial generalization. Ship v0.2 with narrow-scope tagline like FabricationGuard\'s MMLU caveat (out-of-scope rather than failed). Honest registration of partial signal.',
        '',
        '**If most benches < 0.70**: methodology insufficient — single-layer + single-position is too restrictive. Pivot to multi-layer probe ensemble (v0.3 candidate).',
        '',
        'Either way, both v0.1 and v0.2 numbers stay published on ProbeBench. The framework honors honest negative results.',
    ]))

    write_nb(cells, NOTEBOOKS_DIR / '34_reasonguard_v0_2_multibench.ipynb')


# ───────────────────────────────────────────────────────────────────
# NB 35 — Multi-Probe Reward DPO POC
# ───────────────────────────────────────────────────────────────────
def build_nb35():
    cells = []

    cells.append(md([
        '# Multi-Probe Reward DPO — Qwen3.6-27B + FabricationGuard + ReasonGuard',
        '',
        '**Notebook 35 · OpenInterp · 2026-04-29**',
        '',
        'Proof-of-concept: fine-tune Qwen3.6-27B with **DPO using a combined multi-probe reward** built from two ProbeBench-registered probes:',
        '',
        '- **FabricationGuard** at L31/end_question (factual hallucination)',
        '- **ReasonGuard** at L55/mid_think (reasoning faithfulness)',
        '',
        '## Why multi-probe matters',
        '',
        'Goodfire RLFR (Apr 2026) proved single-probe RL works (-58% hallucination). Our extension: **two orthogonal probes simultaneously** — student must satisfy both, drastically harder to game than single-probe. Anti-Goodhart by orthogonal-objective construction, not just by training-time monitoring.',
        '',
        '## Architecture (Goodfire RLFR pattern, multi-probe)',
        '',
        '```',
        '         Student (LoRA-trainable)         Frozen base (probe scorer)',
        'prompt → Qwen3.6-27B + LoRA → gen   →    Qwen3.6-27B (no LoRA)',
        '                                              │',
        '                                              ▼',
        '                                       L31 + L55 residuals',
        '                                              │',
        '                                              ▼',
        '                                      FG probe + RG probe',
        '                                              │',
        '                                              ▼',
        '                                  reward = -(0.5·P_FG + 0.5·P_RG)',
        '                                              │',
        '                          ◀── DPO update via preference pairs',
        '```',
        '',
        '**Key**: gradient never flows through probes. Student can only influence reward by generating different tokens — not by manipulating activations directly.',
        '',
        '## Compute',
        '',
        '- 1× RTX PRO 6000 96GB or H100 80GB',
        '- ~2 hours wall-clock for the full POC',
        '- ~$10-15 on Colab Pro+ credits',
    ]))

    cells.extend(drive_mount_block('35_multiprobe_dpo_poc'))

    cells.append(code(['!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU"']))

    cells.append(md(['## 1. Setup']))
    cells.append(code([
        '# IMPORTANT: torchao must be >=0.16.0 for peft compatibility on Qwen3.6-27B',
        '%pip install -q -U torchao transformers accelerate peft trl datasets safetensors huggingface_hub',
        '%pip install -q -U scikit-learn matplotlib joblib sentencepiece protobuf tqdm',
        'print("installs done — RESTART RUNTIME if peft import fails, then run all cells again")',
    ]))

    cells.append(code([
        'import os, json, time, math, gc',
        'from pathlib import Path',
        'from typing import Optional, Tuple, List',
        'from contextlib import contextmanager',
        'import numpy as np, pandas as pd',
        'import torch',
        'import torch.nn.functional as F',
        'from tqdm.auto import tqdm',
        'import joblib',
        'from huggingface_hub import login, hf_hub_download, HfApi',
        'from datasets import load_dataset, Dataset',
        'from sklearn.linear_model import LogisticRegressionCV',
        'from sklearn.preprocessing import StandardScaler',
        'from sklearn.model_selection import train_test_split',
        'from sklearn.metrics import roc_auc_score',
        'import matplotlib.pyplot as plt',
        '',
        'CFG = {',
        '    "model":            "Qwen/Qwen3.6-27B",',
        '    "fg_repo":          "caiovicentino1/FabricationGuard-linearprobe-qwen36-27b",',
        '    "rg_repo":          "caiovicentino1/ReasoningGuard-linearprobe-qwen36-27b",',
        '    "fg_layer":         31,',
        '    "rg_layer":         55,',
        '    "probe_layers":     [31, 55],',
        '    "corpus_n_simpleqa": 25,',
        '    "corpus_n_gsm8k":   25,',
        '    "cands_per_q":      4,',
        '    "eval_n_simpleqa":  50,',
        '    "eval_n_gsm8k":     50,',
        '    "reward_alpha":     [0.5, 0.5],',
        '    "lora_r":           16,',
        '    "lora_alpha":       32,',
        '    "dpo_lr":           5e-6,',
        '    "dpo_beta":         0.1,',
        '    "random_seed":      42,',
        '    "output_repo":      "caiovicentino1/openinterp-multiprobe-dpo-poc",',
        '}',
        'THINK_OPEN_ID  = 248068',
        'THINK_CLOSE_ID = 248069',
        '',
        'torch.manual_seed(CFG["random_seed"]); np.random.seed(CFG["random_seed"])',
        'import random; random.seed(CFG["random_seed"])',
        '',
        'HF_TOKEN = os.environ.get("HF_TOKEN")',
        'if HF_TOKEN is None:',
        '    import getpass; HF_TOKEN = getpass.getpass("HF token (write scope): ")',
        'login(HF_TOKEN, add_to_git_credential=False)',
        '',
        'device = "cuda"; assert torch.cuda.is_available()',
        'print(f"CUDA: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")',
    ]))

    cells.append(md(['## 2. Load Qwen3.6-27B + register hooks at L31, L55']))
    cells.append(code([
        'from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM',
        '',
        'print(f"Loading {CFG[\'model\']} ...")',
        'tok = AutoTokenizer.from_pretrained(CFG["model"], trust_remote_code=True)',
        'try:',
        '    model = AutoModelForImageTextToText.from_pretrained(',
        '        CFG["model"], dtype=torch.bfloat16, attn_implementation="sdpa",',
        '        device_map={"":device}, trust_remote_code=True)',
        'except Exception:',
        '    model = AutoModelForCausalLM.from_pretrained(',
        '        CFG["model"], dtype=torch.bfloat16, attn_implementation="sdpa",',
        '        device_map={"":device}, trust_remote_code=True)',
        'model.eval()',
        'for p in model.parameters(): p.requires_grad_(False)',
        '',
        'def _block_list(m):',
        '    candidates = [m, getattr(m,"model",None)]',
        '    for s in candidates:',
        '        if s is None: continue',
        '        for path in [("model","language_model","layers"),("language_model","layers"),("model","layers"),("layers",)]:',
        '            cur=s; ok=True',
        '            for p in path:',
        '                if hasattr(cur,p): cur=getattr(cur,p)',
        '                else: ok=False; break',
        '            if ok and hasattr(cur,"__getitem__"): return cur',
        '    raise RuntimeError("layers not found")',
        '',
        'blocks = _block_list(model)',
        '',
        'class MultiLayerHook:',
        '    def __init__(self, blocks, layers):',
        '        self.bufs = {l:None for l in layers}; self.handles=[]',
        '        for l in layers:',
        '            self.handles.append(blocks[l].register_forward_hook(self._make(l)))',
        '    def _make(self, l):',
        '        def hook(_m,_i,out):',
        '            h = out[0] if isinstance(out,tuple) else out',
        '            self.bufs[l] = h.detach()',
        '        return hook',
        '    def pop(self, l):',
        '        b = self.bufs[l]; self.bufs[l]=None; return b',
        '',
        'ml_hook = MultiLayerHook(blocks, CFG["probe_layers"])',
        'print(f"✓ Hooks registered at layers {CFG[\'probe_layers\']}")',
    ]))

    cells.append(md(['## 3. Load both probes']))
    cells.append(code([
        'def load_probe_bundle(repo, fname="probe.joblib"):',
        '    p = hf_hub_download(repo, repo_type="dataset", filename=fname)',
        '    obj = joblib.load(p)',
        '    return obj["probe"], obj["scaler"]',
        '',
        'fg_probe, fg_scaler = load_probe_bundle(CFG["fg_repo"])',
        'rg_probe, rg_scaler = load_probe_bundle(CFG["rg_repo"])',
        'print(f"✓ FabricationGuard L{CFG[\'fg_layer\']}/end_question loaded")',
        'print(f"✓ ReasonGuard       L{CFG[\'rg_layer\']}/mid_think loaded")',
    ]))

    cells.append(md(['## 4. Apply LoRA adapter']))
    cells.append(code([
        'from peft import LoraConfig, get_peft_model, TaskType',
        '',
        'lora_cfg = LoraConfig(',
        '    r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=0.05,',
        '    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],',
        '    task_type=TaskType.CAUSAL_LM, bias="none",',
        ')',
        'model = get_peft_model(model, lora_cfg)',
        'model.print_trainable_parameters()',
    ]))

    cells.append(md(['## 5. Multi-probe scoring (frozen base via disable_adapter)']))
    cells.append(code([
        '@torch.no_grad()',
        'def fg_score(question: str, answer: str) -> float:',
        '    prompt = f"Q: {question}\\nA: {answer}"',
        '    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)',
        '    n = int(enc["attention_mask"].sum().item())',
        '    with model.disable_adapter():',
        '        _ = model(**enc)',
        '    h = ml_hook.pop(CFG["fg_layer"])[0, n-1].float().cpu().numpy()',
        '    return float(fg_probe.predict_proba(fg_scaler.transform([h]))[0, list(fg_probe.classes_).index(1)])',
        '',
        '@torch.no_grad()',
        'def rg_score(question: str, full_answer: str) -> Optional[float]:',
        '    chat = [{"role":"user","content":question}]',
        '    prefix = tok.apply_chat_template(chat, tokenize=False,',
        '                                     add_generation_prompt=True, enable_thinking=True)',
        '    enc = tok(prefix + full_answer, return_tensors="pt", truncation=True, max_length=2048).to(device)',
        '    ids = enc["input_ids"][0].tolist()',
        '    op = next((i for i,t in enumerate(ids) if t == THINK_OPEN_ID), None)',
        '    cl = next((i for i,t in enumerate(ids) if t == THINK_CLOSE_ID), None)',
        '    if op is None or cl is None or cl <= op + 5: return None',
        '    mid = (op + cl) // 2',
        '    with model.disable_adapter():',
        '        _ = model(**enc)',
        '    h = ml_hook.pop(CFG["rg_layer"])[0, mid].float().cpu().numpy()',
        '    return float(rg_probe.predict_proba(rg_scaler.transform([h]))[0, list(rg_probe.classes_).index(1)])',
        '',
        'def combined_reward(q: str, a: str):',
        '    fg = fg_score(q, a)',
        '    rg = rg_score(q, a)',
        '    if rg is not None:',
        '        comb = CFG["reward_alpha"][0]*fg + CFG["reward_alpha"][1]*rg',
        '    else:',
        '        comb = fg',
        '    return {"fg":fg,"rg":rg,"combined":comb,"reward":-comb,"has_think":rg is not None}',
    ]))

    cells.append(md(['## 6. Mixed corpus + generation helper']))
    cells.append(code([
        'sqa = load_dataset("basicv8vc/SimpleQA", split="test").shuffle(seed=42).select(range(CFG["corpus_n_simpleqa"]))',
        'gsm = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42).select(range(CFG["corpus_n_gsm8k"]))',
        'mixed = ([{"q":ex["problem"], "src":"simpleqa"} for ex in sqa] +',
        '         [{"q":ex["question"],"src":"gsm8k"}    for ex in gsm])',
        'np.random.default_rng(42).shuffle(mixed)',
        'print(f"Mixed corpus: {len(mixed)} questions ({sum(1 for x in mixed if x[\'src\']==\'simpleqa\')} factual, {sum(1 for x in mixed if x[\'src\']==\'gsm8k\')} math)")',
        '',
        '@torch.no_grad()',
        'def gen_one(question: str, temp: float = 0.7, max_new: int = 256) -> str:',
        '    msgs = [{"role":"user","content":question}]',
        '    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)',
        '    enc = tok(txt, return_tensors="pt").to(device)',
        '    out = model.generate(**enc, max_new_tokens=max_new, do_sample=(temp>0),',
        '                         temperature=temp if temp>0 else 1.0, top_p=0.9,',
        '                         pad_token_id=tok.pad_token_id or tok.eos_token_id)',
        '    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()',
    ]))

    cells.append(md(['## 7. Build DPO pairs (multi-probe reward picks chosen/rejected)']))
    cells.append(code([
        'pairs, telemetry = [], []',
        'for ex in tqdm(mixed, desc="build DPO pairs"):',
        '    q = ex["q"]',
        '    cands = [gen_one(q, temp=0.7) for _ in range(CFG["cands_per_q"])]',
        '    rs = [combined_reward(q, c) for c in cands]',
        '    best_i  = int(np.argmin([r["combined"] for r in rs]))',
        '    worst_i = int(np.argmax([r["combined"] for r in rs]))',
        '    pairs.append({"prompt": q, "chosen": cands[best_i], "rejected": cands[worst_i]})',
        '    telemetry.append({',
        '        "q": q, "src": ex["src"],',
        '        "fg_best": rs[best_i]["fg"], "fg_worst": rs[worst_i]["fg"],',
        '        "rg_best": rs[best_i]["rg"], "rg_worst": rs[worst_i]["rg"],',
        '        "combined_gap": rs[worst_i]["combined"] - rs[best_i]["combined"],',
        '        "has_think_rate": sum(1 for r in rs if r["has_think"]) / len(rs),',
        '    })',
        'df_t = pd.DataFrame(telemetry)',
        'df_t.to_csv(OUT / "pair_telemetry.csv", index=False)',
        'print(f"\\nMean combined_gap: {df_t[\'combined_gap\'].mean():.3f}")',
        'print(df_t.groupby("src")["combined_gap"].mean())',
        '',
        '# Probe orthogonality check',
        'df_both = df_t.dropna(subset=["rg_best"])',
        'if len(df_both) > 5:',
        '    rho_best  = np.corrcoef(df_both["fg_best"],  df_both["rg_best"])[0,1]',
        '    rho_worst = np.corrcoef(df_both["fg_worst"], df_both["rg_worst"])[0,1]',
        '    print(f"\\nFG-RG Pearson: best={rho_best:+.3f}, worst={rho_worst:+.3f}")',
        '    print(f"  ({\'orthogonal ✅\' if abs(rho_best)<0.4 else \'correlated ⚠️\'})")',
    ]))

    cells.append(md(['## 8. DPO training']))
    cells.append(code([
        'from trl import DPOTrainer, DPOConfig',
        '',
        'ds = Dataset.from_list(pairs).train_test_split(test_size=0.2, seed=CFG["random_seed"])',
        '',
        'dpo_cfg = DPOConfig(',
        '    output_dir=str(OUT / "dpo_run"),',
        '    num_train_epochs=1, per_device_train_batch_size=1,',
        '    gradient_accumulation_steps=4, learning_rate=CFG["dpo_lr"], beta=CFG["dpo_beta"],',
        '    save_steps=20, logging_steps=2, bf16=True, report_to="none",',
        '    max_length=1024, max_prompt_length=512, save_strategy="steps", save_total_limit=2,',
        ')',
        'trainer = DPOTrainer(model=model, args=dpo_cfg,',
        '                     train_dataset=ds["train"], eval_dataset=ds["test"],',
        '                     processing_class=tok)',
        'trainer.train()',
        'trainer.save_model(str(OUT / "lora_final"))',
        'print(f"✓ LoRA saved to {OUT / \'lora_final\'}")',
    ]))

    cells.append(md(['## 9. Evaluation — base vs student on held-out 100 queries']))
    cells.append(code([
        'sqa_e = load_dataset("basicv8vc/SimpleQA", split="test").shuffle(seed=99).select(range(CFG["eval_n_simpleqa"]))',
        'gsm_e = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=99).select(range(CFG["eval_n_gsm8k"]))',
        'eval_qs = ([(ex["problem"],"simpleqa")   for ex in sqa_e] +',
        '           [(ex["question"],"gsm8k")     for ex in gsm_e])',
        '',
        'results = []',
        'for q, src in tqdm(eval_qs, desc="eval"):',
        '    with model.disable_adapter():',
        '        ans_b = gen_one(q, temp=0.0, max_new=512 if src=="gsm8k" else 256)',
        '    ans_s = gen_one(q, temp=0.0, max_new=512 if src=="gsm8k" else 256)',
        '    rb, rs_ = combined_reward(q, ans_b), combined_reward(q, ans_s)',
        '    results.append({"q":q,"src":src,',
        '                    "fg_base":rb["fg"],"fg_stud":rs_["fg"],',
        '                    "rg_base":rb["rg"],"rg_stud":rs_["rg"],',
        '                    "comb_base":rb["combined"],"comb_stud":rs_["combined"]})',
        '',
        'df_e = pd.DataFrame(results); df_e.to_csv(OUT / "eval_results.csv", index=False)',
        '',
        'print("\\n" + "="*60)',
        'print("  Multi-Probe DPO POC — Verdict")',
        'print("="*60)',
        'for src in ["simpleqa","gsm8k"]:',
        '    sub = df_e[df_e.src==src]',
        '    fg_red = (sub["fg_base"].mean()-sub["fg_stud"].mean())/sub["fg_base"].mean()*100',
        '    rg_sub = sub.dropna(subset=["rg_base","rg_stud"])',
        '    rg_red = (rg_sub["rg_base"].mean()-rg_sub["rg_stud"].mean())/rg_sub["rg_base"].mean()*100 if len(rg_sub)>0 else float("nan")',
        '    cm_red = (sub["comb_base"].mean()-sub["comb_stud"].mean())/sub["comb_base"].mean()*100',
        '    print(f"  {src:10s}: FG -{fg_red:.1f}% · RG -{rg_red:.1f}% · combined -{cm_red:.1f}%")',
        'print("="*60)',
    ]))

    cells.append(md(['## 10. HF push']))
    cells.append(code([
        'verdict = {',
        '    "date": time.strftime("%Y-%m-%d"),',
        '    "config": {k:v for k,v in CFG.items() if k!="output_repo"},',
        '    "pairs_n": len(pairs),',
        '    "eval_n": len(df_e),',
        '    "mean_combined_gap_build": float(df_t["combined_gap"].mean()),',
        '    "fg_rg_pearson_orthogonality": float(np.corrcoef(df_both["fg_best"], df_both["rg_best"])[0,1]) if len(df_both)>5 else None,',
        '    "overall_combined_reduction": float((df_e["comb_base"].mean()-df_e["comb_stud"].mean())/df_e["comb_base"].mean()*100),',
        '}',
        '(OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))',
        '',
        'api = HfApi()',
        'api.upload_folder(folder_path=str(OUT), repo_id=CFG["output_repo"],',
        '                  repo_type="dataset", token=HF_TOKEN,',
        '                  commit_message=f"Multi-Probe DPO POC results @ {time.strftime(\'%Y-%m-%d %H:%M\')}")',
        'print(f"\\n✅ Pushed to https://huggingface.co/datasets/{CFG[\'output_repo\']}")',
    ]))

    cells.append(md([
        '## 11. Honest interpretation',
        '',
        'Run **notebook 36 (Anti-Goodhart Fresh Probe Validation)** for the final verdict — fresh probes trained on student-generated samples will tell us whether the reduction is genuine or evasion.',
        '',
        'If real: first OSS demonstration of multi-probe-reward DPO on 27B+ model. Direct extension of Goodfire RLFR (single-probe, -58% halu) to multi-probe orthogonal-objective design.',
    ]))

    write_nb(cells, NOTEBOOKS_DIR / '35_multiprobe_dpo_poc.ipynb')


# ───────────────────────────────────────────────────────────────────
# NB 36 — Anti-Goodhart Fresh Probe Validation
# ───────────────────────────────────────────────────────────────────
def build_nb36():
    cells = []

    cells.append(md([
        '# Anti-Goodhart Fresh-Probe Validation',
        '',
        '**Notebook 36 · OpenInterp · 2026-04-29**',
        '',
        'After running multi-probe DPO (notebook 35), this notebook performs the **anti-Goodhart final check**: train a FRESH probe on student-generated samples and compare AUROCs.',
        '',
        '## The 4-quadrant test',
        '',
        '| Halu rate change | Original probe AUROC | Fresh probe AUROC | Interpretation |',
        '|---|---|---|---|',
        '| ↓ caiu | qualquer | **≥ 0.80** | ✅ Real improvement — halu actually dropped, signal still detectable |',
        '| → mantém | ↓ caiu | ↓ caiu | ❌ **Goodhart confirmed** — student evades probe direction |',
        '| → mantém | ↓ caiu | ≥ 0.80 | 🟡 Partial evasion — signal moved off original direction |',
        '| ↓ caiu | qualquer | < 0.80 | 🟠 Signal eroded — improvement real but probe-relevant signal weakened |',
        '',
        'The killer test: if **fresh probe AUROC ≥ 0.80**, residual stream still contains discriminative info about hallucination. DPO didn\'t destroy signal — it just shifted distribution toward fewer hallucinations. Desired outcome.',
        '',
        'If **fresh probe AUROC < 0.65**, student learned to evade. Abort.',
        '',
        '## Compute',
        '',
        '- 1× RTX PRO 6000 96GB (or any GPU that fits Qwen3.6-27B BF16)',
        '- ~30 min wall-clock (160 generations + sklearn fit)',
    ]))

    cells.extend(drive_mount_block('36_antigoodhart_validation'))

    cells.append(md(['## 1. Setup']))
    cells.append(code([
        '%pip install -q -U transformers accelerate peft datasets safetensors huggingface_hub',
        '%pip install -q -U scikit-learn matplotlib joblib tqdm',
        'import os, json, time, re',
        'from pathlib import Path',
        'from typing import Optional',
        'import numpy as np, pandas as pd, joblib',
        'import torch',
        'from tqdm.auto import tqdm',
        'from huggingface_hub import login, hf_hub_download, HfApi',
        'from datasets import load_dataset',
        'from sklearn.linear_model import LogisticRegressionCV',
        'from sklearn.preprocessing import StandardScaler',
        'from sklearn.model_selection import train_test_split',
        'from sklearn.metrics import roc_auc_score',
        '',
        'CFG = {',
        '    "model":         "Qwen/Qwen3.6-27B",',
        '    "lora_path":     str(DRIVE_ROOT / "openinterp_runs" / "35_multiprobe_dpo_poc" / "lora_final"),',
        '    "fg_repo":       "caiovicentino1/FabricationGuard-linearprobe-qwen36-27b",',
        '    "rg_repo":       "caiovicentino1/ReasoningGuard-linearprobe-qwen36-27b",',
        '    "fg_layer":      31,',
        '    "rg_layer":      55,',
        '    "probe_layers":  [31, 55],',
        '    "fresh_n_simpleqa": 80,',
        '    "fresh_n_gsm8k":    80,',
        '    "random_seed":   777,',
        '    "output_repo":   "caiovicentino1/openinterp-multiprobe-dpo-poc",',
        '}',
        'THINK_OPEN_ID, THINK_CLOSE_ID = 248068, 248069',
        '',
        'torch.manual_seed(CFG["random_seed"]); np.random.seed(CFG["random_seed"])',
        '',
        'HF_TOKEN = os.environ.get("HF_TOKEN")',
        'if HF_TOKEN is None:',
        '    import getpass; HF_TOKEN = getpass.getpass("HF token: ")',
        'login(HF_TOKEN, add_to_git_credential=False)',
        'device = "cuda"; assert torch.cuda.is_available()',
    ]))

    cells.append(md(['## 2. Load model + LoRA from notebook 35']))
    cells.append(code([
        'from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM',
        'from peft import PeftModel',
        '',
        'print(f"Loading {CFG[\'model\']} ...")',
        'tok = AutoTokenizer.from_pretrained(CFG["model"], trust_remote_code=True)',
        'try:',
        '    base = AutoModelForImageTextToText.from_pretrained(',
        '        CFG["model"], dtype=torch.bfloat16, attn_implementation="sdpa",',
        '        device_map={"":device}, trust_remote_code=True)',
        'except Exception:',
        '    base = AutoModelForCausalLM.from_pretrained(',
        '        CFG["model"], dtype=torch.bfloat16, attn_implementation="sdpa",',
        '        device_map={"":device}, trust_remote_code=True)',
        'base.eval()',
        'for p in base.parameters(): p.requires_grad_(False)',
        '',
        'print(f"Loading LoRA from {CFG[\'lora_path\']}")',
        'model = PeftModel.from_pretrained(base, CFG["lora_path"])',
        'model.eval()',
        '',
        'def _block_list(m):',
        '    candidates = [m, getattr(m,"model",None), getattr(m,"base_model",None),',
        '                  getattr(getattr(m,"base_model",None),"model",None) if hasattr(m,"base_model") else None]',
        '    for s in candidates:',
        '        if s is None: continue',
        '        for path in [("model","language_model","layers"),("language_model","layers"),("model","layers"),("layers",)]:',
        '            cur=s; ok=True',
        '            for p in path:',
        '                if hasattr(cur,p): cur=getattr(cur,p)',
        '                else: ok=False; break',
        '            if ok and hasattr(cur,"__getitem__"): return cur',
        '    raise RuntimeError("layers not found")',
        '',
        'blocks = _block_list(model)',
        '',
        'class MultiLayerHook:',
        '    def __init__(self, blocks, layers):',
        '        self.bufs = {l:None for l in layers}; self.handles=[]',
        '        for l in layers:',
        '            self.handles.append(blocks[l].register_forward_hook(self._make(l)))',
        '    def _make(self, l):',
        '        def hook(_m,_i,out):',
        '            h = out[0] if isinstance(out,tuple) else out',
        '            self.bufs[l] = h.detach()',
        '        return hook',
        '    def pop(self, l):',
        '        b = self.bufs[l]; self.bufs[l]=None; return b',
        '',
        'ml_hook = MultiLayerHook(blocks, CFG["probe_layers"])',
        'print("✓ Hooks registered")',
    ]))

    cells.append(md(['## 3. Load original probes']))
    cells.append(code([
        'def load_probe(repo):',
        '    p = hf_hub_download(repo, repo_type="dataset", filename="probe.joblib")',
        '    obj = joblib.load(p)',
        '    return obj["probe"], obj["scaler"]',
        '',
        'fg_probe, fg_scaler = load_probe(CFG["fg_repo"])',
        'rg_probe, rg_scaler = load_probe(CFG["rg_repo"])',
        'print("✓ Original probes loaded")',
    ]))

    cells.append(md(['## 4. Helpers — generate + capture residuals']))
    cells.append(code([
        '@torch.no_grad()',
        'def gen_one(question: str, max_new: int = 256, mode: str = "student") -> str:',
        '    msgs = [{"role":"user","content":question}]',
        '    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)',
        '    enc = tok(txt, return_tensors="pt").to(device)',
        '    if mode == "base":',
        '        with model.disable_adapter():',
        '            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,',
        '                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)',
        '    else:',
        '        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,',
        '                             pad_token_id=tok.pad_token_id or tok.eos_token_id)',
        '    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()',
        '',
        '@torch.no_grad()',
        'def capture_l31(question, answer):',
        '    enc = tok(f"Q: {question}\\nA: {answer}", return_tensors="pt", truncation=True, max_length=1024).to(device)',
        '    n = int(enc["attention_mask"].sum().item())',
        '    with model.disable_adapter():',
        '        _ = model(**enc)',
        '    return ml_hook.pop(31)[0, n-1].float().cpu().numpy()',
        '',
        '@torch.no_grad()',
        'def capture_l55(question, full_answer):',
        '    chat = [{"role":"user","content":question}]',
        '    prefix = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=True)',
        '    enc = tok(prefix + full_answer, return_tensors="pt", truncation=True, max_length=2048).to(device)',
        '    ids = enc["input_ids"][0].tolist()',
        '    op = next((i for i,t in enumerate(ids) if t == THINK_OPEN_ID), None)',
        '    cl = next((i for i,t in enumerate(ids) if t == THINK_CLOSE_ID), None)',
        '    if op is None or cl is None or cl <= op + 5: return None',
        '    mid = (op + cl) // 2',
        '    with model.disable_adapter():',
        '        _ = model(**enc)',
        '    return ml_hook.pop(55)[0, mid].float().cpu().numpy()',
        '',
        'def normalize_answer(s):',
        '    return "".join(c.lower() for c in str(s) if c.isalnum() or c.isspace()).strip()',
        '',
        'NUMBER_RE = re.compile(r"-?\\d+(?:[.,]\\d+)?")',
        'def grade_gsm8k(gen, gold):',
        '    if "####" in gold:',
        '        try: gold_num = float(gold.split("####")[-1].strip().replace(",",""))',
        '        except: return False',
        '    else:',
        '        nums = NUMBER_RE.findall(gold)',
        '        if not nums: return False',
        '        gold_num = float(nums[-1].replace(",",""))',
        '    nums = NUMBER_RE.findall(gen)',
        '    if not nums: return False',
        '    try: gen_num = float(nums[-1].replace(",",""))',
        '    except: return False',
        '    return abs(gen_num - gold_num) < 1e-2',
    ]))

    cells.append(md(['## 5. Collect samples (BASE + STUDENT modes)']))
    cells.append(code([
        'sqa_fresh = load_dataset("basicv8vc/SimpleQA", split="test").shuffle(seed=CFG["random_seed"]).select(range(CFG["fresh_n_simpleqa"]))',
        'gsm_fresh = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=CFG["random_seed"]).select(range(CFG["fresh_n_gsm8k"]))',
        '',
        'def collect(eval_set, label_fn, ques_key, ans_key, src, mode, max_new):',
        '    out = {"fg":{"X":[],"y":[]}, "rg":{"X":[],"y":[]}}',
        '    for ex in tqdm(eval_set, desc=f"{src}/{mode}"):',
        '        q = ex[ques_key]; gold = ex[ans_key]',
        '        if isinstance(gold, list) and gold: gold = gold[0]',
        '        ans = gen_one(q, max_new=max_new, mode=mode)',
        '        halu = int(not label_fn(ans, gold))',
        '        h31 = capture_l31(q, ans)',
        '        out["fg"]["X"].append(h31); out["fg"]["y"].append(halu)',
        '        h55 = capture_l55(q, ans)',
        '        if h55 is not None:',
        '            out["rg"]["X"].append(h55); out["rg"]["y"].append(halu)',
        '    return out',
        '',
        'samples_base, samples_student = ({"fg":{"X":[],"y":[]},"rg":{"X":[],"y":[]}} for _ in range(2))',
        'for mode, dst in [("base", samples_base), ("student", samples_student)]:',
        '    sqa_s = collect(sqa_fresh, lambda a,g: normalize_answer(g) in normalize_answer(a),',
        '                    "problem","answer","simpleqa", mode, 256)',
        '    gsm_s = collect(gsm_fresh, grade_gsm8k, "question","answer","gsm8k", mode, 512)',
        '    for k in ["fg","rg"]:',
        '        dst[k]["X"].extend(sqa_s[k]["X"] + gsm_s[k]["X"])',
        '        dst[k]["y"].extend(sqa_s[k]["y"] + gsm_s[k]["y"])',
        'print("Collection complete")',
    ]))

    cells.append(md(['## 6. Compare AUROCs — original vs fresh probes']))
    cells.append(code([
        'def fresh_probe_auc(X, y, name):',
        '    X = np.array(X); y = np.array(y)',
        '    if len(np.unique(y)) < 2 or sum(y) < 5 or sum(1-y) < 5:',
        '        return None, len(y), float(y.mean())',
        '    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)',
        '    sc = StandardScaler().fit(X_tr)',
        '    cv = min(5, sum(y_tr).item(), sum(1-y_tr).item())',
        '    clf = LogisticRegressionCV(Cs=[0.001,0.01,0.1,1,10], cv=cv, penalty="l2",',
        '                               solver="lbfgs", max_iter=2000, scoring="roc_auc"',
        '                               ).fit(sc.transform(X_tr), y_tr)',
        '    auc = roc_auc_score(y_te, clf.predict_proba(sc.transform(X_te))[:,1])',
        '    return float(auc), len(y), float(y.mean())',
        '',
        'def orig_auc(X, y, probe, scaler):',
        '    X = np.array(X); y = np.array(y)',
        '    if len(np.unique(y)) < 2: return None',
        '    pos_idx = list(probe.classes_).index(1)',
        '    scores = probe.predict_proba(scaler.transform(X))[:, pos_idx]',
        '    return float(roc_auc_score(y, scores))',
        '',
        'print("\\n" + "="*72)',
        'print("  Anti-Goodhart Verdict — fresh probes vs original probes")',
        'print("="*72)',
        '',
        'auc_fg_orig_b = orig_auc(samples_base["fg"]["X"],    samples_base["fg"]["y"],    fg_probe, fg_scaler)',
        'auc_fg_orig_s = orig_auc(samples_student["fg"]["X"], samples_student["fg"]["y"], fg_probe, fg_scaler)',
        'auc_fg_fresh, n_fg, halu_fg_s = fresh_probe_auc(samples_student["fg"]["X"], samples_student["fg"]["y"], "L31")',
        'halu_fg_b = float(np.mean(samples_base["fg"]["y"]))',
        'print("\\nL31/end_question (FabricationGuard):")',
        'print(f"  Halu rate:  base {halu_fg_b*100:.1f}%  →  student {halu_fg_s*100:.1f}%  (Δ = {(halu_fg_s-halu_fg_b)*100:+.1f}pp)")',
        'print(f"  Original probe AUROC:  base {auc_fg_orig_b:.3f}  →  student {auc_fg_orig_s:.3f}")',
        'print(f"  FRESH probe (on student samples):  {auc_fg_fresh:.3f}  (n={n_fg})")',
        '',
        'auc_rg_orig_b = orig_auc(samples_base["rg"]["X"],    samples_base["rg"]["y"],    rg_probe, rg_scaler)',
        'auc_rg_orig_s = orig_auc(samples_student["rg"]["X"], samples_student["rg"]["y"], rg_probe, rg_scaler)',
        'auc_rg_fresh, n_rg, halu_rg_s = fresh_probe_auc(samples_student["rg"]["X"], samples_student["rg"]["y"], "L55")',
        'halu_rg_b = float(np.mean(samples_base["rg"]["y"]))',
        'print("\\nL55/mid_think (ReasonGuard):")',
        'print(f"  Halu rate:  base {halu_rg_b*100:.1f}%  →  student {halu_rg_s*100:.1f}%  (Δ = {(halu_rg_s-halu_rg_b)*100:+.1f}pp)")',
        'print(f"  Original probe AUROC:  base {auc_rg_orig_b:.3f}  →  student {auc_rg_orig_s:.3f}")',
        'print(f"  FRESH probe (on student samples):  {auc_rg_fresh:.3f}  (n={n_rg})")',
        '',
        'def diagnose(halu_b, halu_s, auc_orig_s, auc_fresh_s, name):',
        '    if any(v is None for v in [halu_s, auc_orig_s, auc_fresh_s]): return f"⚠️  {name}: insufficient data"',
        '    halu_dropped = halu_s < halu_b - 0.03',
        '    orig_dropped = auc_orig_s < 0.65',
        '    fresh_high   = auc_fresh_s >= 0.80',
        '    if halu_dropped and fresh_high:           return f"✅ {name}: REAL IMPROVEMENT"',
        '    if not halu_dropped and orig_dropped and not fresh_high: return f"❌ {name}: GOODHART CONFIRMED"',
        '    if not halu_dropped and orig_dropped and fresh_high:     return f"🟡 {name}: PARTIAL EVASION (signal moved off original direction)"',
        '    if halu_dropped and not fresh_high:       return f"🟠 {name}: SIGNAL ERODED"',
        '    if not halu_dropped and not orig_dropped: return f"🟢 {name}: NO CHANGE — DPO did not move the needle"',
        '    return f"?  {name}: mixed signals"',
        '',
        'print("\\n" + "="*72)',
        'print(diagnose(halu_fg_b, halu_fg_s, auc_fg_orig_s, auc_fg_fresh, "L31/end_question"))',
        'print(diagnose(halu_rg_b, halu_rg_s, auc_rg_orig_s, auc_rg_fresh, "L55/mid_think"))',
        'print("="*72)',
    ]))

    cells.append(md(['## 7. Save + push']))
    cells.append(code([
        'verdict = {',
        '    "l31_fg": {"halu_base":halu_fg_b, "halu_student":halu_fg_s,',
        '               "orig_auc_base":auc_fg_orig_b, "orig_auc_student":auc_fg_orig_s,',
        '               "fresh_auc_student":auc_fg_fresh, "n":n_fg},',
        '    "l55_rg": {"halu_base":halu_rg_b, "halu_student":halu_rg_s,',
        '               "orig_auc_base":auc_rg_orig_b, "orig_auc_student":auc_rg_orig_s,',
        '               "fresh_auc_student":auc_rg_fresh, "n":n_rg},',
        '}',
        '(OUT / "antigoodhart_verdict.json").write_text(json.dumps(verdict, indent=2, default=str))',
        'np.savez_compressed(OUT / "fresh_activations_base.npz",',
        '    X_fg=np.array(samples_base["fg"]["X"]), y_fg=np.array(samples_base["fg"]["y"]),',
        '    X_rg=np.array(samples_base["rg"]["X"]), y_rg=np.array(samples_base["rg"]["y"]))',
        'np.savez_compressed(OUT / "fresh_activations_student.npz",',
        '    X_fg=np.array(samples_student["fg"]["X"]), y_fg=np.array(samples_student["fg"]["y"]),',
        '    X_rg=np.array(samples_student["rg"]["X"]), y_rg=np.array(samples_student["rg"]["y"]))',
        '',
        'api = HfApi()',
        'api.upload_folder(folder_path=str(OUT), repo_id=CFG["output_repo"],',
        '                  repo_type="dataset", token=HF_TOKEN,',
        '                  commit_message=f"Anti-Goodhart fresh-probe verdict @ {time.strftime(\'%Y-%m-%d %H:%M\')}")',
        'print(f"✓ Pushed verdict + activations to https://huggingface.co/datasets/{CFG[\'output_repo\']}")',
    ]))

    cells.append(md([
        '## 8. Why this validation matters',
        '',
        'Goodfire RLFR\'s anti-Goodhart guarantee comes from running the probe on a frozen base model — gradient never touches the probe. But "frozen base + LoRA student" still allows the student to generate tokens that, when fed back through the base, produce activations off the probe direction. That\'s the failure mode we test here.',
        '',
        'If **fresh probe AUROC ≥ 0.80** on student samples: the residual stream still has a discriminable hallucination representation. The student didn\'t destroy the signal; it shifted distribution toward fewer hallucinations. **This is the desired outcome.**',
        '',
        'If **fresh probe AUROC < 0.65**: residual stream actively scrambled along the relevant direction. Multi-probe reward didn\'t prevent it. Need to:',
        '1. Increase α weights asymmetrically',
        '2. Add probe rotation (re-train probe every K steps)',
        '3. Add more orthogonal probes (DeceptionGuard, EvalAwarenessGuard)',
        '',
        'Either result is publishable. **Negative result here is the canonical evidence for why multi-probe orthogonality matters.**',
    ]))

    write_nb(cells, NOTEBOOKS_DIR / '36_antigoodhart_validation.ipynb')


if __name__ == '__main__':
    print('=== Building nb 34 ===')
    build_nb34()
    print('\n=== Building nb 35 ===')
    build_nb35()
    print('\n=== Building nb 36 ===')
    build_nb36()
    print('\n✓ All 3 notebooks generated.')
