"""
Builder for nb45_inference_ensemble.ipynb

Inference-time multi-probe ensemble test.

Question: does running FG + RG simultaneously at inference give better detection
signal than either probe alone? If yes, this validates the ProbePack product angle:
ship probe ensembles as middleware that monitor any LLM in production.

Setup:
- Base Qwen3.6-27B (no training — testing inference middleware on the foundation model)
- 50 hold-out prompts (25 GSM8K test + 25 SimpleQA test, seed=11)
- For each prompt: generate at temp=0.7, capture L31 + L55 activations at end-of-think
- Score with FG (L31) and RG (L55) probes
- Compute 4 fusion methods:
  1. weighted_avg = 0.5·fg + 0.5·rg
  2. max = max(fg, rg)
  3. voting = (fg > 0.5) + (rg > 0.5)  → 0/1/2
  4. bayesian_or = 1 - (1 - fg)(1 - rg)  (assumes independence)
- Score ground truth via gold-match (GSM8K) or judge (SimpleQA)
- Compute AUROC of each method as detector of incorrect answers

Verdict:
- If ensemble AUROC > best single probe AUROC by ≥5pp → ensemble adds value
- If ensemble ≈ single → just use single
- If ensemble < single → orthogonality not enough (probes capture different but
  uncorrelated-with-correctness signals)

Compute: ~1.5h (50 generations + scoring + fusion analysis).

Drive: /content/drive/MyDrive/openinterp_runs/45_inference_ensemble/
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
        "# Notebook 45 — Inference-Time Multi-Probe Ensemble",
        "",
        "**Question**: does running FabricationGuard + ReasonGuard simultaneously at inference give better detection signal than either alone?",
        "",
        "If yes, this validates the **ProbePack** product angle: ship probe ensembles as middleware that can monitor ANY LLM in production. No retraining needed — just run inference + apply probe stack on activations.",
        "",
        "**Setup**:",
        "- Base Qwen3.6-27B (testing on foundation, not LoRA-trained)",
        "- 50 hold-out prompts (25 GSM8K test + 25 SimpleQA test, seed=11)",
        "- temp=0.7 generation",
        "- Capture L31 + L55 at end-of-think",
        "- 4 fusion methods: weighted_avg, max, voting, Bayesian-OR",
        "- Ground truth: gold-match (GSM8K) + Claude Haiku judge (SimpleQA)",
        "",
        "**Verdict**:",
        "- 🟢 ensemble AUROC > best single by ≥5pp → ensemble adds value (ship ProbePack)",
        "- 🟡 ensemble ≈ single → diminishing returns, just ship single probes",
        "- 🔴 ensemble < single → probe outputs uncorrelated with each other AND with correctness",
        "",
        "**Compute**: ~1.5h on RTX 6000.",
        "",
        "**Drive**: `/content/drive/MyDrive/openinterp_runs/45_inference_ensemble/`",
    ]))

    # Phase 1
    cells.append(md(["## Phase 1 — Setup + Drive"]))
    cells.append(code([
        "from pathlib import Path",
        "import os, json, time",
        "import torch, numpy as np",
        "",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "DRIVE = Path('/content/drive/MyDrive')",
        "OUT = DRIVE / 'openinterp_runs' / '45_inference_ensemble'",
        "OUT.mkdir(parents=True, exist_ok=True)",
        "print(f'OUT: {OUT}')",
    ]))
    cells.append(code([
        "!pip install -q -U torchao",
        "!pip install -q -U transformers accelerate datasets",
        "!pip install -q -U huggingface_hub",
        "!pip install -q openai scikit-learn joblib matplotlib",
        "print('✓ deps')",
    ]))

    # Phase 2 — model + probes
    cells.append(md(["## Phase 2 — Qwen3.6-27B + FG + RG probes"]))
    cells.append(code([
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "from huggingface_hub import login, hf_hub_download, HfApi, create_repo",
        "import getpass, joblib",
        "",
        "CFG = {",
        "    'model_id':              'Qwen/Qwen3.6-27B',",
        "    'capture_layer_fg':      31,",
        "    'capture_layer_rg':      55,",
        "    'n_gsm8k':               25,",
        "    'n_simpleqa':            25,",
        "    'temperature':           0.7,",
        "    'max_new_tokens':        2048,",
        "    'random_seed':           11,",
        "    'fg_probe_repo':         'caiovicentino1/FabricationGuard-linearprobe-qwen36-27b',",
        "    'rg_probe_repo':         'caiovicentino1/ReasoningGuard-linearprobe-qwen36-27b',",
        "    'judge_model':           'anthropic/claude-haiku-4.5',",
        "    'output_repo':           'caiovicentino1/openinterp-45-inference-ensemble',",
        "    'fusion_alpha':          0.5,  # weighted_avg fg coefficient",
        "    'bootstrap_n':           1000,",
        "}",
        "THINK_CLOSE_ID = 248069",
        "torch.manual_seed(CFG['random_seed']); np.random.seed(CFG['random_seed'])",
        "",
        "HF_TOKEN = os.environ.get('HF_TOKEN') or getpass.getpass('HF token: ')",
        "login(HF_TOKEN, add_to_git_credential=False)",
        "OPENROUTER_KEY = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter API key: ')",
        "os.environ['OPENROUTER_API_KEY'] = OPENROUTER_KEY",
        "",
        "device = 'cuda'",
        "tok = AutoTokenizer.from_pretrained(CFG['model_id'])",
        "model = AutoModelForCausalLM.from_pretrained(",
        "    CFG['model_id'], torch_dtype=torch.bfloat16, device_map='auto',",
        ")",
        "model.eval()",
        "print(f'✓ Base loaded — {torch.cuda.get_device_name(0)}')",
        "",
        "fg_path = hf_hub_download(repo_id=CFG['fg_probe_repo'], filename='probe.joblib', repo_type='dataset')",
        "rg_path = hf_hub_download(repo_id=CFG['rg_probe_repo'], filename='probe.joblib', repo_type='dataset')",
        "fg_artifact = joblib.load(fg_path)",
        "rg_artifact = joblib.load(rg_path)",
        "fg_clf = fg_artifact['probe']; fg_scaler = fg_artifact['scaler']",
        "rg_clf = rg_artifact['probe']; rg_scaler = rg_artifact['scaler']",
        "if not hasattr(fg_clf, 'multi_class'): fg_clf.multi_class = 'auto'",
        "if not hasattr(rg_clf, 'multi_class'): rg_clf.multi_class = 'auto'",
        "",
        "def fg_score(act):",
        "    x = act.float().cpu().numpy().reshape(1, -1)",
        "    return float(fg_clf.predict_proba(fg_scaler.transform(x))[0, 1])",
        "def rg_score(act):",
        "    x = act.float().cpu().numpy().reshape(1, -1)",
        "    return float(rg_clf.predict_proba(rg_scaler.transform(x))[0, 1])",
        "print('✓ FG + RG probes loaded with sklearn 1.5+ patch')",
    ]))

    # Phase 3 — hooks + holdout
    cells.append(md(["## Phase 3 — Hooks + 50 hold-out prompts"]))
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
        "hook_handles = []",
        "for L in [CFG['capture_layer_fg'], CFG['capture_layer_rg']]:",
        "    h = model.model.layers[L].register_forward_hook(make_hook(L))",
        "    hook_handles.append(h)",
        "print(f'✓ Hooks at L{CFG[\"capture_layer_fg\"]}, L{CFG[\"capture_layer_rg\"]}')",
    ]))
    cells.append(code([
        "from datasets import load_dataset",
        "rng = np.random.default_rng(CFG['random_seed'])",
        "",
        "gsm = load_dataset('openai/gsm8k', 'main', split='test')",
        "gsm_idx = rng.choice(len(gsm), size=CFG['n_gsm8k'], replace=False)",
        "gsm_pool = []",
        "for i in gsm_idx:",
        "    ex = gsm[int(i)]",
        "    gold = ex['answer'].split('####')[-1].strip().replace(',', '')",
        "    gsm_pool.append({'id': f'gsm_{i}', 'src': 'gsm8k',",
        "                     'question': ex['question'], 'gold': gold})",
        "",
        "try:",
        "    sqa = load_dataset('basicv8vc/SimpleQA', split='test')",
        "    sqa_idx = rng.choice(len(sqa), size=CFG['n_simpleqa'], replace=False)",
        "    sqa_pool = []",
        "    for i in sqa_idx:",
        "        ex = sqa[int(i)]",
        "        sqa_pool.append({'id': f'sqa_{i}', 'src': 'simpleqa',",
        "                         'question': ex['problem'], 'gold': ex['answer']})",
        "except Exception as e:",
        "    print(f'SimpleQA failed: {e}'); sqa_pool = []",
        "",
        "holdout = gsm_pool + sqa_pool",
        "rng.shuffle(holdout)",
        "print(f'Hold-out: {len(holdout)} prompts')",
        "src_dist = {}",
        "for p in holdout: src_dist[p['src']] = src_dist.get(p['src'], 0) + 1",
        "print(f'  sources: {src_dist}')",
    ]))

    # Phase 4 — generation + dual probe scoring
    cells.append(md(["## Phase 4 — Generate + score with both probes simultaneously"]))
    cells.append(code([
        "from tqdm.auto import tqdm",
        "import gc",
        "",
        "def find_end_think(token_ids):",
        "    ids = token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids)",
        "    for i in range(len(ids) - 1, -1, -1):",
        "        if ids[i] == THINK_CLOSE_ID: return i",
        "    return None",
        "",
        "def generate_and_probe(prompt):",
        "    \"\"\"Generate at temp=0.7 + score with BOTH probes from same activations.\"\"\"",
        "    messages = [{'role': 'user', 'content': prompt}]",
        "    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)",
        "    enc = tok(text, return_tensors='pt')",
        "    ids = enc['input_ids'].to(device)",
        "    amask = enc.get('attention_mask', torch.ones_like(ids)).to(device)",
        "    n_in = ids.shape[1]",
        "    with torch.no_grad():",
        "        gen = model.generate(",
        "            ids, attention_mask=amask,",
        "            max_new_tokens=CFG['max_new_tokens'],",
        "            do_sample=True, temperature=CFG['temperature'], top_p=0.95,",
        "            pad_token_id=tok.eos_token_id,",
        "        )",
        "    full_ids = gen[0]",
        "    output_ids = full_ids[n_in:]",
        "    end_pos = find_end_think(full_ids)",
        "    output_text = tok.decode(output_ids, skip_special_tokens=False)",
        "    if '</think>' in output_text:",
        "        cot = output_text.split('</think>', 1)[0].strip()",
        "        answer = output_text.split('</think>', 1)[1].strip()",
        "    else:",
        "        cot, answer = output_text.strip(), ''",
        "    if end_pos is None:",
        "        return {'cot': cot, 'answer': answer, 'fg': None, 'rg': None, 'has_think': False}",
        "    captured.clear()",
        "    _pos['pos'] = end_pos",
        "    with torch.no_grad():",
        "        _ = model(full_ids.unsqueeze(0).to(device))",
        "    act_fg = captured.get(f'L{CFG[\"capture_layer_fg\"]}')",
        "    act_rg = captured.get(f'L{CFG[\"capture_layer_rg\"]}')",
        "    return {",
        "        'cot': cot, 'answer': answer,",
        "        'fg': fg_score(act_fg) if act_fg is not None else None,",
        "        'rg': rg_score(act_rg) if act_rg is not None else None,",
        "        'has_think': True, 'end_pos': end_pos,",
        "    }",
        "",
        "results_path = OUT / 'generations_with_probes.jsonl'",
        "done_keys = set()",
        "if results_path.exists():",
        "    with open(results_path) as f:",
        "        for line in f:",
        "            try: done_keys.add(json.loads(line)['id'])",
        "            except: continue",
        "    print(f'Resume: {len(done_keys)} done')",
        "",
        "for p in tqdm(holdout, desc='gen+probe'):",
        "    if p['id'] in done_keys: continue",
        "    try:",
        "        torch.manual_seed(hash(p['id']) % (2**32))",
        "        res = generate_and_probe(p['question'])",
        "    except torch.cuda.OutOfMemoryError:",
        "        torch.cuda.empty_cache(); gc.collect()",
        "        print(f'OOM on {p[\"id\"]}'); continue",
        "    record = {**p, **res}",
        "    with open(results_path, 'a') as f:",
        "        f.write(json.dumps(record) + '\\n')",
        "    done_keys.add(p['id'])",
        "",
        "print('\\n✓ Phase 4 complete')",
    ]))

    # Phase 5 — score truth labels
    cells.append(md(["## Phase 5 — Ground truth: gold-match (GSM8K) + judge (SimpleQA)"]))
    cells.append(code([
        "from openai import OpenAI",
        "client = OpenAI(",
        "    base_url='https://openrouter.ai/api/v1',",
        "    api_key=OPENROUTER_KEY,",
        ")",
        "import re",
        "NUM_RE = re.compile(r'-?\\d+(?:[.,]\\d+)?')",
        "",
        "def grade_gsm8k(answer, gold):",
        "    nums = NUM_RE.findall(answer.replace(',', ''))",
        "    if not nums: return False",
        "    try: gold_num = float(gold.replace(',', ''))",
        "    except: return False",
        "    for n in reversed(nums):",
        "        try:",
        "            if abs(float(n) - gold_num) < 1e-3: return True",
        "        except: continue",
        "    return False",
        "",
        "JUDGE = '''Is this answer factually correct? Reply ONE WORD: YES, NO, or UNVERIFIABLE.\\n\\nQ: {q}\\nGold: {g}\\nAnswer: {a}'''",
        "",
        "def judge_sqa(q, g, a):",
        "    try:",
        "        r = client.chat.completions.create(",
        "            model=CFG['judge_model'], max_tokens=80,",
        "            messages=[{'role': 'user',",
        "                       'content': JUDGE.format(q=q, g=g, a=a[:1000])}],",
        "        )",
        "        t = r.choices[0].message.content.strip().upper()",
        "        if t.startswith('YES'): return 'YES'",
        "        if t.startswith('NO'): return 'NO'",
        "        return 'UNVERIFIABLE'",
        "    except Exception as e:",
        "        return f'ERROR'",
        "",
        "with open(results_path) as f:",
        "    records = [json.loads(line) for line in f]",
        "scored_path = OUT / 'scored.jsonl'",
        "scored_keys = set()",
        "if scored_path.exists():",
        "    with open(scored_path) as f:",
        "        for line in f:",
        "            try: scored_keys.add(json.loads(line)['id'])",
        "            except: continue",
        "",
        "for r in tqdm(records, desc='score'):",
        "    if r['id'] in scored_keys: continue",
        "    if not r.get('has_think'): continue",
        "    answer = r['answer'] or r['cot'][:500]",
        "    if r['src'] == 'gsm8k':",
        "        is_correct = grade_gsm8k(answer, r['gold'])",
        "        label = 'CORRECT' if is_correct else 'INCORRECT'",
        "    else:",
        "        label = judge_sqa(r['question'], r['gold'], answer)",
        "        is_correct = label == 'YES'",
        "    out = {**r, 'is_correct': is_correct, 'judge_label': label}",
        "    with open(scored_path, 'a') as f:",
        "        f.write(json.dumps(out) + '\\n')",
        "    scored_keys.add(r['id'])",
        "",
        "print('\\n✓ Phase 5 complete')",
    ]))

    # Phase 6 — fusion + AUROC analysis
    cells.append(md([
        "## Phase 6 — Fusion methods + AUROC comparison",
        "",
        "Compare AUROC of detecting INCORRECT answers using:",
        "1. FG alone",
        "2. RG alone",
        "3. weighted_avg = 0.5·FG + 0.5·RG",
        "4. max = max(FG, RG)",
        "5. voting = (FG > 0.5) + (RG > 0.5) — counts how many flag",
        "6. bayesian_or = 1 - (1 - FG)(1 - RG) — probabilistic fusion",
    ]))
    cells.append(code([
        "import pandas as pd",
        "from sklearn.metrics import roc_auc_score",
        "",
        "with open(scored_path) as f:",
        "    scored = [json.loads(line) for line in f]",
        "df = pd.DataFrame(scored)",
        "df = df[df['has_think'] & df['fg'].notnull() & df['rg'].notnull()].copy()",
        "print(f'Valid records: {len(df)}')",
        "",
        "# Define target: detect INCORRECT answers (positive class)",
        "df['target'] = (~df['is_correct']).astype(int)  # 1 = incorrect, 0 = correct",
        "",
        "df['fusion_weighted_avg'] = CFG['fusion_alpha'] * df['fg'] + (1 - CFG['fusion_alpha']) * df['rg']",
        "df['fusion_max'] = df[['fg', 'rg']].max(axis=1)",
        "df['fusion_voting'] = (df['fg'] > 0.5).astype(int) + (df['rg'] > 0.5).astype(int)",
        "df['fusion_bayesian_or'] = 1 - (1 - df['fg']) * (1 - df['rg'])",
        "",
        "print('\\n=== Score distributions ===')",
        "print(df[['fg', 'rg', 'fusion_weighted_avg', 'fusion_max', 'fusion_voting', 'fusion_bayesian_or']].describe().round(4))",
        "print(f'\\nIncorrect rate (target positive): {df[\"target\"].mean():.2%}')",
    ]))
    cells.append(code([
        "def bootstrap_auroc(y, scores, n=1000, seed=42):",
        "    rng = np.random.default_rng(seed)",
        "    aurocs = []",
        "    for _ in range(n):",
        "        idx = rng.choice(len(y), size=len(y), replace=True)",
        "        ys = y[idx]; ss = scores[idx]",
        "        if len(set(ys)) < 2: continue",
        "        aurocs.append(roc_auc_score(ys, ss))",
        "    arr = np.array(aurocs)",
        "    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))",
        "",
        "y = df['target'].values",
        "methods = ['fg', 'rg', 'fusion_weighted_avg', 'fusion_max', 'fusion_voting', 'fusion_bayesian_or']",
        "results = {}",
        "print('Method                     AUROC     CI 95%        N')",
        "print('-' * 65)",
        "for m in methods:",
        "    scores = df[m].values.astype(float)",
        "    if len(set(y)) < 2:",
        "        print(f'{m:<28} insufficient class diversity')",
        "        continue",
        "    auroc = roc_auc_score(y, scores)",
        "    boot_mean, lo, hi = bootstrap_auroc(y, scores, n=CFG['bootstrap_n'])",
        "    results[m] = {'auroc': auroc, 'ci_lo': lo, 'ci_hi': hi, 'n': len(y)}",
        "    print(f'{m:<28} {auroc:.4f}   [{lo:.3f}, {hi:.3f}]   {len(y)}')",
        "",
        "best_single = max(results['fg']['auroc'], results['rg']['auroc'])",
        "best_ensemble = max(",
        "    results['fusion_weighted_avg']['auroc'],",
        "    results['fusion_max']['auroc'],",
        "    results['fusion_voting']['auroc'],",
        "    results['fusion_bayesian_or']['auroc'],",
        ")",
        "delta = best_ensemble - best_single",
        "print(f'\\n=== HEADLINE ===')",
        "print(f'Best single probe AUROC: {best_single:.4f}')",
        "print(f'Best ensemble AUROC:     {best_ensemble:.4f}')",
        "print(f'Delta:                   {delta:+.4f}')",
        "if delta > 0.05:",
        "    print('🟢 ENSEMBLE ADDS VALUE — ProbePack product validated')",
        "elif delta > 0.01:",
        "    print('🟡 marginal — needs more probes or larger N')",
        "elif abs(delta) < 0.01:",
        "    print('⚪ ensemble ≈ single — diminishing returns')",
        "else:",
        "    print('🔴 ensemble UNDERPERFORMS single — probe outputs uncorrelated with correctness')",
    ]))
    cells.append(code([
        "# Plot per-method AUROC with bars + per-source breakdown",
        "import matplotlib.pyplot as plt",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
        "names = list(results.keys())",
        "aurocs = [results[m]['auroc'] for m in names]",
        "los = [results[m]['ci_lo'] for m in names]",
        "his = [results[m]['ci_hi'] for m in names]",
        "errs = [[a - lo for a, lo in zip(aurocs, los)],",
        "        [hi - a for a, hi in zip(aurocs, his)]]",
        "colors = ['#3b82f6', '#f59e0b', '#10b981', '#10b981', '#10b981', '#10b981']",
        "axes[0].bar(range(len(names)), aurocs, yerr=errs, capsize=4, color=colors, alpha=0.85)",
        "axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='chance')",
        "axes[0].axhline(best_single, color='blue', linestyle=':', alpha=0.5, label='best single probe')",
        "axes[0].set_xticks(range(len(names)))",
        "axes[0].set_xticklabels([n.replace('fusion_', '') for n in names], rotation=20, ha='right')",
        "axes[0].set_ylabel('AUROC (detect incorrect)')",
        "axes[0].set_title('Per-method AUROC with bootstrap CI')",
        "axes[0].set_ylim(0.4, 1.0)",
        "axes[0].legend(loc='lower right')",
        "axes[0].grid(alpha=0.3)",
        "",
        "# Per-source breakdown",
        "for src in df['src'].unique():",
        "    src_df = df[df['src'] == src]",
        "    if len(src_df) < 5 or len(set(src_df['target'])) < 2: continue",
        "    src_aurocs = []",
        "    for m in methods:",
        "        try: src_aurocs.append(roc_auc_score(src_df['target'].values, src_df[m].values.astype(float)))",
        "        except: src_aurocs.append(0.5)",
        "    axes[1].plot(range(len(methods)), src_aurocs, marker='o', label=src, linewidth=2, markersize=8)",
        "axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)",
        "axes[1].set_xticks(range(len(methods)))",
        "axes[1].set_xticklabels([n.replace('fusion_', '') for n in methods], rotation=20, ha='right')",
        "axes[1].set_ylabel('AUROC')",
        "axes[1].set_title('Per-source breakdown')",
        "axes[1].legend(); axes[1].grid(alpha=0.3)",
        "axes[1].set_ylim(0.4, 1.0)",
        "plt.tight_layout()",
        "plt.savefig(OUT / 'fig_ensemble_auroc.png', dpi=150)",
        "plt.show()",
    ]))

    # Phase 7 — verdict + push
    cells.append(md(["## Phase 7 — FINAL VERDICT + push"]))
    cells.append(code([
        "verdict = {",
        "    'experiment': 'nb45 inference-time multi-probe ensemble',",
        "    'hypothesis': 'FG+RG ensemble at inference > best single probe in detecting incorrect answers',",
        "    'n_holdout': len(df),",
        "    'incorrect_rate': float(df['target'].mean()),",
        "    'method_aurocs': {m: results[m] for m in results},",
        "    'best_single': float(best_single),",
        "    'best_ensemble': float(best_ensemble),",
        "    'delta': float(delta),",
        "    'verdict': (",
        "        'real_gain' if delta > 0.05",
        "        else 'marginal' if delta > 0.01",
        "        else 'null' if abs(delta) <= 0.01",
        "        else 'underperforms'",
        "    ),",
        "    'fg_rg_correlation': float(df[['fg', 'rg']].corr().iloc[0, 1]),",
        "}",
        "(OUT / 'FINAL_VERDICT.json').write_text(json.dumps(verdict, indent=2))",
        "print(json.dumps(verdict, indent=2))",
    ]))
    cells.append(code([
        "api = HfApi()",
        "try: create_repo(CFG['output_repo'], repo_type='dataset', private=False, exist_ok=True, token=HF_TOKEN)",
        "except Exception as e: print(e)",
        "",
        "(OUT / 'README.md').write_text(f'''---",
        "license: apache-2.0",
        "tags: [inference-ensemble, multi-probe, qwen36-27b]",
        "---",
        "",
        "# nb45 — Inference-Time Multi-Probe Ensemble",
        "",
        "Tests whether running FabricationGuard + ReasonGuard simultaneously at inference",
        "produces better detection signal than either probe alone.",
        "",
        "Setup: 50 hold-out prompts (25 GSM8K + 25 SimpleQA, seed=11), base Qwen3.6-27B,",
        "temp=0.7. Capture L31 + L55 at end-of-think. 4 fusion methods.",
        "",
        "Target: detect INCORRECT answers (gold-match GSM8K, judge SimpleQA).",
        "",
        "## Verdict legend",
        "- 🟢 real_gain: ensemble AUROC > best single by ≥5pp",
        "- 🟡 marginal: 1-5pp gain",
        "- ⚪ null: ≤1pp difference",
        "- 🔴 underperforms: ensemble < single",
        "",
        "See FINAL_VERDICT.json.",
        "''')",
        "",
        "try:",
        "    api.upload_folder(folder_path=str(OUT), repo_id=CFG['output_repo'],",
        "                      repo_type='dataset', token=HF_TOKEN,",
        "                      commit_message='nb45 inference ensemble',",
        "                      allow_patterns=['README.md', 'FINAL_VERDICT.json',",
        "                                      'fig_*.png', 'scored.jsonl', '_*.txt'])",
        "    print('✓ pushed')",
        "except Exception as e:",
        "    print(f'HF push failed: {e}')",
    ]))

    cells.append(md([
        "## Done",
        "",
        "**If 🟢 real_gain**: ProbePack product validated. Inference-time ensemble works on ANY model — middleware angle. Build `/products/probepack` page + landing.",
        "",
        "**If 🟡 marginal**: ensemble gives some lift but maybe not enough alone. Test with 3rd, 4th probe (DeceptionGuard, EvalAwarenessGuard) when those ship.",
        "",
        "**If ⚪ null**: probes are TOO orthogonal — they detect different things, neither correlates strongly with correctness. Need probes specifically targeting the 'incorrect answer' axis.",
        "",
        "**If 🔴 underperforms**: ensemble adds noise. Investigate per-source — maybe FG good for SimpleQA, RG good for GSM8K, and ensemble averages out signal.",
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

    out_path = NOTEBOOKS_DIR / "nb45_inference_ensemble.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    print(f"✓ wrote {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
