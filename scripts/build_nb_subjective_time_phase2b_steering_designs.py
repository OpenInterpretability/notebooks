"""
Builder for nb_subjective_time_phase2b_steering_designs.ipynb.

Subjective-Time Probe Phase 2B — steering design exploration on cross-domain SWE-bench.

Builds on Phase 2A's positive causal finding (Fisher p=0.0092 on GSM8K).
Phase 2A established mechanical causality at α=+50; Phase 2B asks:

(Q1) Cross-domain transfer: does the GSM8K-trained direction work on SWE-bench?
(Q2) Budget artifact check: would baselines terminate at 2× the cap?
(Q3) Dose-response: where is the basin transition (between α=+30 and α=+50)?
(Q4) Closed-loop adaptive design: probe-as-sensor + threshold trigger
(Q5) Plateau-based intervention: detect overthinking via predicted-fraction stasis
(Q6) Mechanism isolation: is steering effect state-dependent or trajectory-dependent?

Methodology:
- Caveat #1: stratified cross-repo SWE-bench Verified (5 repos × 2 = 10 problems)
- Caveat #2: re-test all 10 baselines at MAX_NEW_TOK=2048
- α-sweep B: weaker steering {+30, +40} on the same 10 prompts
- Design E (closed-loop threshold): {th=0.65, 0.70, 0.85}, commit_α=+50
- Design F (plateau detector): {w=100/δ=0.02, w=50/δ=0.02}, commit_α=+50
- Onset timing: static α=+50 with delayed onset {step 50, 200, 400}

Findings (recorded inline as cells execute, summarized in final verdict cell):
- Cross-repo: probe rescues 19/20 (95%) vs random 6/20 (30%), Fisher p<0.001
- Budget extension: 0/10 baselines terminate at 2048 — rescue is genuine
- α-sweep B: termination 1/10 (α=30), 7/10 (α=40), 9/10 (α=50) — sharp basin transition
- Design E: 1/10 closed-loop rescue at any threshold — single state-attractor model FALSIFIED
- Design F: 0/10 plateau rescue — predicted-fraction does not satisfy plateau criterion
- Onset timing: deterministic isolation of state-dependent vs trajectory-dependent mechanism

NEW METHODOLOGY CONTRIBUTIONS (Phase 2B for the paper):
- Steering-onset-timing diagnostic (state-vs-trajectory mechanism isolation)
- Probe-as-dual-purpose closed-loop design (and its honest failure)

HARD RULES preserved from Phase 2A:
- Random-direction parallel (Phase 7/8)
- α sweep to multiples of ‖residual‖ (Phase 8 structural-rigidity)
- Whitespace-stripped output comparison (Phase 10)
- Drive checkpoint per prompt (resume-safe)

Compute: ~90 min on A100 80GB / RTX 6000 Blackwell 96GB.

Target: paper-8 ("Probe-Guided Anti-Overthinking"; may rename to
"Trajectory-Shaping Probe Steering in Qwen3.6-27B Reasoning"
based on onset-timing outcome).
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
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def build():
    cells = []

    # ---------- header ----------
    cells.append(md([
        "# Subjective-Time Probe Phase 2B — Steering Design Exploration",
        "",
        "**Phase 2A established**: the subjective-time direction at L31 is causally functional",
        "on GSM8K (Fisher p=0.0092, 9/14 shortened by probe@+50 vs 2/14 random).",
        "",
        "**Phase 2B asks 6 questions** for paper-8:",
        "",
        "1. Does the direction transfer cross-domain? (GSM8K → SWE-bench)",
        "2. Is the cross-domain rescue a budget artifact?",
        "3. Where is the basin transition between α=+30 and α=+50?",
        "4. Can closed-loop probe-as-sensor preserve thinking depth?",
        "5. Does plateau-detection identify overthinking?",
        "6. Is the causal effect state-dependent or trajectory-dependent (KV-cache lock-in)?",
        "",
        "**HARD RULES** (all preserved from Phase 2A):",
        "- Random-direction parallel at every α (Phase 7/8)",
        "- Whitespace-stripped output comparison (Phase 10)",
        "- Drive checkpoint per prompt (resume-safe)",
        "- α sweep to multiples of ‖residual‖ (Phase 8)",
        "",
        "**Compute**: ~90 min on A100 80GB / RTX 6000 Blackwell 96GB.",
        "**Model**: Qwen3.6-27B bf16, hybrid Gated-Delta-Net + standard-attention.",
    ]))

    # ---------- 1. Setup ----------
    cells.append(md(["## 1. Setup — Drive, model, tokenizer, probe refit"]))

    cells.append(code([
        "# 1.1 — Drive mount + paths",
        "from pathlib import Path",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "",
        "DRIVE     = Path('/content/drive/MyDrive')",
        "CACHE_PT  = DRIVE / 'openinterp_runs' / 'predictive_sae_v1' / 'cache' / 'residuals_multilayer.pt'",
        "OUT_DIR   = DRIVE / 'openinterp_runs' / 'subjective_time_phase2a' / 'caveat1_cross_repo'",
        "OUT_DIR.mkdir(parents=True, exist_ok=True)",
        "print(f'cache exists:  {CACHE_PT.exists()}')",
        "print(f'output dir:    {OUT_DIR}')",
    ]))

    cells.append(code([
        "# 1.2 — Install deps if missing (Colab)",
        "# Qwen3.6 (model_type='qwen3_5') requires transformers from main branch",
        "!pip uninstall -y -q transformers",
        "!pip install -q --upgrade git+https://github.com/huggingface/transformers.git accelerate sentencepiece datasets scipy",
        "print('⚠️  Restart runtime after this cell (Runtime → Restart session), then re-run from cell 1.')",
    ]))

    cells.append(code([
        "# 1.3 — Load Qwen3.6-27B (bf16, ~54 GB VRAM)",
        "import torch",
        "from transformers import AutoTokenizer, AutoModelForCausalLM",
        "",
        "MODEL_ID = 'Qwen/Qwen3.6-27B'",
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)",
        "model = AutoModelForCausalLM.from_pretrained(",
        "    MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True",
        ")",
        "model.eval()",
        "device = next(model.parameters()).device",
        "L31_LAYER_IDX = 31",
        "THINK_END_ID  = tokenizer.encode('</think>', add_special_tokens=False)[0]",
        "print('model:', type(model).__name__)",
        "print('language_model.layers:', len(model.model.language_model.layers))",
    ]))

    cells.append(code([
        "# 1.4 — Refit Ridge probe at L31 from cached residuals (deterministic, ~5s)",
        "import numpy as np",
        "from sklearn.linear_model import Ridge",
        "from sklearn.metrics import r2_score",
        "",
        "cache = torch.load(CACHE_PT, map_location='cpu')",
        "fractions = [0.1, 0.25, 0.5, 0.75, 1.0]",
        "sub = cache[31]",
        "",
        "Xs, ys, pidx = [], [], []",
        "for f in fractions:",
        "    t = sub[f].numpy()",
        "    Xs.append(t); ys.append(np.full(t.shape[0], f, dtype=np.float32))",
        "    pidx.append(np.arange(t.shape[0]))",
        "X = np.concatenate(Xs); y = np.concatenate(ys); prompt_id = np.concatenate(pidx)",
        "",
        "rng = np.random.default_rng(42)",
        "all_prompts = np.arange(sub[0.1].shape[0]); rng.shuffle(all_prompts)",
        "train_pids = set(all_prompts[:int(0.8 * len(all_prompts))].tolist())",
        "train_mask = np.array([p in train_pids for p in prompt_id])",
        "",
        "probe_full = Ridge(alpha=1.0).fit(X[train_mask], y[train_mask])",
        "r2 = r2_score(y[~train_mask], probe_full.predict(X[~train_mask]))",
        "print(f'L31 Ridge probe R² = {r2:.4f}  (Phase 2A reported 0.858)')",
        "",
        "# Two representations of the same direction:",
        "probe_coef_raw   = torch.from_numpy(probe_full.coef_.astype(np.float32))  # for prediction (sensor)",
        "probe_intercept  = float(probe_full.intercept_)",
        "w_t              = probe_coef_raw.flatten().clone()",
        "probe_w          = (w_t / w_t.norm()).to(device=device, dtype=model.dtype)  # unit-norm for steering",
        "",
        "# Matched random direction (seed=42, identical to Phase 2A)",
        "g = torch.Generator(device='cpu').manual_seed(42)",
        "r = torch.randn(probe_w.shape[0], generator=g)",
        "random_w = (r / r.norm()).to(device=device, dtype=model.dtype)",
        "",
        "print(f'probe_w  norm={probe_w.float().norm().item():.4f}  intercept={probe_intercept:.4f}')",
        "print(f'random_w norm={random_w.float().norm().item():.4f}  cos(probe,random)={(probe_w.float()*random_w.float()).sum().item():.4f}')",
    ]))

    # ---------- 2. Cross-repo sampling ----------
    cells.append(md([
        "## 2. Cross-repo SWE-bench sample",
        "",
        "10 problems stratified across top-5 most-populated non-astropy repos in SWE-bench Verified:",
        "django, sympy, sphinx, matplotlib, scikit-learn. Seed=42 for reproducibility.",
    ]))

    cells.append(code([
        "import random as pyrandom",
        "from collections import defaultdict",
        "from datasets import load_dataset",
        "",
        "pyrandom.seed(42)",
        "swe = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')",
        "",
        "by_repo = defaultdict(list)",
        "for ex in swe:",
        "    if ex['repo'] != 'astropy/astropy':",
        "        by_repo[ex['repo']].append(ex)",
        "repos_ranked = sorted(by_repo.keys(), key=lambda r: -len(by_repo[r]))",
        "target_repos = repos_ranked[:5]",
        "",
        "sampled = []",
        "for r in target_repos:",
        "    sampled.extend(pyrandom.sample(by_repo[r], 2))",
        "",
        "print('=== Cross-repo stratified sample ===')",
        "for r in target_repos:",
        "    print(f'  {r:40s}  pool={len(by_repo[r]):4d}  picked=2')",
        "print(f'Total: {len(sampled)} problems across {len(target_repos)} repos')",
    ]))

    # ---------- 3. Helper functions ----------
    cells.append(md([
        "## 3. Helper functions — prompt builder + 4 steering modes",
        "",
        "All generators share: greedy decode, MAX_NEW_TOK=1024, hook on `model.model.language_model.layers[31]`.",
        "Diff:",
        "- `generate_one(direction, alpha)` — static, applied every token (Phase 2A canonical)",
        "- `generate_closed_loop(threshold, commit_alpha)` — Design E, probe-as-sensor + threshold trigger",
        "- `generate_plateau(window, delta_thresh, commit_alpha)` — Design F, predicted-fraction stasis detector",
        "- `generate_delayed_static(onset_step, alpha)` — diagnostic, static with delayed onset",
    ]))

    cells.append(code([
        "MAX_NEW_TOK = 1024",
        "",
        "def build_prompt(ex):",
        "    stmt = ex['problem_statement'][:4000]",
        "    user_msg = (f'<problem_statement>\\n{stmt}\\n</problem_statement>\\n\\n'",
        "                'Analyze the issue carefully and propose a fix.')",
        "    msgs = [{'role': 'user', 'content': user_msg}]",
        "    return tokenizer.apply_chat_template(",
        "        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True",
        "    )",
        "",
        "@torch.no_grad()",
        "def generate_one(prompt_text, direction=None, alpha=0.0):",
        "    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)",
        "    prompt_len = inputs.input_ids.shape[1]",
        "    hook_handle = None",
        "    if direction is not None:",
        "        d = direction.to(device=device, dtype=model.dtype)",
        "        def _hook(_m, _i, out):",
        "            h = out[0] if isinstance(out, tuple) else out",
        "            h2 = h + alpha * d",
        "            return (h2,) + out[1:] if isinstance(out, tuple) else h2",
        "        layer = model.model.language_model.layers[L31_LAYER_IDX]",
        "        hook_handle = layer.register_forward_hook(_hook)",
        "    try:",
        "        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOK,",
        "                                 do_sample=False, temperature=None, top_p=None,",
        "                                 pad_token_id=tokenizer.eos_token_id)",
        "    finally:",
        "        if hook_handle is not None: hook_handle.remove()",
        "    gen_ids = out_ids[0, prompt_len:].tolist()",
        "    terminated = THINK_END_ID in gen_ids",
        "    thinking_len = (gen_ids.index(THINK_END_ID) + 1) if terminated else len(gen_ids)",
        "    text = tokenizer.decode(gen_ids, skip_special_tokens=False)",
        "    return {'thinking_len': thinking_len, 'terminated': terminated, 'text': text[:2000]}",
    ]))

    cells.append(code([
        "# Design E — closed-loop probe-as-sensor + threshold trigger",
        "@torch.no_grad()",
        "def generate_closed_loop(prompt_text, threshold=0.85, commit_alpha=50,",
        "                         min_decode_steps=50, max_new_tok=1024):",
        "    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)",
        "    prompt_len = inputs.input_ids.shape[1]",
        "    state = {'committed': False, 'committed_at': None, 'trace': [], 'decode_steps': 0}",
        "    coef = probe_coef_raw.to(device).float()",
        "    intc = probe_intercept",
        "    unit = probe_w",
        "",
        "    def _hook(_m, _i, out):",
        "        h = out[0] if isinstance(out, tuple) else out",
        "        if h.shape[1] == 1:",
        "            state['decode_steps'] += 1",
        "            pred = (h[:, -1, :].float() @ coef + intc).item()",
        "            state['trace'].append(pred)",
        "            if (not state['committed']",
        "                and state['decode_steps'] >= min_decode_steps",
        "                and pred >= threshold):",
        "                state['committed'] = True",
        "                state['committed_at'] = state['decode_steps']",
        "            if state['committed']:",
        "                h2 = h + commit_alpha * unit",
        "                return (h2,) + out[1:] if isinstance(out, tuple) else h2",
        "        return out",
        "",
        "    layer = model.model.language_model.layers[L31_LAYER_IDX]",
        "    handle = layer.register_forward_hook(_hook)",
        "    try:",
        "        out_ids = model.generate(**inputs, max_new_tokens=max_new_tok,",
        "                                 do_sample=False, temperature=None, top_p=None,",
        "                                 pad_token_id=tokenizer.eos_token_id)",
        "    finally:",
        "        handle.remove()",
        "    gen_ids = out_ids[0, prompt_len:].tolist()",
        "    terminated = THINK_END_ID in gen_ids",
        "    thinking_len = (gen_ids.index(THINK_END_ID) + 1) if terminated else len(gen_ids)",
        "    text = tokenizer.decode(gen_ids, skip_special_tokens=False)",
        "    return {",
        "        'thinking_len': thinking_len, 'terminated': terminated, 'text': text[:2500],",
        "        'committed_at': state['committed_at'],",
        "        'trace_max': max(state['trace']) if state['trace'] else None,",
        "        'trace_len': len(state['trace']),",
        "    }",
    ]))

    cells.append(code([
        "# Design F — plateau detector (predicted-fraction stasis triggers commit)",
        "@torch.no_grad()",
        "def generate_plateau(prompt_text, window=100, delta_thresh=0.02, commit_alpha=50,",
        "                     min_decode_steps=100, max_new_tok=1024):",
        "    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)",
        "    prompt_len = inputs.input_ids.shape[1]",
        "    state = {'committed': False, 'committed_at': None, 'trace': [], 'decode_steps': 0}",
        "    coef = probe_coef_raw.to(device).float()",
        "    intc = probe_intercept",
        "    unit = probe_w",
        "",
        "    def _hook(_m, _i, out):",
        "        h = out[0] if isinstance(out, tuple) else out",
        "        if h.shape[1] == 1:",
        "            state['decode_steps'] += 1",
        "            pred = (h[:, -1, :].float() @ coef + intc).item()",
        "            state['trace'].append(pred)",
        "            if (not state['committed']",
        "                and state['decode_steps'] >= min_decode_steps",
        "                and len(state['trace']) > window):",
        "                recent = state['trace'][-window:]",
        "                if (max(recent) - min(recent)) < delta_thresh:",
        "                    state['committed'] = True",
        "                    state['committed_at'] = state['decode_steps']",
        "            if state['committed']:",
        "                h2 = h + commit_alpha * unit",
        "                return (h2,) + out[1:] if isinstance(out, tuple) else h2",
        "        return out",
        "",
        "    layer = model.model.language_model.layers[L31_LAYER_IDX]",
        "    handle = layer.register_forward_hook(_hook)",
        "    try:",
        "        out_ids = model.generate(**inputs, max_new_tokens=max_new_tok,",
        "                                 do_sample=False, temperature=None, top_p=None,",
        "                                 pad_token_id=tokenizer.eos_token_id)",
        "    finally:",
        "        handle.remove()",
        "    gen_ids = out_ids[0, prompt_len:].tolist()",
        "    terminated = THINK_END_ID in gen_ids",
        "    thinking_len = (gen_ids.index(THINK_END_ID) + 1) if terminated else len(gen_ids)",
        "    text = tokenizer.decode(gen_ids, skip_special_tokens=False)",
        "    return {",
        "        'thinking_len': thinking_len, 'terminated': terminated, 'text': text[:2500],",
        "        'committed_at': state['committed_at'],",
        "        'trace_max': max(state['trace']) if state['trace'] else None,",
        "    }",
    ]))

    cells.append(code([
        "# Onset timing diagnostic — static α with delayed start (no sensor)",
        "@torch.no_grad()",
        "def generate_delayed_static(prompt_text, onset_step, alpha=50, max_new_tok=1024):",
        "    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)",
        "    prompt_len = inputs.input_ids.shape[1]",
        "    state = {'decode_steps': 0, 'active_from': None}",
        "    unit = probe_w",
        "",
        "    def _hook(_m, _i, out):",
        "        h = out[0] if isinstance(out, tuple) else out",
        "        if h.shape[1] == 1:",
        "            state['decode_steps'] += 1",
        "            if state['decode_steps'] >= onset_step:",
        "                if state['active_from'] is None:",
        "                    state['active_from'] = state['decode_steps']",
        "                h2 = h + alpha * unit",
        "                return (h2,) + out[1:] if isinstance(out, tuple) else h2",
        "        return out",
        "",
        "    layer = model.model.language_model.layers[L31_LAYER_IDX]",
        "    handle = layer.register_forward_hook(_hook)",
        "    try:",
        "        out_ids = model.generate(**inputs, max_new_tokens=max_new_tok,",
        "                                 do_sample=False, temperature=None, top_p=None,",
        "                                 pad_token_id=tokenizer.eos_token_id)",
        "    finally:",
        "        handle.remove()",
        "    gen_ids = out_ids[0, prompt_len:].tolist()",
        "    terminated = THINK_END_ID in gen_ids",
        "    thinking_len = (gen_ids.index(THINK_END_ID) + 1) if terminated else len(gen_ids)",
        "    text = tokenizer.decode(gen_ids, skip_special_tokens=False)",
        "    return {'thinking_len': thinking_len, 'terminated': terminated,",
        "            'text': text[:2000], 'active_from': state['active_from']}",
    ]))

    # ---------- 4. Caveat #1 ----------
    cells.append(md([
        "## 4. Caveat #1 — Cross-repo static steering (Phase 2A canonical mode)",
        "",
        "Replicate Phase 2A's static α=+50 lever on cross-repo SWE-bench. Expected: probe@+50 rescues ~9/10,",
        "random@+50 rescues ~3/10. Fisher exact target p<0.01.",
    ]))

    cells.append(code([
        "ALPHA = 50",
        "results_caveat1 = []",
        "for i, ex in enumerate(sampled, 1):",
        "    prompt = build_prompt(ex)",
        "    print(f'[{i:2d}/10] {ex[\"repo\"]} / {ex[\"instance_id\"]}')",
        "    base = generate_one(prompt, direction=None)",
        "    pro  = generate_one(prompt, direction=probe_w,  alpha=ALPHA)",
        "    rnd  = generate_one(prompt, direction=random_w, alpha=ALPHA)",
        "    row = {",
        "        'i': i, 'repo': ex['repo'], 'instance_id': ex['instance_id'],",
        "        'baseline':   {'len': base['thinking_len'], 'term': base['terminated']},",
        "        'probe_p50':  {'len': pro['thinking_len'],  'term': pro['terminated']},",
        "        'random_p50': {'len': rnd['thinking_len'],  'term': rnd['terminated']},",
        "    }",
        "    print(f'     base term={base[\"terminated\"]} len={base[\"thinking_len\"]:4d}  |  '",
        "          f'probe term={pro[\"terminated\"]} len={pro[\"thinking_len\"]:4d}  |  '",
        "          f'rand  term={rnd[\"terminated\"]} len={rnd[\"thinking_len\"]:4d}')",
        "    results_caveat1.append(row)",
        "    with open(OUT_DIR / 'results_cross_repo.json', 'w') as f:",
        "        json.dump(results_caveat1, f, indent=2)",
    ]))

    cells.append(code([
        "# Aggregate Caveat #1",
        "from collections import defaultdict",
        "by_r = defaultdict(list)",
        "for r in results_caveat1: by_r[r['repo']].append(r)",
        "print(f'{\"repo\":40s}  base_t  probe_t  rand_t   probe_mean_len  rand_mean_len')",
        "for repo, rows in by_r.items():",
        "    bt = sum(r['baseline']['term'] for r in rows)",
        "    pt = sum(r['probe_p50']['term'] for r in rows)",
        "    rt = sum(r['random_p50']['term'] for r in rows)",
        "    pl = [r['probe_p50']['len'] for r in rows if r['probe_p50']['term']]",
        "    rl = [r['random_p50']['len'] for r in rows if r['random_p50']['term']]",
        "    print(f'{repo:40s}  {bt}/{len(rows)}     {pt}/{len(rows)}     {rt}/{len(rows)}     '",
        "          f'{(sum(pl)/len(pl) if pl else 0):8.0f}      {(sum(rl)/len(rl) if rl else 0):8.0f}')",
        "",
        "# Fisher exact",
        "from scipy.stats import fisher_exact",
        "pt_all = sum(r['probe_p50']['term']  for r in results_caveat1)",
        "rt_all = sum(r['random_p50']['term'] for r in results_caveat1)",
        "n = len(results_caveat1)",
        "table = [[pt_all, n - pt_all], [rt_all, n - rt_all]]",
        "odds, pval = fisher_exact(table, alternative='greater')",
        "print(f'\\nFisher exact (probe>random termination): OR={odds:.1f}  p={pval:.4f}')",
    ]))

    # ---------- 5. Caveat #2 ----------
    cells.append(md([
        "## 5. Caveat #2 — Budget extension (MAX_NEW_TOK=2048)",
        "",
        "Test whether the 0/10 baseline-terminate result is a budget artifact. Re-run all 10 baselines with",
        "MAX_NEW_TOK=2048. Expected: 0/10 still fail → overthinking is genuine. >0/10 terminate → soften rescue claim.",
    ]))

    cells.append(code([
        "@torch.no_grad()",
        "def baseline_at_budget(prompt_text, max_new_tok=2048):",
        "    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)",
        "    prompt_len = inputs.input_ids.shape[1]",
        "    out_ids = model.generate(**inputs, max_new_tokens=max_new_tok,",
        "                             do_sample=False, temperature=None, top_p=None,",
        "                             pad_token_id=tokenizer.eos_token_id)",
        "    gen_ids = out_ids[0, prompt_len:].tolist()",
        "    terminated = THINK_END_ID in gen_ids",
        "    thinking_len = (gen_ids.index(THINK_END_ID) + 1) if terminated else len(gen_ids)",
        "    return {'thinking_len': thinking_len, 'terminated': terminated}",
        "",
        "results_caveat2 = []",
        "for i, ex in enumerate(sampled, 1):",
        "    r = baseline_at_budget(build_prompt(ex), max_new_tok=2048)",
        "    print(f'[{i:2d}/10] {ex[\"instance_id\"]:35s}  len={r[\"thinking_len\"]}  term={r[\"terminated\"]}')",
        "    results_caveat2.append({'instance_id': ex['instance_id'], **r})",
        "with open(OUT_DIR / 'caveat2_budget2048.json', 'w') as f:",
        "    json.dump(results_caveat2, f, indent=2)",
        "",
        "n_term = sum(r['terminated'] for r in results_caveat2)",
        "if n_term == 0:",
        "    print(f'\\n🟢 LOCKED: 0/10 terminate at 2048. Rescue is genuine.')",
        "elif n_term <= 3:",
        "    print(f'\\n🟡 PARTIAL: {n_term}/10 terminate at 2048. Soften claim.')",
        "else:",
        "    print(f'\\n🔴 ARTIFACT: {n_term}/10 terminate at 2048. Rescue framing invalid.')",
    ]))

    # ---------- 6. α-sweep B ----------
    cells.append(md([
        "## 6. α-sweep B — weaker alphas {+30, +40}",
        "",
        "Characterize the basin transition between α=+30 (insufficient) and α=+50 (canonical).",
        "Expected: monotonic increase in termination rate with α. Sharp transition near α=+40 → +50",
        "supports the discrete-basin attractor interpretation.",
    ]))

    cells.append(code([
        "ALPHAS_B = [30, 40]",
        "results_alpha_b = []",
        "for i, ex in enumerate(sampled, 1):",
        "    prompt = build_prompt(ex)",
        "    print(f'[{i:2d}/10] {ex[\"repo\"]:25s} / {ex[\"instance_id\"]}')",
        "    row = {'i': i, 'repo': ex['repo'], 'instance_id': ex['instance_id'], 'by_alpha': {}}",
        "    for a in ALPHAS_B:",
        "        out = generate_one(prompt, direction=probe_w, alpha=a)",
        "        row['by_alpha'][a] = {'len': out['thinking_len'], 'term': out['terminated']}",
        "        print(f'     α=+{a:2d}  term={out[\"terminated\"]}  len={out[\"thinking_len\"]:4d}')",
        "    results_alpha_b.append(row)",
        "    with open(OUT_DIR / 'alpha_sweep_quality.json', 'w') as f:",
        "        json.dump(results_alpha_b, f, indent=2)",
        "",
        "print('\\n=== α-sweep aggregate ===')",
        "for a in ALPHAS_B:",
        "    rows = [r['by_alpha'][a] for r in results_alpha_b]",
        "    nt = sum(r['term'] for r in rows)",
        "    lens = [r['len'] for r in rows if r['term']]",
        "    ml = sum(lens)/len(lens) if lens else 0",
        "    print(f'  α=+{a:2d}     {nt}/10      mean_len_term={ml:.0f}')",
        "print('  α=+50     9/10   mean_len_term=269   (from Caveat #1)')",
        "print('  base      0/10   —                  (Caveat #2 confirmed at 2048)')",
    ]))

    # ---------- 7. Design E ----------
    cells.append(md([
        "## 7. Design E — Closed-loop probe-as-sensor (threshold trigger)",
        "",
        "**Hypothesis** (turned out FALSE): the probe is a state-attractor; firing α=+50 when the residual reaches",
        "a 'near-end' state (predicted fraction ≥ 0.85) should cause clean termination.",
        "",
        "**Result**: 1/10 termination at threshold 0.85; max predicted fraction during overthinking is ~0.7-0.8,",
        "never reaching 0.85. Lower thresholds (0.65, 0.70) trigger but still don't terminate.",
        "**Reveals**: late commit does NOT cause termination, even with α=+50 active for 500+ tokens.",
    ]))

    cells.append(code([
        "CL_THRESHOLDS = [0.65, 0.70, 0.85]",
        "results_design_e = []",
        "for i, ex in enumerate(sampled, 1):",
        "    prompt = build_prompt(ex)",
        "    print(f'[{i:2d}/10] {ex[\"repo\"]:25s} / {ex[\"instance_id\"]}')",
        "    row = {'i': i, 'repo': ex['repo'], 'instance_id': ex['instance_id'], 'by_threshold': {}}",
        "    for th in CL_THRESHOLDS:",
        "        out = generate_closed_loop(prompt, threshold=th, commit_alpha=50, min_decode_steps=50)",
        "        row['by_threshold'][th] = {k: v for k, v in out.items() if k != 'text'}",
        "        print(f'     th={th:.2f}  term={out[\"terminated\"]}  len={out[\"thinking_len\"]:4d}  '",
        "              f'commit@={out[\"committed_at\"]}  max_pred={out[\"trace_max\"]:.3f}')",
        "    results_design_e.append(row)",
        "    with open(OUT_DIR / 'design_e_results.json', 'w') as f:",
        "        json.dump(results_design_e, f, indent=2)",
    ]))

    # ---------- 8. Design F ----------
    cells.append(md([
        "## 8. Design F — Plateau detector (predicted-fraction stasis)",
        "",
        "**Hypothesis** (turned out FALSE): if model's predicted fraction stops changing (plateau in a rolling",
        "window), it's stuck in overthinking → trigger commit.",
        "",
        "**Result**: 0/10 plateau-triggered, even with relaxed window (w=50). Predicted fraction oscillates",
        "more than δ=0.02 in any reasonable window during overthinking — too noisy to be a plateau signal.",
    ]))

    cells.append(code([
        "PLATEAU_CONFIGS = [(100, 0.02), (50, 0.02)]",
        "results_design_f = []",
        "for i, ex in enumerate(sampled, 1):",
        "    prompt = build_prompt(ex)",
        "    print(f'[{i:2d}/10] {ex[\"repo\"]:25s} / {ex[\"instance_id\"]}')",
        "    row = {'i': i, 'repo': ex['repo'], 'instance_id': ex['instance_id'], 'by_config': {}}",
        "    for w, d in PLATEAU_CONFIGS:",
        "        out = generate_plateau(prompt, window=w, delta_thresh=d, commit_alpha=50,",
        "                               min_decode_steps=max(80, w))",
        "        key = f'w{w}_d{d}'",
        "        row['by_config'][key] = {k: v for k, v in out.items() if k != 'text'}",
        "        print(f'     {key:12s}  term={out[\"terminated\"]}  len={out[\"thinking_len\"]:4d}  '",
        "              f'commit@={out[\"committed_at\"]}')",
        "    results_design_f.append(row)",
        "    with open(OUT_DIR / 'design_f_results.json', 'w') as f:",
        "        json.dump(results_design_f, f, indent=2)",
    ]))

    # ---------- 9. Onset timing ----------
    cells.append(md([
        "## 9. Onset timing experiment — mechanism isolation (state-vs-trajectory)",
        "",
        "**Question**: is the Design E failure because the SENSOR misfires, or because LATE STEERING is",
        "fundamentally weaker than from-start steering (KV-cache hypothesis)?",
        "",
        "**Setup**: static α=+50 with deterministic onset at decode step {50, 200, 400} — no sensor.",
        "",
        "**Read**:",
        "- If onset=50 ≈ 9/10 termination: SENSOR was the problem; late triggering with the right threshold could work.",
        "- If onset=50 ≈ 5-6/10 and decay monotonically with onset: TRAJECTORY-dependent (KV cache lock-in confirmed).",
        "- If all onsets ≈ 1-2/10: timing is binary; any delay kills the effect.",
    ]))

    cells.append(code([
        "ONSETS = [50, 200, 400]",
        "results_onset = []",
        "for i, ex in enumerate(sampled, 1):",
        "    prompt = build_prompt(ex)",
        "    print(f'[{i:2d}/10] {ex[\"repo\"]} / {ex[\"instance_id\"]}')",
        "    row = {'i': i, 'repo': ex['repo'], 'instance_id': ex['instance_id'], 'by_onset': {}}",
        "    for s in ONSETS:",
        "        out = generate_delayed_static(prompt, onset_step=s, alpha=50)",
        "        row['by_onset'][s] = out",
        "        print(f'     onset@{s:3d}  term={out[\"terminated\"]}  len={out[\"thinking_len\"]:4d}  '",
        "              f'active_from={out[\"active_from\"]}')",
        "    results_onset.append(row)",
        "    with open(OUT_DIR / 'onset_timing_results.json', 'w') as f:",
        "        json.dump([{**r, 'by_onset': {k: {kk: vv for kk, vv in v.items() if kk != 'text'}",
        "                                       for k, v in r['by_onset'].items()}}",
        "                   for r in results_onset], f, indent=2)",
        "",
        "print('\\n=== Onset timing verdict ===')",
        "for s in ONSETS:",
        "    rows = [r['by_onset'][s] for r in results_onset]",
        "    nt = sum(r['terminated'] for r in rows)",
        "    lens = [r['thinking_len'] for r in rows if r['terminated']]",
        "    ml = sum(lens)/len(lens) if lens else 0",
        "    print(f'  onset={s:3d}   {nt}/10   mean_len_term={ml:.0f}')",
        "",
        "print('  (from prior:  static_from_token1 = 9/10 mean 269; closed_loop ≈ 1/10)')",
    ]))

    # ---------- 10. Final verdict ----------
    cells.append(md([
        "## 10. Phase 2B verdict",
        "",
        "Consolidated decision based on all 6 experiments. Maps to paper-8 section revisions:",
        "",
        "| Question | Finding | Paper section affected |",
        "|---|---|---|",
        "| Q1 cross-domain | ✅ 19/20 probe rescue (Fisher p<0.001) | §7 (locked) |",
        "| Q2 budget artifact | ✅ 0/10 at 2048 — genuine | §7 last paragraph |",
        "| Q3 basin transition | ✅ sharp between +30 (1/10) → +40 (7/10) → +50 (9/10) | §5 dose-response strengthened |",
        "| Q4 closed-loop E | ❌ 1/10 — single-state-attractor model FALSIFIED | §7.3 new + §10 SDK reframe |",
        "| Q5 plateau F | ❌ 0/10 — plateau signal too noisy | §7.3 |",
        "| Q6 onset timing | [see cell] | §7.4 + abstract refinement |",
        "",
        "If Q6 confirms KV-cache trajectory-dependence:",
        "- Paper retitle to 'Trajectory-Shaping Probe Steering in Qwen3.6-27B'",
        "- §10 SDK reframe: `agent-probe-guard` anti_overthinking mode must apply steering from gen start (token 1),",
        "  not as post-hoc detector. Becomes a 'compute budget enforcer', not 'adaptive intelligence'.",
        "",
        "If Q6 shows onset=50 still works ~9/10:",
        "- Closed-loop CAN work, just needs earlier triggering. Design E' with very low threshold + early min_decode.",
        "- §10 SDK reframe: closed-loop is viable with calibrated early-trigger threshold.",
    ]))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = NOTEBOOKS_DIR / "nb_subjective_time_phase2b_steering_designs.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    print(f"✓ wrote {out_path}")
    print(f"  cells: {len(cells)}")
    return out_path


if __name__ == "__main__":
    build()
