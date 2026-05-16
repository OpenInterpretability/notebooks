"""
Builder for nb_subjective_time_phase2c_cross_layer_kv_cache.ipynb.

Subjective-Time Probe Phase 2C — cross-layer extension of the KV-cache lock-in finding.

Phase 2B established (at L31) that probe-steering causality is trajectory-dependent:
onset @ token 1 = 9/10 termination; onset @ step 50 = 3/10; onset @ step 200+ = 0/10.
This is a layer-specific finding. Phase 2C asks: does the same mechanism appear
at the OTHER layers where v1 found high-R² subjective-time probes?

Setup: subjective-time v1 probe at L11 (R²=0.84) and L55 (R²=0.82) — refit
from same cache as Phase 2A/2B. Then for each layer:

(A) Static probe@+50 from token 1 on 10 cross-repo SWE-bench prompts — establish
    whether the layer's direction is causal at all.
(B) If layer is causal, run onset-timing {step 50, 200, 400} to test KV-cache lock-in.

Compute: ~50 min on A100 80GB. 80-100 generations depending on which layers lever.

Decision matrix:

| L11 causal? | L55 causal? | Onset pattern at L11/L55 | Interpretation for paper-8 |
|---|---|---|---|
| ✓ | ✓ | Both show same decay as L31 | KV-cache mechanism is LAYER-GENERAL within probe family. §9.8 + §10.3 strengthened. |
| ✓ | ✓ | Different decay shapes per layer | Mechanism is gradient — some layers preserve longer trajectory effect. New paper subsection. |
| ✓ | ✗ (or vice versa) | — | Causality is layer-asymmetric independent of probe accuracy. Significant new finding. |
| ✗ | ✗ | — | L31 is special even within the probe family. KV-cache lock-in is L31-specific (more conservative claim). |

HARD RULES carry over from Phase 2A/2B:
- Random direction parallel at every alpha
- Drive checkpoint per prompt (resume-safe)
- Whitespace-stripped termination via THINK_END_ID
- Greedy decoding only

Target: Phase 2C results → paper-8 v3 (or v2.1 patch) addressing §9.8 limitation.
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

    cells.append(md([
        "# Subjective-Time Probe Phase 2C — Cross-Layer KV-Cache Lock-In",
        "",
        "**Phase 2B established** at L31 that probe-steering causality is trajectory-dependent:",
        "static-from-token-1 → 9/10 termination; delayed onset → 3/10 by step 50, 0/10 by step 200.",
        "",
        "**Phase 2C asks**: does the same trajectory-dependence appear at the *other* layers where",
        "the subjective-time probe achieves high R² (L11=0.84, L55=0.82)?",
        "",
        "**Two-stage design**:",
        "- **Stage A** (causal-effect screen): static probe@+50 from token 1 at L11 and L55.",
        "  Establishes whether either layer's direction is causal at all.",
        "- **Stage B** (KV-cache lock-in test): for any causal layer, sweep onset ∈ {50, 200, 400}",
        "  to test if KV-cache mechanism applies.",
        "",
        "**Decision matrix**:",
        "",
        "| L11 causal? | L55 causal? | Onset pattern | Interpretation |",
        "|---|---|---|---|",
        "| ✓ | ✓ | Both decay like L31 | KV-cache mechanism is LAYER-GENERAL within probe family |",
        "| ✓ | ✓ | Different decays | Mechanism is layer-gradient — refine paper-8 §10 |",
        "| Asymmetric | — | — | Causality is layer-asymmetric independent of probe R². New finding. |",
        "| ✗ | ✗ | — | L31 is special. KV-cache claim narrows to L31-specific. |",
        "",
        "**Compute**: ~50 min on A100 80 GB / RTX 6000 Blackwell 96 GB.",
        "**Self-contained**: assumes fresh Colab (no prior state required).",
    ]))

    cells.append(md(["## 1. Setup — Drive, install, model"]))

    cells.append(code([
        "from pathlib import Path",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "",
        "DRIVE    = Path('/content/drive/MyDrive')",
        "CACHE_PT = DRIVE / 'openinterp_runs' / 'predictive_sae_v1' / 'cache' / 'residuals_multilayer.pt'",
        "OUT_DIR  = DRIVE / 'openinterp_runs' / 'subjective_time_phase2c'",
        "OUT_DIR.mkdir(parents=True, exist_ok=True)",
        "",
        "assert CACHE_PT.exists(), f'cache missing: {CACHE_PT}'",
        "print(f'cache:  {CACHE_PT}')",
        "print(f'output: {OUT_DIR}')",
    ]))

    cells.append(code([
        "# Qwen3.6 (model_type='qwen3_5') requires transformers from main branch",
        "!pip uninstall -y -q transformers",
        "!pip install -q --upgrade git+https://github.com/huggingface/transformers.git accelerate sentencepiece datasets scipy",
        "print('⚠️  Restart runtime after this cell (Runtime → Restart session), then re-run from cell 1.')",
    ]))

    cells.append(code([
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
        "THINK_END_ID = tokenizer.encode('</think>', add_special_tokens=False)[0]",
        "",
        "# Layer access shim — handles both CausalLM (model.model.layers) and",
        "# ConditionalGeneration (model.model.language_model.layers) wrappers",
        "LAYERS = model.model.layers if hasattr(model.model, 'layers') else model.model.language_model.layers",
        "print('model:', type(model).__name__)",
        "print(f'layers: {len(LAYERS)}  (via {\"direct\" if hasattr(model.model, \"layers\") else \"language_model\"})')",
        "print('device:', device, 'dtype:', model.dtype)",
    ]))

    cells.append(md(["## 2. Refit probes at L11, L31, L55 (from cache, ~10s total)"]))

    cells.append(code([
        "import numpy as np",
        "from sklearn.linear_model import Ridge",
        "from sklearn.metrics import r2_score",
        "",
        "cache = torch.load(CACHE_PT, map_location='cpu')",
        "FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]",
        "",
        "def refit_layer(layer_idx, seed=42):",
        "    sub = cache[layer_idx]",
        "    Xs, ys, pidx = [], [], []",
        "    for f in FRACTIONS:",
        "        t = sub[f].numpy()",
        "        Xs.append(t); ys.append(np.full(t.shape[0], f, dtype=np.float32))",
        "        pidx.append(np.arange(t.shape[0]))",
        "    X = np.concatenate(Xs); y = np.concatenate(ys); pid = np.concatenate(pidx)",
        "    rng = np.random.default_rng(seed)",
        "    perm = np.arange(sub[0.1].shape[0]); rng.shuffle(perm)",
        "    train_set = set(perm[:int(0.8 * len(perm))].tolist())",
        "    mtr = np.array([p in train_set for p in pid])",
        "    probe = Ridge(alpha=1.0).fit(X[mtr], y[mtr])",
        "    r2 = r2_score(y[~mtr], probe.predict(X[~mtr]))",
        "    coef = torch.from_numpy(probe.coef_.astype(np.float32))",
        "    unit = (coef / coef.norm()).to(device=device, dtype=model.dtype)",
        "    return {",
        "        'layer': layer_idx,",
        "        'r2': r2,",
        "        'coef_raw': coef,",
        "        'intercept': float(probe.intercept_),",
        "        'unit': unit,",
        "    }",
        "",
        "PROBES = {L: refit_layer(L) for L in [11, 31, 55]}",
        "for L, p in PROBES.items():",
        "    print(f'  L{L}:  R²={p[\"r2\"]:.4f}  coef_norm={p[\"coef_raw\"].norm().item():.4f}  intercept={p[\"intercept\"]:.4f}')",
        "",
        "# Matched random direction (same seed=42, same as Phase 2A/2B)",
        "g = torch.Generator(device='cpu').manual_seed(42)",
        "r = torch.randn(PROBES[31]['unit'].shape[0], generator=g)",
        "RANDOM_W = (r / r.norm()).to(device=device, dtype=model.dtype)",
        "print(f'random_w norm={RANDOM_W.float().norm().item():.4f}')",
    ]))

    cells.append(md(["## 3. Cross-repo sample (same 10 as Phase 2B for direct comparability)"]))

    cells.append(code([
        "import random as pyrandom",
        "from collections import defaultdict",
        "from datasets import load_dataset",
        "",
        "pyrandom.seed(42)",
        "swe = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')",
        "by_repo = defaultdict(list)",
        "for ex in swe:",
        "    if ex['repo'] != 'astropy/astropy':",
        "        by_repo[ex['repo']].append(ex)",
        "repos_ranked = sorted(by_repo.keys(), key=lambda r: -len(by_repo[r]))",
        "target_repos = repos_ranked[:5]",
        "sampled = []",
        "for r in target_repos:",
        "    sampled.extend(pyrandom.sample(by_repo[r], 2))",
        "print(f'Sampled {len(sampled)} problems across {len(target_repos)} repos')",
        "for s in sampled:",
        "    print(f'  {s[\"repo\"]:40s}  {s[\"instance_id\"]}')",
    ]))

    cells.append(md(["## 4. Helpers — prompt builder + generation hooks (layer-parameterized)"]))

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
        "def gen_static(prompt_text, layer_idx, direction, alpha):",
        "    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)",
        "    prompt_len = inputs.input_ids.shape[1]",
        "    handle = None",
        "    if direction is not None:",
        "        d = direction.to(device=device, dtype=model.dtype)",
        "        def _hook(_m, _i, out):",
        "            h = out[0] if isinstance(out, tuple) else out",
        "            h2 = h + alpha * d",
        "            return (h2,) + out[1:] if isinstance(out, tuple) else h2",
        "        handle = LAYERS[layer_idx].register_forward_hook(_hook)",
        "    try:",
        "        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOK,",
        "                                 do_sample=False, temperature=None, top_p=None,",
        "                                 pad_token_id=tokenizer.eos_token_id)",
        "    finally:",
        "        if handle is not None: handle.remove()",
        "    gen_ids = out_ids[0, prompt_len:].tolist()",
        "    term = THINK_END_ID in gen_ids",
        "    tlen = (gen_ids.index(THINK_END_ID) + 1) if term else len(gen_ids)",
        "    return {'thinking_len': tlen, 'terminated': term}",
        "",
        "@torch.no_grad()",
        "def gen_delayed(prompt_text, layer_idx, direction, alpha, onset_step):",
        "    inputs = tokenizer(prompt_text, return_tensors='pt').to(device)",
        "    prompt_len = inputs.input_ids.shape[1]",
        "    state = {'decode_steps': 0, 'active_from': None}",
        "    d = direction.to(device=device, dtype=model.dtype)",
        "    def _hook(_m, _i, out):",
        "        h = out[0] if isinstance(out, tuple) else out",
        "        if h.shape[1] == 1:",
        "            state['decode_steps'] += 1",
        "            if state['decode_steps'] >= onset_step:",
        "                if state['active_from'] is None:",
        "                    state['active_from'] = state['decode_steps']",
        "                h2 = h + alpha * d",
        "                return (h2,) + out[1:] if isinstance(out, tuple) else h2",
        "        return out",
        "    handle = LAYERS[layer_idx].register_forward_hook(_hook)",
        "    try:",
        "        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOK,",
        "                                 do_sample=False, temperature=None, top_p=None,",
        "                                 pad_token_id=tokenizer.eos_token_id)",
        "    finally:",
        "        handle.remove()",
        "    gen_ids = out_ids[0, prompt_len:].tolist()",
        "    term = THINK_END_ID in gen_ids",
        "    tlen = (gen_ids.index(THINK_END_ID) + 1) if term else len(gen_ids)",
        "    return {'thinking_len': tlen, 'terminated': term, 'active_from': state['active_from']}",
    ]))

    cells.append(md([
        "## 5. Stage A — Causal-effect screen at L11 and L55 (static from token 1)",
        "",
        "Establish whether the subjective-time direction at L11 / L55 levers termination at all.",
        "Each layer: baseline + probe@+50 + random@+50. 10 prompts × 3 conditions × 2 layers = 60 gens (~30 min).",
        "",
        "Reuses Phase 2A finding at L31 (9/10 probe rescue, 6/20 random) — not re-run here.",
        "If a layer's probe@+50 doesn't substantively differ from random@+50, that layer is NON-CAUSAL",
        "and we skip Stage B for it.",
    ]))

    cells.append(code([
        "import json",
        "ALPHA = 50",
        "STAGE_A_LAYERS = [11, 55]",
        "",
        "results_stage_a = []",
        "for i, ex in enumerate(sampled, 1):",
        "    prompt = build_prompt(ex)",
        "    row = {'i': i, 'repo': ex['repo'], 'instance_id': ex['instance_id'], 'by_layer': {}}",
        "    print(f'[{i:2d}/10] {ex[\"repo\"]} / {ex[\"instance_id\"]}')",
        "    for L in STAGE_A_LAYERS:",
        "        base = gen_static(prompt, L, None, 0)",
        "        pro  = gen_static(prompt, L, PROBES[L]['unit'], ALPHA)",
        "        rnd  = gen_static(prompt, L, RANDOM_W, ALPHA)",
        "        row['by_layer'][L] = {",
        "            'baseline':   {'len': base['thinking_len'], 'term': base['terminated']},",
        "            'probe_p50':  {'len': pro['thinking_len'],  'term': pro['terminated']},",
        "            'random_p50': {'len': rnd['thinking_len'],  'term': rnd['terminated']},",
        "        }",
        "        print(f'     L{L:2d}  base term={base[\"terminated\"]} len={base[\"thinking_len\"]:4d}  |  '",
        "              f'probe term={pro[\"terminated\"]} len={pro[\"thinking_len\"]:4d}  |  '",
        "              f'rand  term={rnd[\"terminated\"]} len={rnd[\"thinking_len\"]:4d}')",
        "    results_stage_a.append(row)",
        "    with open(OUT_DIR / 'stage_a_causal_screen.json', 'w') as f:",
        "        json.dump(results_stage_a, f, indent=2)",
        "",
        "# Aggregate Stage A",
        "from scipy.stats import fisher_exact",
        "print('\\n=== Stage A aggregate ===')",
        "stage_a_summary = {}",
        "for L in STAGE_A_LAYERS:",
        "    rows = [r['by_layer'][L] for r in results_stage_a]",
        "    bt = sum(r['baseline']['term']   for r in rows)",
        "    pt = sum(r['probe_p50']['term']  for r in rows)",
        "    rt = sum(r['random_p50']['term'] for r in rows)",
        "    pl_t = [r['probe_p50']['len']  for r in rows if r['probe_p50']['term']]",
        "    rl_t = [r['random_p50']['len'] for r in rows if r['random_p50']['term']]",
        "    odds, pval = fisher_exact([[pt, 10-pt], [rt, 10-rt]], alternative='greater')",
        "    causal = (pt >= 5) and (pt - rt >= 3) and (pval < 0.10)",
        "    stage_a_summary[L] = {",
        "        'base_term': bt, 'probe_term': pt, 'random_term': rt,",
        "        'probe_mean_len': (sum(pl_t)/len(pl_t) if pl_t else None),",
        "        'random_mean_len': (sum(rl_t)/len(rl_t) if rl_t else None),",
        "        'fisher_odds': odds, 'fisher_p': pval, 'is_causal': causal,",
        "    }",
        "    print(f'  L{L:2d}:  base={bt}/10  probe={pt}/10  random={rt}/10  '",
        "          f'OR={odds:.1f}  p={pval:.4f}  causal={causal}')",
        "print('  L31 (Phase 2A reference): probe=9/10  random=3/10  OR=14.0  p≈0.02  causal=True')",
        "",
        "with open(OUT_DIR / 'stage_a_summary.json', 'w') as f:",
        "    json.dump(stage_a_summary, f, indent=2)",
    ]))

    cells.append(md([
        "## 6. Stage B — Onset-timing sweep on causal layers",
        "",
        "For each layer that passed Stage A (probe-vs-random termination gap statistically significant),",
        "test KV-cache lock-in: static probe@+50 with onset at decode step ∈ {50, 200, 400}.",
        "",
        "Expected (if mechanism is layer-general): same decay pattern as L31 — onset=50 produces partial rescue,",
        "onset≥200 produces no rescue. Up to 60 generations (~30 min) depending on causal layers.",
    ]))

    cells.append(code([
        "ONSETS = [50, 200, 400]",
        "causal_layers = [L for L, s in stage_a_summary.items() if s['is_causal']]",
        "print(f'Causal layers from Stage A: {causal_layers}')",
        "if not causal_layers:",
        "    print('🟡 No causal layers detected in Stage A. Skipping Stage B.')",
        "    print('   Verdict: KV-cache lock-in claim narrows to L31-specific (more conservative paper claim).')",
        "",
        "results_stage_b = []",
        "for i, ex in enumerate(sampled, 1):",
        "    if not causal_layers: break",
        "    prompt = build_prompt(ex)",
        "    print(f'[{i:2d}/10] {ex[\"repo\"]} / {ex[\"instance_id\"]}')",
        "    row = {'i': i, 'repo': ex['repo'], 'instance_id': ex['instance_id'], 'by_layer_by_onset': {}}",
        "    for L in causal_layers:",
        "        row['by_layer_by_onset'][L] = {}",
        "        for s in ONSETS:",
        "            out = gen_delayed(prompt, L, PROBES[L]['unit'], ALPHA, onset_step=s)",
        "            row['by_layer_by_onset'][L][s] = out",
        "            print(f'     L{L:2d}  onset={s:3d}  term={out[\"terminated\"]}  len={out[\"thinking_len\"]:4d}  '",
        "                  f'active_from={out[\"active_from\"]}')",
        "    results_stage_b.append(row)",
        "    with open(OUT_DIR / 'stage_b_onset_timing.json', 'w') as f:",
        "        json.dump(results_stage_b, f, indent=2)",
        "",
        "# Aggregate Stage B",
        "if causal_layers:",
        "    print('\\n=== Stage B aggregate ===')",
        "    print(f'{\"layer\":>6s}  {\"onset\":>6s}  term_rate  mean_len_term')",
        "    stage_b_summary = {}",
        "    for L in causal_layers:",
        "        stage_b_summary[L] = {}",
        "        for s in ONSETS:",
        "            rows = [r['by_layer_by_onset'][L][s] for r in results_stage_b]",
        "            nt = sum(r['terminated'] for r in rows)",
        "            lt = [r['thinking_len'] for r in rows if r['terminated']]",
        "            ml = sum(lt)/len(lt) if lt else 0",
        "            stage_b_summary[L][s] = {'n_term': nt, 'mean_len_term': ml}",
        "            print(f'  L{L:>3d}  {s:>5d}   {nt}/10      {ml:>6.0f}')",
        "        print('   ----  Phase 2A reference at L31:  token1 = 9/10 mean 269  /  step 50 = 3/10  /  step 200+ = 0/10')",
        "    with open(OUT_DIR / 'stage_b_summary.json', 'w') as f:",
        "        json.dump(stage_b_summary, f, indent=2)",
    ]))

    cells.append(md([
        "## 7. Verdict — does KV-cache lock-in generalize across layers?",
        "",
        "Compare each causal layer's onset-timing decay against the L31 reference. Three possible outcomes:",
        "",
        "**(i) Layer-general**: L11 and/or L55 show the same monotonic decay {token1 high, step50 partial, step200+ zero}.",
        "→ KV-cache mechanism is general within the probe family. Paper-8 §9.8 limitation closed.",
        "→ Paper-8 §10.3 implications strengthened from speculative to empirical.",
        "",
        "**(ii) Layer-gradient**: different layers show different decay rates (e.g., L11 decays slower, L55 faster).",
        "→ Mechanism is layer-dependent. New paper subsection / appendix characterizing the gradient.",
        "",
        "**(iii) L31-specific**: only L31 shows the decay; L11/L55 either not causal OR causal without decay.",
        "→ KV-cache claim narrows. Paper-8 stays as-is with a stronger §10.1 caveat about L31 specificity.",
    ]))

    cells.append(code([
        "# Verdict logic",
        "print('=== PHASE 2C VERDICT ===\\n')",
        "if not causal_layers:",
        "    verdict = 'iii) L31-specific — neither L11 nor L55 causal at α=+50'",
        "elif len(causal_layers) == 1:",
        "    L = causal_layers[0]",
        "    n200 = stage_b_summary[L][200]['n_term']",
        "    if n200 <= 2:",
        "        verdict = f'i) Layer-general (partial): L{L} shows KV-cache decay matching L31 ({n200}/10 at onset 200)'",
        "    else:",
        "        verdict = f'ii) Layer-gradient: L{L} causal but onset 200 still rescues {n200}/10 (vs L31 = 0/10)'",
        "else:",
        "    decays = {L: stage_b_summary[L][200]['n_term'] for L in causal_layers}",
        "    if all(v <= 2 for v in decays.values()):",
        "        verdict = f'i) Layer-general: all of {list(decays.keys())} show KV-cache decay (onset 200 → 0-2/10)'",
        "    else:",
        "        verdict = f'ii) Layer-gradient: decays = {decays} — heterogeneous across layers'",
        "print(verdict)",
        "",
        "with open(OUT_DIR / 'phase_2c_verdict.json', 'w') as f:",
        "    json.dump({",
        "        'verdict': verdict,",
        "        'stage_a_summary': stage_a_summary,",
        "        'stage_b_summary': stage_b_summary if causal_layers else None,",
        "        'l31_reference': {'token1': 9, 'step50': 3, 'step200': 0, 'step400': 0},",
        "    }, f, indent=2)",
        "print(f'\\n✓ saved verdict to {OUT_DIR / \"phase_2c_verdict.json\"}')",
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

    out_path = NOTEBOOKS_DIR / "nb_subjective_time_phase2c_cross_layer_kv_cache.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    print(f"✓ wrote {out_path}")
    print(f"  cells: {len(cells)}")
    return out_path


if __name__ == "__main__":
    build()
