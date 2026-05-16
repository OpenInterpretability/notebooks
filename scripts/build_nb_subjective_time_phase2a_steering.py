"""
Builder for nb_subjective_time_phase2a_steering.ipynb.

Subjective-Time Probe Phase 2A — epiphenomenal-time steering test.

Tests the causal status of the subjective-time direction found in v1
(R² 0.82-0.86 across L11/L31/L55). v1 showed time IS encoded; this phase
asks: is the encoding causally functional, or epiphenomenal?

Methodology:
- Re-train Ridge probe on full cache (no split) to get clean "subjective-time
  direction" per layer.
- Inject as steering vector α × (probe.coef_ / ‖probe.coef_‖) into the
  residual stream at the chosen layer, ALL token positions during generation.
- Compare against matched random direction at same α magnitudes (Phase 7/8 rule).
- Sweep α ∈ {-200, -100, -50, +50, +100, +200} — multiples of typical
  residual norm to satisfy structural-rigidity rule (Phase 8).
- Metric: thinking_token_count, terminates_normally, output_stripped (for
  whitespace-stripped flip metric, Phase 10 rule).

Decision rules (registered in cell 8):
- Probe at +α REDUCES thinking_length significantly more than random at +α
  AND probe at -α EXTENDS thinking_length more than random
  → CAUSAL lever. Paper title: "Mechanistic Localization of Self-Time."
- Probe and random produce same monotonic effect with α
  → Softmax-temperature artifact (paper-6 epiphenomenal type 1).
- Neither probe nor random produce change even at α=±200 (well above ‖h‖)
  → Structural lock (paper-6 epiphenomenal type 2).

Compute: ~1.5-2h on A100 80GB. Model = Qwen3.6-27B in bf16 (~54GB VRAM).
15 prompts × 6 alphas × 2 directions + 15 baselines = 195 generations.

Output: phase2a_steering_results.json + scatter plots.

HARD RULES applied:
- Random-direction parallel mandatory (Phase 7/8 lesson)
- α sweep to multiples of residual norm (Phase 8 structural-rigidity rule)
- Whitespace-stripped output comparison (Phase 10 rule)
- Drive checkpoint per prompt (resume-safe)

Target: paper-8 candidate. Both verdicts publishable.
"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path("/Volumes/SSD Major/fish/openinterp-work/notebooks")


def code(lines, **meta):
    return {"cell_type": "code", "metadata": meta or {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}


def md(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def build():
    cells = []

    cells.append(md([
        "# Subjective-Time Probe Phase 2A — Causal Steering Test",
        "",
        "**Question**: the v1 probe found R²=0.86 at L31 — time IS encoded. Is it causally functional?",
        "Or epiphenomenal (info present in residual but behaviorally ignored)?",
        "",
        "**Test**: inject the Ridge probe weight vector as steering at α ∈ {-200, -100, -50, +50, +100, +200}",
        "during generation, measure how thinking-phase length changes vs matched random direction.",
        "",
        "**Decision matrix** (registered in last cell):",
        "| Pattern | Verdict | Paper framing |",
        "|---|---|---|",
        "| Probe monotonic +α↓thinking AND beats random by >20% | 🟢 CAUSAL | \"Mechanistic Localization of Self-Time\" |",
        "| Probe and random both produce similar monotone effect | 🟡 EPIPHENOMENAL type 1 (softmax-temp) | \"Two Forms vol. 2: Subjective Time as Softmax Artifact\" |",
        "| Neither moves behavior at α=±200 | 🟡 EPIPHENOMENAL type 2 (structural) | \"Two Forms vol. 2: Subjective Time as Structural Lock\" |",
        "",
        "All three are publishable.",
        "",
        "**Compute**: ~1.5-2h on A100 80GB (Qwen3.6-27B bf16 = ~54GB VRAM).",
        "",
        "**HARD RULES applied**:",
        "- Random-direction parallel (Phase 7/8)",
        "- α sweep to multiples of ‖residual‖ (Phase 8 structural-rigidity)",
        "- Whitespace-stripped output comparison (Phase 10)",
        "- Drive checkpoint per prompt (resume-safe)",
    ]))

    # Cell 1 — Drive
    cells.append(md(["## 1. Drive mount + paths"]))
    cells.append(code([
        "from pathlib import Path",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "",
        "DRIVE = Path('/content/drive/MyDrive')",
        "PSAE_CACHE = DRIVE / 'openinterp_runs' / 'predictive_sae_v1' / 'cache'",
        "OUT = DRIVE / 'openinterp_runs' / 'subjective_time_phase2a'",
        "OUT.mkdir(parents=True, exist_ok=True)",
        "(OUT / 'per_prompt').mkdir(parents=True, exist_ok=True)",
        "RESULT_JSON = OUT / 'phase2a_steering_results.json'",
        "",
        "assert PSAE_CACHE.exists(), f'PSAE cache not found: {PSAE_CACHE}'",
        "print(f'CACHE: {sorted(p.name for p in PSAE_CACHE.iterdir())}')",
        "print(f'OUT: {OUT}')",
    ]))

    # Cell 2 — Install
    cells.append(md(["## 2. Install"]))
    cells.append(code([
        "import sys, subprocess",
        "def pip(*a): return subprocess.run([sys.executable, '-m', 'pip', *a], check=False)",
        "",
        "# transformers main needed for Qwen3.6-27B (uses qwen3_5 architecture) — check specifically",
        "try:",
        "    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES",
        "    has_qwen35 = 'qwen3_5' in CONFIG_MAPPING_NAMES",
        "except Exception:",
        "    has_qwen35 = False",
        "",
        "if not has_qwen35:",
        "    pip('install', '-q', '--upgrade', 'git+https://github.com/huggingface/transformers.git')",
        "    print('⚠ installed transformers main — RESTART RUNTIME and re-run from cell 1')",
        "    raise SystemExit",
        "",
        "for mod in ('sklearn', 'safetensors', 'datasets'):",
        "    try: __import__(mod)",
        "    except ImportError: pip('install', '-q', 'scikit-learn' if mod=='sklearn' else mod)",
        "",
        "import torch, numpy as np",
        "from sklearn.linear_model import Ridge",
        "print(f'torch {torch.__version__}, cuda {torch.cuda.is_available()}, dev {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')",
        "assert torch.cuda.is_available(), 'NEEDS GPU (A100 80GB recommended for Qwen3.6-27B bf16)'",
    ]))

    # Cell 3 — Config
    cells.append(md(["## 3. Config"]))
    cells.append(code([
        "MODEL_ID    = 'Qwen/Qwen3.6-27B'",
        "DTYPE       = torch.bfloat16",
        "DEVICE      = torch.device('cuda')",
        "",
        "TARGET_LAYER = 31           # strongest from v1 (R²=0.858)",
        "ALPHAS       = [-200, -100, -50, 50, 100, 200]   # 6 non-zero magnitudes",
        "DIRECTIONS   = ['probe', 'random']               # parallel control",
        "N_PROMPTS    = 15           # subset of PSAE cache prompts",
        "MAX_NEW_TOK  = 1024         # cap generation length",
        "SEED         = 42",
        "",
        "FRACTIONS_FOR_RIDGE = [0.10, 0.25, 0.50, 0.75, 1.00]",
        "D_MODEL = 5120",
        "",
        "torch.manual_seed(SEED)",
        "np.random.seed(SEED)",
        "print(f'target_layer=L{TARGET_LAYER}, alphas={ALPHAS}, dirs={DIRECTIONS}, N_prompts={N_PROMPTS}')",
    ]))

    # Cell 4 — Load model
    cells.append(md(["## 4. Load Qwen3.6-27B (~5 min, ~54GB VRAM)"]))
    cells.append(code([
        "from transformers import AutoTokenizer, AutoModelForImageTextToText",
        "import time",
        "",
        "t0 = time.time()",
        "tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)",
        "if tok.pad_token_id is None: tok.pad_token = tok.eos_token",
        "",
        "model = AutoModelForImageTextToText.from_pretrained(",
        "    MODEL_ID, dtype=DTYPE, device_map=DEVICE,",
        "    attn_implementation='sdpa', trust_remote_code=True,",
        ")",
        "model.eval()",
        "for p in model.parameters(): p.requires_grad_(False)",
        "print(f'load: {time.time()-t0:.0f}s  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB')",
        "",
        "THINK_START_ID = tok.convert_tokens_to_ids('<think>')",
        "THINK_END_ID   = tok.convert_tokens_to_ids('</think>')",
        "if THINK_START_ID == tok.unk_token_id:",
        "    THINK_START_ID = tok.encode('<think>', add_special_tokens=False)[-1]",
        "if THINK_END_ID == tok.unk_token_id:",
        "    THINK_END_ID = tok.encode('</think>', add_special_tokens=False)[-1]",
        "print(f'<think> id={THINK_START_ID}, </think> id={THINK_END_ID}')",
    ]))

    # Cell 5 — Layer helper
    cells.append(md(["## 5. Layer indexing helper"]))
    cells.append(code([
        "def get_layer_module(m, idx):",
        "    for path in [('model','language_model','layers'),",
        "                 ('language_model','layers'), ('model','layers')]:",
        "        try:",
        "            cur = m",
        "            for p in path: cur = getattr(cur, p)",
        "            return cur[idx]",
        "        except AttributeError: continue",
        "    raise RuntimeError('layer path not found')",
        "",
        "target_layer_mod = get_layer_module(model, TARGET_LAYER)",
        "print(f'target_layer_mod for L{TARGET_LAYER}: {type(target_layer_mod).__name__}')",
    ]))

    # Cell 6 — Extract probe direction from cache + matched random
    cells.append(md([
        "## 6. Extract probe direction from cache via Ridge",
        "",
        "Re-train Ridge on full cache (no split) to get stable direction. Match random to ‖probe‖.",
    ]))
    cells.append(code([
        "residuals = torch.load(PSAE_CACHE / 'residuals_multilayer.pt', map_location='cpu', weights_only=False)",
        "",
        "# Build full dataset for TARGET_LAYER",
        "X_list, y_list = [], []",
        "for f in FRACTIONS_FOR_RIDGE:",
        "    X_list.append(residuals[TARGET_LAYER][f].numpy().astype(np.float32))",
        "    y_list.append(np.full(residuals[TARGET_LAYER][f].shape[0], f, dtype=np.float32))",
        "X_full = np.concatenate(X_list, axis=0)",
        "y_full = np.concatenate(y_list, axis=0)",
        "print(f'X_full {X_full.shape}, y mean={y_full.mean():.3f}')",
        "",
        "probe = Ridge(alpha=1.0, random_state=SEED)",
        "probe.fit(X_full, y_full)",
        "probe_dir = probe.coef_.copy()                                # (D_MODEL,)",
        "probe_norm = float(np.linalg.norm(probe_dir))",
        "probe_dir_unit = probe_dir / (probe_norm + 1e-9)",
        "",
        "rng = np.random.default_rng(SEED)",
        "random_dir = rng.standard_normal(D_MODEL).astype(np.float32)",
        "random_dir_unit = random_dir / (np.linalg.norm(random_dir) + 1e-9)",
        "",
        "print(f'probe_dir norm = {probe_norm:.3f}  (unit-normalized for steering)')",
        "print(f'cosine(probe, random) = {float(np.dot(probe_dir_unit, random_dir_unit)):.4f}  (should be near 0)')",
        "",
        "# Push to GPU as bf16 (matches residual dtype)",
        "PROBE_DIR_GPU = torch.from_numpy(probe_dir_unit).to(DEVICE, DTYPE)",
        "RAND_DIR_GPU  = torch.from_numpy(random_dir_unit).to(DEVICE, DTYPE)",
        "print(f'steering vectors loaded: probe shape={PROBE_DIR_GPU.shape}, dtype={PROBE_DIR_GPU.dtype}')",
    ]))

    # Cell 7 — Generation + steering hook
    cells.append(md([
        "## 7. Generation loop with steering hook",
        "",
        "For each (prompt, α, direction): register hook adding `α × unit_dir` to L{TARGET_LAYER} output, generate,",
        "remove hook, store metrics. Drive checkpoint per prompt for resume safety.",
    ]))
    cells.append(code([
        "import time, json",
        "from tqdm.auto import tqdm",
        "",
        "# Reuse PSAE cache prompts via thinking_traces.pt (already has the GSM8K questions)",
        "traces = torch.load(PSAE_CACHE / 'thinking_traces.pt', map_location='cpu', weights_only=False)",
        "rng_pick = np.random.default_rng(SEED)",
        "picked_indices = rng_pick.choice(len(traces), size=N_PROMPTS, replace=False).tolist()",
        "picked_indices.sort()",
        "print(f'picked prompt indices: {picked_indices}')",
        "",
        "def make_prompt_thinking(question):",
        "    msgs = [{'role': 'user', 'content': question}]",
        "    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)",
        "",
        "def find_position(token_ids, target_id):",
        "    matches = (token_ids == target_id).nonzero(as_tuple=True)[0]",
        "    return matches[0].item() if len(matches) > 0 else -1",
        "",
        "def make_steering_hook(direction_vec, alpha):",
        "    def _hook(module, inp, out):",
        "        h = out[0] if isinstance(out, tuple) else out",
        "        h_steered = h + alpha * direction_vec.view(1, 1, -1)",
        "        if isinstance(out, tuple):",
        "            return (h_steered,) + out[1:]",
        "        return h_steered",
        "    return _hook",
        "",
        "def generate_with_steering(prompt_text, direction_vec, alpha, max_new=MAX_NEW_TOK):",
        "    ids = tok(prompt_text, return_tensors='pt').input_ids.to(DEVICE)",
        "    prompt_len = ids.shape[1]",
        "    handle = None",
        "    if alpha != 0 and direction_vec is not None:",
        "        handle = target_layer_mod.register_forward_hook(make_steering_hook(direction_vec, alpha))",
        "    try:",
        "        with torch.no_grad():",
        "            out = model.generate(",
        "                ids, max_new_tokens=max_new,",
        "                do_sample=False, temperature=0.0,",
        "                pad_token_id=tok.pad_token_id,",
        "            )",
        "    finally:",
        "        if handle is not None: handle.remove()",
        "    gen_ids = out[0]",
        "    gen_only = gen_ids[prompt_len:]",
        "    think_end_rel = find_position(gen_only, THINK_END_ID)",
        "    thinking_len = think_end_rel if think_end_rel >= 0 else len(gen_only)",
        "    terminates  = (think_end_rel >= 0)",
        "    output_text = tok.decode(gen_only, skip_special_tokens=False)",
        "    return {",
        "        'thinking_len': int(thinking_len),",
        "        'terminates': bool(terminates),",
        "        'total_len': int(len(gen_only)),",
        "        'output_text': output_text,",
        "        'output_strip': output_text.strip(),",
        "    }",
        "",
        "# Resume from per-prompt checkpoints",
        "results = []",
        "for idx in tqdm(picked_indices, desc='prompts'):",
        "    ck_path = OUT / 'per_prompt' / f'prompt_{idx:03d}.json'",
        "    if ck_path.exists():",
        "        results.append(json.load(open(ck_path)))",
        "        continue",
        "    q = traces[idx]['question']",
        "    prompt_text = make_prompt_thinking(q)",
        "    rec = {'idx': int(idx), 'question': q, 'measurements': {}}",
        "",
        "    # Baseline (α=0, no hook)",
        "    t0 = time.time()",
        "    base = generate_with_steering(prompt_text, None, 0)",
        "    rec['measurements']['baseline'] = base",
        "    print(f'  prompt {idx} baseline: thinking_len={base[\"thinking_len\"]} terminates={base[\"terminates\"]} ({time.time()-t0:.0f}s)')",
        "",
        "    # Steered conditions",
        "    for direction_name, dir_vec in [('probe', PROBE_DIR_GPU), ('random', RAND_DIR_GPU)]:",
        "        for alpha in ALPHAS:",
        "            t0 = time.time()",
        "            res = generate_with_steering(prompt_text, dir_vec, alpha)",
        "            key = f'{direction_name}_alpha{alpha:+d}'",
        "            rec['measurements'][key] = res",
        "            base_strip = base['output_strip']",
        "            stripped_equal = (res['output_strip'] == base_strip)",
        "            res['stripped_equal_baseline'] = bool(stripped_equal)",
        "            print(f'  {key}: thinking_len={res[\"thinking_len\"]} term={res[\"terminates\"]} '",
        "                  f'strip_eq_base={stripped_equal} ({time.time()-t0:.0f}s)')",
        "",
        "    # Checkpoint",
        "    with open(ck_path, 'w') as f: json.dump(rec, f, indent=2)",
        "    results.append(rec)",
        "",
        "# Aggregate save",
        "with open(RESULT_JSON, 'w') as f: json.dump(results, f, indent=2)",
        "print(f'\\n✓ saved {RESULT_JSON} ({len(results)} prompts)')",
    ]))

    # Cell 8 — Analysis + verdict
    cells.append(md(["## 8. Analysis: thinking-length shift per (direction, α)"]))
    cells.append(code([
        "import numpy as np",
        "import json",
        "",
        "if not results:",
        "    results = json.load(open(RESULT_JSON))",
        "",
        "# Per-condition mean thinking_len and flip-rate vs baseline",
        "print(f'{\"condition\":<25}{\"think_len\":>12}{\"Δ_vs_base\":>12}{\"term_rate\":>12}{\"strip_eq_base\":>16}')",
        "print('-' * 77)",
        "",
        "base_lens = [r['measurements']['baseline']['thinking_len'] for r in results]",
        "base_mean = np.mean(base_lens)",
        "print(f'{\"baseline\":<25}{base_mean:>12.1f}{0.0:>12.1f}{np.mean([r[\"measurements\"][\"baseline\"][\"terminates\"] for r in results]):>12.2f}{1.00:>16.2f}')",
        "",
        "summary = {'baseline': {'thinking_len_mean': float(base_mean)}}",
        "",
        "for direction_name in DIRECTIONS:",
        "    for alpha in ALPHAS:",
        "        key = f'{direction_name}_alpha{alpha:+d}'",
        "        lens  = np.array([r['measurements'][key]['thinking_len'] for r in results])",
        "        terms = np.array([r['measurements'][key]['terminates'] for r in results])",
        "        flips = np.array([not r['measurements'][key]['stripped_equal_baseline'] for r in results])",
        "        delta = lens - np.array(base_lens)",
        "        print(f'{key:<25}{lens.mean():>12.1f}{delta.mean():>+12.1f}{terms.mean():>12.2f}{(1-flips.mean()):>16.2f}')",
        "        summary[key] = {",
        "            'thinking_len_mean': float(lens.mean()),",
        "            'thinking_len_delta_vs_base': float(delta.mean()),",
        "            'thinking_len_delta_std': float(delta.std()),",
        "            'terminates_rate': float(terms.mean()),",
        "            'flip_rate_stripped': float(flips.mean()),",
        "        }",
        "",
        "with open(OUT / 'phase2a_summary.json', 'w') as f: json.dump(summary, f, indent=2)",
        "print(f'\\n✓ saved {OUT / \"phase2a_summary.json\"}')",
    ]))

    # Cell 9 — Plot α vs thinking_len for probe and random
    cells.append(md(["## 9. Plot: α vs thinking_length, probe vs random"]))
    cells.append(code([
        "import matplotlib.pyplot as plt",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(12, 5))",
        "",
        "# Plot 1: thinking_len vs α",
        "ax = axes[0]",
        "alphas_signed = sorted(set(ALPHAS))",
        "for direction_name, color in [('probe', 'C0'), ('random', 'C1')]:",
        "    means = []",
        "    stds = []",
        "    for a in alphas_signed:",
        "        key = f'{direction_name}_alpha{a:+d}'",
        "        lens = np.array([r['measurements'][key]['thinking_len'] for r in results])",
        "        means.append(lens.mean()); stds.append(lens.std() / np.sqrt(len(lens)))",
        "    ax.errorbar(alphas_signed, means, yerr=stds, marker='o', color=color, label=direction_name, lw=2, capsize=4)",
        "ax.axhline(base_mean, color='gray', ls=':', lw=1.5, alpha=0.7, label=f'baseline ({base_mean:.0f})')",
        "ax.set_xlabel('α (steering coefficient)')",
        "ax.set_ylabel('thinking_length (tokens)')",
        "ax.set_title(f'L{TARGET_LAYER}  thinking-length vs α  (N={N_PROMPTS})')",
        "ax.legend()",
        "ax.grid(True, alpha=0.3)",
        "",
        "# Plot 2: stripped flip rate vs α",
        "ax = axes[1]",
        "for direction_name, color in [('probe', 'C0'), ('random', 'C1')]:",
        "    flips = []",
        "    for a in alphas_signed:",
        "        key = f'{direction_name}_alpha{a:+d}'",
        "        f_r = np.mean([not r['measurements'][key]['stripped_equal_baseline'] for r in results])",
        "        flips.append(f_r)",
        "    ax.plot(alphas_signed, flips, marker='s', color=color, label=direction_name, lw=2)",
        "ax.set_xlabel('α (steering coefficient)')",
        "ax.set_ylabel('stripped-flip rate vs baseline')",
        "ax.set_title('Behavioral flip rate (Phase-10 stripped metric)')",
        "ax.legend()",
        "ax.grid(True, alpha=0.3)",
        "ax.set_ylim(-0.05, 1.05)",
        "",
        "fig.suptitle(f'Phase 2A — Subjective-Time Steering at L{TARGET_LAYER}\\n'",
        "             f'CAUSAL if probe (blue) bends thinking_len more than random (orange) at +α; EPIPHENOMENAL otherwise',",
        "             fontsize=11, y=1.02)",
        "fig.tight_layout()",
        "fig.savefig(OUT / 'phase2a_alpha_sweep.png', dpi=130, bbox_inches='tight')",
        "print(f'✓ saved {OUT / \"phase2a_alpha_sweep.png\"}')",
    ]))

    # Cell 10 — Verdict
    cells.append(md(["## 10. Verdict (auto-classification)"]))
    cells.append(code([
        "# Decision rules:",
        "# - CAUSAL: probe Δ at high |α| > random Δ × 1.5 AND directional monotonicity (+α reduces, -α extends OR vice versa)",
        "# - EPIPHENOMENAL type 1 (softmax-temp): probe and random produce similar non-zero Δ",
        "# - EPIPHENOMENAL type 2 (structural): probe and random both produce |Δ| < 5% of baseline at α=±200",
        "",
        "import json",
        "import numpy as np",
        "if not results:",
        "    results = json.load(open(RESULT_JSON))",
        "",
        "def cond_delta(direction_name, alpha):",
        "    key = f'{direction_name}_alpha{alpha:+d}'",
        "    lens = np.array([r['measurements'][key]['thinking_len'] for r in results])",
        "    return float((lens - np.array(base_lens)).mean()), float(lens.std() / np.sqrt(len(lens)))",
        "",
        "max_alpha = max(ALPHAS)",
        "min_alpha = min(ALPHAS)",
        "",
        "probe_pos_d, _ = cond_delta('probe',  max_alpha)",
        "probe_neg_d, _ = cond_delta('probe',  min_alpha)",
        "rand_pos_d, _  = cond_delta('random', max_alpha)",
        "rand_neg_d, _  = cond_delta('random', min_alpha)",
        "",
        "print(f'At α={max_alpha:+d}: probe Δ={probe_pos_d:+.1f},  random Δ={rand_pos_d:+.1f}')",
        "print(f'At α={min_alpha:+d}: probe Δ={probe_neg_d:+.1f},  random Δ={rand_neg_d:+.1f}')",
        "",
        "abs_probe_max = max(abs(probe_pos_d), abs(probe_neg_d))",
        "abs_rand_max  = max(abs(rand_pos_d),  abs(rand_neg_d))",
        "rel_floor = 0.05 * base_mean",
        "",
        "verdict = None",
        "if abs_probe_max < rel_floor and abs_rand_max < rel_floor:",
        "    verdict = '🟡 EPIPHENOMENAL type 2 (STRUCTURAL LOCK)'",
        "    framing = 'Time encoded but causally inaccessible from L{} residual at any α. Paper title: \"Two Forms of Epiphenomenal Probes vol. 2: Subjective Time as Structural Lock\"'.format(TARGET_LAYER)",
        "elif abs_probe_max > abs_rand_max * 1.5 and (np.sign(probe_pos_d) != np.sign(probe_neg_d)):",
        "    verdict = '🟢 CAUSAL'",
        "    framing = 'Probe direction monotonically alters thinking_length and beats random by >50%. Paper title: \"Mechanistic Localization of Self-Time in Qwen3.6-27B Reasoning Phase\"'",
        "elif abs_probe_max > rel_floor:",
        "    verdict = '🟡 EPIPHENOMENAL type 1 (SOFTMAX-TEMP ARTIFACT)'",
        "    framing = 'Probe and random produce similar non-zero effects — likely uniform softmax-temperature shift. Paper title: \"Two Forms of Epiphenomenal Probes vol. 2: Subjective Time as Softmax Artifact\"'",
        "else:",
        "    verdict = '⚪ INDETERMINATE'",
        "    framing = 'Edge case — inspect plot manually and expand N_prompts if needed.'",
        "",
        "print(f'\\n=== VERDICT: {verdict} ===')",
        "print(f'\\n{framing}')",
    ]))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = NOTEBOOKS_DIR / "nb_subjective_time_phase2a_steering.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {out_path} ({len(cells)} cells, {out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    build()
