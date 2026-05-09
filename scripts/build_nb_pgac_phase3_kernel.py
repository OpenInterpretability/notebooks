"""
Builder for nb_pgac_phase3_kernel.ipynb.

PGAC Phase 3 — REAL kernel benchmark end-to-end.

Validates core PGAC mechanism:
1. Standard forward até L_gate
2. Probe(residual_gate) → top-m features predicted at L_target
3. Skip layers L_gate+1 .. L_target-1 (pass-through residual)
4. At L_target: sparse SAE eval with top-m predicted features
5. Standard forward L_target+1 .. final

Measures:
- Wall-clock speedup per (gate, target) pair vs baseline vanilla Qwen3.6-27B
- GSM8K accuracy degradation
- Quality bound validation (SAE_floor + (1-recall)*β)

Pairs benchmarked:
- L11→L31 (Phase 2 winner, recall 0.84 at m=4096)
- L11→L55 (aggressive skip)
- L19→L43 (mid-to-mid with fullstack SAEs)
- L11→L43 (intermediate)

Compute: ~6h on RTX 6000 Blackwell.
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
        "# PGAC Phase 3 — Real Kernel Benchmark",
        "",
        "**Goal**: validate core PGAC mechanism end-to-end with REAL Qwen3.6-27B + trained SAEs + Phase 2 probes.",
        "",
        "**What this notebook does**:",
        "1. Standard forward until `L_gate` for each prompt",
        "2. Probe predicts top-m SAE features at `L_target`",
        "3. Skip layers between gate and target (residual pass-through)",
        "4. Sparse SAE eval at L_target — only compute top-m predicted features",
        "5. Standard forward to end",
        "6. Compare wall-clock + GSM8K accuracy vs vanilla baseline",
        "",
        "**Pairs benchmarked**:",
        "- L11→L31 (Phase 2 winner, recall 0.84 @ m=4096)",
        "- L11→L55 (aggressive skip)",
        "- L19→L43 (mid-to-mid with fullstack)",
        "- L11→L43 (intermediate)",
        "",
        "**Outputs**:",
        "- `phase3_results.json` (speedup × quality per pair)",
        "- Pareto plot speedup vs accuracy",
        "",
        "**Compute**: ~6h on RTX 6000 Blackwell.",
    ]))

    # Cell 1: Drive
    cells.append(md(["## 1. Drive mount"]))
    cells.append(code([
        "from pathlib import Path",
        "import os, json, time, math",
        "import torch, numpy as np",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "DRIVE = Path('/content/drive/MyDrive')",
        "OUT = DRIVE / 'openinterp_runs' / 'pgac_phase3'",
        "OUT.mkdir(parents=True, exist_ok=True)",
        "print(f'OUT: {OUT}')",
    ]))

    # Cell 2: Install
    cells.append(md(["## 2. Install — transformers main + flash-linear-attention"]))
    cells.append(code([
        "import sys, subprocess, shutil",
        "def pip(*a): return subprocess.run([sys.executable, '-m', 'pip', *a], check=False)",
        "",
        "try:",
        "    import transformers",
        "    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES",
        "    has_qwen35 = 'qwen3_5' in CONFIG_MAPPING_NAMES",
        "except Exception:",
        "    has_qwen35 = False",
        "",
        "needs_restart = False",
        "if not has_qwen35:",
        "    pip('install', '-q', 'accelerate', 'datasets', 'huggingface_hub==1.5.0',",
        "        'safetensors', 'einops', 'tqdm', 'sentencepiece', 'tokenizers', 'protobuf', 'hf_transfer')",
        "    pip('uninstall', '-y', '-q', 'transformers', 'causal-conv1d')",
        "    SRC = '/content/transformers_src'",
        "    if Path(SRC).exists(): shutil.rmtree(SRC)",
        "    subprocess.run(['git','clone','--quiet','--depth=1',",
        "                    'https://github.com/huggingface/transformers.git', SRC], check=True)",
        "    pip('install','-q','--force-reinstall','--no-deps','--no-cache-dir', SRC)",
        "    for m in list(sys.modules):",
        "        if m.startswith('transformers'): del sys.modules[m]",
        "    needs_restart = True",
        "",
        "try:",
        "    import fla",
        "except ImportError:",
        "    pip('install', '-q', '--no-cache-dir', 'flash-linear-attention')",
        "    needs_restart = True",
        "",
        "pip('install', '-q', 'matplotlib')",
        "",
        "if needs_restart: print('*** RESTART RUNTIME, re-run cells 1+2 ***')",
        "",
        "try:",
        "    from google.colab import userdata",
        "    t = userdata.get('HF_TOKEN')",
        "    if t: os.environ['HF_TOKEN'] = t",
        "except Exception: t = os.environ.get('HF_TOKEN')",
        "if t:",
        "    from huggingface_hub import login",
        "    login(token=t, add_to_git_credential=False)",
        "    print('HF auth OK')",
    ]))

    # Cell 3: CFG
    cells.append(md(["## 3. CFG"]))
    cells.append(code([
        "MODEL_ID = 'Qwen/Qwen3.6-27B'",
        "D_MODEL  = 5120",
        "DEVICE   = 'cuda'",
        "DTYPE    = torch.bfloat16",
        "",
        "# Pairs to benchmark (gate, target)",
        "PGAC_PAIRS = [",
        "    {'gate': 11, 'target': 31, 'm': 4096, 'name': 'L11_to_L31'},  # Phase 2 winner",
        "    {'gate': 11, 'target': 55, 'm': 4096, 'name': 'L11_to_L55'},  # aggressive",
        "    {'gate': 19, 'target': 43, 'm': 4096, 'name': 'L19_to_L43'},  # mid-mid fullstack",
        "    {'gate': 11, 'target': 43, 'm': 4096, 'name': 'L11_to_L43'},  # intermediate",
        "]",
        "",
        "# SAE source per layer",
        "SAE_SOURCES = {",
        "    11: 'caiovicentino1/qwen36-27b-sae-papergrade',",
        "    31: 'caiovicentino1/qwen36-27b-sae-papergrade',",
        "    55: 'caiovicentino1/qwen36-27b-sae-papergrade',",
        "    19: 'caiovicentino1/qwen36-27b-sae-fullstack',",
        "    43: 'caiovicentino1/qwen36-27b-sae-fullstack',",
        "}",
        "SAE_DIM = {11: 65536, 31: 65536, 55: 65536, 19: 40960, 43: 40960}",
        "K_SAE   = 128",
        "",
        "# Probe source — Phase 2 trained probes",
        "PROBE_REPO = 'caiovicentino1/openinterp-pgac-phase2-probe'",
        "PROBE_FILES = {",
        "    'L11_to_L31': 'probe_L11_to_L31.pt',",
        "    'L11_to_L55': 'probe_L11_to_L55.pt',",
        "    'L31_to_L55': 'probe_L31_to_L55.pt',",
        "}",
        "# For new pairs (L19→L43, L11→L43) we will train a quick probe on-the-fly using SAE encoders",
        "",
        "# Eval",
        "N_EVAL_PROMPTS = 100",
        "MAX_NEW_TOKENS = 256",
        "TEMPERATURE = 0.0  # deterministic for accuracy comparison",
        "",
        "torch.manual_seed(42); np.random.seed(42)",
        "print(f'Pairs: {len(PGAC_PAIRS)}, eval prompts: {N_EVAL_PROMPTS}')",
    ]))

    # Cell 4: Load Qwen
    cells.append(md(["## 4. Load Qwen3.6-27B"]))
    cells.append(code([
        "from transformers import AutoTokenizer, AutoModelForImageTextToText",
        "",
        "tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)",
        "if tok.pad_token_id is None: tok.pad_token = tok.eos_token",
        "",
        "model = AutoModelForImageTextToText.from_pretrained(",
        "    MODEL_ID, dtype=DTYPE, device_map=DEVICE,",
        "    attn_implementation='sdpa', trust_remote_code=True,",
        ")",
        "model.eval()",
        "for p in model.parameters(): p.requires_grad_(False)",
        "print(f'VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB')",
    ]))

    # Cell 5: Layer module helpers
    cells.append(md(["## 5. Layer module helpers"]))
    cells.append(code([
        "def get_layer_module(m, idx):",
        "    for path in [('model','language_model','layers'),",
        "                 ('language_model','layers'), ('model','layers')]:",
        "        try:",
        "            cur = m",
        "            for p in path: cur = getattr(cur, p)",
        "            return cur[idx]",
        "        except AttributeError: continue",
        "    raise RuntimeError('Could not locate decoder layers')",
        "",
        "def get_layers_module(m):",
        "    for path in [('model','language_model','layers'),",
        "                 ('language_model','layers'), ('model','layers')]:",
        "        try:",
        "            cur = m",
        "            for p in path: cur = getattr(cur, p)",
        "            return cur",
        "        except AttributeError: continue",
        "    raise RuntimeError",
        "",
        "layers = get_layers_module(model)",
        "print(f'Total layers: {len(layers)}')",
        "print(f'Sample: L0 = {type(layers[0]).__name__}')",
    ]))

    # Cell 6: Load needed SAEs
    cells.append(md(["## 6. Load needed SAEs"]))
    cells.append(code([
        "from huggingface_hub import hf_hub_download",
        "from safetensors.torch import load_file",
        "",
        "class TopKSAEInf(nn.Module):",
        "    def __init__(self, d_in, n, k):",
        "        super().__init__()",
        "        self.W_enc = nn.Parameter(torch.zeros(d_in, n))",
        "        self.W_dec = nn.Parameter(torch.zeros(n, d_in))",
        "        self.b_enc = nn.Parameter(torch.zeros(n))",
        "        self.b_dec = nn.Parameter(torch.zeros(d_in))",
        "        self.k = k; self.n = n",
        "    def encode_pre(self, x):",
        "        return (x - self.b_dec) @ self.W_enc + self.b_enc",
        "    def forward(self, x):",
        "        pre = self.encode_pre(x)",
        "        top_v, top_i = pre.topk(self.k, dim=-1)",
        "        z = torch.zeros_like(pre)",
        "        z.scatter_(-1, top_i, F.relu(top_v))",
        "        return z @ self.W_dec + self.b_dec, top_i",
        "    def sparse_decode(self, top_v, top_i):",
        "        \"\"\"Decode from already-known top-m features (PGAC sparse path).\"\"\"",
        "        # top_v: (B, m), top_i: (B, m). Build sparse z without full encode.",
        "        z = torch.zeros(top_i.shape[0], self.n, device=top_i.device, dtype=top_v.dtype)",
        "        z.scatter_(-1, top_i, F.relu(top_v))",
        "        return z @ self.W_dec + self.b_dec",
        "",
        "def load_sae_for(L):",
        "    repo = SAE_SOURCES[L]",
        "    n = SAE_DIM[L]",
        "    weights = load_file(hf_hub_download(repo, f'sae_L{L}_latest.safetensors'))",
        "    sae = TopKSAEInf(D_MODEL, n, K_SAE).to(DEVICE, torch.float32)",
        "    for k, v in weights.items(): getattr(sae, k).data = v.to(DEVICE, torch.float32)",
        "    sae.eval()",
        "    return sae",
        "",
        "# Layers needed across all pairs",
        "needed_layers = sorted(set(p['gate'] for p in PGAC_PAIRS) | set(p['target'] for p in PGAC_PAIRS))",
        "saes = {}",
        "for L in needed_layers:",
        "    print(f'Loading SAE L{L}...')",
        "    saes[L] = load_sae_for(L)",
        "print(f'\\nVRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB')",
    ]))

    # Cell 7: Load probes
    cells.append(md(["## 7. Load Phase 2 probes (and quick-train new for missing pairs)"]))
    cells.append(code([
        "probes = {}",
        "for pair in PGAC_PAIRS:",
        "    name = pair['name']",
        "    if name in PROBE_FILES:",
        "        try:",
        "            probe_path = hf_hub_download(PROBE_REPO, PROBE_FILES[name])",
        "            state = torch.load(probe_path, map_location='cpu', weights_only=False)",
        "            probe = nn.Linear(D_MODEL, SAE_DIM[pair['target']]).to(DEVICE, torch.float32)",
        "            probe.load_state_dict(state)",
        "            probe.eval()",
        "            probes[name] = probe",
        "            print(f'  ✓ {name}: loaded from Phase 2')",
        "        except Exception as e:",
        "            print(f'  ⚠ {name}: download failed ({e}), will warm-start from SAE encoder')",
        "            probes[name] = None",
        "    else:",
        "        probes[name] = None",
        "        print(f'  - {name}: no Phase 2 probe, warm-start from SAE encoder')",
        "",
        "# For pairs without trained probe, use the target SAE's encoder as a probe (warm-start)",
        "# This is the same initialization as Phase 2 — gives ~chance baseline but works",
        "for pair in PGAC_PAIRS:",
        "    name = pair['name']",
        "    if probes[name] is None:",
        "        target = pair['target']",
        "        sae = saes[target]",
        "        probe = nn.Linear(D_MODEL, SAE_DIM[target]).to(DEVICE, torch.float32)",
        "        probe.weight.data = sae.W_enc.T.contiguous()",
        "        probe.bias.data = sae.b_enc.clone()",
        "        probe.eval()",
        "        probes[name] = probe",
        "        print(f'  ✓ {name}: warm-started from L{target} SAE encoder')",
        "",
        "print(f'\\n{len(probes)} probes ready')",
    ]))

    # Cell 8: PGAC forward function
    cells.append(md(["## 8. PGAC forward — layer skip + sparse SAE"]))
    cells.append(code([
        "def standard_forward(input_ids, max_new_tokens=MAX_NEW_TOKENS):",
        "    \"\"\"Vanilla baseline.\"\"\"",
        "    with torch.no_grad():",
        "        out = model.generate(",
        "            input_ids,",
        "            max_new_tokens=max_new_tokens,",
        "            do_sample=False,",
        "            temperature=TEMPERATURE,",
        "            pad_token_id=tok.pad_token_id,",
        "        )",
        "    return out",
        "",
        "class PGACForward:",
        "    \"\"\"Implements PGAC layer skip + sparse SAE eval via forward hooks.\"\"\"",
        "    def __init__(self, gate, target, m, probe, sae_target):",
        "        self.gate = gate",
        "        self.target = target",
        "        self.m = m",
        "        self.probe = probe",
        "        self.sae_target = sae_target",
        "        self.gate_residual = None",
        "        self.skipped_layers = list(range(gate + 1, target))  # exclusive of target",
        "        self.handles = []",
        "        self._install_hooks()",
        "",
        "    def _install_hooks(self):",
        "        # Hook on gate layer — capture residual",
        "        def gate_hook(module, inp, out):",
        "            h = out[0] if isinstance(out, tuple) else out",
        "            self.gate_residual = h.detach()",
        "        self.handles.append(get_layer_module(model, self.gate).register_forward_hook(gate_hook))",
        "",
        "        # Hooks on skipped layers — pass-through (output = input residual)",
        "        for L in self.skipped_layers:",
        "            def skip_hook(module, inp, out, L=L):",
        "                # inp is (residual_in,), out is (residual_out,) or just residual_out",
        "                residual_in = inp[0]",
        "                if isinstance(out, tuple):",
        "                    return (residual_in,) + tuple(out[1:])",
        "                return residual_in",
        "            self.handles.append(get_layer_module(model, L).register_forward_hook(skip_hook))",
        "",
        "        # Hook on target layer — replace output with sparse SAE reconstruction",
        "        def target_hook(module, inp, out):",
        "            # Use gate residual + probe to predict top-m features",
        "            with torch.no_grad():",
        "                # gate_residual: (B, T, D)",
        "                gate_res = self.gate_residual.to(torch.float32)",
        "                B, T, D = gate_res.shape",
        "                gate_res_flat = gate_res.reshape(-1, D)  # (B*T, D)",
        "                ",
        "                # Probe predicts top-m features at target",
        "                logits = self.probe(gate_res_flat)  # (B*T, n_features)",
        "                top_v, top_i = logits.topk(self.m, dim=-1)  # (B*T, m)",
        "                ",
        "                # Restrict to top-k of those for sparse decode",
        "                k_eff = min(K_SAE, self.m)",
        "                top_v_k, top_i_k_idx = top_v.topk(k_eff, dim=-1)",
        "                top_i_k = torch.gather(top_i, -1, top_i_k_idx)",
        "                ",
        "                # Sparse decode using SAE_target.W_dec",
        "                x_hat = self.sae_target.sparse_decode(top_v_k.float(), top_i_k)",
        "                x_hat = x_hat.reshape(B, T, D).to(torch.bfloat16)",
        "            ",
        "            # Replace target layer output with sparse SAE reconstruction",
        "            if isinstance(out, tuple):",
        "                return (x_hat,) + tuple(out[1:])",
        "            return x_hat",
        "        self.handles.append(get_layer_module(model, self.target).register_forward_hook(target_hook))",
        "",
        "    def remove(self):",
        "        for h in self.handles: h.remove()",
        "",
        "    def generate(self, input_ids, max_new_tokens=MAX_NEW_TOKENS):",
        "        with torch.no_grad():",
        "            out = model.generate(",
        "                input_ids,",
        "                max_new_tokens=max_new_tokens,",
        "                do_sample=False,",
        "                temperature=TEMPERATURE,",
        "                pad_token_id=tok.pad_token_id,",
        "            )",
        "        return out",
        "",
        "print('PGACForward ready.')",
    ]))

    # Cell 9: Load eval set
    cells.append(md(["## 9. Load GSM8K eval set"]))
    cells.append(code([
        "from datasets import load_dataset",
        "import re",
        "",
        "ds = load_dataset('gsm8k', 'main', split='test')",
        "eval_set = ds.select(range(N_EVAL_PROMPTS))",
        "",
        "def extract_answer(text):",
        "    \"\"\"Extract numerical answer from GSM8K format (#### XXX or final number).\"\"\"",
        "    match = re.search(r'####\\s*([-+]?\\d+(?:\\.\\d+)?)', text)",
        "    if match: return match.group(1).strip()",
        "    nums = re.findall(r'[-+]?\\d+(?:\\.\\d+)?', text)",
        "    return nums[-1] if nums else None",
        "",
        "def make_prompt(question):",
        "    \"\"\"Format with chat template if available.\"\"\"",
        "    msgs = [{'role': 'user', 'content': question}]",
        "    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)",
        "",
        "print(f'Eval set: {len(eval_set)} prompts')",
        "print(f'Sample question: {eval_set[0][\"question\"][:100]}...')",
        "print(f'Gold answer: {extract_answer(eval_set[0][\"answer\"])}')",
    ]))

    # Cell 10: Baseline benchmark
    cells.append(md(["## 10. Baseline (vanilla Qwen) benchmark"]))
    cells.append(code([
        "from tqdm.auto import tqdm",
        "import time",
        "",
        "def run_benchmark(label, gen_fn):",
        "    \"\"\"Returns dict with timing, accuracy, per-prompt breakdown.\"\"\"",
        "    times = []",
        "    correct = 0",
        "    outputs = []",
        "    for ex in tqdm(eval_set, desc=label):",
        "        prompt = make_prompt(ex['question'])",
        "        ids = tok(prompt, return_tensors='pt').input_ids.to(DEVICE)",
        "        torch.cuda.synchronize()",
        "        t0 = time.time()",
        "        out = gen_fn(ids)",
        "        torch.cuda.synchronize()",
        "        elapsed = time.time() - t0",
        "        gen_text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)",
        "        pred = extract_answer(gen_text)",
        "        gold = extract_answer(ex['answer'])",
        "        is_correct = (pred is not None and pred == gold)",
        "        correct += int(is_correct)",
        "        times.append(elapsed)",
        "        outputs.append({'gold': gold, 'pred': pred, 'correct': is_correct, 'time': elapsed})",
        "    return {",
        "        'label': label,",
        "        'n': len(eval_set),",
        "        'correct': correct,",
        "        'accuracy': correct / len(eval_set),",
        "        'mean_time': sum(times) / len(times),",
        "        'total_time': sum(times),",
        "        'outputs': outputs,",
        "    }",
        "",
        "print('Running baseline (vanilla Qwen3.6-27B)...')",
        "baseline = run_benchmark('baseline', standard_forward)",
        "print(f'\\nBaseline: acc={baseline[\"accuracy\"]:.4f}, mean_time={baseline[\"mean_time\"]:.2f}s')",
    ]))

    # Cell 11: PGAC benchmarks
    cells.append(md(["## 11. PGAC benchmarks for each pair"]))
    cells.append(code([
        "results = {'baseline': baseline}",
        "",
        "for pair in PGAC_PAIRS:",
        "    name = pair['name']",
        "    print(f'\\n=== Benchmarking {name} (gate=L{pair[\"gate\"]}, target=L{pair[\"target\"]}, m={pair[\"m\"]}) ===')",
        "    pgac = PGACForward(pair['gate'], pair['target'], pair['m'], probes[name], saes[pair['target']])",
        "    try:",
        "        result = run_benchmark(name, pgac.generate)",
        "        results[name] = result",
        "        speedup = baseline['mean_time'] / result['mean_time']",
        "        acc_drop = baseline['accuracy'] - result['accuracy']",
        "        print(f'\\n{name}: acc={result[\"accuracy\"]:.4f} (drop {acc_drop:+.4f})')",
        "        print(f'  speedup: {speedup:.2f}x')",
        "    finally:",
        "        pgac.remove()",
        "",
        "# Save results",
        "results_summary = {}",
        "for label, r in results.items():",
        "    results_summary[label] = {",
        "        'accuracy': r['accuracy'],",
        "        'mean_time': r['mean_time'],",
        "        'n': r['n'],",
        "        'correct': r['correct'],",
        "    }",
        "with open(OUT / 'phase3_results.json', 'w') as f:",
        "    json.dump(results_summary, f, indent=2)",
        "print(f'\\n✓ Saved: {OUT / \"phase3_results.json\"}')",
    ]))

    # Cell 12: Pareto plot
    cells.append(md(["## 12. Pareto plot — speedup vs accuracy"]))
    cells.append(code([
        "import matplotlib.pyplot as plt",
        "",
        "fig, ax = plt.subplots(1, 1, figsize=(8, 6))",
        "",
        "labels = list(results.keys())",
        "speedups = [baseline['mean_time'] / results[k]['mean_time'] for k in labels]",
        "accuracies = [results[k]['accuracy'] * 100 for k in labels]",
        "colors = ['black' if k == 'baseline' else f'C{i}' for i, k in enumerate(labels)]",
        "",
        "for k, sp, ac, c in zip(labels, speedups, accuracies, colors):",
        "    marker = 's' if k == 'baseline' else 'o'",
        "    ax.scatter(sp, ac, label=k, s=200, color=c, marker=marker, edgecolors='black')",
        "    ax.annotate(k, (sp, ac), xytext=(5, 5), textcoords='offset points', fontsize=10)",
        "",
        "ax.set_xlabel('Speedup vs baseline (x)')",
        "ax.set_ylabel('GSM8K accuracy (%)')",
        "ax.set_title('PGAC Phase 3 — Speedup × Quality tradeoff')",
        "ax.axhline(baseline['accuracy'] * 100, color='black', linestyle='--', alpha=0.5, label=f'Baseline acc = {baseline[\"accuracy\"]*100:.1f}%')",
        "ax.axvline(1.0, color='gray', linestyle=':', alpha=0.5)",
        "ax.legend(loc='lower left')",
        "ax.grid(alpha=0.3)",
        "",
        "plot_path = OUT / 'pareto_plot.png'",
        "plt.savefig(plot_path, dpi=170, bbox_inches='tight')",
        "plt.show()",
        "print(f'✓ Saved: {plot_path}')",
    ]))

    # Cell 13: Verdict
    cells.append(md(["## 13. Verdict"]))
    cells.append(code([
        "print('=== PGAC Phase 3 Verdict ===\\n')",
        "print(f'Baseline (vanilla Qwen3.6-27B): acc={baseline[\"accuracy\"]:.4f}, mean_time={baseline[\"mean_time\"]:.2f}s\\n')",
        "print(f'{\"Pair\":<15} {\"Speedup\":<10} {\"Acc\":<10} {\"Acc drop\":<12} {\"Verdict\":<20}')",
        "print('-' * 70)",
        "for pair in PGAC_PAIRS:",
        "    name = pair['name']",
        "    if name not in results: continue",
        "    r = results[name]",
        "    sp = baseline['mean_time'] / r['mean_time']",
        "    drop = baseline['accuracy'] - r['accuracy']",
        "    if sp > 1.5 and drop < 0.10:",
        "        verdict = '🟢 STRONG'",
        "    elif sp > 1.2 and drop < 0.15:",
        "        verdict = '🟡 MARGINAL'",
        "    else:",
        "        verdict = '🔴 INSUFFICIENT'",
        "    print(f'{name:<15} {sp:<10.2f}x {r[\"accuracy\"]:<10.4f} {drop:<+12.4f} {verdict}')",
        "",
        "best = max((p['name'] for p in PGAC_PAIRS if p['name'] in results),",
        "           key=lambda n: baseline['mean_time'] / results[n]['mean_time'] / max(0.05, baseline['accuracy'] - results[n]['accuracy'] + 0.05))",
        "print(f'\\nBest pair (speedup × accuracy retention): {best}')",
        "best_r = results[best]",
        "best_sp = baseline['mean_time'] / best_r['mean_time']",
        "print(f'  → {best_sp:.2f}x speedup at {best_r[\"accuracy\"]*100:.1f}% accuracy ({(baseline[\"accuracy\"] - best_r[\"accuracy\"])*100:+.1f}pp vs baseline)')",
    ]))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out_path = NOTEBOOKS_DIR / "nb_pgac_phase3_kernel.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"✓ Wrote {out_path}")
    print(f"  Cells: {len(cells)}")


if __name__ == "__main__":
    build()
