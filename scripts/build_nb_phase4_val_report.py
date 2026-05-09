"""
Builder for nb_phase4_val_report.ipynb.

PGAC Phase 4 — standalone validation report on 11 trained SAEs.

Loads:
- Qwen3.6-27B (HF)
- 11 SAEs from caiovicentino1/qwen36-27b-sae-fullstack (latest @ 170M tokens)
- 3 papergrade SAEs from caiovicentino1/qwen36-27b-sae-papergrade (L11/L31/L55) for comparison

Computes per layer:
- ve (variance explained) — REAL measured, not running average
- L0 (mean active features per token, should be ~k=128)
- alive% (fraction of features that fired at least once)
- dead% (complementary)

Outputs:
- val_report.json (uploaded to HF repo)
- comparison plot vs papergrade

Compute: ~30 min on RTX 6000 Blackwell. ~1M tokens through Qwen + 14 SAEs.
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
        "# PGAC Phase 4 — Validation Report (standalone)",
        "",
        "Standalone notebook to run val_report.json on the 11 trained SAEs from HF + 3 papergrade.",
        "",
        "**Inputs**:",
        "- Qwen3.6-27B (Hugging Face)",
        "- `caiovicentino1/qwen36-27b-sae-fullstack` — 11 SAEs (Phase 4 v0.1, 170M tokens)",
        "- `caiovicentino1/qwen36-27b-sae-papergrade` — 3 SAEs (L11/L31/L55, 200M tokens) for comparison",
        "",
        "**Outputs**:",
        "- `val_report.json` (uploaded to fullstack repo)",
        "- Comparison plot fullstack vs papergrade",
        "",
        "**Compute**: ~30 min on RTX 6000 Blackwell. 1M tokens through Qwen + 14 SAEs.",
    ]))

    # Cell 1: Drive mount
    cells.append(md(["## 1. Drive mount"]))
    cells.append(code([
        "from pathlib import Path",
        "import os, json, time",
        "import torch, numpy as np",
        "",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "DRIVE = Path('/content/drive/MyDrive')",
        "OUT = DRIVE / 'openinterp_runs' / 'pgac_phase4_val'",
        "OUT.mkdir(parents=True, exist_ok=True)",
        "print(f'OUT: {OUT}')",
    ]))

    # Cell 2: Install (lessons applied: no -U torch, fla mandatory)
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
        "    print('Installing transformers from source...')",
        "    pip('install', '-q', 'accelerate', 'datasets', 'huggingface_hub==1.5.0',",
        "        'safetensors', 'einops', 'tqdm', 'sentencepiece', 'tokenizers', 'protobuf', 'hf_transfer')",
        "    pip('uninstall', '-y', '-q', 'transformers', 'causal-conv1d')",
        "    SRC = '/content/transformers_src'",
        "    if Path(SRC).exists(): shutil.rmtree(SRC)",
        "    subprocess.run(['git','clone','--quiet','--depth=1',",
        "                    'https://github.com/huggingface/transformers.git', SRC], check=True)",
        "    pip('install','-q','--force-reinstall','--no-deps','--no-cache-dir', SRC)",
        "    for m in list(sys.modules):",
        "        if m.startswith('transformers') or m.startswith('huggingface_hub'):",
        "            del sys.modules[m]",
        "    needs_restart = True",
        "else:",
        "    print(f'transformers {transformers.__version__} ✓')",
        "",
        "try:",
        "    import fla",
        "    print('flash-linear-attention ✓')",
        "except ImportError:",
        "    print('Installing flash-linear-attention...')",
        "    pip('install', '-q', '--no-cache-dir', 'flash-linear-attention')",
        "    needs_restart = True",
        "",
        "pip('install', '-q', 'matplotlib')",
        "",
        "if needs_restart:",
        "    print('\\n*** RESTART RUNTIME, then re-run cells 1+2 ***')",
        "",
        "# HF auth",
        "try:",
        "    from google.colab import userdata",
        "    t = userdata.get('HF_TOKEN')",
        "    if t: os.environ['HF_TOKEN'] = t",
        "except Exception:",
        "    t = os.environ.get('HF_TOKEN')",
        "if t:",
        "    from huggingface_hub import login",
        "    login(token=t, add_to_git_credential=False)",
        "    print('HF auth OK')",
    ]))

    # Cell 3: CFG
    cells.append(md(["## 3. CFG"]))
    cells.append(code([
        "MODEL_ID       = 'Qwen/Qwen3.6-27B'",
        "D_MODEL        = 5120",
        "",
        "# Layer coverage: 14 layers total",
        "FULLSTACK_LAYERS = [15, 19, 23, 27, 35, 39, 43, 47, 51, 59, 63]  # 11 from Phase 4",
        "PAPERGRADE_LAYERS = [11, 31, 55]                                  # 3 from papergrade",
        "ALL_LAYERS = sorted(FULLSTACK_LAYERS + PAPERGRADE_LAYERS)",
        "",
        "# SAE configs differ between repos",
        "FULLSTACK_REPO  = 'caiovicentino1/qwen36-27b-sae-fullstack'",
        "PAPERGRADE_REPO = 'caiovicentino1/qwen36-27b-sae-papergrade'",
        "",
        "FULLSTACK_D_SAE  = 40960",
        "FULLSTACK_K      = 128",
        "PAPERGRADE_D_SAE = 65536",
        "PAPERGRADE_K     = 128",
        "",
        "# Validation",
        "VAL_TOKENS = 1_000_000",
        "FWD_BATCH  = 2",
        "SEQ_LEN    = 1024",
        "",
        "DEVICE     = 'cuda'",
        "DTYPE      = torch.bfloat16",
        "",
        "import random",
        "torch.manual_seed(42); random.seed(42); np.random.seed(42)",
        "print(f'Total layers to validate: {len(ALL_LAYERS)} ({len(FULLSTACK_LAYERS)} fullstack + {len(PAPERGRADE_LAYERS)} papergrade)')",
    ]))

    # Cell 4: Load Qwen
    cells.append(md(["## 4. Load Qwen3.6-27B"]))
    cells.append(code([
        "from transformers import AutoTokenizer, AutoModelForImageTextToText",
        "",
        "tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)",
        "if tok.pad_token_id is None:",
        "    tok.pad_token = tok.eos_token",
        "",
        "model = AutoModelForImageTextToText.from_pretrained(",
        "    MODEL_ID, dtype=DTYPE, device_map=DEVICE,",
        "    attn_implementation='sdpa', trust_remote_code=True,",
        ")",
        "model.eval()",
        "for p in model.parameters():",
        "    p.requires_grad_(False)",
        "print(f'VRAM after model: {torch.cuda.memory_allocated()/1e9:.1f} GB')",
    ]))

    # Cell 5: Layer path + LayerTap
    cells.append(md(["## 5. LayerTap multi-layer hook"]))
    cells.append(code([
        "def get_layer_module(m, idx):",
        "    for path in [('model','language_model','layers'),",
        "                 ('language_model','layers'),",
        "                 ('model','layers')]:",
        "        try:",
        "            cur = m",
        "            for p in path:",
        "                cur = getattr(cur, p)",
        "            return cur[idx]",
        "        except AttributeError:",
        "            continue",
        "    raise RuntimeError('Could not locate decoder layers')",
        "",
        "class LayerTap:",
        "    def __init__(self, model, layers):",
        "        self.buf = {L: None for L in layers}",
        "        self.handles = []",
        "        for L in layers:",
        "            mod = get_layer_module(model, L)",
        "            self.handles.append(mod.register_forward_hook(self._mk_hook(L)))",
        "    def _mk_hook(self, L):",
        "        def _h(module, inp, out):",
        "            h = out[0] if isinstance(out, tuple) else out",
        "            self.buf[L] = h.detach().to(torch.bfloat16)",
        "        return _h",
        "    def close(self):",
        "        for h in self.handles:",
        "            h.remove()",
        "",
        "# Smoke test",
        "tap = LayerTap(model, ALL_LAYERS)",
        "ids = tok('The quick brown fox jumps over the lazy dog', return_tensors='pt').input_ids.to(DEVICE)",
        "with torch.no_grad():",
        "    model(ids)",
        "for L in ALL_LAYERS:",
        "    print(f'L{L}: shape={tuple(tap.buf[L].shape)}, norm={tap.buf[L].float().norm():.2f}')",
        "tap.close()",
    ]))

    # Cell 6: Load SAEs
    cells.append(md(["## 6. Load 14 SAEs from HF (11 fullstack + 3 papergrade)"]))
    cells.append(code([
        "from huggingface_hub import hf_hub_download",
        "from safetensors.torch import load_file",
        "import torch.nn as nn",
        "",
        "class TopKSAEInference(nn.Module):",
        "    \"\"\"Inference-only TopK SAE (no aux loss).\"\"\"",
        "    def __init__(self, d_in, n, k):",
        "        super().__init__()",
        "        self.d_in, self.n, self.k = d_in, n, k",
        "        self.W_enc = nn.Parameter(torch.zeros(d_in, n))",
        "        self.W_dec = nn.Parameter(torch.zeros(n, d_in))",
        "        self.b_enc = nn.Parameter(torch.zeros(n))",
        "        self.b_dec = nn.Parameter(torch.zeros(d_in))",
        "    def forward(self, x):",
        "        \"\"\"x: (B, d_in) → returns (x_hat, z, top_i)\"\"\"",
        "        pre = (x - self.b_dec) @ self.W_enc + self.b_enc",
        "        top_v, top_i = pre.topk(self.k, dim=-1)",
        "        z = torch.zeros_like(pre)",
        "        z.scatter_(-1, top_i, torch.relu(top_v))",
        "        x_hat = z @ self.W_dec + self.b_dec",
        "        return x_hat, z, top_i",
        "",
        "def load_sae(repo, layer, d_sae, k):",
        "    weights = load_file(hf_hub_download(repo, f'sae_L{layer}_latest.safetensors'))",
        "    sae = TopKSAEInference(D_MODEL, d_sae, k).to(DEVICE, torch.float32)",
        "    sae.W_enc.data = weights['W_enc'].to(DEVICE, torch.float32)",
        "    sae.W_dec.data = weights['W_dec'].to(DEVICE, torch.float32)",
        "    sae.b_enc.data = weights['b_enc'].to(DEVICE, torch.float32)",
        "    sae.b_dec.data = weights['b_dec'].to(DEVICE, torch.float32)",
        "    sae.eval()",
        "    return sae",
        "",
        "saes = {}",
        "for L in FULLSTACK_LAYERS:",
        "    print(f'Loading L{L} from fullstack...', end=' ')",
        "    saes[L] = load_sae(FULLSTACK_REPO, L, FULLSTACK_D_SAE, FULLSTACK_K)",
        "    print(f'd_sae={FULLSTACK_D_SAE} ✓')",
        "",
        "for L in PAPERGRADE_LAYERS:",
        "    print(f'Loading L{L} from papergrade...', end=' ')",
        "    saes[L] = load_sae(PAPERGRADE_REPO, L, PAPERGRADE_D_SAE, PAPERGRADE_K)",
        "    print(f'd_sae={PAPERGRADE_D_SAE} ✓')",
        "",
        "print(f'\\nVRAM after SAEs: {torch.cuda.memory_allocated()/1e9:.1f} GB')",
    ]))

    # Cell 7: Validation corpus stream
    cells.append(md(["## 7. Validation corpus (1M tokens, fineweb-edu + OpenThoughts + OpenMath)"]))
    cells.append(code([
        "from datasets import load_dataset",
        "",
        "CORPUS_MIX = [",
        "    ('HuggingFaceFW/fineweb-edu',             'sample-10BT', 0.70),",
        "    ('open-thoughts/OpenThoughts-114k',       'default',     0.20),",
        "    ('nvidia/OpenMathInstruct-2',             'default',     0.10),",
        "]",
        "",
        "def _load_stream(repo, split):",
        "    try: return load_dataset(repo, split, split='train', streaming=True)",
        "    except Exception: return load_dataset(repo, split='train', streaming=True)",
        "",
        "def _txt(repo, row):",
        "    if 'fineweb' in repo: return row.get('text', '')",
        "    if 'OpenThoughts' in repo:",
        "        conv = row.get('conversations') or row.get('messages') or []",
        "        if conv:",
        "            msgs = [{'role': m.get('from', m.get('role','user')).replace('human','user').replace('gpt','assistant'),",
        "                     'content': m.get('value', m.get('content',''))} for m in conv]",
        "            try: return tok.apply_chat_template(msgs, tokenize=False, enable_thinking=True)",
        "            except: return '\\n\\n'.join(m['content'] for m in msgs)",
        "        return row.get('text','')",
        "    if 'OpenMath' in repo:",
        "        q = row.get('problem') or row.get('question') or ''",
        "        a = row.get('generated_solution') or row.get('solution') or row.get('answer') or ''",
        "        return f'Problem: {q}\\n\\nSolution: {a}'",
        "    return row.get('text', '')",
        "",
        "def text_stream(seed=43):",
        "    random.seed(seed)",
        "    streams = [(repo, iter(_load_stream(repo, sp)), sp) for repo, sp, _ in CORPUS_MIX]",
        "    weights = [w for _, _, w in CORPUS_MIX]",
        "    while True:",
        "        idx = random.choices(range(len(streams)), weights=weights, k=1)[0]",
        "        repo, it, sp = streams[idx]",
        "        try: row = next(it)",
        "        except StopIteration:",
        "            streams[idx] = (repo, iter(_load_stream(repo, sp)), sp)",
        "            row = next(streams[idx][1])",
        "        txt = _txt(repo, row)",
        "        if txt and len(txt) > 50:",
        "            yield txt",
        "",
        "# Smoke",
        "_g = text_stream()",
        "for _ in range(2):",
        "    s = next(_g)",
        "    print(f'[{len(s)} chars] {s[:80]!r}...')",
    ]))

    # Cell 8: Validation loop
    cells.append(md([
        "## 8. Validation loop — 1M tokens, compute ve/L0/alive% per layer",
    ]))
    cells.append(code([
        "from tqdm.auto import tqdm",
        "",
        "def pack_sequences(text_gen, n_seq, seq_len):",
        "    out, carry = [], []",
        "    while len(out) < n_seq:",
        "        if len(carry) < seq_len:",
        "            carry.extend(tok(next(text_gen), add_special_tokens=False).input_ids)",
        "            continue",
        "        out.append(carry[:seq_len])",
        "        carry = carry[seq_len:]",
        "    return torch.tensor(out, dtype=torch.long, device=DEVICE)",
        "",
        "# Per-layer running stats",
        "stats = {L: {'residual_sq': 0.0, 'var_total': 0.0, 'l0_sum': 0, 'l0_n': 0, 'fired': set()} for L in ALL_LAYERS}",
        "",
        "tap = LayerTap(model, ALL_LAYERS)",
        "text_gen = text_stream()",
        "emitted = 0",
        "t0 = time.time()",
        "pbar = tqdm(total=VAL_TOKENS, unit='tok', unit_scale=True, desc='Validation')",
        "",
        "try:",
        "    while emitted < VAL_TOKENS:",
        "        ids = pack_sequences(text_gen, FWD_BATCH, SEQ_LEN)",
        "        with torch.no_grad():",
        "            model(ids)",
        "        chunk_tokens = FWD_BATCH * SEQ_LEN",
        "        emitted += chunk_tokens",
        "        for L in ALL_LAYERS:",
        "            x = tap.buf[L].reshape(-1, D_MODEL).to(torch.float32)",
        "            xh, z, top_i = saes[L](x)",
        "            s = stats[L]",
        "            s['residual_sq'] += (x - xh).pow(2).sum().item()",
        "            s['var_total']   += (x - x.mean(0)).pow(2).sum().item()",
        "            s['l0_sum']      += (z > 0).sum().item()",
        "            s['l0_n']        += x.shape[0]",
        "            s['fired'].update(top_i.unique().cpu().tolist())",
        "        pbar.update(chunk_tokens)",
        "finally:",
        "    tap.close()",
        "",
        "pbar.close()",
        "elapsed = time.time() - t0",
        "print(f'\\nValidation done in {elapsed/60:.1f} min')",
    ]))

    # Cell 9: Compute report + comparison
    cells.append(md(["## 9. Compute val_report.json + per-layer comparison"]))
    cells.append(code([
        "report = {}",
        "for L in ALL_LAYERS:",
        "    s = stats[L]",
        "    var_expl = 1.0 - s['residual_sq'] / s['var_total']",
        "    l0 = s['l0_sum'] / s['l0_n']",
        "    alive = len(s['fired'])",
        "    n_total = FULLSTACK_D_SAE if L in FULLSTACK_LAYERS else PAPERGRADE_D_SAE",
        "    dead_pct = 100.0 * (1 - alive / n_total)",
        "    source = 'fullstack' if L in FULLSTACK_LAYERS else 'papergrade'",
        "    report[L] = {",
        "        'source': source,",
        "        'd_sae': n_total,",
        "        'var_expl': var_expl,",
        "        'l0': l0,",
        "        'alive': alive,",
        "        'alive_pct': 100.0 * alive / n_total,",
        "        'dead_pct': dead_pct,",
        "    }",
        "",
        "# Print table",
        "print(f'{\"Layer\":<6} {\"Source\":<12} {\"d_sae\":<8} {\"ve\":>8} {\"L0\":>6} {\"alive%\":>8} {\"dead%\":>8}')",
        "print('-' * 60)",
        "for L in sorted(report.keys()):",
        "    r = report[L]",
        "    print(f\"L{L:<5} {r['source']:<12} {r['d_sae']:<8} {r['var_expl']:>8.4f} {r['l0']:>6.1f} {r['alive_pct']:>7.2f}% {r['dead_pct']:>7.2f}%\")",
        "",
        "# Save",
        "report_path = OUT / 'val_report.json'",
        "with open(report_path, 'w') as f:",
        "    json.dump({str(L): v for L, v in report.items()}, f, indent=2)",
        "print(f'\\n✓ Saved: {report_path}')",
        "",
        "# Upload to HF",
        "from huggingface_hub import HfApi",
        "hfapi = HfApi()",
        "hfapi.upload_file(",
        "    path_or_fileobj=str(report_path),",
        "    path_in_repo='val_report.json',",
        "    repo_id=FULLSTACK_REPO,",
        "    commit_message=f'Validation report ({VAL_TOKENS/1e6:.0f}M tokens, 14 layers)',",
        ")",
        "print(f'✓ Uploaded to {FULLSTACK_REPO}/val_report.json')",
    ]))

    # Cell 10: Comparison plot
    cells.append(md(["## 10. Plot — fullstack vs papergrade U-shape"]))
    cells.append(code([
        "import matplotlib.pyplot as plt",
        "",
        "fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))",
        "",
        "layers_sorted = sorted(report.keys())",
        "ve_vals = [report[L]['var_expl'] for L in layers_sorted]",
        "l0_vals = [report[L]['l0'] for L in layers_sorted]",
        "alive_vals = [report[L]['alive_pct'] for L in layers_sorted]",
        "colors = ['#ef4444' if report[L]['source'] == 'papergrade' else '#3b82f6' for L in layers_sorted]",
        "",
        "ax = axes[0]",
        "ax.bar(range(len(layers_sorted)), ve_vals, color=colors)",
        "ax.set_xticks(range(len(layers_sorted)))",
        "ax.set_xticklabels([f'L{L}' for L in layers_sorted], rotation=45)",
        "ax.set_ylabel('Variance explained')",
        "ax.set_title('ve per layer (red=papergrade d=65536, blue=fullstack d=40960)')",
        "ax.axhline(0.7, color='gray', linestyle='--', alpha=0.5, label='ve=0.7')",
        "ax.axhline(0.8, color='gray', linestyle=':', alpha=0.5, label='ve=0.8')",
        "ax.legend(); ax.grid(alpha=0.3)",
        "",
        "ax = axes[1]",
        "ax.bar(range(len(layers_sorted)), l0_vals, color=colors)",
        "ax.set_xticks(range(len(layers_sorted)))",
        "ax.set_xticklabels([f'L{L}' for L in layers_sorted], rotation=45)",
        "ax.set_ylabel('L0 (avg active features)')",
        "ax.set_title('L0 per layer (target: k=128)')",
        "ax.axhline(128, color='red', linestyle='--', label='k=128')",
        "ax.legend(); ax.grid(alpha=0.3)",
        "",
        "ax = axes[2]",
        "ax.bar(range(len(layers_sorted)), alive_vals, color=colors)",
        "ax.set_xticks(range(len(layers_sorted)))",
        "ax.set_xticklabels([f'L{L}' for L in layers_sorted], rotation=45)",
        "ax.set_ylabel('Alive features (%)')",
        "ax.set_title('Alive% per layer (target: ≥90%)')",
        "ax.axhline(90, color='red', linestyle='--', label='90%')",
        "ax.legend(); ax.grid(alpha=0.3)",
        "",
        "plt.tight_layout()",
        "plot_path = OUT / 'val_report_comparison.png'",
        "plt.savefig(plot_path, dpi=170, bbox_inches='tight')",
        "plt.show()",
        "print(f'✓ Saved: {plot_path}')",
        "",
        "# Upload plot",
        "hfapi.upload_file(",
        "    path_or_fileobj=str(plot_path),",
        "    path_in_repo='val_report_comparison.png',",
        "    repo_id=FULLSTACK_REPO,",
        "    commit_message='Validation report plot',",
        ")",
        "print(f'✓ Plot uploaded')",
    ]))

    # Cell 11: Summary verdict
    cells.append(md(["## 11. Summary verdict + Path 2 config recommendation"]))
    cells.append(code([
        "# Identify worst layers (lowest ve) — these are Path 2 retrain targets",
        "fullstack_only = {L: r for L, r in report.items() if r['source'] == 'fullstack'}",
        "by_ve = sorted(fullstack_only.items(), key=lambda x: x[1]['var_expl'])",
        "",
        "print('=== Phase 4 v0.1 Validation Verdict ===\\n')",
        "print('Worst 5 layers (Path 2 retrain candidates):')",
        "for L, r in by_ve[:5]:",
        "    print(f\"  L{L}: ve={r['var_expl']:.4f} L0={r['l0']:.1f} alive={r['alive_pct']:.1f}%\")",
        "print('\\nBest 5 layers:')",
        "for L, r in by_ve[-5:]:",
        "    print(f\"  L{L}: ve={r['var_expl']:.4f} L0={r['l0']:.1f} alive={r['alive_pct']:.1f}%\")",
        "",
        "median_ve = np.median([r['var_expl'] for r in fullstack_only.values()])",
        "median_alive = np.median([r['alive_pct'] for r in fullstack_only.values()])",
        "print(f'\\nFullstack median ve: {median_ve:.4f}')",
        "print(f'Fullstack median alive%: {median_alive:.2f}%')",
        "",
        "papergrade_only = {L: r for L, r in report.items() if r['source'] == 'papergrade'}",
        "for L, r in sorted(papergrade_only.items()):",
        "    print(f'Papergrade L{L}: ve={r[\"var_expl\"]:.4f} (reference)')",
        "",
        "print('\\n=== Path 2 config recommendation ===')",
        "if median_alive >= 90:",
        "    print('alive% OK → AuxK alpha unchanged (1/32)')",
        "else:",
        "    print(f'alive% {median_alive:.1f}% < 90 → bump AuxK alpha to 1/16')",
        "",
        "print('Path 2 should retrain bottom-3 layers from worst list with:')",
        "print(f'  d_sae=65536 (vs 40960)')",
        "print(f'  k_topk=192 or 256 (vs 128) — addresses mid-layer composition density')",
        "print(f'  Same 200M tokens, same lr schedule')",
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
    out_path = NOTEBOOKS_DIR / "nb_phase4_val_report.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"✓ Wrote {out_path}")
    print(f"  Cells: {len(cells)}")


if __name__ == "__main__":
    build()
