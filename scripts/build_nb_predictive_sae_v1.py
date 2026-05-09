"""
Builder for nb_predictive_sae_v1_multilayer.ipynb.

PREDICTIVE SAE v1.5 — Multi-layer feature trajectory prediction.

Hypothesis: linear probes can predict SAE features at end-of-thinking from
features at earlier thinking position, ACROSS multiple layers (L11/L31/L55).

Why multi-layer:
- "Anthropic attribution graphs are static. We add temporal axis ACROSS LAYERS."
- Direct novelty axis vs SSAE / Tracing-the-Traces / DEER
- Reveals BOTH within-generation AND cross-layer feature dynamics
- Tests construct-then-compress hypothesis at within-generation timescale

Pipeline:
1. Generate ~150 GSM8K prompts with thinking enabled
2. Capture L11, L31, L55 residuals at 5 thinking fractions [10%, 25%, 50%, 75%, end]
3. Train linear probes — one per (source_layer × source_fraction) → predict
   same-layer features at end-of-thinking (12 probes total)
4. Eval recall@k per layer × fraction
5. Plot per-layer curves (3 curves)
6. Verdict + early-exit threshold per layer

Layers: L11 (early/input), L31 (mid/compositional), L55 (late/answer-ready)
SAEs: papergrade trio (d=65536, k=128 each)
Dataset: GSM8K test split (150 prompts, 80/20 train/test)
Compute: ~3.5h on RTX 6000 Blackwell

Drive: /content/drive/MyDrive/openinterp_runs/predictive_sae_v1/

Target: NeurIPS MI Workshop 2026 — paper-3.
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
        "# Predictive SAE v1.5 — Multi-Layer Feature Trajectory Prediction",
        "",
        "**Hypothesis**: linear probes predict SAE features at end-of-thinking from",
        "earlier thinking positions, ACROSS multiple layers (L11/L31/L55).",
        "",
        "**Novelty (gap confirmed via search, size M, no scoop)**:",
        "- Anthropic attribution graphs: STATIC → we add temporal axis",
        "- SSAE: step-level current → we predict ahead",
        "- Tracing-the-Traces: raw activations → we use SAE features (interpretable)",
        "- DEER: logit confidence → we use feature signal",
        "- **Combined**: SAE-features-as-target × within-generation × multi-layer × predictive",
        "",
        "**Experimental design**:",
        "- 3 layers × 4 source fractions × N=120 train = 12 probes",
        "- Target per probe: same-layer features at end-of-thinking (fraction=1.00)",
        "- Source fractions: 10%, 25%, 50%, 75% of thinking phase",
        "- Recall@k for k ∈ {128, 256, 512, 1024, 2048, 4096}",
        "",
        "**Layers**:",
        "- **L11** (early/input): captures input-derived features",
        "- **L31** (mid/compositional): vale do U-shape, reasoning composition",
        "- **L55** (late/answer-ready): answer-shaped features",
        "",
        "**Compute**: ~3.5h on RTX 6000 Blackwell.",
        "",
        "**Target**: NeurIPS MI Workshop 2026 — paper-3 (companion to paper-2 grokking).",
    ]))

    # ============================================================
    # Cell 1 — Drive
    # ============================================================
    cells.append(md(["## 1. Drive mount + paths"]))
    cells.append(code([
        "from pathlib import Path",
        "import os, json, time, math, random",
        "import torch, numpy as np",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "DRIVE = Path('/content/drive/MyDrive')",
        "OUT = DRIVE / 'openinterp_runs' / 'predictive_sae_v1'",
        "OUT.mkdir(parents=True, exist_ok=True)",
        "(OUT / 'cache').mkdir(parents=True, exist_ok=True)",
        "(OUT / 'probes').mkdir(parents=True, exist_ok=True)",
        "(OUT / 'figures').mkdir(parents=True, exist_ok=True)",
        "print(f'OUT: {OUT}')",
        "print(f'Existing: {sorted(p.name for p in OUT.iterdir())}')",
    ]))

    # ============================================================
    # Cell 2 — Install
    # ============================================================
    cells.append(md(["## 2. Install (transformers main + flash-linear-attention)"]))
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
        "else:",
        "    print(f'transformers {transformers.__version__} ✓')",
        "",
        "try:",
        "    import fla",
        "    print('flash-linear-attention ✓')",
        "except ImportError:",
        "    pip('install', '-q', '--no-cache-dir', 'flash-linear-attention')",
        "    needs_restart = True",
        "",
        "pip('install', '-q', 'matplotlib')",
        "",
        "if needs_restart:",
        "    print('\\n*** RESTART RUNTIME, then re-run cells 1+2 ***')",
        "",
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

    # ============================================================
    # Cell 3 — CFG
    # ============================================================
    cells.append(md(["## 3. CFG"]))
    cells.append(code([
        "MODEL_ID    = 'Qwen/Qwen3.6-27B'",
        "D_MODEL     = 5120",
        "DEVICE      = 'cuda'",
        "DTYPE       = torch.bfloat16",
        "",
        "# Multi-layer setup — papergrade trio (d=65536)",
        "LAYERS      = [11, 31, 55]",
        "SAE_REPO    = 'caiovicentino1/qwen36-27b-sae-papergrade'",
        "D_SAE       = 65536",
        "K_SAE       = 128",
        "",
        "# Dataset",
        "N_PROMPTS   = 150",
        "TRAIN_FRAC  = 0.80      # 120 train, 30 test",
        "MAX_NEW_TOK = 2048",
        "",
        "# Thinking fractions to capture (relative to think_end)",
        "FRACTIONS   = [0.10, 0.25, 0.50, 0.75, 1.00]  # 1.00 = end (target)",
        "SOURCE_FRACS = [0.10, 0.25, 0.50, 0.75]       # source = predict from these",
        "",
        "# Probe training",
        "N_EPOCHS    = 5",
        "BATCH_SIZE  = 256",
        "LR          = 1e-3",
        "WD          = 1e-5",
        "",
        "# Probe prediction breadth (m)",
        "M_VALUES    = [128, 256, 512, 1024, 2048, 4096]",
        "",
        "SEED        = 42",
        "torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)",
        "",
        "print(f'Layers: {LAYERS}')",
        "print(f'SAE: {SAE_REPO}, d={D_SAE}, k={K_SAE}')",
        "print(f'{N_PROMPTS} prompts × {len(SOURCE_FRACS)} source fracs × {len(LAYERS)} layers = {len(SOURCE_FRACS) * len(LAYERS)} probes')",
    ]))

    # ============================================================
    # Cell 4 — Load Qwen
    # ============================================================
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
        "",
        "THINK_START_ID = tok.convert_tokens_to_ids('<think>')",
        "THINK_END_ID = tok.convert_tokens_to_ids('</think>')",
        "if THINK_START_ID == tok.unk_token_id:",
        "    THINK_START_ID = tok.encode('<think>', add_special_tokens=False)[-1]",
        "if THINK_END_ID == tok.unk_token_id:",
        "    THINK_END_ID = tok.encode('</think>', add_special_tokens=False)[-1]",
        "print(f'<think> id: {THINK_START_ID}, </think> id: {THINK_END_ID}')",
    ]))

    # ============================================================
    # Cell 5 — Layer helpers + multi-layer LayerTap
    # ============================================================
    cells.append(md(["## 5. Multi-layer LayerTap"]))
    cells.append(code([
        "def get_layer_module(m, idx):",
        "    for path in [('model','language_model','layers'),",
        "                 ('language_model','layers'), ('model','layers')]:",
        "        try:",
        "            cur = m",
        "            for p in path: cur = getattr(cur, p)",
        "            return cur[idx]",
        "        except AttributeError: continue",
        "    raise RuntimeError",
        "",
        "class MultiLayerTap:",
        "    \"\"\"Captures full residuals from multiple layers in single forward.\"\"\"",
        "    def __init__(self, model, layers):",
        "        self.layers = layers",
        "        self.captured = {L: None for L in layers}",
        "        self.handles = []",
        "        for L in layers:",
        "            mod = get_layer_module(model, L)",
        "            self.handles.append(mod.register_forward_hook(self._mk_hook(L)))",
        "    def _mk_hook(self, L):",
        "        def _h(module, inp, out):",
        "            h = out[0] if isinstance(out, tuple) else out",
        "            self.captured[L] = h.detach().to(torch.bfloat16)",
        "        return _h",
        "    def close(self):",
        "        for h in self.handles: h.remove()",
        "",
        "tap = MultiLayerTap(model, LAYERS)",
        "ids = tok('The quick brown fox', return_tensors='pt').input_ids.to(DEVICE)",
        "with torch.no_grad():",
        "    model(ids)",
        "for L in LAYERS:",
        "    print(f'L{L}: shape={tuple(tap.captured[L].shape)}, norm={tap.captured[L].float().norm():.2f}')",
        "tap.close()",
    ]))

    # ============================================================
    # Cell 6 — Load 3 papergrade SAEs
    # ============================================================
    cells.append(md(["## 6. Load 3 papergrade SAEs (L11, L31, L55)"]))
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
        "        self.k = k",
        "        self.n = n",
        "    def encode_pre(self, x):",
        "        return (x - self.b_dec) @ self.W_enc + self.b_enc",
        "    @torch.no_grad()",
        "    def topk_features(self, x):",
        "        pre = self.encode_pre(x)",
        "        return pre.topk(self.k, dim=-1)",
        "",
        "saes = {}",
        "for L in LAYERS:",
        "    print(f'Loading SAE L{L}...', end=' ')",
        "    weights = load_file(hf_hub_download(SAE_REPO, f'sae_L{L}_latest.safetensors'))",
        "    sae = TopKSAEInf(D_MODEL, D_SAE, K_SAE).to(DEVICE, torch.float32)",
        "    for k_, v in weights.items(): getattr(sae, k_).data = v.to(DEVICE, torch.float32)",
        "    sae.eval()",
        "    saes[L] = sae",
        "    print(f'd={D_SAE}, k={K_SAE} ✓')",
        "",
        "print(f'\\nVRAM after 3 SAEs: {torch.cuda.memory_allocated()/1e9:.1f} GB')",
    ]))

    # ============================================================
    # Cell 7 — Generation
    # ============================================================
    cells.append(md([
        "## 7. Generation pass — collect thinking traces",
        "",
        "Resume-safe via Drive cache."
    ]))
    cells.append(code([
        "from datasets import load_dataset",
        "from tqdm.auto import tqdm",
        "",
        "def make_prompt_thinking(question):",
        "    msgs = [{'role': 'user', 'content': question}]",
        "    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)",
        "",
        "def find_token_position(token_ids, target_id):",
        "    matches = (token_ids == target_id).nonzero(as_tuple=True)[0]",
        "    return matches[0].item() if len(matches) > 0 else -1",
        "",
        "ds = load_dataset('gsm8k', 'main', split='test')",
        "prompts_set = ds.select(range(N_PROMPTS))",
        "",
        "trace_cache = OUT / 'cache' / 'thinking_traces.pt'",
        "if trace_cache.exists():",
        "    traces = torch.load(trace_cache, weights_only=False)",
        "    print(f'✓ Loaded {len(traces)} traces from cache')",
        "else:",
        "    traces = []",
        "    for ex in tqdm(prompts_set, desc='Generating'):",
        "        prompt = make_prompt_thinking(ex['question'])",
        "        ids = tok(prompt, return_tensors='pt').input_ids.to(DEVICE)",
        "        prompt_len = ids.shape[1]",
        "        with torch.no_grad():",
        "            out = model.generate(",
        "                ids, max_new_tokens=MAX_NEW_TOK,",
        "                do_sample=False, temperature=0.0,",
        "                pad_token_id=tok.pad_token_id,",
        "            )",
        "        gen_ids = out[0]",
        "        gen_only = gen_ids[prompt_len:]",
        "        think_end_rel = find_token_position(gen_only, THINK_END_ID)",
        "        if think_end_rel < 0 or think_end_rel < 50:",
        "            continue",
        "        think_start_full = find_token_position(gen_ids, THINK_START_ID)",
        "        if think_start_full < 0:",
        "            think_start_full = prompt_len",
        "        think_end_full = prompt_len + think_end_rel",
        "        traces.append({",
        "            'question': ex['question'],",
        "            'answer': ex['answer'],",
        "            'gen_ids': gen_ids.cpu(),",
        "            'prompt_len': prompt_len,",
        "            'think_start': think_start_full,",
        "            'think_end': think_end_full,",
        "            'thinking_length': think_end_full - think_start_full,",
        "        })",
        "    torch.save(traces, trace_cache)",
        "    print(f'✓ Saved {len(traces)} traces')",
        "",
        "thinking_lengths = [t['thinking_length'] for t in traces]",
        "print(f'Thinking length: mean={np.mean(thinking_lengths):.0f}, median={np.median(thinking_lengths):.0f}')",
    ]))

    # ============================================================
    # Cell 8 — Capture residuals at all layers × fractions
    # ============================================================
    cells.append(md([
        "## 8. Forward pass — capture residuals at LAYERS × FRACTIONS",
        "",
        "One forward per trace, MultiLayerTap captures all 3 layers simultaneously.",
        "Output structure: residuals[layer][fraction] = (N, D_MODEL).",
    ]))
    cells.append(code([
        "from tqdm.auto import tqdm",
        "",
        "residuals_cache = OUT / 'cache' / 'residuals_multilayer.pt'",
        "if residuals_cache.exists():",
        "    residuals = torch.load(residuals_cache, weights_only=False)",
        "    print(f'✓ Loaded residuals from cache')",
        "else:",
        "    residuals = {L: {f: [] for f in FRACTIONS} for L in LAYERS}",
        "    tap = MultiLayerTap(model, LAYERS)",
        "    try:",
        "        for trace in tqdm(traces, desc='Capturing residuals'):",
        "            gen_ids = trace['gen_ids'].to(DEVICE).unsqueeze(0)",
        "            think_start = trace['think_start']",
        "            think_end = trace['think_end']",
        "            thinking_len = trace['thinking_length']",
        "            with torch.no_grad():",
        "                model(gen_ids)",
        "            for frac in FRACTIONS:",
        "                offset = int(round(thinking_len * frac))",
        "                pos = min(think_start + offset, think_end)",
        "                pos = max(pos, think_start + 1)",
        "                for L in LAYERS:",
        "                    residuals[L][frac].append(",
        "                        tap.captured[L][0, pos].cpu().to(torch.float32)",
        "                    )",
        "    finally:",
        "        tap.close()",
        "    # Stack per (layer, fraction)",
        "    for L in LAYERS:",
        "        for f in FRACTIONS:",
        "            residuals[L][f] = torch.stack(residuals[L][f])",
        "    torch.save(residuals, residuals_cache)",
        "    print(f'✓ Saved residuals')",
        "",
        "for L in LAYERS:",
        "    for f in FRACTIONS:",
        "        print(f'L{L} frac={f}: {residuals[L][f].shape}')",
    ]))

    # ============================================================
    # Cell 9 — Encode features at each (layer, fraction)
    # ============================================================
    cells.append(md(["## 9. SAE encode → top-k features per (layer × fraction)"]))
    cells.append(code([
        "features_cache = OUT / 'cache' / 'features_multilayer.pt'",
        "if features_cache.exists():",
        "    features = torch.load(features_cache, weights_only=False)",
        "    print(f'✓ Loaded features from cache')",
        "else:",
        "    features = {L: {f: None for f in FRACTIONS} for L in LAYERS}",
        "    for L in LAYERS:",
        "        for f in FRACTIONS:",
        "            X = residuals[L][f].to(DEVICE)  # (N, D_MODEL)",
        "            top_v, top_i = saes[L].topk_features(X)",
        "            features[L][f] = {",
        "                'indices': top_i.cpu(),",
        "                'values': top_v.cpu().to(torch.float32),",
        "            }",
        "    torch.save(features, features_cache)",
        "    print(f'✓ Saved features')",
        "",
        "for L in LAYERS:",
        "    for f in FRACTIONS:",
        "        idx = features[L][f]['indices']",
        "        print(f'L{L} frac={f}: indices {idx.shape}, range [{idx.min()}, {idx.max()}]')",
    ]))

    # ============================================================
    # Cell 10 — Train probes
    # ============================================================
    cells.append(md([
        "## 10. Train probes — one per (source_layer × source_fraction)",
        "",
        "Each probe: residual_at(L, frac_source) → feature_active_at(L, frac=1.00).",
        "12 probes total (3 layers × 4 source fractions).",
    ]))
    cells.append(code([
        "from torch.optim.lr_scheduler import CosineAnnealingLR",
        "",
        "N = residuals[LAYERS[0]][FRACTIONS[0]].shape[0]",
        "n_train = int(N * TRAIN_FRAC)",
        "perm = torch.randperm(N, generator=torch.Generator().manual_seed(SEED))",
        "train_idx, test_idx = perm[:n_train], perm[n_train:]",
        "print(f'N total: {N}, train: {n_train}, test: {N - n_train}')",
        "",
        "def make_multihot(top_i, n_features=D_SAE):",
        "    mh = torch.zeros(top_i.shape[0], n_features, dtype=torch.bool)",
        "    mh.scatter_(-1, top_i, True)",
        "    return mh",
        "",
        "def topk_ranking_loss(logits, target_active, k=K_SAE):",
        "    pos_mask = target_active.bool()",
        "    pos_logits = (logits * pos_mask.float()).sum(-1) / k",
        "    neg_logits = (logits * (~pos_mask).float()).sum(-1) / (logits.shape[-1] - k)",
        "    return F.softplus(-(pos_logits - neg_logits)).mean()",
        "",
        "def train_probe(X_train, y_train, init_W=None, init_b=None):",
        "    probe = nn.Linear(D_MODEL, D_SAE).to(DEVICE, torch.float32)",
        "    if init_W is not None:",
        "        probe.weight.data = init_W.contiguous()",
        "        probe.bias.data = init_b.clone()",
        "    opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WD)",
        "    sched = CosineAnnealingLR(opt, T_max=N_EPOCHS)",
        "    X = X_train.to(DEVICE); y = y_train.to(DEVICE)",
        "    for epoch in range(N_EPOCHS):",
        "        idx = torch.randperm(X.shape[0], device=DEVICE)",
        "        ep_loss = 0.0; nb = 0",
        "        for i in range(0, X.shape[0], BATCH_SIZE):",
        "            b = idx[i:i+BATCH_SIZE]",
        "            logits = probe(X[b])",
        "            loss = topk_ranking_loss(logits, y[b])",
        "            opt.zero_grad(); loss.backward(); opt.step()",
        "            ep_loss += loss.item(); nb += 1",
        "        sched.step()",
        "    probe.eval()",
        "    return probe",
        "",
        "@torch.no_grad()",
        "def eval_probe(probe, X_test, y_test_multi, m_values=M_VALUES):",
        "    X = X_test.to(DEVICE)",
        "    y = y_test_multi.to(DEVICE).bool()",
        "    logits = probe(X)",
        "    out = {}",
        "    for m in m_values:",
        "        _, top_m = logits.topk(m, dim=-1)",
        "        pred = torch.zeros_like(logits, dtype=torch.bool)",
        "        pred.scatter_(-1, top_m, True)",
        "        recall = (pred & y).sum(-1).float() / K_SAE",
        "        out[m] = (recall.mean().item(), recall.std().item())",
        "    return out",
        "",
        "# Build target multihots per layer (target = features at frac=1.00 for that layer)",
        "target_multihots = {L: make_multihot(features[L][1.00]['indices']) for L in LAYERS}",
        "",
        "results = {L: {} for L in LAYERS}",
        "for L in LAYERS:",
        "    sae_l = saes[L]",
        "    for frac in SOURCE_FRACS:",
        "        print(f'\\n=== L{L} frac={frac:.0%} → frac=100% ===')",
        "        X_train = residuals[L][frac][train_idx]",
        "        X_test = residuals[L][frac][test_idx]",
        "        y_train = target_multihots[L][train_idx]",
        "        y_test = target_multihots[L][test_idx]",
        "        probe = train_probe(X_train, y_train,",
        "                            init_W=sae_l.W_enc.T,",
        "                            init_b=sae_l.b_enc)",
        "        torch.save({'state_dict': probe.state_dict(), 'layer': L, 'frac': frac},",
        "                   OUT / 'probes' / f'probe_L{L}_f{int(frac*100):03d}.pt')",
        "        metrics = eval_probe(probe, X_test, y_test)",
        "        results[L][frac] = metrics",
        "        print(f'  recall@128={metrics[128][0]:.3f}, recall@4096={metrics[4096][0]:.3f}')",
    ]))

    # ============================================================
    # Cell 11 — Plot
    # ============================================================
    cells.append(md(["## 11. Plot — recall@m vs fraction, per layer"]))
    cells.append(code([
        "import matplotlib.pyplot as plt",
        "",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))",
        "colors = {11: '#10b981', 31: '#3b82f6', 55: '#ef4444'}",
        "labels = {11: 'L11 (early)', 31: 'L31 (mid)', 55: 'L55 (late)'}",
        "",
        "# Plot 1: recall@128 (strict)",
        "ax = axes[0]",
        "for L in LAYERS:",
        "    fracs = sorted(results[L].keys())",
        "    means = [results[L][f][128][0] for f in fracs]",
        "    stds  = [results[L][f][128][1] for f in fracs]",
        "    ax.errorbar([f*100 for f in fracs], means, yerr=stds,",
        "                marker='o', label=labels[L], color=colors[L], linewidth=2, capsize=4)",
        "ax.set_xlabel('Source thinking progress (%)')",
        "ax.set_ylabel('Recall@128 (strict)')",
        "ax.set_title('Same-layer feature trajectory prediction')",
        "ax.axhline(0.7, color='gray', linestyle='--', alpha=0.5, label='Threshold 0.7')",
        "ax.axhline(K_SAE/D_SAE, color='black', linestyle=':', alpha=0.4, label='Random')",
        "ax.set_ylim(0, 1)",
        "ax.legend()",
        "ax.grid(alpha=0.3)",
        "",
        "# Plot 2: recall@4096 (relaxed, Phase 2 sweet spot)",
        "ax = axes[1]",
        "for L in LAYERS:",
        "    fracs = sorted(results[L].keys())",
        "    means = [results[L][f][4096][0] for f in fracs]",
        "    stds  = [results[L][f][4096][1] for f in fracs]",
        "    ax.errorbar([f*100 for f in fracs], means, yerr=stds,",
        "                marker='o', label=labels[L], color=colors[L], linewidth=2, capsize=4)",
        "ax.set_xlabel('Source thinking progress (%)')",
        "ax.set_ylabel('Recall@4096 (Phase 2 sweet spot)')",
        "ax.set_title('Recall@m=4096 — feature breadth')",
        "ax.axhline(0.85, color='gray', linestyle='--', alpha=0.5, label='Threshold 0.85')",
        "ax.set_ylim(0, 1)",
        "ax.legend()",
        "ax.grid(alpha=0.3)",
        "",
        "# Plot 3: per-layer trade-off curve at frac=0.50",
        "ax = axes[2]",
        "for L in LAYERS:",
        "    if 0.50 not in results[L]: continue",
        "    means = [results[L][0.50][m][0] for m in M_VALUES]",
        "    ax.plot(M_VALUES, means, marker='o', label=labels[L], color=colors[L], linewidth=2)",
        "ax.set_xlabel('m (probe top-m)')",
        "ax.set_ylabel('Recall')",
        "ax.set_xscale('log')",
        "ax.set_title('Trade-off curve at frac=50% (mid-thinking)')",
        "ax.legend()",
        "ax.grid(alpha=0.3)",
        "",
        "plt.tight_layout()",
        "fig_path = OUT / 'figures' / 'recall_multilayer.png'",
        "plt.savefig(fig_path, dpi=170, bbox_inches='tight')",
        "plt.show()",
        "print(f'✓ Saved: {fig_path}')",
    ]))

    # ============================================================
    # Cell 12 — Verdict
    # ============================================================
    cells.append(md(["## 12. Verdict + per-layer early-exit decision"]))
    cells.append(code([
        "print('=== Predictive SAE v1.5 — Multi-Layer Verdict ===\\n')",
        "print(f'{\"Layer\":<6} {\"Frac\":<8} {\"Recall@128\":<14} {\"Recall@4096\":<14}')",
        "print('-' * 45)",
        "for L in LAYERS:",
        "    for frac in sorted(results[L].keys()):",
        "        r128 = results[L][frac][128][0]",
        "        r4k = results[L][frac][4096][0]",
        "        print(f'L{L:<5} {frac:<8.0%} {r128:<14.3f} {r4k:<14.3f}')",
        "    print()",
        "",
        "# Decision: which (layer, fraction) gives best early-exit signal?",
        "print('=== Early-exit threshold analysis ===')",
        "best = None",
        "for L in LAYERS:",
        "    for frac in sorted(results[L].keys()):",
        "        r = results[L][frac][128][0]",
        "        if r >= 0.7 and (best is None or frac < best[1]):",
        "            best = (L, frac, r)",
        "if best:",
        "    L, frac, r = best",
        "    print(f'🟢 STRONG: L{L} at frac={frac:.0%} reaches recall@128={r:.3f} ≥ 0.7')",
        "    print(f'   Early-exit at {frac:.0%} of thinking → {(1-frac)*100:.0f}% speedup on think phase')",
        "else:",
        "    # Check 4k threshold",
        "    best_4k = None",
        "    for L in LAYERS:",
        "        for frac in sorted(results[L].keys()):",
        "            r = results[L][frac][4096][0]",
        "            if r >= 0.85 and (best_4k is None or frac < best_4k[1]):",
        "                best_4k = (L, frac, r)",
        "    if best_4k:",
        "        L, frac, r = best_4k",
        "        print(f'🟡 MARGINAL: L{L} at frac={frac:.0%} reaches recall@4096={r:.3f} ≥ 0.85')",
        "        print(f'   Use Phase 2 m=4096 trade-off — early-exit at {frac:.0%}')",
        "    else:",
        "        print('🔴 INSUFFICIENT: linear probe within-generation weak — needs MLP probe or cross-layer setup (v2)')",
        "",
        "print('\\n=== Cross-layer comparison ===')",
        "# Which layer is most predictable?",
        "for frac in SOURCE_FRACS:",
        "    layer_scores = {L: results[L][frac][128][0] for L in LAYERS}",
        "    best_L = max(layer_scores, key=layer_scores.get)",
        "    print(f'  frac={frac:.0%}: L{best_L} best (recall@128={layer_scores[best_L]:.3f})')",
        "",
        "# Save",
        "results_path = OUT / 'predictive_sae_v15_results.json'",
        "results_dump = {",
        "    'layers': LAYERS,",
        "    'fractions': sorted(results[LAYERS[0]].keys()),",
        "    'recall': {",
        "        f'L{L}': {",
        "            str(f): {str(m): list(v) for m, v in metrics.items()}",
        "            for f, metrics in results[L].items()",
        "        }",
        "        for L in LAYERS",
        "    },",
        "    'config': {",
        "        'd_sae': D_SAE, 'k_sae': K_SAE,",
        "        'n_train': n_train, 'n_test': N - n_train, 'seed': SEED,",
        "    },",
        "}",
        "with open(results_path, 'w') as f:",
        "    json.dump(results_dump, f, indent=2)",
        "print(f'\\n✓ Saved: {results_path}')",
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
    out_path = NOTEBOOKS_DIR / "nb_predictive_sae_v1.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"✓ Wrote {out_path}")
    print(f"  Cells: {len(cells)}")


if __name__ == "__main__":
    build()
