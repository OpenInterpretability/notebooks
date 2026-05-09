#!/usr/bin/env python3
"""
Generates notebook 33 — FabricationGuard vs HaluGate head-to-head benchmark.
Single source of truth. Writes the verified .ipynb file.
"""
import json
from pathlib import Path

NB_OUT = Path('/Volumes/SSD Major/fish/openinterp-work/notebooks/33_fabricationguard_vs_halugate.ipynb')

# ---------- helpers ---------------------------------------------------------

def md(text: str) -> dict:
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.rstrip('\n').split('\n')],
    }

def code(text: str) -> dict:
    return {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': [line + '\n' for line in text.rstrip('\n').split('\n')],
    }

# ---------- notebook content -----------------------------------------------

cells = []

# 0. Title -------------------------------------------------------------------
cells.append(md(r'''
# FabricationGuard vs HaluGate — pre-gen probe vs post-gen NLI

**Notebook 33 · ProbeBench v0.0.1 · OpenInterp · April 2026**

A direct head-to-head benchmark of two production hallucination-detection methods on **Qwen3.6-27B**:

- **FabricationGuard** — L2 logistic-regression probe on the residual stream at L31. Trained cross-bench on TruthfulQA + HaluEval + MMLU, scored at the last token of a `Q: …\nA:` prompt. **Pre-generation**: decides to abstain *before* spending tokens on the answer.
- **HaluGate-style NLI baseline** — replication of the methodology used in vLLM Semantic Router v0.1 "Iris" HaluGate (Dec 2025). Two modes:
  - **Grounded** — NLI between the model's answer and a retrieval / gold context. Best when ground truth is available (RAG, function-calling).
  - **Self-consistency** — K=3 sampled answers, pairwise NLI disagreement. Closed-book fallback.

## Hypotheses (falsifiable)

| | Claim | Test |
|---|---|---|
| **H1** | FabricationGuard has lower latency-to-decision than HaluGate when the system needs to abstain pre-generation | Compare TTFT distributions |
| **H2** | FabricationGuard ≥ HaluGate on closed-book QA (no retrieval) | AUROC on SimpleQA, TruthfulQA-MC1, HaluEval closed-book |
| **H3** | HaluGate-grounded ≥ FabricationGuard on retrieval-grounded QA | AUROC on HaluEval-grounded (knowledge field as context) |
| **H4** | FabricationGuard ⊕ HaluGate dominates either alone | Marginal AUROC + confident-wrong rate of the OR-combo |

Final results render at `https://openinterp.org/probebench/comparisons/fabricationguard-vs-halugate`.

## Reproducer

This notebook is end-to-end self-contained. It:

1. Loads Qwen3.6-27B in BF16 (handles multimodal-vs-causal head fallback)
2. Downloads the live FabricationGuard probe artifact from `caiovicentino1/hallucinationguard-v2-linearprobe-qwen36-27b`
3. Loads `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` as the HaluGate NLI backend
4. Runs three datasets × three methods (FG, HG-grounded, HG-CB), bootstraps AUROC CIs, computes ECE / FPR@99TPR / latency CDFs
5. Saves CSVs + plots + JSON summary; optional upload to HF dataset

Total runtime on 1× H100 80GB: **~3-4 hours** for default sizes (200 HaluEval + 100 SimpleQA + 100 TruthfulQA = 400 unique queries).
'''.strip()))

# 1. Compute -----------------------------------------------------------------
cells.append(md(r'''
## Compute requirements

- **Hardware**: 1× H100 80GB (preferred) or A100 80GB. Qwen3.6-27B BF16 ≈ 55 GB + KV cache + NLI model + scratch.
- **Runtime**: ~3-4 hours wall-clock at the default subset sizes.
- **Cost**: ~$5-10 on Colab Pro+ (A100/H100 sessions) or vast.ai.
- **Ablations**: setting `MODEL_ID = "Qwen/Qwen3-8B"` reduces compute 4× but the FabricationGuard artifact is Qwen3.6-27B-specific — only the HaluGate side is meaningful at the smaller size.

If `nvidia-smi` below reports < 60 GB, the notebook will warn and you should either upgrade GPU or skip the FabricationGuard side.
'''.strip()))

# 2. nvidia-smi --------------------------------------------------------------
cells.append(code(r'''
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "no GPU"
'''.strip()))

# 3. install -----------------------------------------------------------------
cells.append(code(r'''
# pip install — NEVER pin transformers (Qwen3.6 needs recent), DO NOT install sae-bench (downgrades numpy)
%pip install -q -U transformers accelerate safetensors huggingface_hub datasets
%pip install -q -U scikit-learn matplotlib seaborn tqdm
%pip install -q -U joblib sentencepiece protobuf
print("installs done")
'''.strip()))

# 3b. DRIVE MOUNT — non-negotiable first cell after install ----------------
cells.append(md('## 0. Drive mount + checkpoint dir (non-negotiable)'))
cells.append(code(r'''
# === DRIVE MOUNT — non-negotiable for any run >30min ===
# All checkpoints, intermediate saves, and final artifacts go to a Drive path.
# Volatile /content/ is forbidden for outputs.
from pathlib import Path
import os, sys

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except Exception as e:
    print(f'Drive mount FAILED: {e}')
    print('You are NOT in Colab, or Drive is unavailable. Refuse to proceed.')
    raise

DRIVE_ROOT = Path('/content/drive/MyDrive')
assert DRIVE_ROOT.exists(), f'{DRIVE_ROOT} not found — Drive mount silently failed'
NB_NAME = '33_fabricationguard_vs_halugate'
OUT = DRIVE_ROOT / 'openinterp_runs' / NB_NAME
OUT.mkdir(parents=True, exist_ok=True)
(OUT / '_dry_run.txt').write_text('drive mount OK')
assert (OUT / '_dry_run.txt').exists()
print(f'✓ Drive checkpoint dir: {OUT}')
print(f'  Existing artifacts: {sorted(p.name for p in OUT.iterdir())}')
'''.strip()))

cells.append(code(r'''
# === Checkpoint cadence ===
CHECKPOINT_EVERY     = 25     # save CSV to Drive every N rows
HF_PUSH_EVERY        = 100    # push to HF every N rows (best-effort)
ENABLE_HF_INCREMENTAL = True
print(f'✓ Checkpoint cadence: Drive every {CHECKPOINT_EVERY} rows; HF every {HF_PUSH_EVERY} (enabled={ENABLE_HF_INCREMENTAL})')
'''.strip()))

# 4. imports + CFG -----------------------------------------------------------
cells.append(md('## 1. Setup, config, HF login'))
cells.append(code(r'''
import os, json, time, math, gc, sys, hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Callable, Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import joblib
from huggingface_hub import login, snapshot_download, hf_hub_download, HfApi, create_repo
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

CFG = {
    'model':            'Qwen/Qwen3.6-27B',
    'probe_repo':       'caiovicentino1/hallucinationguard-v2-linearprobe-qwen36-27b',
    'nli_model':        'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
    'probe_layer':      31,
    'subset_size': {
        'haluval':    200,
        'simpleqa':   100,
        'truthfulqa': 100,
    },
    'gen_max_new_tokens':  100,
    'sc_K':                3,        # self-consistency samples per query
    'sc_temperature':      0.7,
    'sc_top_p':            0.9,
    'random_seed':         42,
    'fg_threshold':        0.5,      # default decision threshold for confident-wrong
    'hg_threshold':        0.5,
    'bootstrap_n':         1000,
    'output_repo':         os.environ.get('HF_USERNAME', 'caiovicentino1') + '/probebench-fg-vs-halugate',
}
LOCAL_OUT = OUT  # alias to Drive checkpoint dir from cell 0 — never /content/
print(f'LOCAL_OUT (Drive): {LOCAL_OUT}')
print(json.dumps(CFG, indent=2, default=str))

torch.manual_seed(CFG['random_seed'])
np.random.seed(CFG['random_seed'])
import random; random.seed(CFG['random_seed'])
'''.strip()))

cells.append(code(r'''
# HF login (write scope — needed only for the optional upload at the end)
HF_TOKEN = os.environ.get('HF_TOKEN')
if HF_TOKEN is None:
    try:
        import getpass
        HF_TOKEN = getpass.getpass('HF token (write scope, optional — Enter to skip): ').strip() or None
    except Exception:
        HF_TOKEN = None
if HF_TOKEN:
    login(HF_TOKEN, add_to_git_credential=False)
    print('HF login: OK')
else:
    print('HF login: skipped (no upload at the end)')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device == 'cuda', 'Need GPU.'
gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'CUDA: {torch.cuda.get_device_name(0)}, {gpu_mem_gb:.1f} GB')
if gpu_mem_gb < 60:
    print('⚠️  < 60 GB — Qwen3.6-27B BF16 may OOM. Reduce subset sizes or fall back to a smaller MODEL_ID.')
'''.strip()))

# 5. Load Qwen --------------------------------------------------------------
cells.append(md('## 2. Load Qwen3.6-27B (robust to multimodal head)'))
cells.append(code(r'''
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM

print(f'Loading {CFG["model"]} ...')
t0 = time.time()
tok = AutoTokenizer.from_pretrained(CFG['model'], trust_remote_code=True)
try:
    # Qwen3.6 family is multimodal — image-text-to-text head exposes the LM head we need
    model = AutoModelForImageTextToText.from_pretrained(
        CFG['model'], dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map={'': device}, trust_remote_code=True,
    )
    print('loaded as ImageTextToText')
except Exception as e:
    print(f'  ImageTextToText load failed ({type(e).__name__}); falling back to CausalLM')
    model = AutoModelForCausalLM.from_pretrained(
        CFG['model'], dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map={'': device}, trust_remote_code=True,
    )
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
print(f'Loaded in {time.time() - t0:.1f}s, model uses {torch.cuda.memory_allocated()/1e9:.1f} GB')
'''.strip()))

cells.append(code(r'''
# Robust transformer-block-list finder (handles multimodal nesting)
def _block_list(m):
    candidates = [m]
    if hasattr(m, 'model'):
        candidates.append(m.model)
    for s in candidates:
        for path in [('model','language_model','layers'),
                     ('language_model','layers'),
                     ('model','layers'),
                     ('layers',)]:
            cur = s; ok = True
            for p in path:
                if hasattr(cur, p):
                    cur = getattr(cur, p)
                else:
                    ok = False; break
            if ok and hasattr(cur, '__getitem__'):
                return cur
    raise RuntimeError('Could not locate transformer block list')

blocks = _block_list(model)
n_layers = len(blocks)
d_model = model.config.hidden_size if hasattr(model.config, 'hidden_size') else model.config.text_config.hidden_size
print(f'Layers: {n_layers}, d_model: {d_model}, probe layer: L{CFG["probe_layer"]}')
assert CFG['probe_layer'] < n_layers, f'probe layer {CFG["probe_layer"]} out of range'
'''.strip()))

cells.append(code(r'''
# Smoke test: simple greedy generation
@torch.no_grad()
def _smoke():
    messages = [{'role': 'user', 'content': 'What is the capital of Brazil?'}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors='pt').to(device)
    out = model.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
print('Smoke generation:', _smoke())
'''.strip()))

# 6. FabricationGuard probe -------------------------------------------------
cells.append(md(r'''
## 3. FabricationGuard probe — load + score function

Probe artifact: `caiovicentino1/hallucinationguard-v2-linearprobe-qwen36-27b` (HF dataset repo). Format: `probe.joblib` is a dict `{probe, scaler, layer}`. The probe was trained at the **last token of a plain `Q: {question}\nA:` prompt** — we replicate that exactly.
'''.strip()))

cells.append(code(r'''
print(f'Downloading {CFG["probe_repo"]} ...')
PROBE_DIR = Path(snapshot_download(CFG['probe_repo'], repo_type='dataset'))
print('Files:', sorted(p.name for p in PROBE_DIR.iterdir()))

# Probe format may be a single joblib bundle OR separate probe.joblib + scaler.joblib
probe_clf = scaler = None
bundle_paths = ['probe.joblib', 'fabricationguard.joblib', 'global_probe.joblib']
for pp in bundle_paths:
    p = PROBE_DIR / pp
    if p.exists():
        obj = joblib.load(p)
        if isinstance(obj, dict) and 'probe' in obj:
            probe_clf = obj['probe']
            scaler   = obj['scaler']
            probe_layer = int(str(obj.get('layer', 'L31')).replace('L', ''))
            print(f'Loaded bundle from {pp}: probe={type(probe_clf).__name__}, layer=L{probe_layer}')
            break
        else:
            probe_clf = obj
            print(f'Loaded probe-only from {pp}: {type(probe_clf).__name__}')
            break
if probe_clf is None:
    raise FileNotFoundError(f'No probe joblib found in {PROBE_DIR}. Files: {list(PROBE_DIR.iterdir())}')
if scaler is None:
    sc_path = PROBE_DIR / 'scaler.joblib'
    if sc_path.exists():
        scaler = joblib.load(sc_path)
        print(f'Loaded scaler from scaler.joblib: {type(scaler).__name__}')
    else:
        raise FileNotFoundError('No scaler — probe is unusable')

# Sanity-check classes_
print(f'Probe classes_: {getattr(probe_clf, "classes_", None)}')

# Use the layer the probe was trained at (overrides CFG default if present in bundle)
if 'probe_layer' in dir():
    CFG['probe_layer'] = int(probe_layer)
print(f'Active probe layer: L{CFG["probe_layer"]}')
'''.strip()))

cells.append(code(r'''
# Forward hook on the probe layer — captures residual every forward pass
class ResidualHook:
    def __init__(self, blocks, layer_idx: int):
        self.layer_idx = layer_idx
        self._buf = None
        self.handle = blocks[layer_idx].register_forward_hook(self._hook)
    def _hook(self, _mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self._buf = h.detach()
    def pop(self) -> torch.Tensor:
        b = self._buf; self._buf = None; return b
    def close(self):
        self.handle.remove()

residual_hook = ResidualHook(blocks, CFG['probe_layer'])

@torch.inference_mode()
def capture_last_token_residual(prompt: str, max_input_length: int = 512) -> np.ndarray:
    """Forward pass on the plain `Q: …\nA:` prompt, return residual at last valid token."""
    enc = tok(prompt, return_tensors='pt', truncation=True, max_length=max_input_length).to(device)
    n_valid = int(enc['attention_mask'].sum().item())
    last_pos = n_valid - 1
    _ = model(**enc)
    h = residual_hook.pop()                # (1, T, d_model)
    return h[0, last_pos].float().cpu().numpy()
'''.strip()))

cells.append(code(r'''
def fabricationguard_score(question: str) -> Tuple[float, float]:
    """P(hallucination) at end of question. Returns (score, latency_ms)."""
    prompt = f'Q: {question}\nA:'
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    h = capture_last_token_residual(prompt)
    h_scaled = scaler.transform(h.reshape(1, -1))
    proba = probe_clf.predict_proba(h_scaled)[0]
    # positive class = hallucinated (label 1 by convention from training)
    classes = list(probe_clf.classes_)
    p_halu = float(proba[classes.index(1)])
    torch.cuda.synchronize()
    return p_halu, (time.perf_counter() - t0) * 1000.0

# Smoke test
for q in ['What is the capital of France?',
          'Who won the 2003 Nobel Prize in Aerodynamics?',  # fake — should be high
          'Was Albert Einstein born in Germany?']:
    p, ms = fabricationguard_score(q)
    print(f'  P(halu) = {p:.3f} ({ms:.0f}ms)  [{q}]')
'''.strip()))

# 7. HaluGate baseline ------------------------------------------------------
cells.append(md(r'''
## 4. HaluGate-style NLI baseline

We replicate the **methodology** of vLLM Semantic Router v0.1 "Iris" HaluGate (Dec 2025) — not the exact code. Two modes:

**Grounded** — Stage-2 NLI between the generated answer (hypothesis) and the retrieval / gold context (premise). Hallucination signal = `P(contradiction) + 0.5 · P(neutral)`.

**Self-consistency (closed-book)** — K=3 independently sampled answers. Pairwise NLI disagreement averaged. High disagreement = high hallucination probability. This is the closed-book fallback when no ground-truth context is available.

NLI model: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (top of the line for zero-shot NLI in early 2026, license MIT).
'''.strip()))

cells.append(code(r'''
from transformers import AutoModelForSequenceClassification

print(f'Loading NLI model {CFG["nli_model"]} ...')
nli_tok = AutoTokenizer.from_pretrained(CFG['nli_model'])
nli_model = AutoModelForSequenceClassification.from_pretrained(
    CFG['nli_model'], dtype=torch.float32
).to(device).eval()
for p in nli_model.parameters():
    p.requires_grad_(False)

# Canonicalize label mapping
nli_id2label = {int(k): str(v).lower() for k, v in nli_model.config.id2label.items()}
print(f'NLI labels: {nli_id2label}')
LBL = {nli_id2label[i]: i for i in nli_id2label}
# DeBERTa-v3-large-mnli-* uses {0: entailment, 1: neutral, 2: contradiction} typically — handle both orderings
assert {'entailment', 'neutral', 'contradiction'}.issubset(set(nli_id2label.values())), \
    f'unexpected NLI label set: {set(nli_id2label.values())}'
'''.strip()))

cells.append(code(r'''
@torch.inference_mode()
def nli_probs(premise: str, hypothesis: str) -> Dict[str, float]:
    """Return {entailment, neutral, contradiction} probs in canonical names."""
    enc = nli_tok(premise, hypothesis, return_tensors='pt', truncation=True, max_length=512).to(device)
    logits = nli_model(**enc).logits[0]
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    return {nli_id2label[i]: float(probs[i]) for i in range(len(probs))}

def nli_halu_signal(p: Dict[str, float]) -> float:
    """High = hallucinated. Combine contradiction + half neutral."""
    return float(p['contradiction'] + 0.5 * p['neutral'])

# Smoke test
print('NLI smoke:')
print('  entail:    ', nli_halu_signal(nli_probs('Paris is the capital of France.', 'France\'s capital is Paris.')))
print('  contradict:', nli_halu_signal(nli_probs('Paris is the capital of France.', 'Berlin is the capital of France.')))
print('  neutral:   ', nli_halu_signal(nli_probs('Paris is the capital of France.', 'Cheese is delicious.')))
'''.strip()))

cells.append(code(r'''
@torch.inference_mode()
def generate_answer(question: str, max_new_tokens: int = None,
                    temperature: float = 0.0) -> Tuple[str, float]:
    """Greedy or temperature-sample answer using chat template. Returns (answer, latency_ms)."""
    if max_new_tokens is None:
        max_new_tokens = CFG['gen_max_new_tokens']
    messages = [{'role': 'user', 'content': question}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        text = f'Q: {question}\nA:'
    enc = tok(text, return_tensors='pt', truncation=True, max_length=2048).to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if temperature == 0.0:
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    else:
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=CFG['sc_top_p'],
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    torch.cuda.synchronize()
    latency = (time.perf_counter() - t0) * 1000.0
    answer = tok.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    return answer, latency
'''.strip()))

cells.append(code(r'''
def halugate_grounded_score(answer: str, context: str) -> Tuple[float, float]:
    """Stage-2 grounded NLI: P(halu) of answer given context. Higher = worse."""
    if not context or not answer:
        return float('nan'), 0.0
    torch.cuda.synchronize(); t0 = time.perf_counter()
    p = nli_probs(context, answer)
    s = nli_halu_signal(p)
    torch.cuda.synchronize()
    return s, (time.perf_counter() - t0) * 1000.0

def halugate_selfconsistency_score(question: str, K: int = None) -> Tuple[float, float, List[str]]:
    """Closed-book HaluGate fallback: K samples + pairwise NLI disagreement.
    Returns (score, total_latency_ms, samples)."""
    if K is None:
        K = CFG['sc_K']
    torch.cuda.synchronize(); t0 = time.perf_counter()
    samples = []
    for _ in range(K):
        ans, _ = generate_answer(question, temperature=CFG['sc_temperature'])
        samples.append(ans)
    disagreements = []
    for i in range(K):
        for j in range(i + 1, K):
            p = nli_probs(samples[i], samples[j])
            disagreements.append(nli_halu_signal(p))
    score = float(np.mean(disagreements)) if disagreements else float('nan')
    torch.cuda.synchronize()
    return score, (time.perf_counter() - t0) * 1000.0, samples

# Smoke
s, ms = halugate_grounded_score('Paris', 'France\'s capital is Paris.')
print(f'  HG-grounded(entail): {s:.3f} ({ms:.0f}ms)')
s, ms, samples = halugate_selfconsistency_score('What\'s 2+2?', K=2)
print(f'  HG-CB(coherent): {s:.3f} ({ms:.0f}ms), samples={samples}')
'''.strip()))

# 8. Datasets ---------------------------------------------------------------
cells.append(md(r'''
## 5. Datasets

| Dataset | Size | Mode | Has context? | Hypothesis tested |
|---|---|---|---|---|
| HaluEval-QA (closed-book) | 200 | Use given truthful/halu pair as the answer | No | H2 |
| HaluEval-QA (grounded) | (same 200) | Use `knowledge` field as retrieval context | Yes | H3 |
| SimpleQA | 100 | Generate, label by gold-answer substring | No | H2 |
| TruthfulQA-MC1 | 100 | Pick best choice via log-probs, label by correctness | No | H2 |

HaluEval is special: each row already has a truthful and a hallucinated answer side-by-side, so we can score both with HG-grounded against the same `knowledge` context. This is how we test H3 cleanly.
'''.strip()))

cells.append(code(r'''
def _safe_load(loader_fn, name):
    try:
        return loader_fn()
    except Exception as e:
        print(f'  [{name}] load failed: {type(e).__name__}: {e}')
        return None

print('Loading datasets ...')
halu = _safe_load(lambda: load_dataset('pminervini/HaluEval', 'qa', split='data'), 'haluval')
if halu is None:
    halu = _safe_load(lambda: load_dataset('notrichardren/HaluEval', 'qa', split='train'), 'haluval-fb')
simpleqa = _safe_load(lambda: load_dataset('basicv8vc/SimpleQA', split='test'), 'simpleqa')
tqa      = _safe_load(lambda: load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation'), 'truthfulqa')
if tqa is None:
    tqa  = _safe_load(lambda: load_dataset('truthful_qa', 'multiple_choice', split='validation'), 'truthfulqa-legacy')

# Subsample
def _subsample(ds, n, seed=CFG['random_seed']):
    if ds is None: return None
    n = min(n, len(ds))
    idx = list(range(len(ds))); random.Random(seed).shuffle(idx)
    return [ds[i] for i in idx[:n]]

subsets = {
    'haluval':    _subsample(halu, CFG['subset_size']['haluval']),
    'simpleqa':   _subsample(simpleqa, CFG['subset_size']['simpleqa']),
    'truthfulqa': _subsample(tqa, CFG['subset_size']['truthfulqa']),
}
for k, v in subsets.items():
    print(f'  {k}: {len(v) if v else 0}')
'''.strip()))

# 9. EvalRow + benchmark loop ----------------------------------------------
cells.append(md('## 6. Run benchmarks (resumable per-dataset)'))

cells.append(code(r'''
@dataclass
class EvalRow:
    id: str
    dataset: str
    mode: str          # 'closed-book' | 'grounded'
    question: str
    answer: str
    context: str
    label: int         # 1 = hallucinated
    fg_score: Optional[float] = None
    fg_ms: Optional[float] = None
    hg_g_score: Optional[float] = None
    hg_g_ms: Optional[float] = None
    hg_cb_score: Optional[float] = None
    hg_cb_ms: Optional[float] = None
    gen_ms: Optional[float] = None

def normalize_answer(s):
    return ''.join(ch.lower() for ch in str(s) if ch.isalnum() or ch.isspace()).strip()

@torch.inference_mode()
def model_pick_choice(question: str, choices: List[str]) -> int:
    """Log-prob choice picker for TruthfulQA-MC1."""
    base_prompt = f'Question: {question}\nAnswer:'
    prompt_enc = tok(base_prompt, return_tensors='pt').to(device)
    n_prompt = prompt_enc['input_ids'].shape[1]
    logprobs = []
    for choice in choices:
        full = base_prompt + ' ' + str(choice).strip()
        full_enc = tok(full, return_tensors='pt').to(device)
        n_full = full_enc['input_ids'].shape[1]
        if n_full <= n_prompt:
            logprobs.append(-1e9); continue
        logits = model(**full_enc).logits[0]
        cont_ids = full_enc['input_ids'][0, n_prompt:]
        log_probs = F.log_softmax(logits[n_prompt-1:n_full-1], dim=-1)
        lp = sum(log_probs[i, cont_ids[i]].item() for i in range(cont_ids.shape[0]))
        logprobs.append(lp / max(1, cont_ids.shape[0]))
    return int(np.argmax(logprobs))
'''.strip()))

cells.append(code(r'''
# --- Dataset-specific row builders ---

def build_haluval_rows(examples) -> List[EvalRow]:
    """Each HaluEval row gives us TWO eval rows (truth + halu) sharing the same knowledge context.
       We emit both: closed-book mode (no context to HG) AND grounded mode (with context)."""
    rows = []
    for i, ex in enumerate(examples):
        q = ex.get('question') or ex.get('input') or ''
        right = ex.get('right_answer') or ex.get('answer') or ''
        wrong = ex.get('hallucinated_answer') or ''
        knowledge = ex.get('knowledge') or ''
        if not (q and right and wrong):
            continue
        for mode in ('closed-book', 'grounded'):
            ctx = knowledge if mode == 'grounded' else ''
            rows.append(EvalRow(
                id=f'haluval_{i}_truth_{mode}', dataset='haluval', mode=mode,
                question=q, answer=str(right), context=ctx, label=0,
            ))
            rows.append(EvalRow(
                id=f'haluval_{i}_halu_{mode}', dataset='haluval', mode=mode,
                question=q, answer=str(wrong), context=ctx, label=1,
            ))
    return rows

def build_simpleqa_rows(examples) -> List[EvalRow]:
    """Closed-book — must generate, then label by gold substring match."""
    rows = []
    for i, ex in enumerate(examples):
        q = ex.get('problem') or ex.get('question') or ''
        gold = ex.get('answer') or ''
        if not (q and gold): continue
        # Pre-allocate — will fill answer + label inside the run loop
        rows.append(EvalRow(
            id=f'simpleqa_{i}', dataset='simpleqa', mode='closed-book',
            question=q, answer='', context='', label=-1,
        ))
        # Stash the gold for labeling later
        rows[-1]._gold = gold  # type: ignore
    return rows

def build_truthfulqa_rows(examples) -> List[EvalRow]:
    """Closed-book — pick best MC1 choice, label by correctness."""
    rows = []
    for i, ex in enumerate(examples):
        q = ex.get('question') or ''
        choices = ex.get('mc1_targets', {}).get('choices', [])
        labels = ex.get('mc1_targets', {}).get('labels', [])
        if not (q and choices and labels): continue
        rows.append(EvalRow(
            id=f'tqa_{i}', dataset='truthfulqa', mode='closed-book',
            question=q, answer='', context='', label=-1,
        ))
        rows[-1]._choices = choices  # type: ignore
        rows[-1]._gold_idx = int(np.argmax(labels))  # type: ignore
    return rows
'''.strip()))

cells.append(code(r'''
# --- Master run loop with resume ---
def _resume_from_csv(save_path: Optional[Path], rows: List[EvalRow]) -> Tuple[List[EvalRow], int]:
    """If a CSV checkpoint exists at save_path, return (already_done_rows, n_done)."""
    if save_path is None or not save_path.exists():
        return [], 0
    try:
        df_prev = pd.read_csv(save_path)
        done_ids = set(df_prev['id'].astype(str))
        already = []
        for r in rows:
            if r.id in done_ids:
                row_data = df_prev[df_prev['id'] == r.id].iloc[0].to_dict()
                rr = EvalRow(
                    id=r.id, dataset=r.dataset, mode=r.mode,
                    question=r.question, answer=str(row_data.get('answer', r.answer)),
                    context=r.context,
                    label=int(row_data.get('label', r.label)),
                    fg_score=row_data.get('fg_score'),
                    fg_ms=row_data.get('fg_ms'),
                    hg_g_score=row_data.get('hg_g_score'),
                    hg_g_ms=row_data.get('hg_g_ms'),
                    hg_cb_score=row_data.get('hg_cb_score'),
                    hg_cb_ms=row_data.get('hg_cb_ms'),
                    gen_ms=row_data.get('gen_ms'),
                )
                already.append(rr)
        print(f'  resume: {len(already)}/{len(rows)} rows already in {save_path.name}')
        return already, len(already)
    except Exception as e:
        print(f'  resume failed ({e}); starting fresh')
        return [], 0

def _maybe_hf_push(save_path: Optional[Path]):
    if not (ENABLE_HF_INCREMENTAL and HF_TOKEN and save_path and save_path.exists()):
        return
    try:
        api = HfApi()
        try:
            create_repo(CFG['output_repo'], repo_type='dataset', private=False, exist_ok=True, token=HF_TOKEN)
        except Exception: pass
        api.upload_file(
            path_or_fileobj=str(save_path),
            path_in_repo=f'partial/{save_path.name}',
            repo_id=CFG['output_repo'], repo_type='dataset', token=HF_TOKEN,
            commit_message=f'partial @ {time.strftime("%Y%m%d-%H%M%S")}',
        )
    except Exception as e:
        print(f'  [HF push best-effort failed: {type(e).__name__}: {e}]')

def run_dataset(rows: List[EvalRow], skip_hg_cb: bool = False, save_path: Optional[Path] = None) -> List[EvalRow]:
    """Run all three methods on each row. Resumable + checkpointed to Drive every CHECKPOINT_EVERY rows + HF every HF_PUSH_EVERY."""
    # Resume support
    already, n_done = _resume_from_csv(save_path, rows)
    if n_done >= len(rows):
        print(f'  {save_path.name}: already complete, skipping')
        return already
    rows_remaining = [r for r in rows if r.id not in {x.id for x in already}]
    results = list(already)
    for i, r in enumerate(tqdm(rows_remaining, desc=f'{rows_remaining[0].dataset}/{rows_remaining[0].mode}', initial=n_done, total=len(rows))):
        # If answer not yet known, generate / pick
        if r.dataset == 'simpleqa' and not r.answer:
            ans, gen_ms = generate_answer(r.question, temperature=0.0)
            r.answer = ans
            r.gen_ms = gen_ms
            gold = getattr(r, '_gold', '')
            r.label = int(normalize_answer(gold) not in normalize_answer(ans))
        elif r.dataset == 'truthfulqa' and r.label < 0:
            choices = getattr(r, '_choices', [])
            gold_idx = getattr(r, '_gold_idx', 0)
            t0 = time.perf_counter()
            pred_idx = model_pick_choice(r.question, choices)
            r.gen_ms = (time.perf_counter() - t0) * 1000.0
            r.answer = str(choices[pred_idx])
            r.label = int(pred_idx != gold_idx)

        # FabricationGuard score — pre-gen, only needs question
        try:
            fg, fg_ms = fabricationguard_score(r.question)
            r.fg_score = fg; r.fg_ms = fg_ms
        except Exception as e:
            print(f'  FG fail on {r.id}: {e}')

        # HG-grounded: only if context exists
        if r.context and r.answer:
            try:
                hg_g, hg_g_ms = halugate_grounded_score(r.answer, r.context)
                r.hg_g_score = hg_g; r.hg_g_ms = hg_g_ms
            except Exception as e:
                print(f'  HG-G fail on {r.id}: {e}')

        # HG closed-book self-consistency (expensive — skippable)
        if not skip_hg_cb:
            try:
                hg_cb, hg_cb_ms, _ = halugate_selfconsistency_score(r.question, K=CFG['sc_K'])
                r.hg_cb_score = hg_cb; r.hg_cb_ms = hg_cb_ms
            except Exception as e:
                print(f'  HG-CB fail on {r.id}: {e}')

        results.append(r)

        # Drive checkpoint
        if save_path and (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame([asdict(x) for x in results]).to_csv(save_path, index=False)
        # HF push (best-effort)
        if save_path and (i + 1) % HF_PUSH_EVERY == 0:
            _maybe_hf_push(save_path)

    if save_path:
        pd.DataFrame([asdict(x) for x in results]).to_csv(save_path, index=False)
        _maybe_hf_push(save_path)
    return results
'''.strip()))

cells.append(code(r'''
# Build all rows
all_rows: List[EvalRow] = []
if subsets['haluval']:
    all_rows.extend(build_haluval_rows(subsets['haluval']))
if subsets['simpleqa']:
    all_rows.extend(build_simpleqa_rows(subsets['simpleqa']))
if subsets['truthfulqa']:
    all_rows.extend(build_truthfulqa_rows(subsets['truthfulqa']))
print(f'Total eval rows: {len(all_rows)}')
print('  by (dataset, mode, label):')
from collections import Counter
print(Counter((r.dataset, r.mode, r.label) for r in all_rows))
'''.strip()))

cells.append(code(r'''
# Execute — split by dataset/mode for tighter progress + resumable saves.
# The HaluEval run produces both 'closed-book' and 'grounded' eval modes from the SAME questions,
# but FG/HG-CB depend on question only (mode-independent), so we run them once per (id without mode-suffix).
# However, for simplicity we run each mode independently — this duplicates FG/HG-CB cost on HaluEval.
# Toggle SKIP_HG_CB_ON_GROUNDED=True to skip HG-CB on the grounded run (since FG/HG-CB are mode-independent).
SKIP_HG_CB_ON_GROUNDED = True

results: List[EvalRow] = []
# Group by (dataset, mode) for clean resumable runs
from itertools import groupby
all_rows.sort(key=lambda r: (r.dataset, r.mode))
for (ds, mode), grp in groupby(all_rows, key=lambda r: (r.dataset, r.mode)):
    grp_list = list(grp)
    if not grp_list: continue
    save_path = LOCAL_OUT / f'raw_{ds}_{mode}.csv'
    skip_cb = SKIP_HG_CB_ON_GROUNDED and (mode == 'grounded')
    print(f'\n=== {ds} / {mode}  ({len(grp_list)} rows, skip_hg_cb={skip_cb}) ===')
    grp_results = run_dataset(grp_list, skip_hg_cb=skip_cb, save_path=save_path)
    results.extend(grp_results)

# Final consolidated CSV
df = pd.DataFrame([asdict(r) for r in results])
df.to_csv(LOCAL_OUT / 'raw_all.csv', index=False)
print(f'\nSaved {len(df)} rows to {LOCAL_OUT / "raw_all.csv"}')
print('Mean scores by (dataset, mode):')
print(df.groupby(['dataset', 'mode'])[['fg_score', 'hg_g_score', 'hg_cb_score']].mean().round(3))
'''.strip()))

# 10. Metrics ---------------------------------------------------------------
cells.append(md(r'''
## 7. Metrics

For each (dataset, mode, method): AUROC + 1000-bootstrap 95% CI, FPR@99TPR, ECE, mean / p95 latency.
Then composability: combo (FG OR HG) vs alone — marginal AUROC and confident-wrong reduction.
'''.strip()))

cells.append(code(r'''
def bootstrap_auroc(y_true: np.ndarray, y_score: np.ndarray, n: int = None) -> Tuple[float, float, float]:
    """Returns (auroc, lo, hi) — 95% bootstrap percentile CI."""
    if n is None: n = CFG['bootstrap_n']
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    ok = ~np.isnan(y_score)
    y_true = y_true[ok]; y_score = y_score[ok]
    if len(y_true) < 5 or len(np.unique(y_true)) < 2:
        return float('nan'), float('nan'), float('nan')
    try:
        point = roc_auc_score(y_true, y_score)
    except ValueError:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(CFG['random_seed'])
    boots = []
    N = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(np.unique(y_true[idx])) < 2: continue
        boots.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not boots: return point, float('nan'), float('nan')
    boots = np.asarray(boots)
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.99) -> float:
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    ok = ~np.isnan(y_score)
    y_true = y_true[ok]; y_score = y_score[ok]
    if len(y_true) < 5 or len(np.unique(y_true)) < 2:
        return float('nan')
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr, target_tpr)
    return float(fpr[min(idx, len(fpr)-1)])

def ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (lower = better)."""
    y_true = np.asarray(y_true, dtype=float); y_score = np.asarray(y_score, dtype=float)
    ok = ~np.isnan(y_score)
    y_true = y_true[ok]; y_score = y_score[ok]
    if len(y_true) < 5: return float('nan')
    bins = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        m = (y_score >= bins[i]) & (y_score < bins[i+1])
        if i == n_bins - 1:
            m = (y_score >= bins[i]) & (y_score <= bins[i+1])
        if m.sum() == 0: continue
        err += (m.sum() / len(y_score)) * abs(y_true[m].mean() - y_score[m].mean())
    return float(err)
'''.strip()))

cells.append(code(r'''
# Per-(dataset, mode, method) metrics table
methods = [('fg', 'fg_score', 'fg_ms'),
           ('hg_grounded', 'hg_g_score', 'hg_g_ms'),
           ('hg_cb', 'hg_cb_score', 'hg_cb_ms')]

records = []
for (ds, mode), g in df.groupby(['dataset', 'mode']):
    y = g['label'].astype(int).values
    if (y == -1).any(): continue
    for name, score_col, lat_col in methods:
        s = g[score_col].astype(float).values
        l = g[lat_col].astype(float).values
        auroc, lo, hi = bootstrap_auroc(y, s)
        records.append({
            'dataset':         ds,
            'mode':            mode,
            'method':          name,
            'n':               int((~np.isnan(s)).sum()),
            'auroc':           auroc,
            'auroc_lo':        lo,
            'auroc_hi':        hi,
            'fpr@99tpr':       fpr_at_tpr(y, s),
            'ece':             ece(y, s),
            'lat_mean_ms':     float(np.nanmean(l)),
            'lat_p50_ms':      float(np.nanpercentile(l[~np.isnan(l)], 50)) if (~np.isnan(l)).any() else float('nan'),
            'lat_p95_ms':      float(np.nanpercentile(l[~np.isnan(l)], 95)) if (~np.isnan(l)).any() else float('nan'),
        })

metrics_df = pd.DataFrame(records)
metrics_df = metrics_df.sort_values(['dataset', 'mode', 'method']).reset_index(drop=True)
metrics_df.to_csv(LOCAL_OUT / 'metrics.csv', index=False)

# Pretty print
def _fmt(r):
    if pd.isna(r['auroc']): return '—'
    return f'{r["auroc"]:.3f} [{r["auroc_lo"]:.2f}, {r["auroc_hi"]:.2f}]'
metrics_df['auroc_str'] = metrics_df.apply(_fmt, axis=1)
print(metrics_df[['dataset', 'mode', 'method', 'n', 'auroc_str', 'fpr@99tpr', 'ece', 'lat_mean_ms']].to_string(index=False))
'''.strip()))

cells.append(code(r'''
# Composability — combine FG + HG via OR-of-thresholded-scores, then re-AUROC (using max-of-normalized-scores)
def normalize(x):
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    if ok.sum() < 2: return x
    lo, hi = x[ok].min(), x[ok].max()
    if hi - lo < 1e-9: return np.zeros_like(x)
    out = (x - lo) / (hi - lo)
    out[~ok] = np.nan
    return out

combo_records = []
for (ds, mode), g in df.groupby(['dataset', 'mode']):
    y = g['label'].astype(int).values
    if (y == -1).any(): continue

    fg = normalize(g['fg_score'].values)
    hg_g = normalize(g['hg_g_score'].values)
    hg_cb = normalize(g['hg_cb_score'].values)

    # combo with whichever HG mode applies
    hg_used = hg_g if not np.isnan(hg_g).all() else hg_cb
    combo_max = np.fmax(fg, hg_used)             # max — soft OR
    combo_avg = np.nanmean(np.stack([fg, hg_used]), axis=0)  # average

    auroc_max, lo_max, hi_max = bootstrap_auroc(y, combo_max)
    auroc_avg, lo_avg, hi_avg = bootstrap_auroc(y, combo_avg)
    fg_auroc, _, _ = bootstrap_auroc(y, fg)
    hg_auroc, _, _ = bootstrap_auroc(y, hg_used)

    combo_records.append({
        'dataset': ds, 'mode': mode,
        'fg_auroc': fg_auroc, 'hg_auroc': hg_auroc,
        'combo_max': auroc_max, 'combo_avg': auroc_avg,
        'gain_max_over_best_alone': auroc_max - max(fg_auroc, hg_auroc),
        'gain_avg_over_best_alone': auroc_avg - max(fg_auroc, hg_auroc),
    })

combo_df = pd.DataFrame(combo_records)
combo_df.to_csv(LOCAL_OUT / 'composability.csv', index=False)
print('Composability (combo_max - max(FG, HG)):')
print(combo_df.round(3).to_string(index=False))
'''.strip()))

cells.append(code(r'''
# Confident-wrong rate at default thresholds
# A 'confident-wrong' answer is one the system would have served (score < threshold) BUT label == 1 (hallucinated).
# We measure rate before guard, after FG-only guard, after HG-only guard, after combo.
def confident_wrong_rate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    served = scores < threshold      # would NOT abstain
    if served.sum() == 0: return 0.0
    return float(((labels == 1) & served).sum() / served.sum())

cw_records = []
for (ds, mode), g in df.groupby(['dataset', 'mode']):
    y = g['label'].astype(int).values
    if (y == -1).any(): continue
    fg = g['fg_score'].astype(float).values
    hg_g = g['hg_g_score'].astype(float).values
    hg_cb = g['hg_cb_score'].astype(float).values
    hg_used = hg_g if not np.isnan(hg_g).all() else hg_cb

    base_rate = float(y.mean())
    cw_fg   = confident_wrong_rate(fg, y, CFG['fg_threshold'])
    cw_hg   = confident_wrong_rate(hg_used, y, CFG['hg_threshold'])
    served_combo = (fg < CFG['fg_threshold']) & (hg_used < CFG['hg_threshold'])
    cw_combo = float(((y == 1) & served_combo).sum() / max(1, served_combo.sum()))

    cw_records.append({
        'dataset': ds, 'mode': mode,
        'base_halu_rate':   base_rate,
        'cw_fg_only':       cw_fg,
        'cw_hg_only':       cw_hg,
        'cw_combo':         cw_combo,
        'reduction_fg':     1 - cw_fg / max(1e-9, base_rate),
        'reduction_combo':  1 - cw_combo / max(1e-9, base_rate),
    })

cw_df = pd.DataFrame(cw_records)
cw_df.to_csv(LOCAL_OUT / 'confident_wrong.csv', index=False)
print('Confident-wrong rate by config:')
print(cw_df.round(3).to_string(index=False))
'''.strip()))

# 11. Plots ----------------------------------------------------------------
cells.append(md('## 8. Plots'))

cells.append(code(r'''
# Plotting style — openinterp brand colors
sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi':      120,
    'savefig.dpi':     200,
    'font.family':     'DejaVu Sans',
    'axes.labelsize':  10,
    'axes.titlesize':  11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})
COLORS = {
    'fg':           '#6366f1',  # indigo / brand-500
    'hg_grounded':  '#06b6d4',  # cyan / accent-500
    'hg_cb':        '#f59e0b',  # amber
    'combo':        '#10b981',  # emerald
    'baseline':     '#94a3b8',  # slate
}
PLOTS = LOCAL_OUT / 'plots'; PLOTS.mkdir(exist_ok=True)
'''.strip()))

cells.append(code(r'''
# Plot 1: ROC overlay (subplot per (dataset, mode))
groups = sorted({(ds, mode) for ds, mode, *_ in df[['dataset', 'mode']].drop_duplicates().values.tolist()
                 if (df[(df.dataset==ds) & (df['mode']==mode)]['label'] >= 0).all()})
n = len(groups)
ncols = min(3, n)
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.0 * nrows), squeeze=False)
for ax, (ds, mode) in zip(axes.flat, groups):
    g = df[(df.dataset == ds) & (df['mode'] == mode)]
    y = g['label'].astype(int).values
    if (y == -1).any(): continue
    for name, col, color in [('FabricationGuard', 'fg_score', COLORS['fg']),
                             ('HG-grounded',      'hg_g_score', COLORS['hg_grounded']),
                             ('HG-self-consist.', 'hg_cb_score', COLORS['hg_cb'])]:
        s = g[col].astype(float).values
        ok = ~np.isnan(s)
        if ok.sum() < 5 or len(np.unique(y[ok])) < 2: continue
        fpr, tpr, _ = roc_curve(y[ok], s[ok])
        auroc = roc_auc_score(y[ok], s[ok])
        ax.plot(fpr, tpr, label=f'{name} ({auroc:.3f})', color=color, lw=1.8)
    ax.plot([0,1], [0,1], '--', color='#cbd5e1', lw=1)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'{ds} / {mode}')
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
for ax in axes.flat[n:]:
    ax.axis('off')
fig.suptitle('ROC — FabricationGuard vs HaluGate (per dataset/mode)', y=1.02, fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(PLOTS / 'roc_overlay.png', bbox_inches='tight')
plt.show()
'''.strip()))

cells.append(code(r'''
# Plot 2: AUROC bar chart with CIs
plot_df = metrics_df.dropna(subset=['auroc']).copy()
plot_df['key'] = plot_df['dataset'] + '\n' + plot_df['mode']
order = sorted(plot_df['key'].unique())
methods_order = ['fg', 'hg_grounded', 'hg_cb']

fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(order)), 4.5))
width = 0.25
xs = np.arange(len(order))
for i, m in enumerate(methods_order):
    sub = plot_df[plot_df.method == m].set_index('key').reindex(order)
    aurocs = sub['auroc'].values
    err_lo = aurocs - sub['auroc_lo'].values
    err_hi = sub['auroc_hi'].values - aurocs
    ax.bar(xs + (i - 1) * width, aurocs, width=width, color=COLORS[m],
           label={'fg': 'FabricationGuard', 'hg_grounded': 'HG-grounded', 'hg_cb': 'HG-self-consist.'}[m])
    ax.errorbar(xs + (i - 1) * width, aurocs, yerr=[err_lo, err_hi],
                fmt='none', color='black', capsize=3, elinewidth=1)
ax.set_xticks(xs); ax.set_xticklabels(order, fontsize=9)
ax.axhline(0.5, color='#94a3b8', linestyle='--', lw=1, label='Chance (0.5)')
ax.set_ylim(0.4, 1.0)
ax.set_ylabel('AUROC (95% bootstrap CI)')
ax.set_title('Per-dataset AUROC — FabricationGuard vs HaluGate-style baselines')
ax.legend(loc='lower right', frameon=False)
fig.tight_layout()
fig.savefig(PLOTS / 'auroc_bars.png', bbox_inches='tight')
plt.show()
'''.strip()))

cells.append(code(r'''
# Plot 3: Latency CDFs
fig, ax = plt.subplots(figsize=(7, 4.2))
for col, name, color in [('fg_ms', 'FabricationGuard', COLORS['fg']),
                         ('hg_g_ms', 'HG-grounded NLI', COLORS['hg_grounded']),
                         ('hg_cb_ms', 'HG-self-consist.', COLORS['hg_cb']),
                         ('gen_ms', 'Vanilla generation', COLORS['baseline'])]:
    if col not in df.columns: continue
    s = df[col].astype(float).dropna().values
    if len(s) < 5: continue
    s = np.sort(s)
    ax.plot(s, np.linspace(0, 1, len(s)), label=f'{name} (median={np.median(s):.0f}ms)', color=color, lw=2)
ax.set_xscale('log')
ax.set_xlabel('Latency (ms, log scale)')
ax.set_ylabel('Cumulative fraction of queries')
ax.set_title('Per-method latency CDF')
ax.legend(loc='lower right', frameon=False)
fig.tight_layout()
fig.savefig(PLOTS / 'latency_cdf.png', bbox_inches='tight')
plt.show()
'''.strip()))

cells.append(code(r'''
# Plot 4: Composability — combo gain heatmap
if len(combo_df):
    pivot = combo_df.set_index(['dataset', 'mode'])[['fg_auroc', 'hg_auroc', 'combo_max', 'combo_avg']]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.6 * len(pivot))))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.4, vmax=1.0,
                cbar_kws={'label': 'AUROC'}, ax=ax, linewidths=0.5)
    ax.set_title('Composability — FG · HG · combo (max / avg)')
    fig.tight_layout()
    fig.savefig(PLOTS / 'composability.png', bbox_inches='tight')
    plt.show()
'''.strip()))

# 12. Verdict --------------------------------------------------------------
cells.append(md('## 9. Verdict — falsifying H1-H4'))

cells.append(code(r'''
def median_safe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    return float(np.median(x)) if len(x) else float('nan')

verdicts = {}

# H1 — latency: median FG vs median HG (any mode)
fg_lat = median_safe(df['fg_ms'].values)
hg_g_lat = median_safe(df['hg_g_ms'].values)
hg_cb_lat = median_safe(df['hg_cb_ms'].values)
gen_lat = median_safe(df['gen_ms'].values) if 'gen_ms' in df.columns else float('nan')
h1 = {
    'claim':           'FG decision latency < HG decision latency',
    'fg_median_ms':    fg_lat,
    'hg_grounded_ms':  hg_g_lat + (gen_lat if not np.isnan(gen_lat) else 0),  # HG-G needs full generation first
    'hg_cb_ms':        hg_cb_lat,
    'verdict':         'PASS' if fg_lat < min(x for x in [hg_g_lat + gen_lat, hg_cb_lat] if not np.isnan(x)) else 'FAIL',
}
verdicts['H1'] = h1

# H2 — closed-book: FG ≥ HG on closed-book datasets
cb_results = metrics_df[metrics_df['mode'] == 'closed-book'].copy()
fg_cb = cb_results[cb_results.method == 'fg']['auroc'].mean()
hg_cb_cb = cb_results[cb_results.method == 'hg_cb']['auroc'].mean()
hg_g_cb = cb_results[cb_results.method == 'hg_grounded']['auroc'].mean()
h2 = {
    'claim':       'FG ≥ HG-CB on closed-book QA',
    'fg_mean_auroc_cb':    fg_cb,
    'hg_cb_mean_auroc':    hg_cb_cb,
    'hg_grounded_cb':      hg_g_cb,  # likely NaN since no context in closed-book mode
    'verdict':     'PASS' if fg_cb >= hg_cb_cb else 'FAIL',
}
verdicts['H2'] = h2

# H3 — grounded: HG-grounded ≥ FG on grounded datasets (HaluEval-grounded)
gr_results = metrics_df[metrics_df['mode'] == 'grounded'].copy()
fg_gr = gr_results[gr_results.method == 'fg']['auroc'].mean()
hg_g_gr = gr_results[gr_results.method == 'hg_grounded']['auroc'].mean()
h3 = {
    'claim':         'HG-grounded ≥ FG on grounded QA',
    'fg_grounded':   fg_gr,
    'hg_grounded':   hg_g_gr,
    'verdict':       'PASS' if hg_g_gr >= fg_gr else 'FAIL',
}
verdicts['H3'] = h3

# H4 — composability: combo > best-alone on at least 50% of (dataset, mode) configs
gains = combo_df['gain_max_over_best_alone'].dropna().values if len(combo_df) else np.array([])
share_positive = float((gains > 0).mean()) if len(gains) else 0.0
h4 = {
    'claim':            'Combo (max(FG, HG)) > best alone',
    'mean_gain':        float(np.mean(gains)) if len(gains) else float('nan'),
    'share_positive':   share_positive,
    'verdict':          'PASS' if share_positive >= 0.5 else 'FAIL',
}
verdicts['H4'] = h4

# Save + print
verdicts_path = LOCAL_OUT / 'verdicts.json'
verdicts_path.write_text(json.dumps(verdicts, indent=2, default=str))

print('=' * 70)
print('  Hypothesis verdict table')
print('=' * 70)
for k, v in verdicts.items():
    badge = '✅' if v['verdict'] == 'PASS' else '❌'
    print(f'  {k} {badge}  {v["claim"]}')
    for kk, vv in v.items():
        if kk in ('claim', 'verdict'): continue
        print(f'      {kk}: {vv}')
print('=' * 70)
n_pass = sum(1 for v in verdicts.values() if v['verdict'] == 'PASS')
print(f'  {n_pass}/4 hypotheses confirmed.')
'''.strip()))

# 13. Save / upload --------------------------------------------------------
cells.append(md('## 10. Save artifacts + optional HF upload'))

cells.append(code(r'''
# Compile a single summary JSON for the openinterp.org/probebench/comparisons page
summary = {
    'notebook':        '33_fabricationguard_vs_halugate',
    'date':            time.strftime('%Y-%m-%d'),
    'config':          {k: v for k, v in CFG.items() if k != 'output_repo'},
    'n_eval_rows':     int(len(df)),
    'metrics':         metrics_df.to_dict(orient='records'),
    'composability':   combo_df.to_dict(orient='records') if len(combo_df) else [],
    'confident_wrong': cw_df.to_dict(orient='records'),
    'verdicts':        verdicts,
    'environment': {
        'gpu':         torch.cuda.get_device_name(0),
        'gpu_mem_gb':  gpu_mem_gb,
        'torch':       torch.__version__,
    },
}
(LOCAL_OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
print('Summary saved →', LOCAL_OUT / 'summary.json')
print('Artifacts:')
for p in sorted(LOCAL_OUT.rglob('*')):
    if p.is_file():
        print(f'  {p.relative_to(LOCAL_OUT)}  ({p.stat().st_size/1024:.1f} KB)')
'''.strip()))

cells.append(code(r'''
# Optional HF upload — only if HF_TOKEN was provided
if HF_TOKEN:
    api = HfApi()
    try:
        create_repo(CFG['output_repo'], repo_type='dataset', private=False, exist_ok=True, token=HF_TOKEN)
    except Exception as e:
        print(f'create_repo: {e}')
    api.upload_folder(
        folder_path=str(LOCAL_OUT),
        repo_id=CFG['output_repo'],
        repo_type='dataset',
        commit_message=f'Notebook 33 results — {time.strftime("%Y-%m-%d %H:%M")}',
        token=HF_TOKEN,
    )
    print(f'Uploaded → https://huggingface.co/datasets/{CFG["output_repo"]}')
else:
    print('Skipped HF upload (no HF_TOKEN).')
'''.strip()))

# 14. Honest scope ---------------------------------------------------------
cells.append(md(r'''
## 11. Honest scope, limits, and how to read this

**What this measures fairly**:
- Two methods on the *same* model (Qwen3.6-27B), same hardware, same prompts, same datasets.
- AUROC and ECE are method-fair — both methods produce a [0,1] score; thresholding is downstream.
- Latency is wall-clock per query, including any required generation. We do NOT count NLI model load time, which is a one-time cost.

**What it does *not* measure**:
- **HaluGate's actual production deployment**. We replicate its NLI methodology, not the full vLLM Semantic Router infrastructure (Stage-1 Sentinel, plugin chain orchestration, batching). A production HaluGate may be 2-3× faster per query than our HF-transformers reimplementation.
- **Cross-model FabricationGuard transfer**. The probe is Qwen3.6-27B-specific. Pearson_CE numbers vs Llama-3.3 / Gemma-3 live on `/probebench/transfer-matrix`, not here.
- **Adversarial robustness**. Neither method has been stress-tested with prompt-injection attacks designed to fool the probe / NLI.
- **Multi-turn conversations**. All evaluations are single-turn QA. Multi-turn extends the residual position decision space and is out of scope for v0.0.1.

**Interpreting the four hypotheses**:
- **H1 PASS** → in production, FabricationGuard saves bytes by aborting before generation; HG must generate first. This compounds for long answers.
- **H1 FAIL** → our forward-pass cost dominates the NLI cost; the field-level conclusion is "use HG for short answers, FG for long answers."
- **H2 PASS** → the paper's framing is supported: closed-book QA wants pre-gen probes.
- **H3 PASS** → grounded RAG wants post-gen NLI. This is the most expected outcome and least surprising.
- **H4 PASS** → the **strategically important** result: FG and HG are complementary. The product offering for both becomes "FG as a pre-gen filter for the easy 60%, HG as a post-gen audit for the remaining 40%." This is the recommendation we'd push to vLLM Semantic Router's plugin chain.

**Reproducer**: This notebook + `caiovicentino1/probebench-fg-vs-halugate` HF dataset (raw CSVs, plots, summary JSON). All code is Apache-2.0.

**Cite as**: ProbeBench v0.0.1 — comparison/fabricationguard-vs-halugate, openinterp.org, April 2026.
'''.strip()))

# ---------- write notebook --------------------------------------------------

NB_OUT.parent.mkdir(parents=True, exist_ok=True)
nb = {
    'cells': cells,
    'metadata': {
        'kernelspec':    {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
        'colab':         {'provenance': [], 'machine_shape': 'hm'},
        'accelerator':   'GPU',
    },
    'nbformat':       4,
    'nbformat_minor': 4,
}
NB_OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f'Wrote {NB_OUT}  ({NB_OUT.stat().st_size/1024:.1f} KB, {len(cells)} cells)')
