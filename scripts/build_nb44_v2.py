"""
Builder for nb44_v2_behavior_eval.ipynb

DECISIVE behavior eval at n=100/source (vs nb44 v1 pilot at n=25). Tighter
paired bootstrap CIs to resolve the +5.5pp directional signal from v1.

This is the gate experiment for paper-3 (GRPO) verdict and additional
validation for paper-2 (DPO behavior shift).

3 conditions paired per prompt:
- base (Qwen3.6-27B, no LoRA)
- nb37 v2 DPO (multi-probe FG+RG combined preference, 200 steps)
- nb43 GRPO (multi-probe continuous reward, 500 steps, U-shape end-state)

n=100 GSM8K test + n=100 SimpleQA test (different seed than nb37 training pool
+ different from nb44 v1 pilot pool to avoid contamination).

Decoding: temp=0.7, paired (same generation seed per prompt across conditions).

LoRA loading: safe_load_qwen36_lora() from openinterp v0.2.1+ — auto-strip
.language_model. infix + auto-verify logit-diff > 0.01.

Judge: Claude Haiku via OpenRouter.

Metrics:
1. GSM8K correctness (gold numerical match)
2. SimpleQA factual rate (judge YES/NO/UNVERIFIABLE)
3. Combined correctness across both tasks
4. Paired bootstrap CI on each pairwise delta:
   - DPO vs base
   - GRPO vs base
   - DPO vs GRPO

Verdict per comparison:
- 🟢 ship: delta > 0 with CI excluding 0
- 🟡 directional: positive delta but CI crosses 0
- ⚪ null: |delta| within CI noise
- 🔴 regression: delta < 0 with CI excluding 0

Compute: ~5h on RTX 6000 Blackwell. Cost: ~$10 OpenRouter judge calls.

Drive: /content/drive/MyDrive/openinterp_runs/44_v2_behavior_eval/

Note on nb43 GRPO: final LoRA captures POST-PEAK drift (peak at idx 1084
of 1303 training trajectories ~77%). Behavior eval on final LoRA may
underestimate GRPO's peak performance. We use what's available.
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
        "# Notebook 44 v2 — Behavior Eval (n=100/source decisive)",
        "",
        "**The decisive question**: does multi-probe DPO or GRPO produce a Qwen3.6-27B that is *measurably better* than base on real tasks?",
        "",
        "v1 was a pilot (n=25/source, n=50 total). Found SimpleQA +15.4pp DPO directional but CI [−8, +36] crosses 0. **Underpowered**.",
        "",
        "v2 scales to n=100/source = 200 prompts × 3 conditions = 600 generations. Paired bootstrap CI ~±5-10pp, decisive for 15pp effects.",
        "",
        "## Setup",
        "",
        "**Conditions** (paired per prompt — same generation seed across conditions):",
        "- **base** — Qwen3.6-27B no LoRA",
        "- **DPO** — nb37 v2 multi-probe FG+RG combined preference, 200 steps",
        "- **GRPO** — nb43 multi-probe continuous reward, 500 steps (U-shape end-state)",
        "",
        "**Prompts**: 100 GSM8K test + 100 SimpleQA test (seed=44, different than v1 + nb37 training)",
        "",
        "**Decoding**: temp=0.7, paired across conditions",
        "",
        "**LoRA loading**: `openinterp.safe_load_qwen36_lora()` v0.2.1+ — auto-strip `.language_model.` infix + auto-verify",
        "",
        "**Judge**: Claude Haiku via OpenRouter",
        "",
        "## Metrics",
        "",
        "1. GSM8K correctness (gold numerical match)",
        "2. SimpleQA factual rate (judge YES rate)",
        "3. Combined correctness across both tasks",
        "4. Paired bootstrap CI on each pairwise delta",
        "",
        "## Verdict (per comparison)",
        "",
        "- 🟢 ship: delta > 0 with CI excluding 0",
        "- 🟡 directional: delta > 0 but CI crosses 0",
        "- ⚪ null: |delta| within CI noise",
        "- 🔴 regression: delta < 0 with CI excluding 0",
        "",
        "**Compute**: ~5h Blackwell. **Cost**: ~$10 OpenRouter.",
        "",
        "**Drive**: `/content/drive/MyDrive/openinterp_runs/44_v2_behavior_eval/`",
        "",
        "**Caveat — nb43 GRPO final LoRA**: captures POST-PEAK drift (peak at idx 1084/1303 ~77% of training). Eval may underestimate GRPO's true peak performance. We use what was saved.",
    ]))

    # === Phase 1 — Setup ===
    cells.append(md(["## Phase 1 — Setup + Drive"]))
    cells.append(code([
        "from pathlib import Path",
        "import os, json, time",
        "import torch, numpy as np",
        "",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "DRIVE = Path('/content/drive/MyDrive')",
        "OUT = DRIVE / 'openinterp_runs' / '44_v2_behavior_eval'",
        "OUT.mkdir(parents=True, exist_ok=True)",
        "print(f'OUT: {OUT}')",
        "print(f'Existing: {sorted(p.name for p in OUT.iterdir())}')",
    ]))
    cells.append(code([
        "!pip install -q -U torchao",
        "!pip install -q -U transformers accelerate datasets peft",
        "!pip install -q -U huggingface_hub",
        "!pip install -q -U openinterp  # v0.2.1+ for safe_load_qwen36_lora",
        "!pip install -q openai scikit-learn joblib matplotlib",
        "import openinterp",
        "print(f'✓ openinterp v{openinterp.__version__}')",
        "assert openinterp.__version__ >= '0.2.1', 'Need openinterp v0.2.1+ for safe_load_qwen36_lora'",
    ]))

    # === Phase 2 — Config + base model ===
    cells.append(md(["## Phase 2 — Configuration + Base model"]))
    cells.append(code([
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "from huggingface_hub import login, snapshot_download, HfApi, create_repo",
        "from openinterp import safe_load_qwen36_lora",
        "import getpass",
        "",
        "CFG = {",
        "    'model_id':            'Qwen/Qwen3.6-27B',",
        "    'n_gsm8k':              100,",
        "    'n_simpleqa':           100,",
        "    'temperature':          0.7,",
        "    'max_new_tokens':       2048,",
        "    'random_seed':          44,",
        "    'dpo_repo':             'caiovicentino1/openinterp-37v2-multiprobe-dpo-extended',",
        "    'dpo_checkpoint':       'checkpoint-200',",
        "    'grpo_repo':            'caiovicentino1/openinterp-43-multiprobe-grpo-full',",
        "    'grpo_checkpoint':      'lora_final',",
        "    'judge_model':          'anthropic/claude-haiku-4.5',",
        "    'output_repo':          'caiovicentino1/openinterp-44-v2-behavior-eval',",
        "    'bootstrap_n':          2000,",
        "}",
        "torch.manual_seed(CFG['random_seed']); np.random.seed(CFG['random_seed'])",
        "",
        "HF_TOKEN = os.environ.get('HF_TOKEN') or getpass.getpass('HF token: ')",
        "login(HF_TOKEN, add_to_git_credential=False)",
        "OPENROUTER_KEY = os.environ.get('OPENROUTER_API_KEY') or getpass.getpass('OpenRouter API key: ')",
        "os.environ['OPENROUTER_API_KEY'] = OPENROUTER_KEY",
        "",
        "device = 'cuda'",
        "tok = AutoTokenizer.from_pretrained(CFG['model_id'])",
        "base_model = AutoModelForCausalLM.from_pretrained(",
        "    CFG['model_id'], torch_dtype=torch.bfloat16, device_map='auto',",
        ")",
        "base_model.eval()",
        "print(f'✓ Base loaded — {torch.cuda.get_device_name(0)}')",
    ]))

    # === Phase 3 — Load adapters via safe loader ===
    cells.append(md([
        "## Phase 3 — Load DPO + GRPO adapters via `safe_load_qwen36_lora`",
        "",
        "Encapsulates the `.language_model.` infix bug fix. Auto-verifies logit-diff > 0.01.",
    ]))
    cells.append(code([
        "# LoRA adapters live in Drive (not HF datasets — only metadata is public).",
        "# Adjust folder names if yours are different.",
        "DPO_DRIVE_DIR = '37v2_multiprobe_dpo_extended'",
        "GRPO_DRIVE_DIR = '43_multiprobe_grpo_full'",
        "",
        "# Both pipelines save final adapter as 'lora_final'",
        "dpo_path = DRIVE / 'openinterp_runs' / DPO_DRIVE_DIR / 'lora_final'",
        "grpo_path = DRIVE / 'openinterp_runs' / GRPO_DRIVE_DIR / 'lora_final'",
        "",
        "if not dpo_path.exists():",
        "    print(f'⚠️  DPO path missing: {dpo_path}')",
        "    print(f'  Try: !ls /content/drive/MyDrive/openinterp_runs/')",
        "if not grpo_path.exists():",
        "    print(f'⚠️  GRPO path missing: {grpo_path}')",
        "",
        "assert dpo_path.exists(), 'DPO adapter not found — check folder name'",
        "assert grpo_path.exists(), 'GRPO adapter not found — check folder name'",
        "",
        "# Verify adapter files",
        "for label, p in [('DPO', dpo_path), ('GRPO', grpo_path)]:",
        "    files = sorted(f.name for f in p.iterdir())",
        "    has_config = 'adapter_config.json' in files",
        "    has_model = any(f.startswith('adapter_model') for f in files)",
        "    status = '✓' if (has_config and has_model) else '⚠️'",
        "    print(f'{status} {label} ({p.name}): {files}')",
    ]))
    cells.append(code([
        "# Load DPO model (auto strip + verify)",
        "print('=== Loading DPO model ===')",
        "dpo_model = safe_load_qwen36_lora(",
        "    base_model_id=CFG['model_id'],",
        "    adapter_path=dpo_path,",
        "    base_model=base_model,  # reuse loaded base",
        "    tokenizer=tok,",
        "    verify=True,",
        ")",
        "print('✓ DPO model loaded + verified')",
        "",
        "print('\\n=== Loading GRPO model ===')",
        "# Need a fresh base_model since DPO modifies it via PEFT injection",
        "del dpo_model",
        "torch.cuda.empty_cache()",
        "base_model = AutoModelForCausalLM.from_pretrained(",
        "    CFG['model_id'], torch_dtype=torch.bfloat16, device_map='auto',",
        ")",
        "base_model.eval()",
        "grpo_model = safe_load_qwen36_lora(",
        "    base_model_id=CFG['model_id'],",
        "    adapter_path=grpo_path,",
        "    base_model=base_model,",
        "    tokenizer=tok,",
        "    verify=True,",
        ")",
        "print('✓ GRPO model loaded + verified')",
        "",
        "# Strategy for inference:",
        "# Use a SINGLE PeftModel and switch adapters per condition.",
        "# Or load 3 separate models (memory cost ~3x). On Blackwell with bf16, 27B = ~55GB,",
        "# so 3 instances would exceed VRAM. We'll load adapters ONE AT A TIME per condition pass.",
        "",
        "del grpo_model",
        "torch.cuda.empty_cache()",
        "print('\\n✓ Ready for sequential adapter switching')",
    ]))

    # === Phase 4 — Holdout prompts ===
    cells.append(md(["## Phase 4 — Hold-out prompts (n=100 GSM8K + 100 SimpleQA)"]))
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
        "    gsm_pool.append({'id': f'gsm_v2_{i}', 'src': 'gsm8k',",
        "                     'question': ex['question'], 'gold': gold})",
        "",
        "try:",
        "    sqa = load_dataset('basicv8vc/SimpleQA', split='test')",
        "    sqa_idx = rng.choice(len(sqa), size=CFG['n_simpleqa'], replace=False)",
        "    sqa_pool = []",
        "    for i in sqa_idx:",
        "        ex = sqa[int(i)]",
        "        sqa_pool.append({'id': f'sqa_v2_{i}', 'src': 'simpleqa',",
        "                         'question': ex['problem'], 'gold': ex['answer']})",
        "except Exception as e:",
        "    print(f'SimpleQA load failed: {e}'); sqa_pool = []",
        "",
        "holdout = gsm_pool + sqa_pool",
        "rng.shuffle(holdout)",
        "print(f'Hold-out: {len(holdout)} prompts')",
        "src_dist = {}",
        "for p in holdout: src_dist[p['src']] = src_dist.get(p['src'], 0) + 1",
        "print(f'  sources: {src_dist}')",
    ]))

    # === Phase 5 — Generate per condition (sequential adapter switch) ===
    cells.append(md([
        "## Phase 5 — Generate 3 conditions × 200 prompts (paired)",
        "",
        "Outer loop: condition (base / DPO / GRPO). Inner loop: prompts.",
        "For each (prompt, condition): generate at temp=0.7 with paired seed.",
        "Resume-safe per (prompt_id, condition).",
    ]))
    cells.append(code([
        "from tqdm.auto import tqdm",
        "import gc",
        "",
        "def make_generator_fn(model):",
        "    def gen(prompt):",
        "        messages = [{'role': 'user', 'content': prompt}]",
        "        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)",
        "        enc = tok(text, return_tensors='pt')",
        "        ids = enc['input_ids'].to(device)",
        "        amask = enc.get('attention_mask', torch.ones_like(ids)).to(device)",
        "        n_in = ids.shape[1]",
        "        with torch.no_grad():",
        "            gen_ids = model.generate(",
        "                ids, attention_mask=amask,",
        "                max_new_tokens=CFG['max_new_tokens'],",
        "                do_sample=True, temperature=CFG['temperature'], top_p=0.95,",
        "                pad_token_id=tok.eos_token_id,",
        "            )",
        "        full_ids = gen_ids[0]",
        "        output_text = tok.decode(full_ids[n_in:], skip_special_tokens=False)",
        "        if '</think>' in output_text:",
        "            cot = output_text.split('</think>', 1)[0].strip()",
        "            answer = output_text.split('</think>', 1)[1].strip()",
        "        else:",
        "            cot, answer = output_text.strip(), ''",
        "        return {'cot': cot, 'answer': answer, 'has_think': '</think>' in output_text}",
        "    return gen",
        "",
        "results_path = OUT / 'generations.jsonl'",
        "done_keys = set()",
        "if results_path.exists():",
        "    with open(results_path) as f:",
        "        for line in f:",
        "            try: done_keys.add((json.loads(line)['id'], json.loads(line)['condition']))",
        "            except: continue",
        "    print(f'Resume: {len(done_keys)} (id, condition) pairs done')",
        "",
        "def run_condition(model, name):",
        "    print(f'\\n=== Condition: {name} ===')",
        "    fn = make_generator_fn(model)",
        "    pbar = tqdm(holdout, desc=name)",
        "    for p in pbar:",
        "        if (p['id'], name) in done_keys: continue",
        "        try:",
        "            torch.manual_seed(hash(p['id']) % (2**32))",
        "            res = fn(p['question'])",
        "        except torch.cuda.OutOfMemoryError:",
        "            torch.cuda.empty_cache(); gc.collect()",
        "            print(f'OOM on {p[\"id\"]}'); continue",
        "        record = {**p, 'condition': name, **res}",
        "        with open(results_path, 'a') as f:",
        "            f.write(json.dumps(record) + '\\n')",
        "        done_keys.add((p['id'], name))",
        "",
        "# Condition 1: base",
        "run_condition(base_model, 'base')",
        "torch.cuda.empty_cache(); gc.collect()",
        "",
        "# Condition 2: DPO (reload base + apply DPO adapter)",
        "del base_model",
        "torch.cuda.empty_cache()",
        "_base = AutoModelForCausalLM.from_pretrained(CFG['model_id'], torch_dtype=torch.bfloat16, device_map='auto')",
        "_base.eval()",
        "dpo_model = safe_load_qwen36_lora(CFG['model_id'], dpo_path, base_model=_base, tokenizer=tok, verify=True)",
        "run_condition(dpo_model, 'dpo')",
        "del dpo_model, _base",
        "torch.cuda.empty_cache(); gc.collect()",
        "",
        "# Condition 3: GRPO",
        "_base = AutoModelForCausalLM.from_pretrained(CFG['model_id'], torch_dtype=torch.bfloat16, device_map='auto')",
        "_base.eval()",
        "grpo_model = safe_load_qwen36_lora(CFG['model_id'], grpo_path, base_model=_base, tokenizer=tok, verify=True)",
        "run_condition(grpo_model, 'grpo')",
        "del grpo_model, _base",
        "torch.cuda.empty_cache(); gc.collect()",
        "",
        "print('\\n✓ Phase 5 complete')",
    ]))

    # === Phase 6 — Score with judge ===
    cells.append(md([
        "## Phase 6 — Score: GSM8K gold-match + SimpleQA judge",
    ]))
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
        "            messages=[{'role': 'user', 'content': JUDGE.format(q=q, g=g, a=a[:1000])}],",
        "        )",
        "        t = r.choices[0].message.content.strip().upper()",
        "        if t.startswith('YES'): return 'YES'",
        "        if t.startswith('NO'): return 'NO'",
        "        return 'UNVERIFIABLE'",
        "    except Exception as e:",
        "        return 'ERROR'",
        "",
        "with open(results_path) as f:",
        "    records = [json.loads(line) for line in f]",
        "scored_path = OUT / 'scored.jsonl'",
        "scored_keys = set()",
        "if scored_path.exists():",
        "    with open(scored_path) as f:",
        "        for line in f:",
        "            try:",
        "                rec = json.loads(line)",
        "                scored_keys.add((rec['id'], rec['condition']))",
        "            except: continue",
        "    print(f'Resume scoring: {len(scored_keys)} done')",
        "",
        "for r in tqdm(records, desc='score'):",
        "    key = (r['id'], r['condition'])",
        "    if key in scored_keys: continue",
        "    if not r.get('has_think'): continue",
        "    answer = (r.get('answer') or r.get('cot', ''))[:1500]",
        "    if r['src'] == 'gsm8k':",
        "        is_correct = grade_gsm8k(answer, r['gold'])",
        "        label = 'CORRECT' if is_correct else 'INCORRECT'",
        "    else:",
        "        label = judge_sqa(r['question'], r['gold'], answer)",
        "        is_correct = label == 'YES'",
        "    out = {**r, 'is_correct': bool(is_correct), 'judge_label': label}",
        "    with open(scored_path, 'a') as f:",
        "        f.write(json.dumps(out) + '\\n')",
        "    scored_keys.add(key)",
        "",
        "print('\\n✓ Phase 6 complete')",
    ]))

    # === Phase 7 — Paired bootstrap analysis ===
    cells.append(md(["## Phase 7 — Paired bootstrap CI analysis"]))
    cells.append(code([
        "import pandas as pd",
        "with open(scored_path) as f:",
        "    scored = [json.loads(line) for line in f]",
        "df = pd.DataFrame(scored)",
        "df = df[df['has_think']].copy()",
        "df['correct'] = df['is_correct'].astype(int)",
        "",
        "print('=== Per condition × source correctness rates ===')",
        "print(df.groupby(['condition', 'src'])['correct'].agg(['mean', 'count']).round(3))",
        "",
        "# Pivot to paired format: rows = prompts, columns = conditions",
        "paired = df.pivot_table(index='id', columns='condition', values='correct', aggfunc='first')",
        "paired = paired.dropna()  # only keep prompts with all 3 conditions",
        "print(f'\\nPaired prompts (all 3 conditions): {len(paired)}')",
        "",
        "# Also keep src for stratified bootstrap",
        "src_map = df.groupby('id')['src'].first().to_dict()",
        "paired['src'] = paired.index.map(src_map)",
    ]))
    cells.append(code([
        "def paired_delta_ci(pair_a, pair_b, n=2000, seed=42):",
        "    \"\"\"Paired bootstrap on (b - a) mean.\"\"\"",
        "    rng = np.random.default_rng(seed)",
        "    diffs = (pair_b - pair_a).values",
        "    n_obs = len(diffs)",
        "    boots = []",
        "    for _ in range(n):",
        "        idx = rng.choice(n_obs, n_obs, replace=True)",
        "        boots.append(diffs[idx].mean())",
        "    arr = np.array(boots)",
        "    return float(diffs.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))",
        "",
        "def emoji_for_delta(delta, lo, hi, threshold=0.0):",
        "    if delta > 0 and lo > threshold: return '🟢'",
        "    if delta > 0 and hi <= threshold: return '🔴'  # upper bound below 0",
        "    if delta < 0 and hi < threshold: return '🔴'",
        "    if delta > 0 and hi > threshold: return '🟡'  # directional but CI crosses 0",
        "    return '⚪'",
        "",
        "comparisons = [",
        "    ('dpo', 'base', 'DPO vs base'),",
        "    ('grpo', 'base', 'GRPO vs base'),",
        "    ('dpo', 'grpo', 'DPO vs GRPO'),",
        "]",
        "",
        "results = {}",
        "print('\\n=== Overall paired bootstrap CI on correctness deltas ===\\n')",
        "for a, b, label in comparisons:",
        "    if b not in paired.columns or a not in paired.columns: continue",
        "    delta, lo, hi = paired_delta_ci(paired[b], paired[a])",
        "    em = emoji_for_delta(delta, lo, hi)",
        "    results[label] = {'delta': delta, 'ci_lo': lo, 'ci_hi': hi, 'verdict': em}",
        "    print(f'  {label:<20} Δ={delta:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  {em}')",
        "",
        "print('\\n=== Per-source breakdown ===')",
        "for src in paired['src'].unique():",
        "    sub = paired[paired['src'] == src]",
        "    print(f'\\n{src} (n={len(sub)}):')",
        "    for a, b, label in comparisons:",
        "        if b not in sub.columns or a not in sub.columns: continue",
        "        delta, lo, hi = paired_delta_ci(sub[b], sub[a])",
        "        em = emoji_for_delta(delta, lo, hi)",
        "        print(f'  {a} vs {b}: Δ={delta:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  {em}')",
    ]))
    cells.append(code([
        "# Aggregate verdict",
        "best_verdict = None",
        "for label, r in results.items():",
        "    if r['verdict'] == '🟢':",
        "        best_verdict = label",
        "        break",
        "if best_verdict:",
        "    summary = f'🟢 {best_verdict}: Δ={results[best_verdict][\"delta\"]:+.4f} CI EXCLUDES 0'",
        "elif any(r['verdict'] == '🟡' for r in results.values()):",
        "    summary = '🟡 Directional signal in some comparison but CI crosses 0 — needs more data'",
        "elif all(r['verdict'] == '⚪' for r in results.values()):",
        "    summary = '⚪ NULL — no condition meaningfully different from base'",
        "else:",
        "    summary = '🔴 Regression detected'",
        "print(f'\\n=== AGGREGATE VERDICT ===')",
        "print(summary)",
    ]))

    # === Phase 8 — Viz + push ===
    cells.append(md(["## Phase 8 — Visualization + HF push"]))
    cells.append(code([
        "import matplotlib.pyplot as plt",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))",
        "",
        "# Left: per-condition correctness rate by source",
        "rates = df.groupby(['condition', 'src'])['correct'].mean().unstack()",
        "rates.T.plot(kind='bar', ax=axes[0], color=['#9ca3af', '#10b981', '#f59e0b'], width=0.7, alpha=0.85)",
        "axes[0].set_ylabel('Correctness rate'); axes[0].set_xlabel('Source')",
        "axes[0].set_title('Per-condition correctness rate'); axes[0].grid(alpha=0.3)",
        "axes[0].legend(title='condition'); axes[0].set_ylim(0, 1)",
        "",
        "# Right: paired bootstrap CI on deltas",
        "labels = list(results.keys())",
        "deltas = [results[l]['delta'] for l in labels]",
        "los = [results[l]['ci_lo'] for l in labels]",
        "his = [results[l]['ci_hi'] for l in labels]",
        "errs = [[d-lo for d,lo in zip(deltas, los)], [hi-d for hi,d in zip(his, deltas)]]",
        "colors = ['#10b981' if r['verdict'] == '🟢' else ('#ef4444' if r['verdict'] == '🔴' else '#9ca3af') for r in [results[l] for l in labels]]",
        "axes[1].bar(labels, deltas, yerr=errs, color=colors, alpha=0.85, capsize=8)",
        "axes[1].axhline(0, color='black', alpha=0.5)",
        "axes[1].set_ylabel('Δ correctness rate (paired bootstrap)')",
        "axes[1].set_title('Pairwise deltas with 95% CI'); axes[1].grid(alpha=0.3)",
        "axes[1].tick_params(axis='x', rotation=15)",
        "",
        "plt.tight_layout()",
        "plt.savefig(OUT / 'fig_behavior_eval_v2.png', dpi=170, bbox_inches='tight')",
        "plt.show()",
    ]))
    cells.append(code([
        "verdict_obj = {",
        "    'experiment': 'nb44 v2 behavior eval (n=100/source)',",
        "    'n_paired': len(paired),",
        "    'n_per_source_target': CFG['n_gsm8k'],",
        "    'conditions': sorted(paired.columns.tolist()),",
        "    'overall_correctness_rate': df.groupby('condition')['correct'].mean().to_dict(),",
        "    'pairwise_deltas': results,",
        "    'aggregate_verdict': summary,",
        "}",
        "(OUT / 'FINAL_VERDICT.json').write_text(json.dumps(verdict_obj, indent=2, default=str))",
        "print(json.dumps(verdict_obj, indent=2, default=str))",
    ]))
    cells.append(code([
        "api = HfApi()",
        "try: create_repo(CFG['output_repo'], repo_type='dataset', private=False, exist_ok=True, token=HF_TOKEN)",
        "except Exception as e: print(e)",
        "",
        "(OUT / 'README.md').write_text(f'''---",
        "license: apache-2.0",
        "tags: [behavior-eval, qwen36-27b, dpo, grpo, paired-bootstrap]",
        "---",
        "",
        "# nb44 v2 — Behavior Eval (n=100/source)",
        "",
        "**Verdict**: {{summary}}",
        "",
        "Decisive behavior eval at n=100 GSM8K + n=100 SimpleQA test prompts × 3 conditions",
        "(base / DPO / GRPO). Paired bootstrap CIs on correctness deltas.",
        "",
        "Conditions:",
        "- base: Qwen3.6-27B no LoRA",
        "- DPO: nb37 v2 multi-probe FG+RG combined preference, 200 steps",
        "- GRPO: nb43 multi-probe continuous reward, 500 steps (final LoRA, post-peak drift)",
        "",
        "LoRA loading: `openinterp.safe_load_qwen36_lora()` v0.2.1 — auto strip + verify.",
        "''')",
        "",
        "try:",
        "    api.upload_folder(folder_path=str(OUT), repo_id=CFG['output_repo'],",
        "                      repo_type='dataset', token=HF_TOKEN,",
        "                      commit_message=f'nb44 v2 behavior eval: {summary}',",
        "                      allow_patterns=['README.md', 'FINAL_VERDICT.json',",
        "                                      'fig_*.png', 'scored.jsonl'])",
        "    print('✓ pushed')",
        "except Exception as e:",
        "    print(f'HF push failed: {e}')",
    ]))

    cells.append(md([
        "## Done",
        "",
        "Decision tree:",
        "- 🟢 DPO or GRPO > base: paper-3 ships positive. Update paper-2 grokking writeup with this behavior validation. Tweet chapter 5.",
        "- 🟡 directional only: more data needed. Could fold into existing paper-2 LessWrong post as preliminary behavior signal.",
        "- ⚪ null: paper-3 honest negative. ICML MI Workshop template fits.",
        "- 🔴 regression: paper-3 walk-back. Investigate which condition regresses; combine with nb43 U-shape finding (peak idx 1084) — argue training-dynamics matters.",
        "",
        "Cost: ~5h compute + ~$10 OpenRouter judge calls.",
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

    out_path = NOTEBOOKS_DIR / "nb44_v2_behavior_eval.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    print(f"✓ wrote {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
