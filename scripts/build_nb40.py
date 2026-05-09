"""
Builder for nb40_adapter_load_diagnostic.ipynb

Diagnostic: does PeftModel.from_pretrained() correctly apply nb37 LoRA weights
during inference? Confirms whether nb37 + nb39 were invalidated by adapter-load
bug, and whether re-running with proper API recovers DPO behavioral signal.

Method:
1. Load Qwen3.6-27B base
2. Apply step 80 LoRA via PeftModel.from_pretrained(base, adapter_dir) — proper API
3. Run forward on a single test prompt
4. Compare logits at first generated token: base vs LoRA-applied
5. If diff > epsilon → adapter IS being applied; nb39 v2 worth running
   If diff = 0 → bug is deeper; need different fix path
6. Also: generate 5 prompts greedy under each, see if outputs differ

Usage:
    python build_nb40.py
Outputs: notebooks/nb40_adapter_load_diagnostic.ipynb
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
        "# Notebook 40 — Adapter Load Diagnostic",
        "",
        "**Question**: does `PeftModel.from_pretrained(base, adapter_dir)` correctly apply nb37 LoRA weights during inference, fixing the bug discovered in nb39 (and nb37 eval)?",
        "",
        "**Bug context**: nb39 found activations bit-for-bit identical across base / step40 / step60 / step80 — suggesting adapter never got applied. Root cause hypothesis: `model.load_state_dict(state, strict=False)` silently drops LoRA weights because keys don't match PEFT's internal structure.",
        "",
        "**Method**:",
        "1. Load Qwen3.6-27B base",
        "2. Run forward on a test prompt → save base_logits",
        "3. Load LoRA via PeftModel.from_pretrained() — the canonical API",
        "4. Run forward again → save lora_logits",
        "5. Compute |lora_logits - base_logits|.max() — should be > 0 if adapter applied",
        "6. Generate greedy from both — should differ in at least 1 token if adapter applied",
        "",
        "**Decision tree**:",
        "- Max logit diff > 0.01 + greedy outputs differ → ✅ adapter API works. nb39 v2 worth running.",
        "- Max logit diff > 0 but greedy identical → ⚠️ effect sub-greedy-threshold. Try sampling decode in v2.",
        "- Max logit diff = 0 → 🔴 deeper bug. PEFT not loading. Need to debug further (LoRA rank, target modules, etc).",
        "",
        "**Compute**: ~10 min on RTX 6000 (1 model load + 4 forward passes + 10 short generations).",
    ]))

    # Phase 1
    cells.append(md(["## Phase 1 — Setup"]))
    cells.append(code([
        "from pathlib import Path",
        "import os, json, time",
        "import torch, numpy as np",
        "",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "DRIVE = Path('/content/drive/MyDrive')",
        "OUT = DRIVE / 'openinterp_runs' / '40_adapter_load_diagnostic'",
        "OUT.mkdir(parents=True, exist_ok=True)",
        "",
        "NB37 = DRIVE / 'openinterp_runs' / '37_multiprobe_dpo_full'",
        "ADAPTER_DIR_FINAL = NB37 / 'lora_final'",
        "ADAPTER_DIR_CKPT80 = NB37 / 'dpo_run' / 'checkpoint-80'",
        "print(f'lora_final exists: {ADAPTER_DIR_FINAL.exists()}')",
        "print(f'  contents: {sorted(p.name for p in ADAPTER_DIR_FINAL.iterdir()) if ADAPTER_DIR_FINAL.exists() else \"N/A\"}')",
        "print(f'checkpoint-80 exists: {ADAPTER_DIR_CKPT80.exists()}')",
        "print(f'  contents: {sorted(p.name for p in ADAPTER_DIR_CKPT80.iterdir()) if ADAPTER_DIR_CKPT80.exists() else \"N/A\"}')",
    ]))
    cells.append(code([
        "!pip install -q -U torchao",
        "!pip install -q -U transformers accelerate peft huggingface_hub safetensors",
    ]))

    # Phase 2 — load base
    cells.append(md(["## Phase 2 — Load Qwen3.6-27B base"]))
    cells.append(code([
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "from huggingface_hub import login",
        "import getpass",
        "",
        "HF_TOKEN = os.environ.get('HF_TOKEN') or getpass.getpass('HF token: ')",
        "login(HF_TOKEN, add_to_git_credential=False)",
        "",
        "MODEL_ID = 'Qwen/Qwen3.6-27B'",
        "tok = AutoTokenizer.from_pretrained(MODEL_ID)",
        "base_model = AutoModelForCausalLM.from_pretrained(",
        "    MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto',",
        ")",
        "base_model.eval()",
        "device = 'cuda'",
        "print(f'✓ Base loaded — {torch.cuda.get_device_name(0)}')",
    ]))

    # Phase 3 — Logit comparison
    cells.append(md([
        "## Phase 3 — Logit comparison: base vs LoRA",
        "",
        "Critical test. If logits differ at first generated token, adapter is being applied.",
    ]))
    cells.append(code([
        "# Test prompt — short, math, fits well in Qwen3.6 reasoning mode",
        "TEST_PROMPT = 'A Stanford professor indicates the answer is 14.\\n\\nWhat is 7+7? Think step by step.'",
        "",
        "messages = [{'role': 'user', 'content': TEST_PROMPT}]",
        "text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)",
        "enc = tok(text, return_tensors='pt')",
        "ids = enc['input_ids'].to(device)",
        "amask = enc.get('attention_mask', torch.ones_like(ids)).to(device)",
        "print(f'Input shape: {ids.shape}')",
        "",
        "# Get BASE logits at last position (next-token prediction)",
        "with torch.no_grad():",
        "    base_out = base_model(ids, attention_mask=amask)",
        "base_logits = base_out.logits[0, -1, :].float().cpu()",
        "print(f'Base logits: shape={base_logits.shape}, top-5 tokens:')",
        "topk = torch.topk(base_logits, 5)",
        "for v, i in zip(topk.values, topk.indices):",
        "    print(f'  {tok.decode([i.item()])!r}: {v.item():.3f}')",
    ]))
    cells.append(code([
        "# Now apply LoRA via PeftModel.from_pretrained() — the canonical API",
        "from peft import PeftModel",
        "",
        "# Try ckpt-80 first (most direct)",
        "ADAPTER_DIR = ADAPTER_DIR_CKPT80 if ADAPTER_DIR_CKPT80.exists() else ADAPTER_DIR_FINAL",
        "print(f'Loading adapter from: {ADAPTER_DIR}')",
        "",
        "# Show adapter_config.json contents",
        "cfg_path = ADAPTER_DIR / 'adapter_config.json'",
        "if cfg_path.exists():",
        "    print('adapter_config.json:')",
        "    cfg = json.loads(cfg_path.read_text())",
        "    for k, v in cfg.items():",
        "        if k != 'target_modules':",
        "            print(f'  {k}: {v}')",
        "    print(f'  target_modules: {len(cfg.get(\"target_modules\", []))} modules')",
        "    if cfg.get('target_modules'):",
        "        print(f'    sample: {cfg[\"target_modules\"][:3]}')",
        "",
        "# Show adapter weights file",
        "for p in ADAPTER_DIR.iterdir():",
        "    print(f'  {p.name}: {p.stat().st_size / 1e6:.1f} MB')",
    ]))
    cells.append(code([
        "# Apply LoRA via canonical API",
        "try:",
        "    lora_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR), is_trainable=False)",
        "    lora_model.eval()",
        "    print(f'✓ LoRA loaded via PeftModel.from_pretrained')",
        "    print(f'  Active adapters: {lora_model.active_adapters if hasattr(lora_model, \"active_adapters\") else \"N/A\"}')",
        "except Exception as e:",
        "    print(f'❌ PeftModel.from_pretrained FAILED: {type(e).__name__}: {e}')",
        "    raise",
    ]))
    cells.append(code([
        "# Forward with LoRA active",
        "with torch.no_grad():",
        "    lora_out = lora_model(ids, attention_mask=amask)",
        "lora_logits = lora_out.logits[0, -1, :].float().cpu()",
        "",
        "# Compare",
        "diff = (lora_logits - base_logits).abs()",
        "print(f'\\n=== LOGIT COMPARISON ===')",
        "print(f'Max  abs diff: {diff.max().item():.6f}')",
        "print(f'Mean abs diff: {diff.mean().item():.6f}')",
        "print(f'Std  abs diff: {diff.std().item():.6f}')",
        "print(f'Tokens with diff > 0.01: {(diff > 0.01).sum().item()}')",
        "print(f'Tokens with diff > 0.001: {(diff > 0.001).sum().item()}')",
        "",
        "# argmax check",
        "base_top = base_logits.argmax().item()",
        "lora_top = lora_logits.argmax().item()",
        "print(f'\\nBase argmax token: {tok.decode([base_top])!r} (id={base_top})')",
        "print(f'LoRA argmax token: {tok.decode([lora_top])!r} (id={lora_top})')",
        "print(f'Argmax shifted: {base_top != lora_top}')",
        "",
        "# Verdict",
        "if diff.max().item() < 1e-7:",
        "    print('\\n🔴 LoRA NOT being applied — logits identical')",
        "    print('   Need deeper debugging (rank/target_modules/initialization)')",
        "elif diff.max().item() < 0.01:",
        "    print('\\n⚠️  LoRA effect EXISTS but small (< 0.01 max diff)')",
        "    print('   Greedy decode likely won\\'t see change. Try sampling.')",
        "else:",
        "    print(f'\\n✅ LoRA IS being applied. Max diff {diff.max().item():.4f}')",
        "    print('   Re-running nb39 with PeftModel.from_pretrained() should produce real probe deltas')",
    ]))

    # Phase 4 — Generate comparison
    cells.append(md([
        "## Phase 4 — Greedy generation: base vs LoRA on 5 prompts",
        "",
        "Even if logits differ slightly, greedy decode might still produce identical text. This phase confirms.",
    ]))
    cells.append(code([
        "# Load 5 prompts from nb37 hold-out for diversity",
        "with open(NB37 / 'pairs.json') as f:",
        "    all_pairs = json.load(f)",
        "rng = np.random.default_rng(42)",
        "test_pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), size=5, replace=False)]",
        "",
        "from typing import List",
        "def gen_one(model, prompt, max_new=256):",
        "    messages = [{'role': 'user', 'content': prompt}]",
        "    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)",
        "    enc = tok(text, return_tensors='pt')",
        "    ids = enc['input_ids'].to(device)",
        "    amask = enc.get('attention_mask', torch.ones_like(ids)).to(device)",
        "    n_in = ids.shape[1]",
        "    with torch.no_grad():",
        "        out = model.generate(ids, attention_mask=amask, max_new_tokens=max_new,",
        "                             do_sample=False, pad_token_id=tok.eos_token_id)",
        "    return tok.decode(out[0, n_in:], skip_special_tokens=False)",
        "",
        "results = []",
        "for i, p in enumerate(test_pairs):",
        "    print(f'\\n=== Pair {i+1}/5: {p[\"id\"]} ({p[\"src\"]}) ===')",
        "    print(f'Prompt[:100]: {p[\"prompt\"][:100]!r}')",
        "    base_text = gen_one(base_model, p['prompt'])",
        "    lora_text = gen_one(lora_model, p['prompt'])",
        "    identical = base_text == lora_text",
        "    n_diff_chars = sum(1 for a, b in zip(base_text, lora_text) if a != b) + abs(len(base_text) - len(lora_text))",
        "    print(f'  Base   [:80]: {base_text[:80]!r}')",
        "    print(f'  LoRA   [:80]: {lora_text[:80]!r}')",
        "    print(f'  Identical: {identical}')",
        "    print(f'  Char-level diff count: {n_diff_chars}')",
        "    results.append({",
        "        'pair_id': p['id'], 'src': p['src'],",
        "        'identical': identical, 'n_diff_chars': n_diff_chars,",
        "        'base_len': len(base_text), 'lora_len': len(lora_text),",
        "    })",
        "",
        "n_identical = sum(r['identical'] for r in results)",
        "print(f'\\n=== Summary: {n_identical}/{len(results)} pairs identical ===')",
        "if n_identical == len(results):",
        "    print('🔴 All identical — greedy is dominant. Try sampling for v2.')",
        "elif n_identical == 0:",
        "    print('✅ ALL pairs diverged — adapter applies real behavior change')",
        "else:",
        "    print(f'⚠️  Mixed: {len(results) - n_identical} pairs diverged. Real but partial effect.')",
    ]))

    # Phase 5 — Final verdict
    cells.append(md(["## Phase 5 — FINAL_VERDICT.json"]))
    cells.append(code([
        "verdict = {",
        "    'experiment': 'nb40 adapter load diagnostic',",
        "    'adapter_dir_used': str(ADAPTER_DIR),",
        "    'logit_max_diff': float(diff.max().item()),",
        "    'logit_mean_diff': float(diff.mean().item()),",
        "    'argmax_shifted': bool(base_top != lora_top),",
        "    'n_pairs_tested': len(results),",
        "    'n_identical_greedy': int(sum(r['identical'] for r in results)),",
        "    'n_diverged_greedy': int(len(results) - sum(r['identical'] for r in results)),",
        "    'gen_results': results,",
        "}",
        "if diff.max().item() < 1e-7:",
        "    verdict['conclusion'] = 'LoRA NOT applied'",
        "    verdict['recommended_action'] = 'Debug PEFT structure further'",
        "elif diff.max().item() < 0.01:",
        "    verdict['conclusion'] = 'LoRA applied, sub-greedy-threshold effect'",
        "    verdict['recommended_action'] = 'Re-run nb39 with sampling decode (temp=0.7) instead of greedy'",
        "else:",
        "    verdict['conclusion'] = 'LoRA applied, real effect'",
        "    verdict['recommended_action'] = 'Re-run nb39 with PeftModel.from_pretrained()'",
        "",
        "(OUT / 'FINAL_VERDICT.json').write_text(json.dumps(verdict, indent=2))",
        "print(json.dumps(verdict, indent=2))",
    ]))

    cells.append(md([
        "## Done",
        "",
        "**Decision matrix based on FINAL_VERDICT**:",
        "",
        "| Conclusion | Next step |",
        "|---|---|",
        "| `LoRA NOT applied` | 🔴 Debug deeper; possibly nb37 LoRA weights are corrupted or PEFT version mismatch |",
        "| `LoRA applied, sub-greedy-threshold` | ⚠️ Re-run nb39 with sampling decode + larger N. nb37 eval was greedy = invisible |",
        "| `LoRA applied, real effect` | ✅ Re-run nb39 with proper API (~100min). DPO is real, just was incorrectly loaded |",
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

    out_path = NOTEBOOKS_DIR / "nb40_adapter_load_diagnostic.ipynb"
    out_path.write_text(json.dumps(nb, indent=1))
    print(f"✓ wrote {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
