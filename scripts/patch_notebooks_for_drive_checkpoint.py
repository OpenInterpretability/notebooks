#!/usr/bin/env python3
"""
Patch notebooks 32 and 33 with the 5-layer Drive checkpoint defense:
  1. Drive mount with hard fail-loud
  2. OUTPUT_DIR on Drive (not /content/)
  3. Resume-from-checkpoint at the start of long loops
  4. np.save / pd.to_csv every N steps
  5. HF Hub incremental push every 50 steps

Outputs:
  - notebooks/32_reasoningguard_proof_qwen36_27b_v2.ipynb
  - notebooks/33_fabricationguard_vs_halugate_v2.ipynb (regenerated via build_nb33.py with patch)

Validates: JSON parse, nbformat, Python AST per code cell, no /content/ paths
left for outputs.
"""
import json, ast, sys
from pathlib import Path

ROOT = Path('/Volumes/SSD Major/fish/openinterp-work')
NB_IN  = ROOT / 'notebooks/32_reasoningguard_proof_qwen36_27b.ipynb'
NB_OUT = ROOT / 'notebooks/32_reasoningguard_proof_qwen36_27b_v2.ipynb'

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

def src_of(cell):
    return ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

# ---------- the canonical Drive checkpoint preamble (REUSE FOR ALL FUTURE NOTEBOOKS) ----

DRIVE_MOUNT_CELL = code(r'''
# === DRIVE MOUNT — non-negotiable for any run >30min ===
# Mounts Google Drive and asserts /content/drive/MyDrive exists. Fails loud if not.
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
NB_NAME = '32_reasoningguard_v2'
OUT = DRIVE_ROOT / 'openinterp_runs' / NB_NAME
OUT.mkdir(parents=True, exist_ok=True)
(OUT / '_dry_run.txt').write_text('drive mount OK at ' + str(Path.cwd()))
assert (OUT / '_dry_run.txt').exists()
print(f'✓ Drive checkpoint dir: {OUT}')
print(f'  Contents so far: {sorted(p.name for p in OUT.iterdir())}')
'''.strip())

CHECKPOINT_HELPERS_CELL = code(r'''
# === Checkpoint helpers — save every N rollouts to Drive + optional HF push ===
import numpy as np, json, pickle, time
from typing import Any

CHECKPOINT_EVERY     = 10              # save to Drive every N rollouts
HF_PUSH_EVERY        = 50              # push to HF every N rollouts (optional)
ENABLE_HF_INCREMENTAL = True           # toggle off if HF rate-limits

def save_partial(state: dict, tag: str = 'rollouts'):
    """Atomic save (temp file + rename) of a dict of arrays/lists to Drive."""
    p = OUT / f'{tag}.npz'
    tmp = p.with_suffix('.npz.tmp')
    # Convert nested lists to arrays where possible
    flat = {}
    for k, v in state.items():
        if isinstance(v, list):
            try:
                flat[k] = np.asarray(v)
            except Exception:
                flat[k] = np.asarray([pickle.dumps(x) for x in v])  # opaque pickle
        elif isinstance(v, np.ndarray):
            flat[k] = v
        else:
            flat[k] = np.asarray(v)
    np.savez_compressed(tmp, **flat)
    tmp.rename(p)
    return p

def load_partial(tag: str = 'rollouts') -> dict:
    """Load partial state from Drive if it exists. Returns {} if missing."""
    p = OUT / f'{tag}.npz'
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}

def hf_push_partial(api, repo_id: str, hf_token: str, tag: str = 'rollouts'):
    """Push current Drive snapshot to HF — best-effort, never blocks training."""
    if not ENABLE_HF_INCREMENTAL:
        return
    p = OUT / f'{tag}.npz'
    if not p.exists():
        return
    try:
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=f'partial/{tag}_{int(time.time())}.npz',
            repo_id=repo_id,
            repo_type='dataset',
            token=hf_token,
            commit_message=f'partial {tag} @ {time.strftime("%Y%m%d-%H%M%S")}',
        )
    except Exception as e:
        print(f'  [HF push failed, continuing]: {type(e).__name__}: {e}')

print(f'✓ Checkpoint cadence: Drive every {CHECKPOINT_EVERY} rollouts; HF every {HF_PUSH_EVERY} (enabled={ENABLE_HF_INCREMENTAL})')
'''.strip())

# Replacement for the ROLLOUT loop cell — adds resume + per-N save + HF push
ROLLOUT_LOOP_CELL = code(r'''
# === Rollout loop — RESUMABLE + CHECKPOINTED ===
from tqdm.auto import tqdm
from huggingface_hub import HfApi, create_repo
import numpy as np

# Initialize / resume
existing = load_partial(tag='rollouts')
if existing:
    print(f'Resuming from existing rollouts.npz — {len(existing.get("y_combined", []))} prior rows')
    # Reconstruct data dict from npz format. Layout convention:
    #   y_<bench>           — labels (1 = halu)
    #   res_<bench>_<pos>_L<layer>  — residual arrays
    #   q_<bench>           — questions (str array)
    #   resp_<bench>        — responses (str array)
    data = {bench: {pos: {l: [] for l in CFG['probe_layers']} for pos in CFG['probe_positions']}
            for bench in subsets}
    for bench in subsets:
        data[bench]['y']        = list(existing.get(f'y_{bench}', np.array([])))
        data[bench]['response'] = list(existing.get(f'resp_{bench}', np.array([])))
        data[bench]['question'] = list(existing.get(f'q_{bench}', np.array([])))
        for pos in CFG['probe_positions']:
            for l in CFG['probe_layers']:
                arr = existing.get(f'res_{bench}_{pos}_L{l}', np.array([]))
                data[bench][pos][l] = list(arr) if arr.size else []
else:
    data = {bench: {pos: {l: [] for l in CFG['probe_layers']} for pos in CFG['probe_positions']}
            for bench in subsets}
    for bench in subsets:
        data[bench]['y']        = []
        data[bench]['response'] = []
        data[bench]['question'] = []

# HF api for incremental push (best-effort)
hf_api = HfApi()
try:
    create_repo(CFG['hf_results_repo'], exist_ok=True, private=False, token=HF_TOKEN, repo_type='dataset')
except Exception as e:
    print(f'  [create_repo: {e}]')

def _serialize_data() -> dict:
    """Flatten the nested data dict into a flat npz-friendly state."""
    out = {}
    for bench in data:
        out[f'y_{bench}']    = np.asarray(data[bench]['y'], dtype=np.int8)
        out[f'q_{bench}']    = np.asarray(data[bench]['question'], dtype=object)
        out[f'resp_{bench}'] = np.asarray(data[bench]['response'], dtype=object)
        for pos in CFG['probe_positions']:
            for l in CFG['probe_layers']:
                arr = data[bench][pos][l]
                if len(arr) > 0:
                    out[f'res_{bench}_{pos}_L{l}'] = np.asarray(arr, dtype=np.float32)
    out['_timestamp']   = np.array(time.time())
    out['_n_completed'] = np.array(sum(len(data[b]['y']) for b in data))
    return out

def process_question(question, ground_truth, grade_fn, bench_name):
    try:
        r = thinking_rollout_with_residuals(question)
    except Exception as e:
        print(f'  [{bench_name}] generation failed: {e}'); return False
    correct = grade_fn(r['response_decoded_clean'], ground_truth)
    is_hallucinated = int(not correct)
    data[bench_name]['y'].append(is_hallucinated)
    data[bench_name]['response'].append(r['response_decoded_clean'][:1500])
    data[bench_name]['question'].append(question[:300])
    for l in CFG['probe_layers']:
        for pos in CFG['probe_positions']:
            data[bench_name][pos][l].append(r['residuals'].get(l, {}).get(pos, np.zeros(d_model)))
    return True

# Run per-benchmark with resume + checkpoint
for bench, items in subsets.items():
    if not items: continue
    n_already = len(data[bench]['y'])
    if n_already >= len(items):
        print(f'  {bench}: already complete ({n_already}/{len(items)}), skipping')
        continue
    if n_already > 0:
        print(f'  {bench}: resuming from row {n_already}')

    grade_fn = {'gsm8k': grade_gsm8k, 'math': grade_math, 'strategyqa': grade_strategyqa}[bench]
    ques_key = {'gsm8k': 'question', 'math': 'problem', 'strategyqa': 'question'}[bench]
    ans_key  = {'gsm8k': 'answer', 'math': 'solution', 'strategyqa': 'answer'}[bench]

    pbar = tqdm(items[n_already:], desc=f'{bench} rollouts', initial=n_already, total=len(items))
    for i, q in enumerate(pbar, start=n_already):
        ques = q.get(ques_key) or q.get('question')
        truth = q.get(ans_key)
        if ans_key == 'solution' and truth is None:
            truth = q.get('answer')
        if isinstance(truth, list) and truth: truth = truth[0]
        ok = process_question(ques, truth, grade_fn, bench)
        # Drive checkpoint
        if ok and (i + 1) % CHECKPOINT_EVERY == 0:
            save_partial(_serialize_data(), tag='rollouts')
        # HF push (less frequent)
        if ok and (i + 1) % HF_PUSH_EVERY == 0:
            hf_push_partial(hf_api, CFG['hf_results_repo'], HF_TOKEN, tag='rollouts')

    # Save at end of bench
    save_partial(_serialize_data(), tag='rollouts')
    hf_push_partial(hf_api, CFG['hf_results_repo'], HF_TOKEN, tag='rollouts')
    correct_rate = 100 * (1 - np.mean(data[bench]['y']))
    print(f'  {bench} correct rate: {correct_rate:.1f}%  ({len(data[bench]["y"])} samples)')

# Convert lists → arrays after all benches done
for bench in data:
    if not data[bench]['y']: continue
    data[bench]['y'] = np.array(data[bench]['y'])
    for pos in CFG['probe_positions']:
        for l in CFG['probe_layers']:
            data[bench][pos][l] = np.array(data[bench][pos][l])
    print(f'  {bench}: N={len(data[bench]["y"])}, hallucination rate {100*data[bench]["y"].mean():.1f}%')

# Final checkpoint
save_partial(_serialize_data(), tag='rollouts_final')
hf_push_partial(hf_api, CFG['hf_results_repo'], HF_TOKEN, tag='rollouts_final')
ml_hook.close()
print(f'\n✓ Rollouts complete and persisted to {OUT}')
'''.strip())

# ---------- patch the notebook ---------------------------------------------

def patch_nb32():
    nb = json.loads(NB_IN.read_text())
    cells = nb['cells']

    # 1. Insert DRIVE_MOUNT_CELL right after the install cell (cell 1)
    #    so it's the FIRST thing that runs after pip
    cells.insert(2, md('## 0. Drive mount + checkpoint dir (non-negotiable)'))
    cells.insert(3, DRIVE_MOUNT_CELL)
    cells.insert(4, CHECKPOINT_HELPERS_CELL)

    # 2. Patch the config cell (was cell 3, now cell 6 after our 3 inserts)
    #    to use OUT (Drive) instead of /content/reasoningguard_out
    cfg_idx = next(i for i, c in enumerate(cells)
                   if c['cell_type'] == 'code' and 'CFG = {' in src_of(c) and 'probe_layers' in src_of(c))
    src = src_of(cells[cfg_idx])
    src = src.replace(
        "LOCAL_OUT = Path('/content/reasoningguard_out')\nLOCAL_OUT.mkdir(parents=True, exist_ok=True)",
        "LOCAL_OUT = OUT  # alias to Drive checkpoint dir from cell 0\n"
        "print(f'LOCAL_OUT (Drive): {LOCAL_OUT}')"
    )
    cells[cfg_idx]['source'] = [line + '\n' for line in src.rstrip('\n').split('\n')]

    # 3. Replace the rollout loop cell (find by 'gsm8k rollouts' marker)
    rollout_idx = next(i for i, c in enumerate(cells)
                       if c['cell_type'] == 'code' and "desc='gsm8k rollouts'" in src_of(c))
    cells[rollout_idx] = ROLLOUT_LOOP_CELL

    # Update first markdown cell's title to flag this is v2
    if cells[0]['cell_type'] == 'markdown':
        title_src = src_of(cells[0])
        title_src = title_src.replace(
            'ReasoningGuard — Proof of Concept on Qwen3.6-27B',
            'ReasoningGuard v2 — RESUMABLE / DRIVE-CHECKPOINTED PoC on Qwen3.6-27B'
        )
        # Add a checkpoint banner at the top
        banner = ('> **v2 patch (2026-04-28)**: Drive mount mandatory at top, '
                  'rollouts auto-save every 10 steps, resume-from-checkpoint built in. '
                  'v1 lost 8h09min of compute due to no persistence.\n\n')
        cells[0]['source'] = [banner + '\n'] + [line + '\n' for line in title_src.split('\n')]

    nb['cells'] = cells
    NB_OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f'Wrote {NB_OUT}  ({NB_OUT.stat().st_size/1024:.1f} KB, {len(cells)} cells)')
    return NB_OUT


def validate(nb_path: Path) -> bool:
    """Validate: JSON, nbformat, Python AST, no /content/ output paths leaked."""
    nb = json.loads(nb_path.read_text())
    try:
        import nbformat
        nbformat.validate(nbformat.from_dict(nb))
    except Exception as e:
        print(f'  ✗ nbformat: {e}')
        return False

    bad_paths = []
    syntax_errors = []
    has_drive_mount = False
    has_save_partial = False
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code': continue
        src = src_of(c)
        cleaned = '\n'.join('pass  # ' + l if l.lstrip().startswith(('%','!')) else l for l in src.splitlines())
        try:
            ast.parse(cleaned)
        except SyntaxError as e:
            syntax_errors.append((i, e))
        if "drive.mount" in src: has_drive_mount = True
        if "save_partial(" in src: has_save_partial = True
        # Check for forbidden /content/ output paths (NOT the drive subpath, NOT comments)
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith('#'): continue
            if "'/content/" in stripped and "/content/drive/" not in stripped and 'OUT' not in stripped:
                # Exclude /content/drive paths and any line that's already going to OUT
                if 'reasoningguard_out' in stripped or 'nb33_out' in stripped or 'reasoning_out' in stripped:
                    bad_paths.append((i, line))

    print(f'  cells: {len(nb["cells"])}')
    print(f'  drive mount cell present: {has_drive_mount}')
    print(f'  save_partial used: {has_save_partial}')
    print(f'  syntax errors: {len(syntax_errors)}')
    print(f'  /content/ output paths leaked: {len(bad_paths)}')
    if bad_paths:
        for i, line in bad_paths[:5]:
            print(f'    cell [{i}]: {line.strip()}')
    return not syntax_errors and has_drive_mount and has_save_partial


if __name__ == '__main__':
    print('=== Patching notebook 32 ===')
    out = patch_nb32()
    print('\n=== Validating notebook 32 v2 ===')
    ok = validate(out)
    print('  →', 'OK' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
