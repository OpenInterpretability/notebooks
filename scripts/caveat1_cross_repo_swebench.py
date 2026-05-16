# =============================================================
# Caveat #1: cross-repo stratified SWE-bench Verified
# Run AFTER the Phase 2A astropy block + Caveat #2 in the same notebook.
# Reuses: model, tokenizer, device, probe_w (L31 unit-norm),
#         random_w (matched random unit-norm), L31_LAYER_IDX (=31),
#         the forward-hook helper used for steering.
# =============================================================

import json
import os
import random as pyrandom
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset

# ---- config ----
SEED          = 42
N_REPOS       = 5      # exclude astropy (already covered)
N_PER_REPO    = 2      # 5 x 2 = 10 new problems, matches astropy N
ALPHA         = 50
MAX_NEW_TOK   = 1024   # same as original SWE-bench block; Caveat #2 already proved >1024 still fails
OUT_DIR       = Path("/content/drive/MyDrive/openinterp_runs/subjective_time_phase2a/caveat1_cross_repo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- stratified sample ----
pyrandom.seed(SEED)
swe = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

by_repo = defaultdict(list)
for ex in swe:
    if ex["repo"] != "astropy/astropy":
        by_repo[ex["repo"]].append(ex)

# pick top-N most-populated repos for stable sampling
repos_ranked = sorted(by_repo.keys(), key=lambda r: -len(by_repo[r]))
target_repos = repos_ranked[:N_REPOS]

sampled = []
for r in target_repos:
    picks = pyrandom.sample(by_repo[r], N_PER_REPO)
    sampled.extend(picks)

print("=== Caveat #1: cross-repo stratified sample ===")
for r in target_repos:
    print(f"  {r:40s}  pool={len(by_repo[r]):4d}  picked={N_PER_REPO}")
print(f"Total: {len(sampled)} problems across {len(target_repos)} repos\n")


# ---- prompt builder (identical to original SWE-bench block) ----
def build_prompt(ex):
    stmt = ex["problem_statement"][:4000]
    user_msg = (
        f"<problem_statement>\n{stmt}\n</problem_statement>\n\n"
        "Analyze the issue carefully and propose a fix."
    )
    msgs = [{"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )


# ---- generation under steering hook ----
THINK_END_ID = tokenizer.encode("</think>", add_special_tokens=False)[0]

@torch.no_grad()
def generate_one(prompt_text, direction=None, alpha=0.0):
    """direction=None => baseline; else add alpha*direction at L31 every token."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    hook_handle = None
    if direction is not None:
        d = direction.to(device=device, dtype=model.dtype)
        def _hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + alpha * d
            return (h,) + out[1:] if isinstance(out, tuple) else h
        # Qwen3.5/3.6 multimodal layout: language_model holds the 64 decoder layers
        layer = model.model.language_model.layers[L31_LAYER_IDX]
        hook_handle = layer.register_forward_hook(_hook)

    try:
        out_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOK,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    gen_ids = out_ids[0, prompt_len:].tolist()
    terminated = THINK_END_ID in gen_ids
    if terminated:
        thinking_len = gen_ids.index(THINK_END_ID) + 1
    else:
        thinking_len = len(gen_ids)
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    return {"thinking_len": thinking_len, "terminated": terminated, "text": text[:2000]}


# ---- main loop ----
results = []
for i, ex in enumerate(sampled, 1):
    instance_id = ex["instance_id"]
    repo        = ex["repo"]
    prompt      = build_prompt(ex)
    print(f"[{i:2d}/{len(sampled)}] {repo} / {instance_id}")

    base = generate_one(prompt, direction=None)
    pro  = generate_one(prompt, direction=probe_w,  alpha=ALPHA)
    rnd  = generate_one(prompt, direction=random_w, alpha=ALPHA)

    row = {
        "i": i, "repo": repo, "instance_id": instance_id,
        "baseline":     {"len": base["thinking_len"], "term": base["terminated"]},
        "probe_p50":    {"len": pro["thinking_len"],  "term": pro["terminated"]},
        "random_p50":   {"len": rnd["thinking_len"],  "term": rnd["terminated"]},
    }
    print(f"     base term={base['terminated']} len={base['thinking_len']:4d}  |  "
          f"probe term={pro['terminated']} len={pro['thinking_len']:4d}  |  "
          f"rand  term={rnd['terminated']} len={rnd['thinking_len']:4d}")
    results.append(row)

    # checkpoint each row (resilient to disconnect)
    with open(OUT_DIR / "results_cross_repo.json", "w") as f:
        json.dump(results, f, indent=2)


# ---- aggregate ----
print("\n=== Caveat #1 aggregate ===")
print(f"{'repo':40s}  base_term  probe_term  rand_term   probe_len  rand_len")
by_r = defaultdict(list)
for r in results:
    by_r[r["repo"]].append(r)

agg = {}
for repo, rows in by_r.items():
    bt = sum(r["baseline"]["term"]   for r in rows)
    pt = sum(r["probe_p50"]["term"]  for r in rows)
    rt = sum(r["random_p50"]["term"] for r in rows)
    pl = [r["probe_p50"]["len"]  for r in rows if r["probe_p50"]["term"]]
    rl = [r["random_p50"]["len"] for r in rows if r["random_p50"]["term"]]
    agg[repo] = dict(n=len(rows), base_term=bt, probe_term=pt, rand_term=rt,
                     probe_mean_len=(sum(pl)/len(pl) if pl else None),
                     rand_mean_len=(sum(rl)/len(rl) if rl else None))
    print(f"{repo:40s}  {bt}/{len(rows)}        {pt}/{len(rows)}         {rt}/{len(rows)}        "
          f"{(sum(pl)/len(pl) if pl else 0):6.0f}    {(sum(rl)/len(rl) if rl else 0):6.0f}")

# overall
total          = len(results)
base_term_all  = sum(r["baseline"]["term"]   for r in results)
probe_term_all = sum(r["probe_p50"]["term"]  for r in results)
rand_term_all  = sum(r["random_p50"]["term"] for r in results)
print(f"\n{'OVERALL':40s}  {base_term_all}/{total}      {probe_term_all}/{total}        {rand_term_all}/{total}")

summary = dict(n=total, base_term=base_term_all, probe_term=probe_term_all,
               rand_term=rand_term_all, per_repo=agg, target_repos=target_repos,
               alpha=ALPHA, seed=SEED, max_new_tok=MAX_NEW_TOK)
with open(OUT_DIR / "summary_cross_repo.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ saved to {OUT_DIR}")

# ---- verdict heuristic ----
if probe_term_all >= int(0.8 * total) and rand_term_all <= int(0.4 * total):
    print(f"\n🟢 STRONG: probe rescues {probe_term_all}/{total} cross-repo "
          f"(random only {rand_term_all}/{total}). Caveat #1 cleared.")
elif probe_term_all >= int(0.5 * total):
    print(f"\n🟡 PARTIAL: probe rescues {probe_term_all}/{total} cross-repo. "
          "Astropy result generalizes weakly — examine per-repo table.")
else:
    print(f"\n🔴 WEAK: probe only rescues {probe_term_all}/{total} cross-repo. "
          "Astropy may be the special case.")
