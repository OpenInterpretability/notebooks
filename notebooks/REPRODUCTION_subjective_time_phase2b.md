# Reproduction guide — Subjective-Time Probe Phase 2B (steering design exploration)

Companion to [`nb_subjective_time_phase2b_steering_designs.ipynb`](./nb_subjective_time_phase2b_steering_designs.ipynb).
Builds on Phase 2A (positive causal probe at L31). Tests 6 follow-up questions for paper-8.

## What you need

- **Model**: `Qwen/Qwen3.6-27B` (HF, hybrid Gated-Delta-Net + standard-attention).
- **GPU**: A100 80 GB or RTX 6000 Blackwell Pro 96 GB. bf16 inference, ~54 GB peak VRAM.
- **Cache**: Phase 2A's residual cache at `Drive/openinterp_runs/predictive_sae_v1/cache/residuals_multilayer.pt`
  (or re-generate from the Phase 2A notebook). Used only for refitting the L31 Ridge probe; ~5 s.
- **Compute time**: ~90 min for the full 6-experiment sweep on A100 80GB.

## Order of execution (cells 1–10)

| # | Cell block | What runs | Output file |
|---|---|---|---|
| 1 | Setup | Drive mount, model load, tokenizer, probe refit | — |
| 2 | Cross-repo sample | `princeton-nlp/SWE-bench_Verified` stratified pick (5 repos × 2 = 10) | — |
| 3 | Helpers | `generate_one`, `generate_closed_loop`, `generate_plateau`, `generate_delayed_static` | — |
| 4 | Caveat #1 — cross-repo static probe@+50 + random | 30 gens, ~15 min | `results_cross_repo.json` |
| 5 | Caveat #2 — re-test baselines at MAX_NEW_TOK=2048 | 10 gens, ~5 min | `caveat2_budget2048.json` |
| 6 | α-sweep B {+30, +40} | 20 gens, ~10 min | `alpha_sweep_quality.json` |
| 7 | Design E — closed-loop thresholds {0.65, 0.70, 0.85} | 30 gens, ~15 min | `design_e_results.json` |
| 8 | Design F — plateau detector {w100, w50} | 20 gens, ~10 min | `design_f_results.json` |
| 9 | Onset timing — static α=+50 with onset {50, 200, 400} | 30 gens, ~15 min | `onset_timing_results.json` |
| 10 | Verdict | Aggregates all 6 experiments, maps to paper-8 sections | — |

All outputs land in `Drive/openinterp_runs/subjective_time_phase2a/caveat1_cross_repo/`.

## Local mount path (Mac, PT-BR localization)

```
~/Library/CloudStorage/GoogleDrive-caiosanford@gmail.com/Meu Drive/openinterp_runs/subjective_time_phase2a/caveat1_cross_repo/
```

Colab `/content/drive/MyDrive/` ⇄ local `Meu Drive/` (not "My Drive").

## Resume-on-disconnect

Every experiment cell checkpoints per prompt: the JSON file is overwritten after each row.
If Colab disconnects mid-run, re-run the same cell — duplicate work is roughly linear with `len(sampled) - len(loaded)`,
which you can wrap with a `if r['instance_id'] in already_done: continue` guard if needed.

## What each output file contains

```
results_cross_repo.json        # 10 rows: baseline / probe@+50 / random@+50 lens + term flags
caveat2_budget2048.json        # 10 rows: baseline at MAX_NEW_TOK=2048 lens + term flags
alpha_sweep_quality.json       # 10 rows × 2 alphas {30, 40}: lens + term flags
design_e_results.json          # 10 rows × 3 thresholds: lens, term, committed_at, trace_max
design_f_results.json          # 10 rows × 2 plateau configs: lens, term, committed_at
onset_timing_results.json      # 10 rows × 3 onsets {50, 200, 400}: lens, term, active_from
```

Text snapshots of generated outputs are inlined in each row's `'text'` field (truncated to 2000–2500 chars).
For full-text inspection re-run with `generate_*(..., max_new_tok=1024)` and dump the raw decode separately.

## Determinism notes

- Greedy decoding everywhere (`do_sample=False, temperature=None`).
- All `random_w` seeded with `torch.manual_seed(42)` after `.randn` — identical Gaussian direction across reruns.
- Stratified SWE-bench sample seeded with `pyrandom.seed(42)`.
- Probe train/test split seeded with `np.random.default_rng(42)`.
- Refit R² should match Phase 2A's reported 0.858 to ±0.001.

Non-deterministic factors (small):
- Attention impl differences across `fla` / `sdpa` / `flash-attn 2` packages can shift probe AUROC by ~0.05 between
  envs (see [feedback_probe_env_coupling.md](../../../) memory). Refit at inference env if AUROC drops.
- bf16 vs fp16 vs fp32 — bf16 reproducible as long as the same CUDA / pytorch versions are used.

## Companion artifacts (separate repos)

- Paper draft: `openinterpretability-web/content/papers/probe-guided-anti-overthinking.md`
- Phase 2A precursor: `notebooks/nb_subjective_time_phase2a_steering.ipynb`
- v1 probe: `notebooks/nb_subjective_time_probe_v1.ipynb`
- SAEs (NOT used here; reference only): HF `caiovicentino1/qwen36-27b-sae-papergrade`

## Decision tree (what to do when each cell finishes)

```
Caveat #1: probe_rescue >= 8/10 and random <= 4/10?
  → 🟢 cross-domain claim locked, proceed
  → 🔴 cross-domain falsified, paper rewrite required

Caveat #2: 0/10 terminate at 2048?
  → 🟢 rescue genuine, §7 last paragraph holds as-is
  → 🟡/🔴 soften "rescue" framing to "compression"

α-sweep B: termination rate monotone with α from 30 → 40 → 50?
  → 🟢 confirms basin transition near +50 (§5 strengthened)

Design E (closed-loop) + Design F (plateau):
  → If 0-2/10 across thresholds: closed-loop FALSIFIED — §7.3 new section, §10 SDK reframe
  → If 5+/10 at some threshold: SDK adaptive mode viable — §10 SDK ships closed-loop config

Onset timing:
  → onset=50 ≈ 9/10: SENSOR was the problem (Design E' viable with right threshold)
  → onset=50 << 9/10, decay with onset: KV cache lock-in confirmed (paper retitle, SDK = budget enforcer)
```

## Hard rules applied (Phase 2A inheritance)

- **Phase 7/8 — Random direction parallel**: every probe steering condition mirrored with `random_w` at same α.
- **Phase 8 — Structural rigidity α-sweep**: α extended to multiples of ‖residual‖ (200) in Phase 2A; Phase 2B trims to {30, 40, 50} for SWE-bench focus.
- **Phase 10 — Whitespace-stripped flip metric**: termination measured via `'</think>' in gen_ids`, not via raw text equality.
- **Phase 6c — Random-feature / shuffled-target / constant-mean baselines**: applied to probe v1 itself (R² 0.86 vs 0.07 / -1.0 / 0.0). Not re-applied here; reference v1 notebook.
- **Drive checkpoint per prompt**: every loop overwrites the JSON after each row.

## Citation

If you reproduce or extend this work, cite the paper draft when it lands:

```
Vicentino, C. (2026). Probe-Guided Anti-Overthinking: A Causal Termination Basin in
Qwen3.6-27B Reasoning. NeurIPS 2026 MI Workshop submission.
```

Apache-2.0. Single-author, double-blind for submission; non-blind acknowledgements after acceptance.
