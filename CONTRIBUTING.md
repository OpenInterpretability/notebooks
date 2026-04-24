# Contributing to `notebooks`

Thanks for contributing a notebook to the OpenInterpretability training/interpretability library.

## Add a new notebook — checklist

1. **Open an issue first** with the proposed title + target model + platform. Avoid duplicate work.
2. Naming: `NN_short_descriptive_name.ipynb` where `NN` is the next free number. Keep it kebab-case.
3. Place in `notebooks/` at the repo root.

## Hard constraints (apply to every notebook)

- **`dtype=torch.bfloat16`**, never the deprecated `torch_dtype=`.
- **`attn_implementation='sdpa'`**, never `flash-attn` (no CUDA-build-time deps for users).
- **HF token via Colab secret** (`userdata.get('HF_TOKEN')`) or **Kaggle secret** (`UserSecretsClient().get_secret('HF_TOKEN')`) — never hard-coded.
- **Resume-safe** if it runs >20 min: checkpoint to HF repo every N tokens/steps, detect existing checkpoint at start.
- **Self-contained**: everything a user needs to reproduce — prompts, hyperparameters, dataset loading — should be in-notebook.
- **Honest output**: if a metric underperforms, report it. No cherry-picking seeds.

## Format

- Structure cells as **markdown → code → markdown → code → …**
- Each section opens with a markdown heading explaining what the code does.
- Keep cell count sane (~12-20 for most notebooks).
- Final cell writes a JSON report + uploads to the user's HF SAE repo where possible.

## Output schemas we care about

If your notebook emits a JSON that other tools consume, match the schema:

| Tool | Schema |
|---|---|
| Trace Theater | [`lib/trace-data.ts` TraceScenario](https://github.com/OpenInterpretability/web/blob/main/lib/trace-data.ts) |
| Circuit Canvas | [`lib/circuit-data.ts` CircuitData](https://github.com/OpenInterpretability/web/blob/main/lib/circuit-data.ts) |
| InterpScore leaderboard | [`lib/leaderboard.ts` LeaderboardEntry](https://github.com/OpenInterpretability/web/blob/main/lib/leaderboard.ts) |

## Review criteria

- ✅ Runs start-to-finish on the declared platform with user's own HF account
- ✅ Under the declared runtime
- ✅ Primary sources cited in header markdown
- ✅ JSON output validates against the schema (if applicable)
- ❌ Any hard-coded API keys, tokens, or Drive paths that aren't tutorial-safe

## PR template will check

- cell count + runtime + platform
- constraints checklist
- representative output

## Local validation before PR

```bash
python3 -c "import json; json.load(open('notebooks/NN_yours.ipynb'))"   # JSON valid
```

## Questions

[Open a Discussion](https://github.com/OpenInterpretability/notebooks/discussions) — especially if you're not sure where your notebook belongs in the ladder.
