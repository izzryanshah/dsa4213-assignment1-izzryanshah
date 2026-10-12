# Project rules for DSA4213 Assignment 1 (NLP for Data Science)

These rules apply to every task in this repository, not just the one that created them.

- The user is learning PyTorch from scratch. Explain every new file and any non-obvious line in plain language. Prefer simple, readable code over clever code.
- Use word-level tokenization everywhere, with one shared vocabulary that includes `<unk>` and `<pad>`.
- Train/validation/test splits are created once, saved to disk, and reused from then on. The test set is only used for final evaluation, never for tuning.
- Set and record a random seed in every script.
- All neural models use embedding dimension 100, to match GloVe 100d.
- Language-model losses must ignore padding tokens.
- Language models may only use preceding tokens to predict each target (no leakage from future tokens).
- Every script reads its settings from a config file (see "Record-keeping rules" below) and saves its results to `results/` as JSON or CSV.
- Never write report text or interpret results for the user — that is their job.
- After each task, append a short dated entry to `TOOL_USE.md` describing what was written or changed.

## Record-keeping rules

These apply to every future task.

1. **No hard-coded design values in scripts.** Every run reads one config file from `configs/` (one per run) holding: architecture, layers, hidden size, embedding setting, context length, learning rate, batch size, epochs, dropout, vocabulary size, subsample size and seed (script-specific extras go in the config's `options` block). Scripts take `--config`, and command-line arguments may override it. Helpers: `src/runconfig.py`.
2. **Every run saves `results/<run>.json`** with: the full config actually used, the seed, the git commit hash, hardware (GPU name, or CPU name when there is no GPU), date, per-epoch losses and final metrics.
3. **Generation output files store their decoding settings** (temperature, top-k, max tokens, seed) alongside the text.
4. **Never make a design choice silently.** Before choosing any design value, present the user 2-3 options with their trade-offs and wait for their choice. Then append to `DECISIONS.md`: date, decision, value and where it is set (file and line), who chose it, options considered, the user's reason. A choice made without an explicit user instruction must be marked `NEEDS REVIEW`.
5. `src/summarise_runs.py` collects every `results/*.json` into `results/all_runs.csv`, one row per run with its settings and metrics. Re-run it after adding runs.
6. **After every run, append an entry to `results/RESULTS_LOG.md`:** date, script, exact command, key numbers and file path (numbers only; no interpretation).
7. **After every task or run:** (a) run `python3 src/update_results.py --config configs/update_results.json` to rebuild `RESULTS.md` and `results/figures/`; (b) append the entry to `results/RESULTS_LOG.md`; (c) tell the user in 2-3 lines what is new in `RESULTS.md`. Facts only, no interpretation. Never edit anything between the `my-notes` marker lines in `RESULTS.md`; that text belongs to the user.
8. **`MY_TASKS.md` is the one list of what the user must do themselves** (rebuilt by `src/update_results.py`). After every task, run `update_results.py` so it is current. If the task added anything that needs the user (a decision, a run to start, a file to check, notes to write), end the reply with a line starting `NEW FOR YOU:` naming it. Never tick boxes in `MY_TASKS.md` and never edit text the user wrote there.
