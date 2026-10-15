# Project rules for DSA4213 Assignment 1 (NLP for Data Science)

## Project scope and your role (read this first)

This repository is one student's individual assignment (30% of the course), due **Fri 16 Oct 2026, 11:59 pm Singapore time**. The full brief is restated in `ASSIGNMENT.md` (source: `Assignment1_2026.pdf`). Read it before starting any task.

- **Part I (40 pts):** on one corpus (IMDB reviews, word-level), train and compare an n-gram model, an RNN, an LSTM and a causal Transformer (all done). Neural models use random trainable 100-d embeddings. Report test perplexity, training time, model size, inference throughput, loss curves, and generated text from the same prompts with matched decoding.
- **Part II (30 pts):** pick one Part I architecture (the LSTM, decision R1); compare random learned embeddings, own embeddings (Skip-gram/CBOW/GloVe) kept fixed, and public GloVe 100d kept fixed. Everything else identical.
- **Part III (20 pts):** frozen language-model representations feed a simple classifier; compare at least two representation choices (Part I LSTM vs Transformer, mean pooling, logistic regression; done).
- **Report (10 pts):** the user writes the PDF (at most 6 pages; `report/` holds assistant-drafted text written at the user's explicit request, for them to verify and rewrite); the repo must reproduce its results (`README.md`).

**Your job:** write and test the code, run what can run, record every result and decision, keep `RESULTS.md` and `MY_TASKS.md` current, and tell the user what they must do. **Not your job:** writing report text, interpreting results, choosing design values for the user (offer 2-3 options and wait), ticking boxes in `MY_TASKS.md`, or writing in the *My notes* blocks.

**Where things are:** `ASSIGNMENT.md` (the brief), `MY_TASKS.md` (what the user must do, with deadlines), `RESULTS.md` (all results, auto-built), `DECISIONS.md` (every design choice and its status), `results/RESULTS_LOG.md` (run log), `TOOL_USE.md` (what each task changed), `configs/` (one config per run), `src/` (scripts), `tests/`, `report/` (draft report text and `VERIFY.md`, only when the user asks), `README.md` (how to reproduce). Run-name convention: Part II runs start with `p2_`, Part III runs with `p3_`.

**Be honest about progress:** `RESULTS.md` and `MY_TASKS.md` show what exists; never describe code or runs that do not exist as done.

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
9. **COLAB-ONLY RUNS. Never run these on this machine; they run only in Colab on a GPU:** training any neural language model (RNN, LSTM, Transformer, including the frozen-embedding LSTM runs); `benchmark.py` (timings must come from the same Colab GPU for all models); generation from neural models; `extract_features.py`; downloading GloVe. For these, write and fix the code, then: (1) optionally run a smoke test of under 1 minute on a tiny subset (a few batches, CPU) only to check the code runs; never treat its output as a result and never save it to `results/`; (2) add the exact command as a new labelled cell in `notebooks/run_colab.ipynb`, saving checkpoints and results to Google Drive; (3) add it to "Runs to do" in `MY_TASKS.md` with the expected runtime, and end the reply with `NEW FOR YOU: run <name> in Colab`. Everything else (data scripts, n-gram, tests such as `test_causality.py`, word2vec, `classify.py`, figures, `update_results.py`) may run locally. Any Colab-only run found on a CPU record is listed in `MY_TASKS.md` as needing a Colab re-run (`update_results.py` does this automatically); never delete the local result.
