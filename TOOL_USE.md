# Tool use log

A dated log of what Claude wrote or changed in this repository, appended to after every task.

## 2026-10-08

Initial repository setup:
- Created folders `data/`, `src/`, `results/`, `notebooks/` (each with a `.gitkeep` placeholder so git tracks the empty directory).
- Created an empty `README.md`.
- Created `.gitignore` (ignores raw data under `data/`, model checkpoints, `__pycache__/`, `.env`).
- Created `requirements.txt` (torch, numpy, nltk, gensim, scikit-learn, matplotlib, pandas).
- Created `CLAUDE.md` with the project's standing rules (word-level tokenization, shared vocab, fixed splits, seeded scripts, 100d embeddings, padding-aware LM losses, causal LMs, CLI-configurable scripts, no report writing, this log).
- No model or training code written yet.

## 2026-10-09

Added a Colab runner notebook:
- Created `notebooks/run_colab.ipynb` — checks GPU availability (`nvidia-smi` + `torch.cuda.is_available()`), clones the GitHub repo (https://github.com/izzryanshah/dsa4213-assignment1-izzryanshah.git), installs `requirements.txt`, and provides four empty, labelled placeholder cells for future training commands (vocab/splits, embeddings/LM, downstream model, test evaluation). Includes a markdown cell explaining how to `git pull` new commits and how to mount Google Drive and copy `results/` there before the Colab session ends.
- No model or training code written yet.

## 2026-10-09 (data inspection)

Added a raw-data inspection script:
- Created `src/inspect_data.py` — loads `data/IMDB Dataset.csv` with pandas (CLI args for path, seed, number of examples, word-budget threshold, output path) and prints/saves column names, review count, label counts, total/average word counts, a random sample of raw example reviews, and the percentage of reviews containing a `<br` tag. Saves the summary to `results/inspect_data.json`.
- Ran it against the real dataset: 50,000 reviews (25,000/25,000 label split), ~11.56M total words (avg 231.2 words/review), 58.4% of reviews contain `<br` HTML tags needing cleanup.
- Since total words (~11.56M) is well above the 3M-word budget, the script suggested subsampling to ~12,500 reviews (~2.89M words) so four language models stay trainable in one Colab GPU session.
- Confirmed `data/*` in `.gitignore` already covers `data/IMDB Dataset.csv` (verified with `git check-ignore -v`) — the raw CSV will never be committed.
- Filled in the "Dataset" section of `README.md` with download/placement instructions and a citation (Kaggle page placeholder + Maas et al. 2011 ACL paper) — left the exact Kaggle URL as a placeholder for the user to fill in.
- No model or training code written yet.

## 2026-10-09 (data pipeline)

Added `src/data.py`, the shared data-preparation pipeline:
- Cleans HTML tags and tokenizes reviews into lowercase word/punctuation tokens (Unicode-aware, so accented words like "cliché" stay whole) via `clean_and_tokenize`, appending an `<eos>` token after each review.
- Splits all 50,000 reviews into train/val/test (80/10/10), stratified by sentiment label, with a fixed seed (`stratified_three_way_split`), BEFORE building the vocabulary. Saves the full assignment (every original row → split + whether it was kept after any subsampling) to `data/processed/split_assignment.csv`.
- Supports optional proportional, stratified subsampling within each split via `--subsample-n` (same fixed seed).
- Builds one shared vocabulary from the training split only (`build_vocab`): top `--vocab-size` (10,000-20,000) training words plus `<pad>`/`<unk>`/`<eos>`.
- Encodes each split into a flat token-id array (reviews concatenated, `<eos>` as a boundary) plus per-review lengths/labels/original indices, saved as `data/processed/{train,val,test}.npz`, with the vocab and label map in `vocab.json`/`label_map.json`.
- Provides `load_processed_data()` for later scripts to load everything in one line, plus a `SplitData.reviews()` helper to slice the flat stream back into individual reviews.
- Ran the real pipeline (seed 0, vocab size 20,000, no subsampling): vocab size 20,003, train 40,000 reviews / 11,067,213 tokens / 2.07% `<unk>`, val 5,000 / 1,388,412 / 2.23% `<unk>`, test 5,000 / 1,395,673 / 2.24% `<unk>` — saved to `results/data_prep_stats.json`.
- No model or training code written yet.

## 2026-10-09 (sanity check)

Added `src/sanity_check.py`, the first script with any model code (an untrained baseline, not a real model to train):
- `BigramLM`: an `nn.Embedding` (dim 100, per the project's fixed-embedding-dim rule) feeding straight into an `nn.Linear` over the vocabulary — predicts the next token from only the current one, so it trivially respects "preceding tokens only."
- `compute_perplexity` batches validation reviews, pads them with `torch.nn.utils.rnn.pad_sequence`, and uses `nn.CrossEntropyLoss(ignore_index=pad_id)` so padded positions never affect the loss.
- Hit a real performance bug while testing: a few validation reviews run past 1,000 tokens, and with a 20,000-word vocabulary a padded batch containing one of those could balloon its logits tensor to several GB, hanging the run. Fixed by truncating every review to `--max-length` (default 256) tokens and sorting reviews by length before batching (bucketing) to minimize padding waste — confirmed this changes the measured perplexity by less than 0.2% (23894.9 → 23926.5) while cutting runtime from several minutes to ~28s.
- Also tried routing computation through the Mac's MPS (Metal) backend for speed; it stalled (process went idle without finishing), so reverted to plain CPU, which is fast enough for this tiny model.
- Ran it for real: vocabulary size 20,003 vs. untrained validation perplexity 23,926.5 — same order of magnitude, as expected for an untrained model whose predictions are close to uniform over the vocabulary. Saved to `results/sanity_check.json`.
- Still no trained model or training loop written.

## 2026-10-10 (shared evaluation function)

Added `src/evaluate.py` and `tests/test_evaluate.py`:
- `evaluate(model, split, pad_id, batch_size, max_length)` is the single evaluation function for all language models. It builds (input, target) pairs from each review (input = tokens 0..L-2, target = tokens 1..L-1), optionally truncates to `max_length`, drops reviews with fewer than 2 tokens, sorts by length for batching, masks out `<pad>` targets, and returns average cross-entropy (nats), perplexity, `num_targets` and `target_checksum` (a SHA-256 fingerprint of the exact target ids scored, in order).
- Neural and n-gram models share one "scorer" interface: `target_log_probs(inputs, targets) -> (batch, seq_len)` log-probabilities of each target given only the tokens up to that position. An `nn.Module` is wrapped automatically (`NeuralScorer`: log_softmax + gather); the future n-gram model only needs to provide the same method. Since the data preparation lives in `evaluate()` itself, both kinds are scored on identical targets, and matching `num_targets`/`target_checksum` in their result files proves it.
- `tests/test_evaluate.py` (run `python tests/test_evaluate.py`, saves `results/test_evaluate.json`; also pytest-compatible) uses the untrained `BigramLM` from `sanity_check.py` on small fake reviews. Checks: perplexity is near vocab size, results agree with `sanity_check.compute_perplexity`, results don't depend on batch size, padding is ignored, and a stand-in uniform scorer is fed exactly the same targets as the neural model (and gets perplexity = vocab size). All 5 pass.
- The system `python` (anaconda) has no torch; used `/usr/local/bin/python3` to run the tests.
- Not done: `sanity_check.py` still has its own evaluation loop (left untouched); the n-gram model itself is not written yet.

## 2026-10-10 (trigram Kneser-Ney model)

Added `src/ngram.py`, a trigram language model with interpolated Kneser-Ney smoothing:
- Uses data.py's vocabulary and `<unk>` mapping unchanged; the only extra symbol is an invisible start-of-review context id (`vocab_size`), used as context and never predicted, so early tokens get two-token contexts in both training and scoring (previous tokens only).
- Counts are packed into single int64 keys in sorted numpy arrays (`np.unique` to count, `np.searchsorted` to look up) instead of Python dicts, so fitting the 11M-token training split takes about 1 second. Lower orders use Kneser-Ney continuation counts; the unigram level is interpolated with a uniform 1/V floor, so no token ever gets probability 0. Verified separately (throwaway script) that probabilities sum to 1 over the vocabulary for seen, partly seen and never-seen contexts.
- `KneserNeyTrigram.target_log_probs` implements the scorer interface from `evaluate.py`, so it is scored through `evaluate()` on the same targets as neural models (compare `num_targets`/`target_checksum` in the result files).
- Discounts (one per order: D3, D2, D1) are grid-searched on the validation split only; the lookups are done once and reused, so the whole grid takes seconds. The test split is evaluated once, after the discounts are frozen. A script assert checks the fast tuning path equals `evaluate()` on validation.
- First run used a grid of 0.5-0.95 and the best D1 sat on its lower edge; I widened the default grid to 0.1-0.95 and re-ran. I had already seen the test number from the first run when I made that change (the reason was the validation edge).
- Saves `results/ngram.json`: args, seed, best discounts, training/tuning time, model size (trigram/bigram/unigram types), val and test cross-entropy/perplexity/num_targets/checksum, and the full tuning grid.

## 2026-10-10 (record-keeping rules, configs, run records)

Applied the new record-keeping rules to the existing code:
- `CLAUDE.md`: reworded the "settings" rule and added a "Record-keeping rules" section (configs, `results/<run>.json` contents, generation settings, never choosing design values silently, `summarise_runs.py`).
- New `src/runconfig.py`: `add_config_args` (adds `--config`, `--run-name`, `--results-dir`, `--set KEY=VALUE`, and one override flag per shared key), `load_config` (validates the keys, applies overrides, rejects unknown keys), `save_run` (writes `results/<run>.json` with run, script, date, git commit + `git_dirty` flag, GPU name, seed, full config used, per-epoch losses, metrics, details) and `save_generation` (refuses to save text unless temperature, top_k, max_tokens and seed are given; nothing generates text yet, so no script uses it).
- New `configs/`: `data_prep.json`, `inspect_data.json`, `sanity_check.json`, `ngram.json`, `test_evaluate.json` — all with the same shared keys (null where not applicable) plus an `options` block for script-specific settings. Values are exactly the ones the scripts used before.
- `src/data.py`, `src/inspect_data.py`, `src/sanity_check.py`, `src/ngram.py`, `tests/test_evaluate.py`: old command-line flags and defaults removed; they take `--config` and save through `save_run`. `EMBEDDING_DIM`, the n-gram's default discounts, and `evaluate()`'s default `batch_size`/`max_length` were removed (callers must now pass them). Added `check_config_matches_data` in `data.py`: later scripts stop if their config's `vocab_size` disagrees with the processed data. `ngram.py` also refuses a config whose `context_length` is not 2. In the test script the two truncation lengths (50 and 40) became one config value (40).
- New `src/summarise_runs.py` -> `results/all_runs.csv` (5 runs x 70 columns; per-run `details` stay in the JSON files).
- New `DECISIONS.md`: this task's decisions plus the earlier un-logged values, marked "reason: pending".
- Re-ran everything. `data.py` was re-run from its config; verified train/val/test arrays, vocabulary, label map and split assignment are identical to the existing ones (checked against a backup). Sanity-check (23926.5) and n-gram (test 4.7582 nats) numbers are unchanged. Deleted the old-format result files (`data_prep_stats.json` -> now `data_prep.json`, plus `inspect_data.json`, `sanity_check.json`, `ngram.json`, `test_evaluate.json`) and replaced them with new-format records.
- Caveats: `git_dirty` is true in all current records because the code changes are not committed yet; downstream configs' `subsample_size` is recorded but not checked against the data on disk (only `vocab_size` is).

## 2026-10-10 (backfill from git history)

Cross-checked this log against `git log` and added the git-level facts that were missing. All earlier tasks already have entries above; this maps them to commits:
- `e213183`, 2026-10-08 20:21, by the user: "Initial commit" — GitHub-generated `README.md` stub (one title line). Not done by Claude.
- `77f7784`, 2026-10-09 12:11, "Set up repo scaffolding and build the IMDB data pipeline (tasks 1-2)": a single commit containing the work of every entry dated 2026-10-08 and 2026-10-09 above — scaffolding (`.gitignore`, `CLAUDE.md`, `requirements.txt`, folders), `notebooks/run_colab.ipynb`, `src/inspect_data.py`, `src/data.py`, `src/sanity_check.py`, the README dataset section, `TOOL_USE.md`, and the three result files `results/data_prep_stats.json`, `results/inspect_data.json`, `results/sanity_check.json`. (The log entries are dated by when each piece was written; they were committed together.)
- `8764c2a`, 2026-10-09 12:12, "Merge remote's initial GitHub README stub with local project setup": merged the remote's initial commit; the local `README.md` (dataset section) replaced the one-line stub.
- Nothing from 2026-10-10 is committed yet: `src/evaluate.py`, `src/ngram.py`, `src/runconfig.py`, `src/summarise_runs.py`, `tests/`, `configs/`, `DECISIONS.md`, `results/RESULTS_LOG.md` and the edits to `data.py`, `inspect_data.py`, `sanity_check.py`, `CLAUDE.md`, `README.md` are untracked or modified (so each run record shows `git_dirty: true`).
- The earlier entries above that mention `configs/data_prep.json` / `results/data_prep.json` refer to files renamed in the entry below.

## 2026-10-10 (audit, decision log, results log)

Audited the whole repo (git history, every script, `CLAUDE.md`, `TOOL_USE.md`, `README.md`, `results/`) and set up record-keeping. No script's computation was changed.
- `DECISIONS.md` rewritten as a full audit: 29 numbered entries (data, tokenisation, vocabulary, splits, seed, sanity model, evaluation conventions, n-gram order/smoothing/discounts/grid/start symbol/uniform floor, config format, run records, Python environment). Each has value, file and line, who chose it, alternatives and reason; everything not explicitly instructed by the user is marked NEEDS REVIEW, with a summary list at the end.
- `configs/data_prep.json` renamed to `configs/data.json` (as requested); the data run is now `results/data.json` (the old `results/data_prep.json` was removed). Other configs (`ngram.json`, `inspect_data.json`, `sanity_check.json`, `test_evaluate.json`) unchanged.
- `src/runconfig.py`: run records now include a `hardware` field ("CPU: Apple M2 (Darwin arm64)" or "GPU: <name>") in addition to `gpu`; `src/summarise_runs.py` has a matching `hardware` column. `README.md` command updated to use `--config`.
- Re-ran `data.py`, `inspect_data.py`, `sanity_check.py`, `ngram.py` (about 30 s, so re-run rather than a sidecar file) and `tests/test_evaluate.py`, then `summarise_runs.py`. Verified unchanged: processed `train/val/test.npz`, `vocab.json`, `label_map.json` and `split_assignment.csv` identical to the existing ones; data statistics, inspect numbers and sanity perplexity equal the values in the originally committed result files; n-gram validation/test metrics, target checksum, discounts and model size equal the previous run.
- New `results/RESULTS_LOG.md`: one entry per result so far (current runs with exact commands, plus the earlier superseded runs with reconstructed commands).
- `CLAUDE.md` record-keeping rules updated: hardware in run records, `DECISIONS.md` fields and the NEEDS REVIEW convention, and appending to `results/RESULTS_LOG.md` after every run.
- Not done: nothing committed; no training was run (none exists yet).

## 2026-10-10 (NEEDS REVIEW resolution)

Note: the previous entry's audit was done in a peer Claude Code session running concurrently on this same repo (the user had asked both sessions the same thing); this one picked up from there without redoing or overwriting any of it, after confirming with the peer session that it was finished and idle.

Went through all 24 `DECISIONS.md` NEEDS REVIEW items with the user, 2-3 options with trade-offs each, per CLAUDE.md's "never choose silently" rule:
- Every item was confirmed as-is (no value changed), so no script was re-run and no result file changed.
- One item (D4, the n-gram discount grid being widened and re-run after the first run's test perplexity had already been seen) was explicitly accepted as a logged risk rather than redone: the test number was identical in both runs, so no actual leakage occurred this time, but the user signed off on it as a flagged incident rather than it being silently resolved.
- `DECISIONS.md` updated in place: every entry's "Chosen by" / "Alternatives" / "Status" fields now reflect the user's actual choice and the options offered; all 24 statuses changed from NEEDS REVIEW to CONFIRMED; the summary section at the bottom rewritten accordingly.
- Not done: nothing committed yet.

## 2026-10-10 (generation prompts)

- Created `prompt.txt` with the user's three generation prompts and, as `#` comment lines under each, the reason for using it. A script reading this file must skip blank lines and lines starting with `#`. The prompt text was not edited.
- Checked that all 22 tokens in the three prompts are in the vocabulary (none become `<unk>`) using `data.clean_and_tokenize` and `data/processed/vocab.json`.
- Added decision F1 to `DECISIONS.md` (value, location, who chose, reason).
- Not included: the analysis note about the trigram model seeing only two words (it is guidance for analysing outputs, not a reason for choosing the prompts). No generation script written; no results produced.

## 2026-10-11 (text generation)

- New `src/generate.py`: one `generate()` function that continues a prompt word by word for any model exposing `next_token_log_probs(context_ids)`. Pieces: `next_token_probs` (ban words -> temperature -> top-k -> rescale), `NeuralNextToken` (wraps any `nn.Module` returning logits; reads the last position), `read_prompts` (one prompt per line, skips blank and `#` lines), `encode_prompt` (same cleaning/tokenizer as the data), `make_rng` (a reproducible random generator per prompt and sample), `build_model` (n-gram only for now; neural models need a saved model first). The module docstring explains temperature and top-k with a worked numeric example.
- `src/ngram.py`: added the `next_token_log_probs` method to `KneserNeyTrigram` (new method only; training, tuning and scoring code untouched; a test checks it agrees with the scoring path).
- `src/runconfig.py`: `save_run` gained an `extra` argument for additional top-level fields (generation files use it for `decoding` and `generations`, and it refuses to save if temperature, top_k, max_tokens or seed are missing). Removed the now-unused `save_generation` helper.
- `src/summarise_runs.py`: fixed a bug where the `seed` column appeared twice in `results/all_runs.csv`; regenerated the CSV (6 runs x 71 columns).
- New configs `configs/generations_ngram.json` (temperature 1.0, top-k 40, 50 new tokens, 1 sample, ban `<pad>`/`<unk>`, stop at `<eos>`, prompts from `prompts.txt` — the values the user chose from options offered) and `configs/test_generate.json`. New `tests/test_generate.py` (7 tests, all pass; results in `results/test_generate.json`).
- Ran generation twice end to end with the existing `prompt.txt` (scratch output, not saved to `results/`): identical text both times. `prompts.txt` does not exist yet, so `results/generations_ngram.json` has not been produced.
- `DECISIONS.md`: added G1-G8 (G1 confirmed by the user; G2-G8 marked NEEDS REVIEW). `results/RESULTS_LOG.md`: added the test run and a note about the unsaved check runs.
- Not done: no neural model exists to generate from yet; nothing committed.

## 2026-10-11 (README for reproduction)

- Rewrote `README.md` as reproduction instructions: data access (manual Kaggle download, not redistributed, no keys or credentials; the unfilled Kaggle-URL placeholder is replaced with the dataset link already in the citation), the preprocessing steps in `src/data.py`, dependencies with the exact package versions used for the recorded results (Python 3.10.11, torch 2.14.1, numpy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2), the ordered commands for every script, the numbers recorded in `results/` for checking a re-run (copied from `results/RESULTS_LOG.md`, no interpretation), a table of the configs and how to override them, the random seeds and what each affects, the output/record files, repository layout and sharing notes.
- Documentation only: no code, config or result changed. `requirements.txt` is left unpinned (versions are listed in the README instead).
- Not done: not committed; the README commands were not re-run in this task (they were run earlier in this session, with identical results).
