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

## 2026-10-11 (n-gram generation run)

- `prompts.txt` (the user's file; same three prompts as the earlier `prompt.txt`, which the user renamed) was checked: all prompt words are in the vocabulary. Ran `src/generate.py --config configs/generations_ngram.json`; saved `results/generations_ngram.json` (decoding settings + texts), refreshed `results/all_runs.csv` (7 runs).
- Docs updated to the new file name: `README.md` (prompts section), `DECISIONS.md` (F1 location, G2 status), `results/RESULTS_LOG.md` (new entry; replaced the "no generation saved yet" note).
- No code or config changed. Not committed (the `prompt.txt` -> `prompts.txt` rename and the new result are uncommitted, so the run record has `git_dirty: true`).

## 2026-10-11 (run_part1_ngram.sh)

- Added `run_part1_ngram.sh`: runs, in order, `data.py`, `inspect_data.py`, `sanity_check.py`, `ngram.py`, `test_evaluate.py`, `test_generate.py`, `generate.py` (n-gram) and `summarise_runs.py`, each with its config from `configs/`, stopping at the first error (`set -euo pipefail`). Uses `python3` (override with `PYTHON=...`); checks that the dataset CSV and `prompts.txt` exist before starting.
- Only syntax-checked (`bash -n`); the script was not executed, so no results were overwritten. No design values added; it only calls the existing commands from `README.md`. Not committed.

## 2026-10-11 (training-window Dataset and DataLoader)

- Added to `src/data.py`: `WindowDataset` (cuts a split's token stream into non-overlapping windows of `context_length` tokens; target = input shifted by one), `make_dataloader` (seeded shuffle for train), and a `--show-batch` mode that prints one batch's shapes and saves `results/show_batch.json`. `torch` is now imported at the top of `data.py`.
- Added `configs/show_batch.json` (context length 64, batch size 64, seed 0) and `tests/test_windows.py` (tiny made-up stream: window count, shift-by-one, no overlap, seeded shuffle repeatable).
- Choices (context length, stride, tail, batch size) were chosen by the user and logged in `DECISIONS.md` section H; H5 (shuffle/last batch) is an assistant default marked NEEDS REVIEW. Logged the run in `results/RESULTS_LOG.md`; refreshed `results/all_runs.csv`.
- Not committed.

## 2026-10-11 (training script)

- Added `src/train.py`: one script for all neural LMs, chosen by `--model` (table `MODELS`, only `bigram` for now). Seeded; Adam with gradient-norm clipping; per-epoch training loss and validation loss (validation via `evaluate.py`); best-validation checkpoint saved to `checkpoints/<run>.pt`; early stopping; one test evaluation at the end with the best checkpoint; `results/<run>.json` with epoch losses, training seconds, parameter count, hardware/GPU. The logits are reshaped `(B, T, V)` -> `(B*T, V)` and targets `(B, T)` -> `(B*T,)` before `CrossEntropyLoss(ignore_index=PAD_ID)`.
- Added `configs/train_bigram.json`, `configs/test_train.json` and `tests/test_train.py` (tiny synthetic data: loss decreases, early stop, checkpoint is the best epoch, repeatable under the same seed, parameter count).
- Values chosen by the user are in `DECISIONS.md` section I; I5 (and the eval batch size in I4) are assistant defaults marked NEEDS REVIEW.
- The full bigram training run was NOT started (about 16 minutes per epoch on CPU). Not committed.

## 2026-10-12 (automatic results document)

- Added `src/update_results.py` and `configs/update_results.json`: rebuild `RESULTS.md` (sections Data, Part I, Part II, Part III, Settings, Checks) and `results/figures/*.png` from `results/*.json`, `DECISIONS.md` and `results/RESULTS_LOG.md`. Per run it shows command, date, git commit, hardware, settings, key numbers, figures; generated text is side by side per prompt; loss curves are drawn for every trained run. Each section ends with a "My notes" block between marker comments that is copied back unchanged on every rebuild.
- Checks section: untrained perplexity vs vocabulary size, test set used only after tuning/fit (static `ast` scan), padding ignored in the loss, causality test and frozen-embedding check (both "NOT YET AVAILABLE" until those tests/runs exist).
- `src/runconfig.py`: `save_run` now also stores the exact `command` in each run record.
- `notebooks/run_colab.ipynb`: new Drive-setup cell defining `save_results()` (rebuilds RESULTS.md, copies `results/` and `RESULTS.md` to Drive, restores them at session start); each training cell is followed by a `save_results()` cell; old end-of-notebook mount/copy cells removed.
- `CLAUDE.md`: new rule 7. `DECISIONS.md`: new section J (J1-J6, all NEEDS REVIEW). Logged in `results/RESULTS_LOG.md` (also added the missing `train_bigram` entry). Not committed.

## 2026-10-12 (full bigram training run logged)

- Ran `python3 src/train.py --config configs/train_bigram.json --model bigram` (started 2026-10-11 21:49, record written 2026-10-12 02:26) -> `results/train_bigram.json`, checkpoint `checkpoints/train_bigram.pt`. Key numbers are in `results/RESULTS_LOG.md` (entry "bigram language model training"); I corrected that entry's command line to the exact command above.
- Re-ran `src/summarise_runs.py` (`results/all_runs.csv`, 10 runs). `RESULTS.md` already lists `train_bigram`.
- Not done: `python3 src/update_results.py --config configs/update_results.json` failed with `ModuleNotFoundError: No module named 'my_tasks'` (line 49 of the untracked `src/update_results.py`), so `RESULTS.md` was not rebuilt by me. Not committed.

## 2026-10-12 (MY_TASKS.md)

- Added `src/my_tasks.py` (called by `src/update_results.py`, which now also rebuilds `MY_TASKS.md` on every run) and new settings in `configs/update_results.json` (deadline, timezone, schedule, requirement list, run-script rules).
- `MY_TASKS.md` has six sections: Waiting on me now, Decisions to review, Runs to do, Results to interpret, Report and submission, Failed checks. Every item is an unticked checkbox; the script never writes a tick, and keeps ticks (matched by hidden `<!--id:...-->` tags) and an "Added by me" block per section. Overdue items (planned day before today, Singapore time) are listed first.
- `update_results.py`: `parse_decisions` now also returns each decision's options and line number; `render_checks` also returns the check results. `CLAUDE.md` rule 8 (`NEW FOR YOU:` line). `DECISIONS.md` section K (K1-K3, NEEDS REVIEW). Not committed.

## 2026-10-12 (assignment brief read; parts corrected)

- Read `Assignment1_2026.pdf`. Part I = n-gram/RNN/LSTM/Transformer, Part II = embedding ablation, Part III = downstream classifier; the earlier part mapping (J1) was wrong and is revised: Parts I and II each get a perplexity/cost table (parameters, training seconds, inference tokens/s, hardware) and side-by-side generated text. Run-name convention: `p2_*`, `p3_*`.
- `src/update_results.py`: two new checks (all models scored on identical test targets; generation runs use matched decoding and prompts). `src/my_tasks.py`: requirement kinds `architectures` and `check`; `extra_items` replace `planned_runs`. `configs/update_results.json`: the 23 requirements from the brief and the not-yet-possible steps (blocked runs, Colab setup, GloVe download).
- `notebooks/run_colab.ipynb`: `save_results()` also copies `checkpoints/` to and from Drive (Part III needs the trained models).

## 2026-10-12 (RNN and LSTM models)

- Added `src/models.py`: `RNNLanguageModel` (tanh) and `LSTMLanguageModel`: embedding (100, random, trainable, `<pad>` row zero) -> dropout -> recurrent layers -> dropout -> linear. The module docstring explains the difference between them.
- `src/train.py`: registered `rnn` and `lstm` in `MODELS` (builder reads `embedding_dim`, `hidden_size`, `layers`, `dropout`, `embedding_init` from the config and stops on nulls). The `--hidden-size`, `--layers` and `--dropout` options already existed through `runconfig.add_config_args`, so no new flags were needed.
- Added `configs/train_rnn.json`, `configs/train_lstm.json` (hidden 256, 2 layers, dropout 0.3), `configs/test_models.json`, `tests/test_models.py` (shape, no future leakage, `<pad>` row, trainable embedding).
- Smoke test: 1 epoch of each model on a 60/40/40-review slice copied to a scratch folder (saved splits untouched), plus one run with command-line overrides. All ran. Logged in `results/RESULTS_LOG.md`. Choices in `DECISIONS.md` section L (L3, L4 NEEDS REVIEW).
- No full training run was started. Not committed.

## 2026-10-12 (commit, push and Colab notebook)

- Committed (72fd68d) and pushed everything from the training, model and results-tooling work. Left out of the commit on purpose: `MY_TASKS.md` (personal task list; generated by `src/my_tasks.py`, which is committed). `RESULTS.md` is committed (it contains the user's "My notes" text).
- `notebooks/run_colab.ipynb`: `REPO_URL` set to the HTTPS URL of the origin repo; replaced the template placeholder cells with: copy `data/processed/` from Drive (`DATA_DRIVE_DIR`, an assistant-chosen default path, change if different), train the LSTM (`configs/train_lstm.json`), train the RNN (`configs/train_rnn.json`), each followed by `save_results()`; removed the separate "evaluate on test" section (`train.py` scores test once at the end). Not run on Colab; not yet committed or pushed.
- Follow-up: set both `DRIVE_DEST` (results, checkpoints, RESULTS.md) and `DATA_DRIVE_DIR` (`<folder>/processed`) in the notebook to the user's Drive folder `MyDrive/Education/University/dsa4213-assignment1`. The `processed` subfolder name is an assistant guess. Not committed.

## 2026-10-12 (Colab RNN and LSTM results logged)

- Unzipped the user's Drive download into a scratch folder, read the records and copied only `results/train_lstm.json` and `results/train_rnn.json` into `results/` (the zip's other files were not copied over the local ones; its `RESULTS_LOG.md` was identical to the local one).
- Added two entries to `results/RESULTS_LOG.md` (commands taken from `notebooks/run_colab.ipynb`), re-ran `src/summarise_runs.py` and `src/update_results.py` (rebuilds `RESULTS.md` and `results/figures/`). No code changed. Not committed.

## 2026-10-12 (ASSIGNMENT.md)

- Added `ASSIGNMENT.md`: the brief's requirements and scope (parts, points, constraints, submission, grading, integrity rules), restated from `Assignment1_2026.pdf`. No code changed.

## 2026-10-12 (project scope for future agents)

- `CLAUDE.md`: new first section "Project scope and your role": the assignment's three parts and report, the agent's job versus the user's, a map of the key files, the run-name convention, and what exists so far. No code changed.

## 2026-10-12 (Colab-only rule)

- `CLAUDE.md` rule 9 (Colab-only runs: neural training, `benchmark.py`, neural generation, `extract_features.py`, GloVe download); replaced the dated "what exists" line with a pointer to `RESULTS.md`/`MY_TASKS.md`.
- Checked what ran locally: only `train_bigram` (neural training on CPU: Apple M2, 16,610 s). `train_rnn` and `train_lstm` ran on a Tesla T4. No neural generation, benchmark, feature extraction or GloVe download exists locally. Nothing deleted.
- `src/my_tasks.py` + `configs/update_results.json`: any Colab-only run whose record shows a CPU is listed in `MY_TASKS.md` as needing a Colab re-run (new run name with suffix `_colab`, local result kept); `done_if_result` drops finished static items (RNN, LSTM); Colab-only notes added to the benchmark, generation, GloVe, feature-extraction and Part II items; removed the outdated Colab-setup item (the repo URL is set).
- `notebooks/run_colab.ipynb`: new cells "4. Re-run the bigram baseline" (`--run-name train_bigram_colab`) followed by `save_results()`.

## 2026-10-12 (Transformer language model)

- `src/models.py`: new `TransformerLanguageModel` (token embedding 100-d, learned positional embedding, N pre-norm decoder blocks via `nn.TransformerEncoder` with a causal mask built by `torch.triu`, final LayerNorm, linear output). The mask is explained line by line in the code.
- `src/train.py`: `build_transformer` and a `transformer` entry in `MODELS`; `num_heads` and `max_positions` are read from the config's `options`.
- `configs/train_transformer.json` (new); `configs/test_models.json` gained `num_heads`, `max_positions`.
- `tests/test_models.py`: now also tests the Transformer (output shape, no future leakage, `<pad>` row zero, trainable embedding, gradients). Passed.
- `notebooks/run_colab.ipynb`: new "4. Train the Transformer" cells; bigram re-run renumbered to 5.
- `configs/update_results.json`: removed the "tell Claude to build the Transformer" item; updated the Transformer run command.
- `DECISIONS.md`: new section M (M1, M2 confirmed; M3, M4 NEEDS REVIEW).
- Smoke test (CPU, 40 fake reviews, scratch folder, nothing saved to `results/`): ran through `fit()`, loss went down.

## 2026-10-13 (Colab results imported)

- Copied `results/train_transformer.json` and `results/train_bigram_colab.json` from the Colab results zip (scratch extraction, compared against local files first; only these two were new). No code changed.
- Ran `update_results.py` (RESULTS.md, figures, MY_TASKS.md rebuilt); added two entries to `results/RESULTS_LOG.md`.

## 2026-10-13 (causality test)

- `tests/test_causality.py` (new): for every model class in `src/models.py` (found automatically) and every position t, replaces the token at t and checks the softmax distributions at positions before t are unchanged (and that position t does change). Saves `results/test_causality.json`.
- `configs/test_causality.json` (new); `DECISIONS.md` N1 (NEEDS REVIEW).
- Checked the test can fail: with the Transformer's mask replaced by "mask nothing" (scratch script, nothing saved) it fails at position 1.

## 2026-10-13 (benchmark)

- `src/benchmark.py` (new): loads the best checkpoint of the RNN, LSTM and Transformer (architecture rebuilt from the saved training config) and refits the n-gram; times `evaluate()` over the whole test set (batch 64, context 256) with untimed warm-up passes and `torch.cuda.synchronize()` before and after each timed pass, 3 repeats; checks all models scored identical targets; writes `results/benchmark.json` and `results/part1_costs.csv`. The docstring explains why warm-up and synchronisation matter.
- `configs/benchmark.json` (new); `notebooks/run_colab.ipynb`: new "5. Benchmark" cells (bigram re-run renumbered to 6); `configs/update_results.json`: removed the hand-written inference task (the auto-generated "Run `benchmark`" task replaces it); `DECISIONS.md` section O (O1 confirmed, O2 NEEDS REVIEW).
- Smoke test only (CPU, 40 test reviews, random-weight checkpoints made in a scratch folder, output not in `results/`): all four models ran and scored identical targets. Not a result. The real run is Colab-only (CLAUDE.md rule 9).

## 2026-10-13 (benchmark results imported)

- Copied `results/benchmark.json` and `results/part1_costs.csv` from the Colab results zip (scratch extraction, compared against local files first; only these two were new). No code changed.
- Ran `update_results.py`; added a `results/RESULTS_LOG.md` entry.

## 2026-10-13 (matched generation for all four models)

- `src/generate.py`: `build_neural` loads the RNN, LSTM or Transformer (architecture from the training record, best checkpoint, parameter-count check); `build_model` now routes to it. Decoding code unchanged.
- `configs/generations_{rnn,lstm,transformer}.json` (new): same decoding values as `configs/generations_ngram.json`.
- `src/combine_generations.py` + `configs/combine_generations.json` (new): checks all four `results/generations_<model>.json` share decoding settings, prompts and samples, then writes `results/part1_generations.md` grouped by prompt. No commentary.
- `notebooks/run_colab.ipynb`: new "6. Generate text" cells (bigram re-run is now 7). `DECISIONS.md` section P (P1 confirmed, P2 NEEDS REVIEW).
- Smoke test only (CPU, random-weight checkpoints in a scratch folder, output not in `results/`): the three loaders and the combiner ran; the combiner refused a file whose temperature differed. Not a result. The real neural generation is Colab-only (CLAUDE.md rule 9).
- `configs/update_results.json`: removed the hand-written generations task (auto-generated "Run `generations_<model>`" and "Run `combine_generations`" tasks replace it).

## 2026-10-13 (generation results imported)

- Copied `results/generations_{rnn,lstm,transformer}.json`, `results/combine_generations.json` and `results/part1_generations.md` from the Colab results zip (scratch extraction, compared against local files first; only these five were new). Re-ran the combiner locally into a scratch folder: output identical to the Colab file. No code changed.
- Ran `update_results.py`; added a `results/RESULTS_LOG.md` entry.

## 2026-10-13 (own Skip-gram embeddings)

- `src/train_word2vec.py` + `configs/word2vec_cost.json` (new): trains gensim Skip-gram (dim 100, seed 0, window 5, min_count 1, epochs 10) on the TRAINING split only, one sentence per review; times vocabulary build and training separately; writes `data/embeddings/w2v_100d.npy` (shape 20003 x 100, row i = vocabulary id i, `<pad>` row zeros; stops if any other word has no vector) and `results/word2vec_cost.json`.
- Ran locally (allowed: word2vec is not Colab-only). Sanity check only: nearest neighbours of "good", "terrible", "actor", "boring" are sensible. gensim printed many "Exception ignored in our_dot_float" lines (not investigated; output looks fine). Reproducibility with a fixed seed was not tested by a second run.
- `DECISIONS.md` section Q (Q1 confirmed, Q2 NEEDS REVIEW). Ran `summarise_runs.py` and `update_results.py`.

## 2026-10-13 (Part II architecture)

- `DECISIONS.md` section R: R1 records the user's choice of the LSTM for Part II. No code changed.

## 2026-10-13 (word2vec: coverage and missing-word rule)

- `src/embedding_utils.py` (new): `fill_missing_rows` (random normal at the real vectors' scale, `<pad>` zero, seeded) and `coverage` (% of vocabulary types and training tokens with a real vector). Shared so the GloVe script builds its table the same way.
- `src/train_word2vec.py`: now reports coverage before filling, fills any word without a vector with that rule (previously it stopped), and saves the coverage and fill details to `results/word2vec_cost.json`. Trains on the same training token stream as the LMs (`train.npz`; `<unk>` is a token; no subsample configured).
- Re-ran locally: coverage 20,002/20,002 types (100%), 100% of tokens, 0 rows filled; vectors identical to the previous run (same seed, one worker). Fill function checked on a small synthetic example only.
- `DECISIONS.md`: Q2 confirmed (option A), R2 added. Ran `update_results.py`.

## 2026-10-13 (public GloVe loader)

- `src/load_glove.py` + `configs/glove_coverage.json` (new): reads `glove.6B.100d.txt`, keeps only words in our vocabulary, builds a (20,003 x 100) table in vocabulary order, reports coverage (% of types and % of training tokens, with and without `<unk>`/`<eos>`), fills missing rows with the shared R2 rule, lists the 20 most frequent missing words, and saves `data/embeddings/glove_100d.npy` and `results/glove_coverage.json`.
- Not run on real GloVe: downloading GloVe is Colab-only (CLAUDE.md rule 9) and no file is on this machine, so there is no real coverage figure yet. Smoke test only, on a made-up GloVe-format file in a scratch folder (60% of words): the loader, coverage, fill and top-missing list ran; output not in `results/` or `data/embeddings/`. Not a result.
- `notebooks/run_colab.ipynb`: new section 8 (download, `load_glove.py`, copy to Drive, restore `w2v_100d.npy`, `save_results()`).
- `README.md`: new section "Public GloVe vectors" with download instructions and the citation. `configs/update_results.json`: the GloVe task now has its command and time. `DECISIONS.md`: S1 NEEDS REVIEW.
