# Results log

One entry per result: date, script, exact command, key numbers, file path.
**Append a new entry after every future run** (rule in `CLAUDE.md`). Numbers are copied from the run's `results/<run>.json`; this log does not interpret them.

All current-format runs below were executed on `CPU: Apple M2 (Darwin arm64)` with git commit `8764c2a` plus uncommitted changes (`git_dirty: true` in each record), using `/usr/local/bin/python3`.

---

## Current runs (config-driven; full record in each JSON file)

### 2026-10-10 22:43 — data preparation
- **Script:** `src/data.py`
- **Command:** `/usr/local/bin/python3 src/data.py --config configs/data.json`
- **Key numbers:** vocabulary 20,003 entries (20,000 words + `<pad>`/`<unk>`/`<eos>`); train 40,000 reviews / 11,067,213 tokens / `<unk>` rate 2.0730%; val 5,000 / 1,388,412 / 2.2256%; test 5,000 / 1,395,673 / 2.2388%.
- **Files:** `results/data.json`; processed data in `data/processed/` (git-ignored). Rebuilt data verified identical (array-by-array) to the earlier build.

### 2026-10-10 22:43 — raw data inspection
- **Script:** `src/inspect_data.py`
- **Command:** `/usr/local/bin/python3 src/inspect_data.py --config configs/inspect_data.json`
- **Key numbers:** 50,000 reviews; labels 25,000 positive / 25,000 negative; 11,557,847 whitespace-split words (average 231.16 per review); 29,200 reviews (58.4%) contain `<br`; subsample suggestion for the 3,000,000-word budget: 12,500 reviews.
- **File:** `results/inspect_data.json`

### 2026-10-10 22:44 — untrained bigram sanity check
- **Script:** `src/sanity_check.py`
- **Command:** `/usr/local/bin/python3 src/sanity_check.py --config configs/sanity_check.json`
- **Key numbers:** validation perplexity (untrained) 23,926.5; average next-token loss 10.0827 nats; 983,447 scored tokens (reviews truncated to 256 tokens); vocabulary size 20,003.
- **File:** `results/sanity_check.json`

### 2026-10-10 22:44 — trigram Kneser-Ney language model
- **Script:** `src/ngram.py`
- **Command:** `/usr/local/bin/python3 src/ngram.py --config configs/ngram.json`
- **Key numbers:**
  - test: cross-entropy 4.7582 nats, perplexity 116.54, 1,390,673 scored tokens (checksum `fd778f4c…`)
  - validation: cross-entropy 4.7617 nats, perplexity 116.95, 1,383,412 scored tokens
  - discounts chosen on validation (D3, D2, D1): 0.9, 0.8, 0.1 (grid 0.1-0.95, 1,000 combinations)
  - training (counting) time 0.84 s; tuning time 22.8 s
  - stored counts 6,445,342 (5,031,011 trigram + 1,394,329 bigram + 20,002 unigram)
- **File:** `results/ngram.json`

### 2026-10-10 22:44 — tests for `evaluate.py`
- **Script:** `tests/test_evaluate.py`
- **Command:** `/usr/local/bin/python3 tests/test_evaluate.py --config configs/test_evaluate.json`
- **Key numbers:** 5 of 5 tests passed.
- **File:** `results/test_evaluate.json`

### 2026-10-11 00:03 — tests for `generate.py`
- **Script:** `tests/test_generate.py`
- **Command:** `/usr/local/bin/python3 tests/test_generate.py --config configs/test_generate.json`
- **Key numbers:** 7 of 7 tests passed.
- **File:** `results/test_generate.json`

### 2026-10-11 01:05 — n-gram text generation
- **Script:** `src/generate.py`
- **Command:** `/usr/local/bin/python3 src/generate.py --config configs/generations_ngram.json`
- **Settings:** temperature 1.0, top-k 40, 50 max new tokens, 1 sample per prompt, banned `<pad>`/`<unk>`, stop at `<eos>`, seed 0; 3 prompts from `prompts.txt`; n-gram discounts (D3, D2, D1) 0.9, 0.8, 0.1 read from `results/ngram.json`.
- **Key numbers:** 3 generations; 1 ended with `<eos>`; mean 41.0 new tokens.
- **File:** `results/generations_ngram.json` (decoding settings and texts stored inside). Run record shows `git_dirty: true` (`prompt.txt` -> `prompts.txt` rename not yet committed).

### 2026-10-10 — summary table
- **Script:** `src/summarise_runs.py`
- **Command:** `/usr/local/bin/python3 src/summarise_runs.py`
- **Output:** one row per run above (5 runs).
- **File:** `results/all_runs.csv`

---

## Earlier runs (before the config system) — superseded, numbers re-confirmed

These were run with the old command-line interface. Their result files were overwritten by the current-format runs above, which reproduce the same numbers. The exact original commands were not recorded; the ones shown are reconstructed from the scripts' usage text at the time and from `TOOL_USE.md`.

### 2026-10-09 — data preparation (original)
- **Script / command:** `src/data.py`; reconstructed: `python src/data.py --seed 0` with the default vocabulary size 20000.
- **Key numbers:** identical to the current data-preparation entry above (committed in `77f7784`).
- **File:** was `results/data_prep_stats.json` (now `results/data.json`).

### 2026-10-09 — raw data inspection (original)
- **Script / command:** `src/inspect_data.py`; reconstructed: `python src/inspect_data.py` (defaults: seed 0, 3 examples, 3,000,000-word budget).
- **Key numbers:** identical to the current inspection entry above.
- **File:** `results/inspect_data.json` (overwritten in the new format).

### 2026-10-09 — untrained bigram sanity check (original)
- **Script / command:** `src/sanity_check.py`; reconstructed: `python src/sanity_check.py` (defaults: seed 0, batch size 32, max length 256).
- **Key numbers:** validation perplexity 23,926.5, vocabulary size 20,003, 983,447 scored tokens — identical to the current entry.
- **File:** `results/sanity_check.json` (overwritten in the new format).

### 2026-10-10 — trigram Kneser-Ney, first run (discount grid 0.5-0.95)
- **Script / command:** `src/ngram.py` with its then-default settings (old CLI, no config).
- **Key numbers:** discounts (D3, D2, D1) = 0.9, 0.8, 0.5; validation cross-entropy 4.7617; test cross-entropy 4.7582 nats, perplexity 116.5. Best D1 sat on the lower edge of the grid, so the grid was widened to 0.1-0.95 and the run repeated (see `DECISIONS.md` D4).
- **File:** `results/ngram.json` (overwritten by the later runs).

### 2026-10-10 — trigram Kneser-Ney, second run (discount grid 0.1-0.95, old CLI)
- **Script / command:** `src/ngram.py` with default settings (old CLI, no config).
- **Key numbers:** discounts 0.9, 0.8, 0.1; test cross-entropy 4.7582 nats, perplexity 116.5; training time 1.0 s; 6,445,342 stored counts.
- **File:** `results/ngram.json` (overwritten by the current-format run above, which reproduces these numbers exactly).

---

## Note, 2026-10-11 — earlier unsaved check runs
Before `prompts.txt` existed, `src/generate.py` was run end to end twice (n-gram, `configs/generations_ngram.json` with `options.prompts_path=prompt.txt`, output to a scratch directory) to check it works and is reproducible (identical text both times). Those outputs were not saved to `results/`. The saved run is the 2026-10-11 01:05 entry above.

### 2026-10-11 — training-window loader, one-batch shape check
- **Script / command:** `src/data.py`; `python3 src/data.py --config configs/show_batch.json --show-batch` (seed 0)
- **Key numbers:** x shape (64, 64) int64; y shape (64, 64) int64; train windows 172,925; batches per epoch 2,702; y equals x shifted by one: True. Also `python3 tests/test_windows.py`: all window tests passed.
- **File:** `results/show_batch.json`

### 2026-10-11 — training-script test (synthetic data; no real training run yet)
- **Script / command:** `tests/test_train.py`; `python3 tests/test_train.py --config configs/test_train.json` (seed 0)
- **Key numbers:** all checks passed; 7 epochs run, best epoch 5, early stop with patience 2. Separate end-to-end check of `src/train.py` on a 60-review slice of the real data (3 epochs, scratch output, not saved to `results/`): 4,020,603 trainable parameters, CPU. Measured about 0.36 s per training batch on the full data (2,702 batches per epoch).
- **File:** `results/test_train.json`

### 2026-10-12 02:26 — bigram language model training (entry added later, 2026-10-12)
- **Script:** `src/train.py`
- **Command:** `python3 src/train.py --config configs/train_bigram.json --model bigram` (launched 2026-10-11 21:49 in the background with `nohup`; the command is from the session, not from the run record).
- **Key numbers:** 20 epochs run (early stop, patience 3); best epoch 17; best validation loss 5.0973; test cross-entropy 5.0895 nats, test perplexity 162.31; 4,020,603 trainable parameters; training time 16,610.5 s; CPU: Apple M2.
- **File:** `results/train_bigram.json`

### 2026-10-12 — results document and builder script
- **Script:** `src/update_results.py` (new), `src/summarise_runs.py`
- **Command:** `python3 src/update_results.py --config configs/update_results.json`, then `python3 src/summarise_runs.py`
- **Key numbers:** 10 run records read; 3 figures written to `results/figures/`; `RESULTS.md` has 6 sections (Data, Part I, Part II, Part III, Settings, Checks); Settings lists 54 decisions, 13 flagged NEEDS REVIEW; Checks: 3 PASS, 2 NOT YET AVAILABLE; `results/all_runs.csv` has 10 rows.
- **File:** `RESULTS.md`, `results/figures/`, `results/all_runs.csv`

### 2026-10-12 — task list built
- **Script:** `src/update_results.py` (now also writes `MY_TASKS.md` via `src/my_tasks.py`)
- **Command:** `python3 src/update_results.py --config configs/update_results.json`
- **Key numbers:** `MY_TASKS.md` lists 4 waiting items, 16 decisions to review, 0 runs to do, 4 sections to interpret, 9 report requirements, 0 failed checks. No new run record.
- **File:** `MY_TASKS.md`

### 2026-10-12 — rebuild after reading the assignment brief
- **Script:** `src/update_results.py`
- **Command:** `python3 src/update_results.py --config configs/update_results.json`
- **Key numbers:** 10 run records; Checks: 3 PASS, 1 FAIL (all models scored on identical test targets: `ngram` 1,390,673 targets vs `train_bigram` 981,353 targets, different checksums), 3 NOT YET AVAILABLE. No new run record.
- **File:** `RESULTS.md`, `MY_TASKS.md`

### 2026-10-12 06:19 UTC — LSTM language model training (Colab)
- **Script / command:** `src/train.py`; `python -u src/train.py --config configs/train_lstm.json --model lstm` (command from `notebooks/run_colab.ipynb`; seed 0; git commit `7b9bbe6`, `git_dirty: false`).
- **Key numbers:** hidden 256, 2 layers, dropout 0.3; 20 epochs run (epoch cap), not stopped early; best epoch 20; best validation loss 4.3990; test cross-entropy 4.3876 nats, test perplexity 80.45; 8,033,999 trainable parameters; training time 3,849.7 s; Tesla T4.
- **File:** `results/train_lstm.json`

### 2026-10-12 07:16 UTC — RNN language model training (Colab)
- **Script / command:** `src/train.py`; `python -u src/train.py --config configs/train_rnn.json --model rnn` (command from `notebooks/run_colab.ipynb`; seed 0; git commit `7b9bbe6`, `git_dirty: true`).
- **Key numbers:** hidden 256, 2 layers, dropout 0.3; 20 epochs run (epoch cap), not stopped early; best epoch 20; best validation loss 4.7053; test cross-entropy 4.6947 nats, test perplexity 109.37; 7,364,303 trainable parameters; training time 3,387.6 s; Tesla T4.
- **File:** `results/train_rnn.json`

### 2026-10-12 — Colab-only audit (no new run)
- **Script:** `src/update_results.py`
- **Command:** `python3 src/update_results.py --config configs/update_results.json`
- **Key numbers:** 13 run records. Colab-only runs found on a CPU: 1 (`train_bigram`, CPU: Apple M2, 16,610.5 s). `train_rnn` and `train_lstm`: GPU: Tesla T4. Pending Colab re-run listed in `MY_TASKS.md`: `train_bigram` as `train_bigram_colab`.
- **File:** `MY_TASKS.md`

### 2026-10-12 — Model tests including the Transformer (no training run)
- **Script:** `tests/test_models.py`
- **Command:** `python3 tests/test_models.py --config configs/test_models.json`
- **Key numbers:** passed: rnn, lstm, transformer (seed 0).
- **File:** `results/test_models.json`

### 2026-10-12 15:05 UTC — Transformer language model training (Colab)
- **Script / command:** `src/train.py`; `python -u src/train.py --config configs/train_transformer.json --model transformer` (seed 0; git commit `b59b5ed`, `git_dirty: false`).
- **Key numbers:** hidden (feed-forward) 256, 2 layers, 4 heads, max_positions 256, dropout 0.3; 20 epochs run (epoch cap), not stopped early; best epoch 20; best validation loss 5.0639; test cross-entropy 5.0559 nats, test perplexity 156.94; 4,231,115 trainable parameters; training time 2,246.0 s; Tesla T4.
- **File:** `results/train_transformer.json`

### 2026-10-12 15:38 UTC — Bigram baseline re-run (Colab)
- **Script / command:** `src/train.py`; `python -u src/train.py --config configs/train_bigram.json --model bigram --run-name train_bigram_colab` (seed 0; git commit `b59b5ed`, `git_dirty: true`).
- **Key numbers:** 20 epochs run, `stopped_early: true`; best epoch 17; best validation loss 5.0973; test cross-entropy 5.0895 nats, test perplexity 162.31; 4,020,603 trainable parameters; training time 1,970.1 s; Tesla T4.
- **File:** `results/train_bigram_colab.json`

### 2026-10-13 — Causality test
- **Script:** `tests/test_causality.py`
- **Command:** `python3 tests/test_causality.py --config configs/test_causality.json`
- **Key numbers:** passed for LSTMLanguageModel, RNNLanguageModel, TransformerLanguageModel; 12 positions each; largest change before the edited position 0.0 for all three (seed 0).
- **File:** `results/test_causality.json`

### 2026-10-12 17:46 UTC — Inference benchmark (Colab)
- **Script / command:** `src/benchmark.py`; `python src/benchmark.py --config configs/benchmark.json` (seed 0; git commit `bd9eb67`, `git_dirty: false`; Tesla T4; torch CPU threads 1).
- **Key numbers:** workload = test set, batch size 64, context length 256, 981,353 target tokens (same checksum `4f77927a1cd9...` for all models), 2 warm-up passes, 3 timed repeats. Tokens/second, mean (std): GPU: RNN 229,944 (1,200), LSTM 164,100 (2,997), Transformer 286,163 (7,950). CPU: n-gram 411,334 (73,730), RNN 4,291 (45), LSTM 4,027 (16), Transformer 4,960 (109). Perplexity on this workload: n-gram 111.31, RNN 109.37, LSTM 80.45, Transformer 156.94.
- **Files:** `results/benchmark.json`, `results/part1_costs.csv`

### 2026-10-13 06:27 UTC — Neural generation and combined file (Colab)
- **Script / command:** `src/generate.py --config configs/generations_<model>.json` for `rnn`, `lstm`, `transformer`, then `src/combine_generations.py --config configs/combine_generations.json` (seed 0; git commit `5771854`, `git_dirty: false`; Tesla T4, generation on cuda).
- **Key numbers:** decoding identical for all four models: temperature 1.0, top-k 40, 50 max new tokens, seed 0, 1 sample, banned `<pad>` and `<unk>`, stop at `<eos>`. 3 prompts each. Ended with `<eos>`: RNN 0 of 3 (mean 50.0 new tokens), LSTM 1 of 3 (35.3), Transformer 1 of 3 (48.7). The n-gram file is the earlier `results/generations_ngram.json` (CPU, commit `ee7c0d9`).
- **Files:** `results/generations_rnn.json`, `results/generations_lstm.json`, `results/generations_transformer.json`, `results/combine_generations.json`, `results/part1_generations.md`

### 2026-10-13 15:05 +08 — Own Skip-gram word2vec (local CPU)
- **Script / command:** `src/train_word2vec.py`; `python3 src/train_word2vec.py --config configs/word2vec_cost.json` (seed 0; git commit `5771854`, `git_dirty: true`; Apple M2, workers 1).
- **Key numbers:** training split only, 40,000 reviews, 11,067,213 tokens. Skip-gram, dim 100, window 5, min_count 1, epochs 10, negative 5, sample 1e-3. Vocabulary build 0.63 s; training 266.74 s; total 267.36 s. Vectors shape 20,003 x 100, `<pad>` row zero.
- **Files:** `results/word2vec_cost.json`, `data/embeddings/w2v_100d.npy`

### 2026-10-13 15:30 +08 — Own Skip-gram word2vec, re-run with coverage (local CPU)
- **Script / command:** `src/train_word2vec.py`; `python3 src/train_word2vec.py --config configs/word2vec_cost.json` (seed 0; Apple M2, workers 1).
- **Key numbers:** 40,000 training reviews, 11,067,213 tokens. Skip-gram, dim 100, window 5, min_count 1, epochs 10. Vocabulary build 0.7 s; training 273.8 s. Coverage: 20,002 of 20,002 vocabulary types (100.00%), 100.00% of training tokens; 0 rows filled randomly. Output identical to the previous run (np.array_equal True).
- **Files:** `results/word2vec_cost.json` (replaces the earlier record), `data/embeddings/w2v_100d.npy`
