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

### 2026-10-13 07:41 UTC — Public GloVe 100d coverage (Colab)
- **Script / command:** `src/load_glove.py`; `python src/load_glove.py --config configs/glove_coverage.json` (seed 0; git commit `bd539e5`, `git_dirty: false`; Tesla T4 session, script itself does not use the GPU).
- **Key numbers:** file `glove.6B.100d.txt` (400,000 lines read). Vocabulary types with a vector: 19,914 of 20,002 (99.56%); training tokens covered: 97.52% (11,067,213 tokens). Excluding `<unk>` and `<eos>`: 99.57% of types, 99.95% of tokens. 88 rows filled randomly (std 0.5027); `<pad>` row zero. Most frequent missing: `<unk>` 229,425 (2.073% of tokens), `<eos>` 40,000 (0.361%), `\x96` 2,060, `\x97` 490, `hadn` 418, `´` 348, `\x91` 93, `imho` 64, `feinstone` 43, `hynkel` 37 (full 20 in the file).
- **File:** `results/glove_coverage.json` (the matrix `data/embeddings/glove_100d.npy` is on Drive, not in the results zip)

### 2026-10-14 06:50 and 07:47 UTC — Part II LSTM with frozen embeddings (Colab)
- **Script / command:** `src/train.py`; `python -u src/train.py --config configs/p2_lstm_w2v.json --model lstm --embedding w2v` and `... configs/p2_lstm_glove.json --model lstm --embedding glove` (seed 0; git commit `5d822d7`, `git_dirty: true` on both; Tesla T4).
- **Key numbers:** both ran all 20 epochs (no early stop; best epoch 20). Test perplexity / cross-entropy / best validation loss / training seconds: w2v 81.42 / 4.3997 / 4.4105 / 3,394; GloVe 90.28 / 4.5029 / 4.5135 / 3,398. Trainable parameters 6,033,699 of 8,033,999 in total. `frozen_embedding_max_abs_change` 0.0 for both. Test targets 981,353, checksum `4f77927a1cd9...` (same as Part I). For reference, Part I `train_lstm` (random, trainable): 80.45 / 4.3876 / 4.3990 / 3,850.
- **Files:** `results/p2_lstm_w2v.json`, `results/p2_lstm_glove.json`

### 2026-10-14 — Part III labelled classification data
- **Script / command:** `src/prepare_classification.py`; `python3 src/prepare_classification.py --config configs/p3_prepare_classification.json` (seed 0, no randomness; local CPU). Max length 256, first-N truncation, `<eos>` dropped.
- **Key numbers (train / val / test):** reviews 40,000 / 5,000 / 5,000; positive 50.00% / 50.00% / 50.00%; truncated 37.0% / 37.0% / 38.2%; mean kept length 196.1 / 197.1 / 196.7; `<unk>` rate of kept tokens 1.996% / 2.122% / 2.151%; of full reviews 2.081% / 2.234% / 2.247%.
- **Files:** `results/p3_prepare_classification.json`, `data/classification/{train,val,test}.npz`

### 2026-10-14 09:36 UTC — Part II generation from the two frozen-embedding LSTMs (Colab)
- **Script / command:** `src/generate.py`; `python -u src/generate.py --config configs/generations_p2_lstm_w2v.json` and `... configs/generations_p2_lstm_glove.json`, then `src/combine_generations.py --config configs/combine_generations_p2.json` (git commit `1e38d0a`, `git_dirty: false`; Tesla T4).
- **Key numbers:** decoding temperature 1.0, top-k 40, 50 new tokens, seed 0, 1 sample, banned `<pad>`/`<unk>`, stop at `<eos>`; identical for both runs (the merge check passed).
- **Files:** `results/generations_p2_lstm_w2v.json`, `results/generations_p2_lstm_glove.json`, `results/combine_generations_p2.json`, `results/part2_generations.md`

### 2026-10-14 09:37 UTC — Part III frozen-feature extraction, mean pooling (Colab)
- **Script / command:** `src/extract_features.py`; `python -u src/extract_features.py --config configs/p3_feat_lstm_mean.json` and `... configs/p3_feat_transformer_mean.json` (seed 0; git commit `1e38d0a`, `git_dirty: false`; Tesla T4).
- **Key numbers:** LSTM (8,033,999 parameters): features 40,000 / 5,000 / 5,000 x 256; seconds train / val / test 9.8 / 1.2 / 1.2; train feature mean 0.0207, std 0.1299. Transformer (4,231,115 parameters): features x 100; seconds 6.0 / 0.7 / 0.7; train mean 0.0106, std 0.3800. `weights_unchanged` true for both.
- **Files:** `results/p3_feat_lstm_mean.json`, `results/p3_feat_transformer_mean.json`, `data/features/p3_feat_*_mean/{train,val,test}.npz` (not in git)

### 2026-10-14 — Part III logistic-regression classifier on frozen features
- **Script / command:** `src/classify.py`; `python3 src/classify.py --config configs/p3_classify.json` (seed 0; local CPU, Apple M2; run before commit, so `git_dirty: true`). Grid C in [0.001 ... 100], selection by validation macro-F1, features standardised.
- **Key numbers:** `p3_feat_lstm_mean` (256 features): best C 0.01; validation accuracy 0.8154; test accuracy 0.8192, macro-F1 0.8192; 904 of 5,000 test errors. `p3_feat_transformer_mean` (100 features): best C 0.1; validation accuracy 0.7624; test accuracy 0.7744, macro-F1 0.7744; 1,128 of 5,000 test errors. All fits converged.
- **Files:** `results/part3.csv`, `results/part3_errors_p3_feat_lstm_mean.csv`, `results/part3_errors_p3_feat_transformer_mean.csv`, `results/p3_classify.json`

### 2026-10-14 — Part III error-review material and ranking table
- **Script / command:** `src/error_sample.py`; `python3 src/error_sample.py --config configs/p3_error_sample.json` (seed 0; local CPU).
- **Key numbers:** `p3_feat_lstm_mean`: errors 904 (mean 203.8 tokens, 45.0% truncated, mean `<unk>` rate 2.34%, 87.7% with a negation word) vs correct 4,096 (195.1, 37.0%, 2.02%, 87.4%). `p3_feat_transformer_mean`: errors 1,128 (202.0, 41.8%, 2.29%, 89.3%) vs correct 3,872 (195.1, 37.5%, 2.02%, 87.0%). Ranking: LSTM perplexity 80.45 (rank 1), accuracy 0.8192 (rank 1); Transformer perplexity 156.94 (rank 2), accuracy 0.7744 (rank 2).
- **Files:** `results/part3_error_sample_p3_feat_lstm_mean.csv`, `results/part3_error_sample_p3_feat_transformer_mean.csv`, `results/part3_error_stats.csv`, `results/part3_ranking.csv`, `results/p3_error_sample.json`

### 2026-10-14 — Report tables and loss-curve figures
- **Script / command:** `src/make_figures.py`; `python3 src/make_figures.py --config configs/make_figures.json` (local CPU; reads existing results only).
- **Key numbers:** Part I test perplexity (same 981,353 targets): n-gram 111.31, RNN 109.37, LSTM 80.45, Transformer 156.94. Part II test perplexity: random 80.45, own Skip-gram 81.42, GloVe 90.28; GloVe coverage 99.56% of types / 97.52% of tokens; word2vec training 273.8 s (local CPU). Part III test accuracy / macro-F1: LSTM features 0.8192 / 0.8192, Transformer features 0.7744 / 0.7744.
- **Files:** `results/tables/part{1,2,3}_table.{csv,tex,md}`, `results/figures/part1_loss_curves.pdf`, `results/figures/part2_loss_curves.pdf`, `results/make_figures.json`

### 2026-10-15 — Part III: PCA to 100 dimensions of the LSTM features
- **Script / command:** `src/classify.py`; `python3 src/classify.py --config configs/p3_classify_pca.json` (seed 0; local CPU; run before commit, `git_dirty: true`). PCA fit on train only, then the unchanged procedure.
- **Key numbers:** `p3_feat_lstm_mean_pca100`: 100 features, PCA explained variance (sum) 0.9788; best C 0.01; validation accuracy 0.8124; test accuracy 0.8128, macro-F1 0.8128; 936 of 5,000 test errors. For reference, unreduced LSTM (256): test accuracy 0.8192, 904 errors.
- **Files:** `results/p3_classify_pca.json`, `results/part3_pca.csv`, `results/part3_errors_p3_feat_lstm_mean_pca100.csv`

### 2026-10-15 — Part III: McNemar's test, LSTM vs Transformer test predictions
- **Script / command:** `src/mcnemar.py`; `python3 src/mcnemar.py --config configs/p3_mcnemar.json` (no randomness; local CPU).
- **Key numbers:** 5,000 test reviews: both right 3,531; only LSTM right 565; only Transformer right 341; both wrong 563. Chi-square (continuity-corrected, 1 df) 54.8885, p = 1.28e-13; exact binomial p = 9.6e-14.
- **File:** `results/p3_mcnemar.json`

### 2026-10-15 08:31 UTC — Part III extension: mean-pooled features for the RNN and the two Part II LSTMs (Colab)
- **Script / command:** `src/extract_features.py`; `python -u src/extract_features.py --config configs/p3_feat_rnn_mean.json`, `... p3_feat_lstm_w2v_mean.json`, `... p3_feat_lstm_glove_mean.json` (seed 0; git commit `d0aacb1`, `git_dirty: false`; Tesla T4).
- **Key numbers:** RNN (7,364,303 parameters): features 40,000 / 5,000 / 5,000 x 256; seconds train / val / test 3.8 / 0.3 / 0.3; train feature mean -0.0210, std 0.5076. LSTM frozen w2v (6,033,699 trainable): x 256; 10.0 / 1.3 / 1.2 s; mean 0.0105, std 0.1333. LSTM frozen GloVe: x 256; 10.2 / 1.3 / 1.3 s; mean 0.0065, std 0.1294. `weights_unchanged` true for all three.
- **Files:** `results/p3_feat_rnn_mean.json`, `results/p3_feat_lstm_w2v_mean.json`, `results/p3_feat_lstm_glove_mean.json` (the feature arrays in `data/features/` were not in the results zip)

### 2026-10-15 08:32 UTC — Part III diagnostic: test perplexity in 64-token windows (Colab)
- **Script / command:** `src/rescore_windows.py`; `python -u src/rescore_windows.py --config configs/p3_rescore_windows.json` (seed 0; git commit `d0aacb1`, `git_dirty: true`; Tesla T4).
- **Key numbers** (stored Part I value / recomputed 256 tokens / 64-token windows): RNN 109.3664 / 109.3664 / 111.1136; LSTM 80.4494 / 80.4494 / 83.0588; Transformer 156.9419 / 156.9419 / 101.0028. All three: 981,353 targets, checksum `4f77927a1cd9...` (equal to Part I and to the one-piece run).
- **Files:** `results/p3_rescore_windows.json`, `results/part3_window_perplexity.csv`

### 2026-10-15 — Part III: classifier on five representations, and LM perplexity vs classifier table
- **Script / command:** `src/classify.py`; `python3 src/classify.py --config configs/p3_classify_five.json`, then `src/rank_table.py`; `python3 src/rank_table.py --config configs/p3_rank_table.json` (seed 0; local CPU; run before commit, `git_dirty: true`). Same procedure for all five.
- **Key numbers** (LM test perplexity; best C; validation macro-F1; test accuracy = test macro-F1; test errors of 5,000): LSTM 80.4494; 0.01; 0.8154; 0.8192; 904. Transformer 156.9419; 0.1; 0.7624; 0.7744; 1,128. RNN 109.3664; 0.1; 0.7676; 0.7782; 1,109. LSTM frozen w2v 81.4249; 1; 0.8486; 0.8592; 704. LSTM frozen GloVe 90.2807; 1; 0.8138; 0.8276; 862. Spearman rho (perplexity vs test accuracy, and vs test macro-F1; n = 5) -0.7, two-sided p = 0.1881 for both. The LSTM and Transformer rows equal the earlier `p3_classify` results.
- **Files:** `results/p3_classify_five.json`, `results/part3_five.csv`, `results/part3_five_errors_<variant>.csv` (x5), `results/p3_rank_table.json`, `results/part3_lm_vs_classifier.csv`

### 2026-10-15 — Part III: McNemar's test, frozen Skip-gram LSTM and frozen GloVe LSTM vs random-embedding LSTM
- **Script / command:** `src/mcnemar.py`; `python3 src/mcnemar.py --config configs/p3_mcnemar_w2v.json` and `... configs/p3_mcnemar_glove.json` (no randomness; local CPU; A = the embedding variant, B = random-embedding LSTM `p3_feat_lstm_mean`; error files `results/part3_five_errors_<variant>.csv`).
- **Key numbers** (5,000 test reviews; both right / only A right / only B right / both wrong; continuity-corrected chi-square, 1 df; p; exact binomial p): Skip-gram vs random: 3,861 / 435 / 235 / 469; 59.1061; p = 1.49e-14; exact 9.52e-15. GloVe vs random: 3,781 / 357 / 315 / 547; 2.5015; p = 0.1137; exact 0.1137.
- **Files:** `results/p3_mcnemar_w2v.json`, `results/p3_mcnemar_glove.json`
