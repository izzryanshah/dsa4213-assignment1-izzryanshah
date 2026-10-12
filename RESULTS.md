# Results

Rebuilt by `src/update_results.py` on 2026-10-13T01:50:14+08:00 from 17 run records in `results/`, `DECISIONS.md` and `results/RESULTS_LOG.md`. Newest run record: 2026-10-13T00:17:56+08:00. Do not edit outside the *My notes* blocks; they are rebuilt each time.

Contents: [Data](#data) · [Part I](#part-i) · [Part II](#part-ii) · [Part III](#part-iii) · [Settings](#settings) · [Checks](#checks)

## Data

#### `data`

| Field | Value |
|---|---|
| Command | `/usr/local/bin/python3 src/data.py --config configs/data.json` (copied from results/RESULTS_LOG.md) |
| Script | `src/data.py` |
| Date | 2026-10-10T22:43:28+08:00 |
| Git commit | `8764c2a (uncommitted changes at run time)` |
| Hardware | CPU: Apple M2 (Darwin arm64) |
| Seed | 0 |
| Result file | `results/data.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | — |
| `layers` | — |
| `hidden_size` | — |
| `embedding_dim` | — |
| `embedding_init` | — |
| `context_length` | — |
| `learning_rate` | — |
| `batch_size` | — |
| `epochs` | — |
| `dropout` | — |
| `vocab_size` | 20,000 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.data_path` | data/IMDB Dataset.csv |
| `options.output_dir` | data/processed |
| `options.val_frac` | 0.1000 |
| `options.test_frac` | 0.1000 |

**Key numbers**

| Metric | Value |
|---|---|
| `vocab_size_with_special_tokens` | 20,003 |
| `splits.train.num_reviews` | 40,000 |
| `splits.train.num_tokens` | 11,067,213 |
| `splits.train.unk_rate` | 0.0207 |
| `splits.val.num_reviews` | 5,000 |
| `splits.val.num_tokens` | 1,388,412 |
| `splits.val.unk_rate` | 0.0223 |
| `splits.test.num_reviews` | 5,000 |
| `splits.test.num_tokens` | 1,395,673 |
| `splits.test.unk_rate` | 0.0224 |

![data_split_tokens.png](results/figures/data_split_tokens.png)

#### `inspect_data`

| Field | Value |
|---|---|
| Command | `/usr/local/bin/python3 src/inspect_data.py --config configs/inspect_data.json` (copied from results/RESULTS_LOG.md) |
| Script | `src/inspect_data.py` |
| Date | 2026-10-10T22:43:31+08:00 |
| Git commit | `8764c2a (uncommitted changes at run time)` |
| Hardware | CPU: Apple M2 (Darwin arm64) |
| Seed | 0 |
| Result file | `results/inspect_data.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | — |
| `layers` | — |
| `hidden_size` | — |
| `embedding_dim` | — |
| `embedding_init` | — |
| `context_length` | — |
| `learning_rate` | — |
| `batch_size` | — |
| `epochs` | — |
| `dropout` | — |
| `vocab_size` | — |
| `subsample_size` | — |
| `seed` | 0 |
| `options.data_path` | data/IMDB Dataset.csv |
| `options.n_examples` | 3 |
| `options.word_budget` | 3,000,000 |
| `options.example_chars` | 500 |
| `options.round_to` | 500 |

**Key numbers**

| Metric | Value |
|---|---|
| `column_names` | `["review", "sentiment"]` |
| `num_reviews` | 50,000 |
| `label_counts.positive` | 25,000 |
| `label_counts.negative` | 25,000 |
| `total_words` | 11,557,847 |
| `avg_review_length` | 231.1569 |
| `num_reviews_with_br_tag` | 29,200 |
| `pct_reviews_with_br_tag` | 58.4000 |
| `subsample_suggestion_reviews` | 12,500 |

#### `show_batch`

| Field | Value |
|---|---|
| Command | `python3 src/data.py --config configs/show_batch.json --show-batch` (copied from results/RESULTS_LOG.md) |
| Script | `src/data.py --show-batch` |
| Date | 2026-10-11T20:22:22+08:00 |
| Git commit | `7ef446d (uncommitted changes at run time)` |
| Hardware | CPU: Apple M2 (Darwin arm64) |
| Seed | 0 |
| Result file | `results/show_batch.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | — |
| `layers` | — |
| `hidden_size` | — |
| `embedding_dim` | — |
| `embedding_init` | — |
| `context_length` | 64 |
| `learning_rate` | — |
| `batch_size` | 64 |
| `epochs` | — |
| `dropout` | — |
| `vocab_size` | 20,000 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.processed_dir` | data/processed |

**Key numbers**

| Metric | Value |
|---|---|
| `x_shape` | `[64, 64]` |
| `y_shape` | `[64, 64]` |
| `x_dtype` | torch.int64 |
| `y_dtype` | torch.int64 |
| `num_train_windows` | 172,925 |
| `num_train_batches` | 2,702 |
| `y_is_x_shifted_by_one` | True |

### My notes

<!-- my-notes:data:start -->

<!-- my-notes:data:end -->

## Part I

#### `generations_ngram`

| Field | Value |
|---|---|
| Command | `/usr/local/bin/python3 src/generate.py --config configs/generations_ngram.json` (copied from results/RESULTS_LOG.md) |
| Script | `src/generate.py` |
| Date | 2026-10-11T01:05:20+08:00 |
| Git commit | `ee7c0d9 (uncommitted changes at run time)` |
| Hardware | CPU: Apple M2 (Darwin arm64) |
| Seed | 0 |
| Result file | `results/generations_ngram.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | trigram_kneser_ney |
| `layers` | — |
| `hidden_size` | — |
| `embedding_dim` | — |
| `embedding_init` | — |
| `context_length` | 2 |
| `learning_rate` | — |
| `batch_size` | — |
| `epochs` | — |
| `dropout` | — |
| `vocab_size` | 20,000 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.model` | ngram |
| `options.processed_dir` | data/processed |
| `options.ngram_results_path` | results/ngram.json |
| `options.prompts_path` | prompts.txt |
| `options.temperature` | 1.0000 |
| `options.top_k` | 40 |
| `options.max_new_tokens` | 50 |
| `options.num_samples` | 1 |
| `options.banned_tokens` | `["&lt;pad&gt;", "&lt;unk&gt;"]` |
| `options.stop_at_eos` | True |

**Key numbers**

| Metric | Value |
|---|---|
| `num_prompts` | 3 |
| `num_generations` | 3 |
| `num_ended_with_eos` | 1 |
| `mean_new_tokens` | 41.0000 |

**Details**

| Name | Value |
|---|---|
| `model` | ngram |
| `prompts_path` | prompts.txt |
| `ngram_results_path` | results/ngram.json |
| `ngram_run_date` | 2026-10-10T22:44:32+08:00 |
| `ngram_run_git_commit` | 8764c2a535ca4f8980bd2386e6aebbb01af0f22b |
| `discounts.d3` | 0.9000 |
| `discounts.d2` | 0.8000 |
| `discounts.d1` | 0.1000 |

**Decoding settings**

| Setting | Value |
|---|---|
| `temperature` | 1.0000 |
| `top_k` | 40 |
| `max_tokens` | 50 |
| `seed` | 0 |
| `num_samples` | 1 |
| `banned_tokens` | `["&lt;pad&gt;", "&lt;unk&gt;"]` |
| `stop_at_eos` | True |

#### `ngram`

| Field | Value |
|---|---|
| Command | `/usr/local/bin/python3 src/ngram.py --config configs/ngram.json` (copied from results/RESULTS_LOG.md) |
| Script | `src/ngram.py` |
| Date | 2026-10-10T22:44:32+08:00 |
| Git commit | `8764c2a (uncommitted changes at run time)` |
| Hardware | CPU: Apple M2 (Darwin arm64) |
| Seed | 0 |
| Result file | `results/ngram.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | trigram_kneser_ney |
| `layers` | — |
| `hidden_size` | — |
| `embedding_dim` | — |
| `embedding_init` | — |
| `context_length` | 2 |
| `learning_rate` | — |
| `batch_size` | 32 |
| `epochs` | — |
| `dropout` | — |
| `vocab_size` | 20,000 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.processed_dir` | data/processed |
| `options.max_length` | — |
| `options.discount_grid` | `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]` |

**Key numbers**

| Metric | Value |
|---|---|
| `val.avg_cross_entropy` | 4.7617 |
| `val.perplexity` | 116.9503 |
| `val.num_targets` | 1,383,412 |
| `val.target_checksum` | 3db6b4677ec60f9fe6ed02f5be7cd8d3a496c3fddf2bed091a56e3cbcabc7ac6 |
| `test.avg_cross_entropy` | 4.7582 |
| `test.perplexity` | 116.5364 |
| `test.num_targets` | 1,390,673 |
| `test.target_checksum` | fd778f4c313dbc0796c6f97a40326932017a8e742b6f6905c422d4ab929d1818 |

**Details**

| Name | Value |
|---|---|
| `vocab_size_with_special_tokens` | 20,003 |
| `best_discounts.d3` | 0.9000 |
| `best_discounts.d2` | 0.8000 |
| `best_discounts.d1` | 0.1000 |
| `train_seconds` | 0.8364 |
| `tune_seconds` | 22.7996 |
| `model_size.trigram_types` | 5,031,011 |
| `model_size.bigram_types` | 1,394,329 |
| `model_size.unigram_types` | 20,002 |
| `model_size.total_stored_counts` | 6,445,342 |
| `model_size.trigram_context_rows` | 1,396,523 |
| `model_size.bigram_context_rows` | 20,001 |
| `tuning_grid` | (1000 entries; see the run's JSON file) |

![ngram_tuning.png](results/figures/ngram_tuning.png)

#### `sanity_check`

| Field | Value |
|---|---|
| Command | `/usr/local/bin/python3 src/sanity_check.py --config configs/sanity_check.json` (copied from results/RESULTS_LOG.md) |
| Script | `src/sanity_check.py` |
| Date | 2026-10-10T22:44:00+08:00 |
| Git commit | `8764c2a (uncommitted changes at run time)` |
| Hardware | CPU: Apple M2 (Darwin arm64) |
| Seed | 0 |
| Result file | `results/sanity_check.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | bigram_lm |
| `layers` | — |
| `hidden_size` | — |
| `embedding_dim` | 100 |
| `embedding_init` | random |
| `context_length` | 1 |
| `learning_rate` | — |
| `batch_size` | 32 |
| `epochs` | — |
| `dropout` | — |
| `vocab_size` | 20,000 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.processed_dir` | data/processed |
| `options.max_length` | 256 |

**Key numbers**

| Metric | Value |
|---|---|
| `val_perplexity_untrained` | 23,926.52 |
| `val_avg_nll_untrained` | 10.0827 |
| `val_num_target_tokens` | 983,447 |

**Details**

| Name | Value |
|---|---|
| `vocab_size_with_special_tokens` | 20,003 |

#### `test_causality`

| Field | Value |
|---|---|
| Command | `/usr/local/bin/python3 tests/test_causality.py --config configs/test_causality.json` |
| Script | `tests/test_causality.py` |
| Date | 2026-10-13T00:17:56+08:00 |
| Git commit | `66bc88f (uncommitted changes at run time)` |
| Hardware | CPU: Apple M2 (Darwin arm64) |
| Seed | 0 |
| Result file | `results/test_causality.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | — |
| `layers` | 2 |
| `hidden_size` | 16 |
| `embedding_dim` | 100 |
| `embedding_init` | random |
| `context_length` | — |
| `learning_rate` | — |
| `batch_size` | 4 |
| `epochs` | — |
| `dropout` | 0.3000 |
| `vocab_size` | 50 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.seq_len` | 12 |
| `options.num_heads` | 4 |
| `options.max_positions` | 16 |
| `options.tolerance` | 0.0000 |

**Key numbers**

| Metric | Value |
|---|---|
| `passed` | True |
| `models.LSTMLanguageModel.positions_tested` | 12 |
| `models.LSTMLanguageModel.max_change_before_t` | 0.0000 |
| `models.RNNLanguageModel.positions_tested` | 12 |
| `models.RNNLanguageModel.max_change_before_t` | 0.0000 |
| `models.TransformerLanguageModel.positions_tested` | 12 |
| `models.TransformerLanguageModel.max_change_before_t` | 0.0000 |

#### `test_evaluate`

| Field | Value |
|---|---|
| Command | `/usr/local/bin/python3 tests/test_evaluate.py --config configs/test_evaluate.json` (copied from results/RESULTS_LOG.md) |
| Script | `tests/test_evaluate.py` |
| Date | 2026-10-11T00:04:03+08:00 |
| Git commit | `8764c2a (uncommitted changes at run time)` |
| Hardware | CPU: Apple M2 (Darwin arm64) |
| Seed | 0 |
| Result file | `results/test_evaluate.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | bigram_lm |
| `layers` | — |
| `hidden_size` | — |
| `embedding_dim` | 100 |
| `embedding_init` | random |
| `context_length` | 1 |
| `learning_rate` | — |
| `batch_size` | 16 |
| `epochs` | — |
| `dropout` | — |
| `vocab_size` | 997 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.num_fake_reviews` | 60 |
| `options.max_fake_review_length` | 80 |
| `options.truncate_length` | 40 |

**Key numbers**

| Metric | Value |
|---|---|
| `all_passed` | True |
| `num_passed` | 5 |
| `num_tests` | 5 |

**Individual tests**

| Test | Result |
|---|---|
| `test_untrained_perplexity_is_near_vocab_size` | passed |
| `test_matches_sanity_check_implementation` | passed |
| `test_result_does_not_depend_on_batch_size` | passed |
| `test_neural_and_ngram_style_scorer_see_same_targets` | passed |
| `test_padding_is_ignored` | passed |

#### `test_generate`

| Field | Value |
|---|---|
| Command | `/usr/local/bin/python3 tests/test_generate.py --config configs/test_generate.json` (copied from results/RESULTS_LOG.md) |
| Script | `tests/test_generate.py` |
| Date | 2026-10-11T00:03:36+08:00 |
| Git commit | `8764c2a (uncommitted changes at run time)` |
| Hardware | CPU: Apple M2 (Darwin arm64) |
| Seed | 0 |
| Result file | `results/test_generate.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | bigram_lm |
| `layers` | — |
| `hidden_size` | — |
| `embedding_dim` | 100 |
| `embedding_init` | random |
| `context_length` | 1 |
| `learning_rate` | — |
| `batch_size` | — |
| `epochs` | — |
| `dropout` | — |
| `vocab_size` | 47 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.num_fake_reviews` | 200 |
| `options.max_fake_review_length` | 30 |
| `options.num_draws` | 3,000 |

**Key numbers**

| Metric | Value |
|---|---|
| `all_passed` | True |
| `num_passed` | 7 |
| `num_tests` | 7 |

**Individual tests**

| Test | Result |
|---|---|
| `test_temperature_and_topk_match_the_docstring_example` | passed |
| `test_banned_words_are_never_sampled` | passed |
| `test_topk_keeps_only_the_k_likeliest_allowed_words` | passed |
| `test_same_seed_same_text_different_seed_different_text` | passed |
| `test_stops_at_eos_and_does_not_include_it` | passed |
| `test_neural_wrapper_gives_valid_distribution` | passed |
| `test_ngram_next_token_matches_scoring_path` | passed |

#### `test_models`

| Field | Value |
|---|---|
| Command | `/usr/local/bin/python3 tests/test_models.py --config configs/test_models.json` |
| Script | `tests/test_models.py` |
| Date | 2026-10-12T21:57:48+08:00 |
| Git commit | `9afc2c8 (uncommitted changes at run time)` |
| Hardware | CPU: Apple M2 (Darwin arm64) |
| Seed | 0 |
| Result file | `results/test_models.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | — |
| `layers` | 2 |
| `hidden_size` | 16 |
| `embedding_dim` | 100 |
| `embedding_init` | random |
| `context_length` | — |
| `learning_rate` | — |
| `batch_size` | 3 |
| `epochs` | — |
| `dropout` | 0.3000 |
| `vocab_size` | 50 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.seq_len` | 7 |
| `options.num_heads` | 4 |
| `options.max_positions` | 16 |

**Key numbers**

| Metric | Value |
|---|---|
| `passed` | True |
| `models` | `["rnn", "lstm", "transformer"]` |

#### `test_train`

| Field | Value |
|---|---|
| Command | `python3 tests/test_train.py --config configs/test_train.json` (copied from results/RESULTS_LOG.md) |
| Script | `tests/test_train.py` |
| Date | 2026-10-11T21:48:02+08:00 |
| Git commit | `7ef446d (uncommitted changes at run time)` |
| Hardware | CPU: Apple M2 (Darwin arm64) |
| Seed | 0 |
| Result file | `results/test_train.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | bigram |
| `layers` | — |
| `hidden_size` | — |
| `embedding_dim` | 100 |
| `embedding_init` | — |
| `context_length` | 8 |
| `learning_rate` | 0.0100 |
| `batch_size` | 4 |
| `epochs` | 30 |
| `dropout` | — |
| `vocab_size` | 20 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.patience` | 2 |
| `options.grad_clip` | 1.0000 |
| `options.eval_max_length` | 16 |
| `options.eval_batch_size` | 4 |

**Key numbers**

| Metric | Value |
|---|---|
| `passed` | True |
| `epochs_run` | 7 |
| `best_epoch` | 5 |

#### `train_bigram`

| Field | Value |
|---|---|
| Command | `python3 src/train.py --config configs/train_bigram.json --model bigram` (copied from results/RESULTS_LOG.md) |
| Script | `src/train.py` |
| Date | 2026-10-12T02:26:40+08:00 |
| Git commit | `7ef446d (uncommitted changes at run time)` |
| Hardware | CPU: Apple M2 (Darwin arm64) |
| Seed | 0 |
| Result file | `results/train_bigram.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | bigram |
| `layers` | — |
| `hidden_size` | — |
| `embedding_dim` | 100 |
| `embedding_init` | — |
| `context_length` | 64 |
| `learning_rate` | 0.0010 |
| `batch_size` | 64 |
| `epochs` | 20 |
| `dropout` | — |
| `vocab_size` | 20,000 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.processed_dir` | data/processed |
| `options.checkpoint_dir` | checkpoints |
| `options.patience` | 3 |
| `options.grad_clip` | 1.0000 |
| `options.eval_max_length` | 256 |
| `options.eval_batch_size` | 64 |

**Key numbers**

| Metric | Value |
|---|---|
| `best_epoch` | 17 |
| `best_val_loss` | 5.0973 |
| `test_cross_entropy` | 5.0895 |
| `test_perplexity` | 162.3054 |
| `train_seconds` | 16,610.52 |
| `num_parameters` | 4,020,603 |
| `epochs_run` | 20 |
| `stopped_early` | True |

**Details**

| Name | Value |
|---|---|
| `checkpoint_path` | checkpoints/train_bigram.pt |
| `test_num_targets` | 981,353 |
| `test_target_checksum` | 4f77927a1cd952f56697be95540ea0fc002013d9f8278d3f95c1290e6c305605 |

**Loss per epoch**

| Epoch | Train loss | Validation loss |
|---|---|---|
| 1 | 5.9058 | 5.3227 |
| 2 | 5.2628 | 5.2130 |
| 3 | 5.1755 | 5.1715 |
| 4 | 5.1295 | 5.1487 |
| 5 | 5.0989 | 5.1339 |
| 6 | 5.0764 | 5.1240 |
| 7 | 5.0590 | 5.1168 |
| 8 | 5.0446 | 5.1119 |
| 9 | 5.0327 | 5.1081 |
| 10 | 5.0223 | 5.1047 |
| 11 | 5.0134 | 5.1022 |
| 12 | 5.0054 | 5.1010 |
| 13 | 4.9983 | 5.0997 |
| 14 | 4.9917 | 5.0985 |
| 15 | 4.9859 | 5.0981 |
| 16 | 4.9804 | 5.0981 |
| 17 | 4.9754 | 5.0973 |
| 18 | 4.9708 | 5.0983 |
| 19 | 4.9663 | 5.0978 |
| 20 | 4.9623 | 5.0980 |

![train_bigram_loss.png](results/figures/train_bigram_loss.png)

#### `train_bigram_colab`

| Field | Value |
|---|---|
| Command | `/usr/bin/python3 src/train.py --config configs/train_bigram.json --model bigram --run-name train_bigram_colab` |
| Script | `src/train.py` |
| Date | 2026-10-12T15:38:17+00:00 |
| Git commit | `b59b5ed (uncommitted changes at run time)` |
| Hardware | GPU: Tesla T4 |
| Seed | 0 |
| Result file | `results/train_bigram_colab.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | bigram |
| `layers` | — |
| `hidden_size` | — |
| `embedding_dim` | 100 |
| `embedding_init` | — |
| `context_length` | 64 |
| `learning_rate` | 0.0010 |
| `batch_size` | 64 |
| `epochs` | 20 |
| `dropout` | — |
| `vocab_size` | 20,000 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.processed_dir` | data/processed |
| `options.checkpoint_dir` | checkpoints |
| `options.patience` | 3 |
| `options.grad_clip` | 1.0000 |
| `options.eval_max_length` | 256 |
| `options.eval_batch_size` | 64 |

**Key numbers**

| Metric | Value |
|---|---|
| `best_epoch` | 17 |
| `best_val_loss` | 5.0973 |
| `test_cross_entropy` | 5.0895 |
| `test_perplexity` | 162.3082 |
| `train_seconds` | 1,970.07 |
| `num_parameters` | 4,020,603 |
| `epochs_run` | 20 |
| `stopped_early` | True |

**Details**

| Name | Value |
|---|---|
| `checkpoint_path` | checkpoints/train_bigram_colab.pt |
| `test_num_targets` | 981,353 |
| `test_target_checksum` | 4f77927a1cd952f56697be95540ea0fc002013d9f8278d3f95c1290e6c305605 |

**Loss per epoch**

| Epoch | Train loss | Validation loss |
|---|---|---|
| 1 | 5.9058 | 5.3227 |
| 2 | 5.2628 | 5.2130 |
| 3 | 5.1755 | 5.1715 |
| 4 | 5.1295 | 5.1487 |
| 5 | 5.0989 | 5.1339 |
| 6 | 5.0764 | 5.1240 |
| 7 | 5.0590 | 5.1168 |
| 8 | 5.0446 | 5.1119 |
| 9 | 5.0327 | 5.1081 |
| 10 | 5.0223 | 5.1047 |
| 11 | 5.0134 | 5.1022 |
| 12 | 5.0054 | 5.1010 |
| 13 | 4.9983 | 5.0997 |
| 14 | 4.9917 | 5.0985 |
| 15 | 4.9859 | 5.0981 |
| 16 | 4.9804 | 5.0981 |
| 17 | 4.9754 | 5.0973 |
| 18 | 4.9708 | 5.0983 |
| 19 | 4.9663 | 5.0978 |
| 20 | 4.9623 | 5.0981 |

![train_bigram_colab_loss.png](results/figures/train_bigram_colab_loss.png)

#### `train_lstm`

| Field | Value |
|---|---|
| Command | `/usr/bin/python3 src/train.py --config configs/train_lstm.json --model lstm` |
| Script | `src/train.py` |
| Date | 2026-10-12T06:19:48+00:00 |
| Git commit | `7b9bbe6` |
| Hardware | GPU: Tesla T4 |
| Seed | 0 |
| Result file | `results/train_lstm.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | lstm |
| `layers` | 2 |
| `hidden_size` | 256 |
| `embedding_dim` | 100 |
| `embedding_init` | random |
| `context_length` | 64 |
| `learning_rate` | 0.0010 |
| `batch_size` | 64 |
| `epochs` | 20 |
| `dropout` | 0.3000 |
| `vocab_size` | 20,000 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.processed_dir` | data/processed |
| `options.checkpoint_dir` | checkpoints |
| `options.patience` | 3 |
| `options.grad_clip` | 1.0000 |
| `options.eval_max_length` | 256 |
| `options.eval_batch_size` | 64 |

**Key numbers**

| Metric | Value |
|---|---|
| `best_epoch` | 20 |
| `best_val_loss` | 4.3990 |
| `test_cross_entropy` | 4.3876 |
| `test_perplexity` | 80.4494 |
| `train_seconds` | 3,849.67 |
| `num_parameters` | 8,033,999 |
| `epochs_run` | 20 |
| `stopped_early` | False |

**Details**

| Name | Value |
|---|---|
| `checkpoint_path` | checkpoints/train_lstm.pt |
| `test_num_targets` | 981,353 |
| `test_target_checksum` | 4f77927a1cd952f56697be95540ea0fc002013d9f8278d3f95c1290e6c305605 |

**Loss per epoch**

| Epoch | Train loss | Validation loss |
|---|---|---|
| 1 | 6.0026 | 5.4246 |
| 2 | 5.3714 | 5.0313 |
| 3 | 5.1241 | 4.8615 |
| 4 | 5.0003 | 4.7660 |
| 5 | 4.9214 | 4.6971 |
| 6 | 4.8639 | 4.6512 |
| 7 | 4.8192 | 4.6100 |
| 8 | 4.7819 | 4.5770 |
| 9 | 4.7511 | 4.5490 |
| 10 | 4.7238 | 4.5255 |
| 11 | 4.7010 | 4.5028 |
| 12 | 4.6801 | 4.4843 |
| 13 | 4.6624 | 4.4693 |
| 14 | 4.6460 | 4.4563 |
| 15 | 4.6319 | 4.4450 |
| 16 | 4.6186 | 4.4318 |
| 17 | 4.6069 | 4.4247 |
| 18 | 4.5962 | 4.4136 |
| 19 | 4.5866 | 4.4058 |
| 20 | 4.5779 | 4.3990 |

![train_lstm_loss.png](results/figures/train_lstm_loss.png)

#### `train_rnn`

| Field | Value |
|---|---|
| Command | `/usr/bin/python3 src/train.py --config configs/train_rnn.json --model rnn` |
| Script | `src/train.py` |
| Date | 2026-10-12T07:16:34+00:00 |
| Git commit | `7b9bbe6 (uncommitted changes at run time)` |
| Hardware | GPU: Tesla T4 |
| Seed | 0 |
| Result file | `results/train_rnn.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | rnn |
| `layers` | 2 |
| `hidden_size` | 256 |
| `embedding_dim` | 100 |
| `embedding_init` | random |
| `context_length` | 64 |
| `learning_rate` | 0.0010 |
| `batch_size` | 64 |
| `epochs` | 20 |
| `dropout` | 0.3000 |
| `vocab_size` | 20,000 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.processed_dir` | data/processed |
| `options.checkpoint_dir` | checkpoints |
| `options.patience` | 3 |
| `options.grad_clip` | 1.0000 |
| `options.eval_max_length` | 256 |
| `options.eval_batch_size` | 64 |

**Key numbers**

| Metric | Value |
|---|---|
| `best_epoch` | 20 |
| `best_val_loss` | 4.7053 |
| `test_cross_entropy` | 4.6947 |
| `test_perplexity` | 109.3664 |
| `train_seconds` | 3,387.55 |
| `num_parameters` | 7,364,303 |
| `epochs_run` | 20 |
| `stopped_early` | False |

**Details**

| Name | Value |
|---|---|
| `checkpoint_path` | checkpoints/train_rnn.pt |
| `test_num_targets` | 981,353 |
| `test_target_checksum` | 4f77927a1cd952f56697be95540ea0fc002013d9f8278d3f95c1290e6c305605 |

**Loss per epoch**

| Epoch | Train loss | Validation loss |
|---|---|---|
| 1 | 5.5889 | 5.0822 |
| 2 | 5.2293 | 4.9397 |
| 3 | 5.1378 | 4.8771 |
| 4 | 5.0866 | 4.8365 |
| 5 | 5.0530 | 4.8066 |
| 6 | 5.0274 | 4.7932 |
| 7 | 5.0076 | 4.7730 |
| 8 | 4.9924 | 4.7609 |
| 9 | 4.9798 | 4.7703 |
| 10 | 4.9707 | 4.7452 |
| 11 | 4.9615 | 4.7330 |
| 12 | 4.9533 | 4.7257 |
| 13 | 4.9480 | 4.7225 |
| 14 | 4.9422 | 4.7172 |
| 15 | 4.9367 | 4.7127 |
| 16 | 4.9322 | 4.7151 |
| 17 | 4.9288 | 4.7109 |
| 18 | 4.9262 | 4.7081 |
| 19 | 4.9227 | 4.7057 |
| 20 | 4.9216 | 4.7053 |

![train_rnn_loss.png](results/figures/train_rnn_loss.png)

#### `train_transformer`

| Field | Value |
|---|---|
| Command | `/usr/bin/python3 src/train.py --config configs/train_transformer.json --model transformer` |
| Script | `src/train.py` |
| Date | 2026-10-12T15:05:12+00:00 |
| Git commit | `b59b5ed` |
| Hardware | GPU: Tesla T4 |
| Seed | 0 |
| Result file | `results/train_transformer.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | transformer |
| `layers` | 2 |
| `hidden_size` | 256 |
| `embedding_dim` | 100 |
| `embedding_init` | random |
| `context_length` | 64 |
| `learning_rate` | 0.0010 |
| `batch_size` | 64 |
| `epochs` | 20 |
| `dropout` | 0.3000 |
| `vocab_size` | 20,000 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.processed_dir` | data/processed |
| `options.checkpoint_dir` | checkpoints |
| `options.patience` | 3 |
| `options.grad_clip` | 1.0000 |
| `options.eval_max_length` | 256 |
| `options.eval_batch_size` | 64 |
| `options.num_heads` | 4 |
| `options.max_positions` | 256 |

**Key numbers**

| Metric | Value |
|---|---|
| `best_epoch` | 20 |
| `best_val_loss` | 5.0639 |
| `test_cross_entropy` | 5.0559 |
| `test_perplexity` | 156.9419 |
| `train_seconds` | 2,246.04 |
| `num_parameters` | 4,231,115 |
| `epochs_run` | 20 |
| `stopped_early` | False |

**Details**

| Name | Value |
|---|---|
| `checkpoint_path` | checkpoints/train_transformer.pt |
| `test_num_targets` | 981,353 |
| `test_target_checksum` | 4f77927a1cd952f56697be95540ea0fc002013d9f8278d3f95c1290e6c305605 |

**Loss per epoch**

| Epoch | Train loss | Validation loss |
|---|---|---|
| 1 | 5.6914 | 5.3427 |
| 2 | 5.2608 | 5.2243 |
| 3 | 5.1314 | 5.1779 |
| 4 | 5.0580 | 5.1563 |
| 5 | 5.0070 | 5.1350 |
| 6 | 4.9698 | 5.1299 |
| 7 | 4.9405 | 5.1224 |
| 8 | 4.9173 | 5.1122 |
| 9 | 4.8978 | 5.1022 |
| 10 | 4.8812 | 5.0984 |
| 11 | 4.8670 | 5.0906 |
| 12 | 4.8546 | 5.0864 |
| 13 | 4.8443 | 5.0854 |
| 14 | 4.8348 | 5.0830 |
| 15 | 4.8262 | 5.0814 |
| 16 | 4.8189 | 5.0769 |
| 17 | 4.8121 | 5.0719 |
| 18 | 4.8065 | 5.0717 |
| 19 | 4.8006 | 5.0687 |
| 20 | 4.7950 | 5.0639 |

![train_transformer_loss.png](results/figures/train_transformer_loss.png)

### Language-model comparison

Validation perplexity of a neural run is exp(best validation loss); the n-gram figure is as recorded. Inference throughput shows — until a run records `inference_tokens_per_second`.

| Run | Architecture | Validation perplexity | Test perplexity | Model size | Training seconds | Inference tokens/s | Hardware |
|---|---|---|---|---|---|---|---|
| `ngram` | trigram_kneser_ney | 116.9503 | 116.5364 | 6,445,342 stored counts | 0.8364 | — | CPU: Apple M2 (Darwin arm64) |
| `train_bigram` | bigram | 163.5824 | 162.3054 | 4,020,603 parameters | 16,610.52 | — | CPU: Apple M2 (Darwin arm64) |
| `train_bigram_colab` | bigram | 163.5852 | 162.3082 | 4,020,603 parameters | 1,970.07 | — | GPU: Tesla T4 |
| `train_lstm` | lstm | 81.3690 | 80.4494 | 8,033,999 parameters | 3,849.67 | — | GPU: Tesla T4 |
| `train_rnn` | rnn | 110.5351 | 109.3664 | 7,364,303 parameters | 3,387.55 | — | GPU: Tesla T4 |
| `train_transformer` | transformer | 158.2141 | 156.9419 | 4,231,115 parameters | 2,246.04 | — | GPU: Tesla T4 |

### Generated text by prompt

Decoding settings for each column are in the run records above (`results/<run>.json`, field `decoding`).

**Prompt:** `this movie was one of the best`

|  | `generations_ngram` |
|---|---|
| Continuation | way to end . it is the very end when she was always going to happen and then the final girl can ' t . a . i guess i was a little less that five or whatever , you need is a very strong and is very weak script |
| New tokens | 50 |
| Ended with `<eos>` | False |

**Prompt:** `i was really disappointed by the acting and`

|  | `generations_ngram` |
|---|---|
| Continuation | timing of the story lines for their lives in the same age as his wife and kid , and just plain sucks . |
| New tokens | 23 |
| Ended with `<eos>` | True |

**Prompt:** `the story follows a young man who`

|  | `generations_ngram` |
|---|---|
| Continuation | played by the end , when his wife that it would look for the entire scenes and the acting is good and the ending , which is about the characters are either . this , i would have added to the original . unfortunately , this is the worst ever |
| New tokens | 50 |
| Ended with `<eos>` | False |

### My notes

<!-- my-notes:part1:start -->

<!-- my-notes:part1:end -->

## Part II

No runs yet.

![val_loss_all_runs.png](results/figures/val_loss_all_runs.png)

### Language-model comparison

No trained language models yet.

### Generated text by prompt

No generation runs yet.

### My notes

<!-- my-notes:part2:start -->

<!-- my-notes:part2:end -->

## Part III

No runs yet.

### Runs not assigned to a section

These runs match no rule in `part_rules` (`configs/update_results.json`).

#### `benchmark`

| Field | Value |
|---|---|
| Command | `/usr/bin/python3 src/benchmark.py --config configs/benchmark.json` |
| Script | `src/benchmark.py` |
| Date | 2026-10-12T17:46:34+00:00 |
| Git commit | `bd9eb67` |
| Hardware | GPU: Tesla T4 |
| Seed | 0 |
| Result file | `results/benchmark.json` |

**Settings used**

| Setting | Value |
|---|---|
| `architecture` | — |
| `layers` | — |
| `hidden_size` | — |
| `embedding_dim` | — |
| `embedding_init` | — |
| `context_length` | 256 |
| `learning_rate` | — |
| `batch_size` | 64 |
| `epochs` | — |
| `dropout` | — |
| `vocab_size` | 20,000 |
| `subsample_size` | — |
| `seed` | 0 |
| `options.processed_dir` | data/processed |
| `options.warmup_passes` | 2 |
| `options.repeats` | 3 |
| `options.devices` | `["cuda", "cpu"]` |
| `options.max_reviews` | — |
| `options.perplexity_rel_tolerance` | 0.0010 |
| `options.costs_csv` | results/part1_costs.csv |
| `options.models` | `[{"name": "ngram", "train_result": "results/ngram.json"}, {"name": "rnn", "train_result": "results/train_rnn.json", "checkpoint": "checkpoints/train_rnn.pt"}, {"name": "lstm", "train_result": "results/train_lstm.json", "checkpoint": "checkpoints/train_lstm.pt"}, {"name": "transformer", "train_result": "results/train_transformer.json", "checkpoint": "checkpoints/train_transformer.pt"}]` |

**Key numbers**

| Metric | Value |
|---|---|
| `ngram.cpu.tokens_per_second_mean` | 411,333.96 |
| `ngram.cpu.tokens_per_second_std` | 73,730.15 |
| `rnn.cuda.tokens_per_second_mean` | 229,944.45 |
| `rnn.cuda.tokens_per_second_std` | 1,199.79 |
| `rnn.cpu.tokens_per_second_mean` | 4,290.51 |
| `rnn.cpu.tokens_per_second_std` | 44.9155 |
| `lstm.cuda.tokens_per_second_mean` | 164,100.08 |
| `lstm.cuda.tokens_per_second_std` | 2,996.51 |
| `lstm.cpu.tokens_per_second_mean` | 4,027.21 |
| `lstm.cpu.tokens_per_second_std` | 15.5268 |
| `transformer.cuda.tokens_per_second_mean` | 286,163.33 |
| `transformer.cuda.tokens_per_second_std` | 7,949.74 |
| `transformer.cpu.tokens_per_second_mean` | 4,959.89 |
| `transformer.cpu.tokens_per_second_std` | 109.0474 |

**Details**

| Name | Value |
|---|---|
| `num_targets` | 981,353 |
| `target_checksum` | 4f77927a1cd952f56697be95540ea0fc002013d9f8278d3f95c1290e6c305605 |
| `costs_csv` | results/part1_costs.csv |
| `per_model.ngram.cpu.tokens_per_second_mean` | 411,333.96 |
| `per_model.ngram.cpu.tokens_per_second_std` | 73,730.15 |
| `per_model.ngram.cpu.seconds_per_repeat` | `[2.7233127239996975, 2.5977159550002398, 1.9790375080010563]` |
| `per_model.ngram.cpu.perplexity` | 111.3124 |
| `per_model.rnn.cuda.tokens_per_second_mean` | 229,944.45 |
| `per_model.rnn.cuda.tokens_per_second_std` | 1,199.79 |
| `per_model.rnn.cuda.seconds_per_repeat` | `[4.2684084929987876, 4.2453272539987665, 4.289845125998909]` |
| `per_model.rnn.cuda.perplexity` | 109.3664 |
| `per_model.rnn.cpu.tokens_per_second_mean` | 4,290.51 |
| `per_model.rnn.cpu.tokens_per_second_std` | 44.9155 |
| `per_model.rnn.cpu.seconds_per_repeat` | `[226.0493249200008, 229.60479648399996, 230.5752105540014]` |
| `per_model.rnn.cpu.perplexity` | 109.3663 |
| `per_model.lstm.cuda.tokens_per_second_mean` | 164,100.08 |
| `per_model.lstm.cuda.tokens_per_second_std` | 2,996.51 |
| `per_model.lstm.cuda.seconds_per_repeat` | `[5.8768716100003076, 5.972537378000197, 6.095219308999731]` |
| `per_model.lstm.cuda.perplexity` | 80.4494 |
| `per_model.lstm.cpu.tokens_per_second_mean` | 4,027.21 |
| `per_model.lstm.cpu.tokens_per_second_std` | 15.5268 |
| `per_model.lstm.cpu.seconds_per_repeat` | `[242.86494289700022, 243.47310186100003, 244.7108190759991]` |
| `per_model.lstm.cpu.perplexity` | 80.4492 |
| `per_model.transformer.cuda.tokens_per_second_mean` | 286,163.33 |
| `per_model.transformer.cuda.tokens_per_second_std` | 7,949.74 |
| `per_model.transformer.cuda.seconds_per_repeat` | `[3.340338027999678, 3.4220091059996776, 3.5310044449997804]` |
| `per_model.transformer.cuda.perplexity` | 156.9419 |
| `per_model.transformer.cpu.tokens_per_second_mean` | 4,959.89 |
| `per_model.transformer.cpu.tokens_per_second_std` | 109.0474 |
| `per_model.transformer.cpu.seconds_per_repeat` | `[195.43262368999967, 195.3237406190001, 203.01145059699957]` |
| `per_model.transformer.cpu.perplexity` | 156.9369 |
| `torch_cpu_threads` | 1 |
| `smoke_test` | False |

### My notes

<!-- my-notes:part3:start -->

<!-- my-notes:part3:end -->

## Settings

Source: `DECISIONS.md` (copied as written there).

### NEEDS REVIEW (22)

| ID | Decision | Status |
|---|---|---|
| G2 | Prompts file and its format | file name CONFIRMED; format NEEDS REVIEW. `prompts.txt` (the F1 prompts, renamed by the user from `prompt.txt`) exists and was used for `results/generations_ngram.json`. |
| G3 | Random seed per generation | NEEDS REVIEW |
| G4 | Order of the sampling dials, and their edge cases | NEEDS REVIEW |
| G5 | How generated text is stored | NEEDS REVIEW |
| G6 | How the n-gram model is prepared for generation | NEEDS REVIEW |
| G7 | How neural models plug in | NEEDS REVIEW |
| G8 | Generation files are full run records | NEEDS REVIEW |
| J1 | Which runs belong to Data / Part I / Part II / Part III (revised 2026-10-12 after reading the assignment brief) | NEEDS REVIEW (naming convention and bigram placement only) |
| J2 | "Close to the vocabulary size" threshold | NEEDS REVIEW |
| J3 | How the exact command of each run is known | NEEDS REVIEW |
| J4 | How the test-set and padding checks are evidenced | NEEDS REVIEW |
| J5 | How "My notes" are protected | NEEDS REVIEW |
| J6 | Figures, and `update_results.py` not saving a run record | NEEDS REVIEW |
| K1 | Report and submission requirement list (revised 2026-10-12 from the brief) | NEEDS REVIEW |
| K2 | Schedule and planned days | NEEDS REVIEW |
| K3 | What counts as "waiting on me now", and how interpretation items are counted | NEEDS REVIEW |
| L3 | Dropout placement and layout | NEEDS REVIEW |
| L4 | Smoke-test data | NEEDS REVIEW |
| M3 | Block style and other defaults | NEEDS REVIEW |
| M4 | Training settings copied from the LSTM | NEEDS REVIEW |
| N1 | Test settings | NEEDS REVIEW |
| O2 | Other benchmark details | NEEDS REVIEW |

### All design choices (68)

| ID | Decision | Value | Where | Chosen by | Status |
|---|---|---|---|---|---|
| A1 | Dataset file | IMDB Dataset of 50K Movie Reviews, `data/IMDB Dataset.csv` (Kaggle, 50,000 reviews, 25,000 positive / 25,000 negative). | `configs/data.json:16`, `configs/inspect_data.json:16`; documented in `README.md:5-19`. | User (downloaded it manually; `README.md` Kaggle link). The assignment's required dataset is not recorded in the repo. | CONFIRMED (dataset), README placeholder still to fix. |
| A2 | Subsample size: none for counting-based work; decided per-script for future neural training | `null` (no subsampling) for `data.py`/vocabulary/n-gram, which all use the full 50,000 reviews. Whether to subsample is deferred to each future neural-LM training script individually, not fixed globally. | `configs/data.json:13`; handled at `src/data.py:280-287`. | User, prompt (2026-10-10, reviewing NEEDS REVIEW): picked "keep full data for counting models, subsample only for neural training later" over keeping full data for everything, or subsampling to ~12,500 globally now. | CONFIRMED |
| A3 | Word budget for the subsample suggestion | 3,000,000 words; suggestion rounded down to a multiple of 500; shows 3 example reviews, 500 characters each. | `configs/inspect_data.json:17-20`; used at `src/inspect_data.py:97-103`. | Unknown (earlier session) for the 3M figure; User, prompt (2026-10-10, reviewing NEEDS REVIEW) confirmed "keep as-is" over changing the number. | CONFIRMED |
| A4 | Word counting in the inspect script | rough whitespace-split word count (not the project tokenizer). | `src/inspect_data.py:68`. | Assistant default, originally; User, prompt (2026-10-10, reviewing NEEDS REVIEW) confirmed "keep as-is" over switching to the real tokenizer. | CONFIRMED |
| A5 | Text cleaning: strip HTML tags, lowercase | every `&lt;...&gt;` tag (e.g. `&lt;br /&gt;`) is replaced by a space; text is lowercased; nothing else is cleaned (no number, URL, HTML-entity, or accent handling). | `src/data.py:58` (tag regex), `src/data.py:71` (`.lower()`). | User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep as-is" over adding number normalization or broader cleaning (HTML entities, etc.). | CONFIRMED |
| A6 | Tokenisation rule | word-level, using the regex `\w+\|[^\s\w]` — words (Unicode-aware) and each punctuation character as its own token. (So "don't" becomes `don`, `'`, `t`.) | `src/data.py:65`. | *Word-level* = User, CLAUDE.md:6. *The specific regex* = User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep as-is" over keeping numbers as single tokens or switching to a library tokenizer (e.g. NLTK). | CONFIRMED |
| A7 | Special tokens and their ids | `&lt;pad&gt;`=0, `&lt;unk&gt;`=1, `&lt;eos&gt;`=2. | `src/data.py:50-51`. | `&lt;pad&gt;` and `&lt;unk&gt;` existing = User, CLAUDE.md:6. Their ids and the extra `&lt;eos&gt;` token = User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep as-is" over adding a `&lt;bos&gt;` token too. | CONFIRMED |
| A8 | End-of-document marker | one `&lt;eos&gt;` token appended to the end of every review; no start-of-review token in the data. `&lt;eos&gt;` is a normal predicted token for language models. | `src/data.py:177`. | User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep current convention" over adding a symmetric `&lt;bos&gt;`. | CONFIRMED |
| A9 | Vocabulary: size, source, and rule | the 20,000 most frequent words of the TRAINING split, plus the 3 special tokens (20,003 total); no minimum-frequency cut; everything else maps to `&lt;unk&gt;`. Allowed range enforced: 10,000-20,000. | `configs/data.json:12` (size); `src/data.py:149-165` (built from training reviews only); `src/data.py:260` (range check). | *One shared vocabulary with `&lt;unk&gt;`/`&lt;pad&gt;`* = User, CLAUDE.md:6. *Size 20,000* = User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep 20,000" over reducing to 10,000 or another value. *Training-only source* = treated as settled (standard practice to avoid leaking val/test information into the vocabulary; not re-litigated as a genuine alternative). | CONFIRMED |
| A10 | Split ratios and stratification | 80% train / 10% validation / 10% test, stratified by sentiment label, done before the vocabulary is built. | `configs/data.json:18-19`; `src/data.py:119-134`. | *Fixed splits, created once, saved and reused* = User, CLAUDE.md:7. *The 80/10/10 ratio and stratification* = User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep 80/10/10, stratified" over 90/5/5 or a custom ratio. | CONFIRMED |
| A11 | Random seed | 0, in every config (data split, example picking, model initialisation, n-gram). | `configs/data.json:14`, `configs/inspect_data.json:14`, `configs/sanity_check.json:14`, `configs/ngram.json:14`, `configs/test_evaluate.json`. | *Setting and recording a seed* = User, CLAUDE.md:8. *The value 0* = User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep seed 0" over picking a different value; open to multi-seed averaging later for final neural models without redoing these results. | CONFIRMED |
| B1 | Untrained bigram sanity model | bigram language model (embedding -&gt; linear layer over the vocabulary), randomly initialised with PyTorch's default init, not trained. | `src/sanity_check.py:43-67`; `configs/sanity_check.json:2,6,7`. | Unknown (earlier session; the user asked for a sanity check of the data pipeline) for the original choice; User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep the bigram sanity check" over also adding a uniform-distribution baseline. | CONFIRMED |
| B2 | Embedding dimension 100 | 100. | `configs/sanity_check.json:5`. | User, CLAUDE.md:9 ("All neural models use embedding dimension 100, to match GloVe 100d"). | CONFIRMED |
| B3 | Sanity-check truncation and batch size | reviews truncated to 256 tokens; batch size 32; reviews sorted by length before batching. | `configs/sanity_check.json:9,17`; `src/sanity_check.py:89,98`. | Assistant default, introduced as a performance fix (a padded batch with a very long review made the logits tensor several GB). User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep 256/32" after seeing the verified &lt;0.2% effect on the result. | CONFIRMED |
| B4 | Loss ignores padding, uses only preceding tokens | `CrossEntropyLoss(ignore_index=pad_id)` in the sanity check; masking of padding in `evaluate()`; each target predicted from earlier tokens only. | `src/sanity_check.py:78`; `src/evaluate.py:138`. | User, CLAUDE.md:10-11. | CONFIRMED |
| C1 | What counts as a target token | for a review `[t0 ... tL-1]` the model reads `t0..tL-2` and is scored on `t1..tL-1`. So the first token of each review is never scored, and `&lt;eos&gt;` and `&lt;unk&gt;` ARE scored as ordinary targets. Reviews with fewer than 2 tokens are skipped. | `src/evaluate.py:104` (skip), `:126`, `:130` (shift), `:138` (padding mask). | Assistant default, originally. User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep as-is" over excluding `&lt;eos&gt;`/`&lt;unk&gt;` from scoring. | CONFIRMED |
| C2 | Perplexity definition and scoring truncation | perplexity = exp(mean negative log-likelihood per scored token, natural log). No truncation by default (`max_length` is passed explicitly by each caller); currently sanity_check.py uses 256, ngram.py uses null (full reviews) — their numbers are not directly comparable to each other as a result. | `src/evaluate.py:149`; `configs/ngram.json:17` (`max_length`: null), `configs/sanity_check.json:17` (256). | Assistant default, originally. User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep per-script truncation as-is for now" — each script's number stands alone; when comparing two real models later, the same `max_length` must be passed to both (already supported via config). | CONFIRMED (policy for future comparisons: pass matching `max_length` explicitly, rather than a fixed global standard) |
| C3 | Scorer interface | all models are scored by one `evaluate()`; neural models via `NeuralScorer` (log-softmax + gather), the n-gram model via its own `target_log_probs`. A SHA-256 fingerprint of the scored targets (`target_checksum`) is saved so two models' results can be shown to use identical targets. | `src/evaluate.py:49-70` (NeuralScorer), `:116` (fingerprint). | Assistant default, in response to the user's prompt (2026-10-10) that both models be scored on exactly the same targets. User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep the shared evaluate() + checksum design" over dropping the checksum. | CONFIRMED |
| D1 | N-gram order | trigram (2 tokens of context). | `configs/ngram.json:7`; enforced at `src/ngram.py:277-278`. | User, prompt (2026-10-10: "a trigram language model"). | CONFIRMED |
| D2 | Smoothing method | interpolated Kneser-Ney (absolute discounting; lower orders use continuation counts). | `src/ngram.py:219-241`. | The user allowed "interpolated Kneser-Ney smoothing (or linear interpolation of unigram, bigram and trigram)". The choice between the two was Assistant default, originally. User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep Kneser-Ney" over switching to or also computing linear interpolation. | CONFIRMED |
| D3 | One discount per order (D3, D2, D1) | three separate absolute discounts (trigram, bigram, unigram level), each tuned. Chosen values: D3=0.9, D2=0.8, D1=0.1. | `src/ngram.py:219-241` (formulas), `:255-262` (tuning); chosen values are saved in `results/ngram.json` under `details.best_discounts`. | Per-order structure = Assistant default, originally. The values = result of the validation grid search the user asked for ("Tune any smoothing or interpolation weights on the VALIDATION set only"). User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep 3 discounts as-is, note D1 is uninformative" over a single shared discount or the closed-form n1/(n1+2n2) estimator. | CONFIRMED |
| D4 | Discount grid and how it was tuned | exhaustive grid over D3, D2, D1 ∈ {0.1, 0.2, ..., 0.9, 0.95} (1,000 combinations), minimising validation cross-entropy. The test split is evaluated once, after the discounts are fixed. | `configs/ngram.json:18`; `src/ngram.py:255-262` (grid search), `:304-306` (tuning on validation), `:316-318` (test evaluated after). | Validation-only tuning = User, prompt (2026-10-10) and CLAUDE.md:7. The grid values = Assistant default. | CONFIRMED (accepted with the risk explicitly logged, not silently resolved) |
| D5 | Start-of-review context symbol | one invisible start symbol (id = vocabulary size) precedes each review as context for the first scored token; never predicted. In training, only predictions of tokens 1..L-1 are counted. | `src/ngram.py:156` (symbol), `:99-118` (training events), `:117` (context at review start). | Assistant default, originally. User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep the invisible BOS symbol" over backing off to the bigram for the first token. | CONFIRMED |
| D6 | Uniform floor over the vocabulary | the unigram level is interpolated with a uniform distribution over all `vocab_size` = 20,003 entries (including `&lt;pad&gt;`, which is never a target). | `src/ngram.py:222-224`. | Assistant default, originally. User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep the uniform floor over all 20,003 entries" over excluding `&lt;pad&gt;`. | CONFIRMED |
| D7 | What "model size" and "training time" mean | size = number of distinct stored trigram + bigram + unigram entries (6,445,342), with the context-summary rows reported separately; lower orders store continuation counts, not raw counts. Training time = time to count n-grams from the training split, excluding data loading and discount tuning (tuning time is reported separately). | `src/ngram.py:190-201` (size), `:295-298` (timing). | Assistant default (interpretation of the user's "training time and model size (number of stored n-gram counts)"), originally. User, prompt (2026-10-10, reviewing NEEDS REVIEW): confirmed "keep these definitions" over folding tuning time into training time. | CONFIRMED |
| D8 | Evaluation batch size | 32 (speed only; verified not to change the result). | `configs/ngram.json:9`. | Assistant default, originally; User, prompt (2026-10-10, reviewing NEEDS REVIEW) confirmed "keep 32" (trivial, no effect on results). | CONFIRMED |
| E1 | Config files: JSON, one shared schema | JSON files in `configs/`, all containing the same 13 keys (null where not applicable) plus an `options` block. | `configs/*.json`; `src/runconfig.py:53-68`. | JSON and shared schema = User, prompt (2026-10-10, selected from options offered). The `options` block for script-specific settings = Assistant default, originally; User, prompt (2026-10-10, reviewing NEEDS REVIEW) confirmed "keep the options block" over formalizing per-script schemas. | CONFIRMED (format, schema, and `options` block) |
| E2 | Run records | every run writes `results/&lt;run&gt;.json` with: config used, seed, git commit (+ a flag for uncommitted code), hardware (CPU or GPU name), date, per-epoch losses, metrics, details. | `src/runconfig.py:144-224`. | User, prompts (2026-10-10). | CONFIRMED |
| E3 | Python used to run scripts | `/usr/local/bin/python3` (the default `python` on this machine, from Anaconda, has no PyTorch). | not stated | Assistant default (environment workaround), originally; User, prompt (2026-10-10, reviewing NEEDS REVIEW) confirmed "keep using /usr/local/bin/python3 as-is" over setting up a dedicated project virtual environment. | CONFIRMED |
| F1 | Generation prompts | three prompts: `this movie was one of the best` / `i was really disappointed by the acting and` / `the story follows a young man who`. | `prompts.txt:5`, `:8`, `:11` (reasons are in the `#` comment line under each prompt). | User, prompt (2026-10-10). The prompts were suggested in a chat response that the user pasted and adopted; no prompt-reading or generation script exists yet. | CONFIRMED |
| G1 | Default decoding settings (2026-10-11) | temperature 1.0; top-k 40; maximum new tokens 50; 1 sample per prompt; never sample `&lt;pad&gt;` or `&lt;unk&gt;`; stop when the model produces `&lt;eos&gt;`; seed 0 (the project seed, see A11). | `configs/generations_ngram.json:20-25`; used in `src/generate.py:103-142`. The same values are meant to be used for every model so outputs are comparable. | User, prompt (2026-10-11), each picked from options offered. | CONFIRMED |
| G2 | Prompts file and its format | prompts are read from `prompts.txt` (a file the user is writing); one prompt per line; blank lines and lines starting with `#` are skipped. The prompts are cleaned and tokenized with the same function as the training data; words not in the vocabulary become `&lt;unk&gt;`. | `configs/generations_ngram.json:19`; `src/generate.py:150-161` (`read_prompts`), `:164-170` (`encode_prompt`). | file name `prompts.txt` = User, prompt (2026-10-11). The line format and `#` comment rule = Assistant default (chosen so the file can carry `#` reason lines under each prompt; see F1). | file name CONFIRMED; format NEEDS REVIEW. `prompts.txt` (the F1 prompts, renamed by the user from `prompt.txt`) exists and was used for `results/generations_ngram.json`. |
| G3 | Random seed per generation | each (prompt, sample number) gets its own random generator seeded from `[seed, sample number, CRC32 of the prompt text]`, so output depends only on those three things. | `src/generate.py:144-146`. | Assistant default. | NEEDS REVIEW |
| G4 | Order of the sampling dials, and their edge cases | banned words are removed first, then temperature is applied, then top-k (so top-k keeps the k likeliest *allowed* words), then the probabilities are rescaled. Temperature must be greater than 0 (0 would mean "always the likeliest word" and is not supported); top-k 0 means off. | `src/generate.py:103-121`. | Assistant default. | NEEDS REVIEW |
| G5 | How generated text is stored | the stopping `&lt;eos&gt;` is not included in the stored tokens (a flag `ended_with_eos` records it); text is the prompt tokens plus new tokens joined with single spaces, so punctuation is separate (e.g. `can ' t`); no detokenization. | `src/generate.py:137-141` (eos), `:237-238` (text). | Assistant default. | NEEDS REVIEW |
| G6 | How the n-gram model is prepared for generation | the counts are re-fitted from the training split (about 1 second); the discounts are NOT re-tuned but read from the saved n-gram run (`results/ngram.json`, `details.best_discounts`), whose date and commit are stored in the generation record. | `src/generate.py:174-191`; `configs/generations_ngram.json:18`. | Assistant default. | NEEDS REVIEW |
| G7 | How neural models plug in | any `nn.Module` with `forward(ids) -&gt; logits` is wrapped by `NeuralNextToken`, which feeds the whole context (prompt + generated so far) each step and reads the last position's logits. No neural model is loadable yet (`build_model` stops with a message). | `src/generate.py:87-97`, `:174-190`. | Assistant default. | NEEDS REVIEW |
| G8 | Generation files are full run records | `results/generations_&lt;model&gt;.json` is a normal run record (config, seed, commit, hardware, date, metrics) with the decoding settings (temperature, top_k, max_tokens, seed, plus the other settings) and the generated texts as extra top-level fields. The file name comes from the config file name (`configs/generations_ngram.json` -&gt; `generations_ngram`). | `src/generate.py:252-263`; `src/runconfig.py:189-225` (`extra`). | Assistant default (to satisfy both "every run saves its record" and "generation files store their decoding settings"). | NEEDS REVIEW |
| H1 | Context length | 64 tokens per window. | `configs/show_batch.json:7` (`context_length`); future training configs must set their own. | User, prompt (2026-10-11), from options offered. | CONFIRMED |
| H2 | Window stride | non-overlapping windows (stride = context length): window i starts at token i*L; each token is a target exactly once. | `src/data.py:255-296` (`WindowDataset`, `start = i * self.context_length`). | User, prompt (2026-10-11), from options offered. | CONFIRMED |
| H3 | Leftover tail of a split's token stream | the tail shorter than L+1 tokens is dropped (no padding); windows run across review boundaries (reviews are separated by `&lt;eos&gt;`). | `src/data.py:276-278` (`num_windows = (N - 1) // L`). | User, prompt (2026-10-11), from options offered. | CONFIRMED |
| H4 | Batch size | 64. | `configs/show_batch.json:9` (`batch_size`). | User, prompt (2026-10-11), from options offered. | CONFIRMED |
| H5 | Shuffling and last batch | train loader shuffles each epoch using a `torch.Generator` seeded from the config seed; val/test loaders do not shuffle; the final batch of an epoch may be smaller than `batch_size` (not dropped). | `src/data.py:299-315` (`make_dataloader`). | Assistant default (shown in the design the user approved, but not offered as options); User, prompt (2026-10-11, reviewing NEEDS REVIEW) confirmed "keep those". | CONFIRMED |
| I1 | Early stopping and epoch cap | stop after 3 epochs without validation improvement; at most 20 epochs. | `configs/train_bigram.json:12` (`epochs`), `:20` (`options.patience`); used in `src/train.py` (`fit`). | User, prompt (2026-10-11), from options offered. | CONFIRMED |
| I2 | Learning rate and gradient clipping | Adam, learning rate 1e-3; gradients clipped to total norm 1.0. | `configs/train_bigram.json:10` (`learning_rate`), `:21` (`options.grad_clip`); `src/train.py` (`fit`, `train_one_epoch`). | User, prompt (2026-10-11), from options offered. | CONFIRMED |
| I3 | Models behind `--model` | only `bigram` (the existing `BigramLM`) for now; others are added with one line in `MODELS`. | `src/train.py` (`MODELS`). | User, prompt (2026-10-11), from options offered. | CONFIRMED |
| I4 | Validation/test review length | reviews cut to their first 256 tokens before scoring with `evaluate()`; eval batch size 64. | `configs/train_bigram.json:22-23` (`options.eval_max_length` = user's choice; `options.eval_batch_size` = assistant default, affects speed only). | `eval_max_length`: User, prompt (2026-10-11), from options offered. | CONFIRMED |
| I5 | Training-script defaults | (a) the reported training loss is the token-weighted running average over the epoch's batches (computed while the weights are changing, dropout on), so it is not directly comparable to validation loss; (b) a checkpoint is saved only when validation loss is strictly lower than the best so far; (c) the best weights go to `checkpoints/&lt;run&gt;.pt` (git-ignored); (d) device is CUDA if available, else CPU; (e) training time is wall-clock seconds around the epoch loop (including validation, excluding data loading and the final test). | `src/train.py` (`train_one_epoch`, `fit`, `main`); `configs/train_bigram.json:19` (`options.checkpoint_dir`). | Assistant default (shown in the design the user approved, not offered as options); User, prompt (2026-10-11, reviewing NEEDS REVIEW) confirmed "keep those". | CONFIRMED |
| J1 | Which runs belong to Data / Part I / Part II / Part III (revised 2026-10-12 after reading the assignment brief) | the brief defines the parts: Part I = n-gram, RNN, LSTM, Transformer comparison; Part II = three embedding settings on one architecture; Part III = downstream classifier on representations. Rules in order: `data`, `inspect_data`, `show_batch` -&gt; Data; any run name containing `p2_` -&gt; Part II; containing `p3_` -&gt; Part III; `sanity_check`, `ngram`, `train_*`, `test_*`, `generations_*` -&gt; Part I. **Naming convention for later runs: Part II runs start with `p2_`, Part III runs with `p3_`.** The bigram run (`train_bigram`) is not required by the brief; it is listed in Part I as a baseline. Each of Parts I and II shows its own perplexity/cost table and side-by-side generated text; Part III shows its run records. | `configs/update_results.json`, `options.part_rules`, `options.part_comparison_extra_runs` (runs from another part to include in a part's comparison, e.g. the reused Part I model in Part II). | Parts: the brief (`Assignment1_2026.pdf`). Run-name convention and placing `train_bigram` in Part I: Assistant default. | NEEDS REVIEW (naming convention and bigram placement only) |
| J2 | "Close to the vocabulary size" threshold | untrained validation perplexity divided by vocabulary size must be between 0.5 and 2.0 to PASS. | `configs/update_results.json`, `options.untrained_ppl_to_vocab_ratio_range`. | Assistant default. | NEEDS REVIEW |
| J3 | How the exact command of each run is known | `save_run` now stores a `command` field (interpreter + arguments) in every new `results/&lt;run&gt;.json`. For older records, the command is copied from `results/RESULTS_LOG.md`; if not found there it is shown as "not recorded". | `src/runconfig.py` (`save_run`); lookup in `src/update_results.py` (`find_command`). | Assistant default. | NEEDS REVIEW |
| J4 | How the test-set and padding checks are evidenced | static scan of the source with Python `ast`: in `src/ngram.py` and `src/train.py`, `data.test` must be used exactly once, on a line after the `tune_discounts(...)` / `fit(...)` call, and no other `src/` file may use it. Padding: every `CrossEntropyLoss` in `src/train.py` and `src/sanity_check.py` must pass `ignore_index`, and the recorded test `test_padding_is_ignored` must be `passed`. Frozen embeddings: PASS needs a run with a pretrained `embedding_init` whose `details.frozen_embedding_max_abs_change` equals 0; `src/train.py` does not write that key yet, so a future Part II change must add it. Causality: any recorded test with "causal" in its name. | `configs/update_results.json` (`test_set_markers`, `padding_loss_files`, `padding_test_*`, `causality_test_keyword`, `frozen_embedding_detail_key`, `not_frozen_embedding_inits`); logic in `src/update_results.py` (`check_*`). | Assistant default. | NEEDS REVIEW |
| J5 | How "My notes" are protected | each section ends with `### My notes` and a pair of HTML-comment marker lines; text between the markers is copied back unchanged on every rebuild. If the markers are missing from the old file, the script stops (override with `--force`). | `src/update_results.py` (`extract_notes`, `notes_block`). | Assistant default. | NEEDS REVIEW |
| J6 | Figures, and `update_results.py` not saving a run record | matplotlib PNGs in `results/figures/`: tokens per split, n-gram discount tuning (D3 axis, D2/D1 at their chosen values), train/validation loss per trained run, and validation loss of all trained runs when there are two or more. Colours are the first slots of the dataviz skill's validated categorical palette. Runs named `test_*` are not plotted. `update_results.py` writes `RESULTS.md` and the figures but no `results/update_results.json`, because it is not an experiment run. | `configs/update_results.json` (`figure_colours`, `test_run_prefix`); `src/update_results.py` (`fig_*`). | Assistant default. | NEEDS REVIEW |
| K1 | Report and submission requirement list (revised 2026-10-12 from the brief) | 23 requirements taken from `Assignment1_2026.pdf` (Part I, II, III bullets, the report contents list, 6-page limit, citations, README, no credentials, submission, late policy), each tagged with the brief's points where known. Automatic evidence only for: the four Part I model families being present, the frozen-embedding check, README keywords, repo state. Everything else shows "not done" until the user ticks it. The split of each part's bullets into separate items is the assistant's. | `configs/update_results.json`, `options.report_requirements`. | Content: the brief. Granularity: Assistant default. | NEEDS REVIEW |
| K2 | Schedule and planned days | `options.schedule` is empty, so every item shows "Planned: not set" and nothing can be overdue. Deadline is Fri 16 Oct 2026, 23:59 Singapore time (from the user's prompt). | `configs/update_results.json`, `options.schedule`, `options.deadline`, `options.timezone`. | Deadline: User, prompt (2026-10-12). Empty schedule: Assistant default (no schedule was available to read). | NEEDS REVIEW |
| K3 | What counts as "waiting on me now", and how interpretation items are counted | Waiting = empty schedule, unresolved NEEDS REVIEW decisions, configs with no results yet, sections with results but empty *My notes*, failing checks, uncommitted or unpushed git changes. "Results to interpret" has one item per section (Data, Part I, Part II, Part III), because *My notes* exist per section, not per run. "Runs to do" lists configs in `configs/` with no `results/&lt;run&gt;.json` plus `options.extra_items` (steps from the brief that no script can detect, such as scripts still to be written); time is taken from earlier runs of the same script. | `src/my_tasks.py` (`build_items`); `configs/update_results.json` (`tasks_ignore_configs`, `run_script_rules`, `extra_items`). | Assistant default. | NEEDS REVIEW |
| L1 | Default size of the RNN and LSTM | hidden size 256, 2 recurrent layers, dropout 0.3, embedding dimension 100 (randomly initialised, trainable). | `configs/train_rnn.json` and `configs/train_lstm.json`, lines 3-4 (`layers`, `hidden_size`) and 11 (`dropout`); overridable with `--hidden-size`, `--layers`, `--dropout`. | User, prompt (2026-10-12), from options offered. | CONFIRMED |
| L2 | RNN nonlinearity | tanh. | `src/models.py` (`RNNLanguageModel`, `nonlinearity="tanh"`). | User, prompt (2026-10-12), from options offered. | CONFIRMED |
| L3 | Dropout placement and layout | dropout after the embedding, between stacked recurrent layers (PyTorch's `dropout=`, set to 0 when there is 1 layer) and before the output layer; unidirectional recurrent layers; no weight tying; the final hidden state is not carried between windows. | `src/models.py` (`forward` of both classes). | Assistant default (shown in the design the user approved, not offered as options). | NEEDS REVIEW |
| L4 | Smoke-test data | the first 60 train / 40 val / 40 test reviews of the saved splits, copied to a scratch folder (not `data/processed`), 1 epoch, output to a scratch results folder. | scratchpad only (not a repository file). | Assistant default for how to make the "small subset" (the user asked for a small subset). | NEEDS REVIEW |
| M1 | Size: blocks, heads, feed-forward width | 2 decoder blocks, 4 attention heads (25 dims each), feed-forward width 256, dropout 0.3, embedding dimension 100 (random, trainable). | `configs/train_transformer.json` lines 3-4 (`layers`, `hidden_size`), 11 (`dropout`) and `options.num_heads`. | User, prompt (2026-10-12), from options offered ("proceed with your recommended set"). | CONFIRMED |
| M2 | Positional embedding table size | `max_positions` 256 (learned positional embeddings), while training windows stay at `context_length` 64. Positions 64-255 are never updated in training (they stay at their random start values), but evaluation (`eval_max_length` 256) feeds sequences up to 256. | `configs/train_transformer.json`, `options.max_positions`. | User, prompt (2026-10-12), from options offered (256 / 64 with sliding window / train at 256). | CONFIRMED |
| M3 | Block style and other defaults | pre-norm blocks (LayerNorm before attention and feed-forward) plus one final LayerNorm; GELU activation; dropout after the embeddings and inside each block; positional embedding starts at PyTorch's default random values; no weight tying; causal mask built with `torch.triu` and passed to `nn.TransformerEncoder`. | `src/models.py`, class `TransformerLanguageModel`. | Assistant default (shown in the design the user approved, not offered as options). | NEEDS REVIEW |
| M4 | Training settings copied from the LSTM | learning rate 0.001, batch size 64, epochs 20, context length 64, patience 3, grad clip 1.0, vocabulary 20000, seed 0. | `configs/train_transformer.json`. | Assistant default (copied from `configs/train_lstm.json` so the comparison is matched; not offered as options). | NEEDS REVIEW |
| N1 | Test settings | random sequences of length 12, batch 4, tolerance 1e-6 on the largest change in any probability; every position t from 0 to 11 is changed in turn; models are in eval mode (dropout off); every class in `models.py` is tested, found automatically. The bigram (in `sanity_check.py`) is not covered. | `configs/test_causality.json` (`batch_size`, `options.seq_len`, `options.tolerance`). | Assistant default (the user specified what to check, not these values; not offered as options). | NEEDS REVIEW |
| O1 | Workload, warm-up and devices | batch size 64 and context length 256 (the same as the evaluation settings, so the speed is for the exact scoring that gave the perplexities); 2 full warm-up passes; 3 timed repeats (mean and standard deviation); every neural model timed on the GPU and on the CPU, the n-gram on the CPU only (NumPy). | `configs/benchmark.json` (`batch_size`, `context_length`, `options.warmup_passes`, `options.repeats`, `options.devices`). | User, prompt (2026-10-13): 3 repeats; the rest from options offered ("proceed with your recommended set"). | CONFIRMED |
| O2 | Other benchmark details | the timed region is the whole `evaluate()` call (sorting, padding and scoring included); the n-gram is refit from the training split with its saved discounts, untimed; the n-gram "parameters" are its stored counts, with a column saying so; n-gram training and tuning seconds are separate columns; a run stops if a model's perplexity differs from its saved test perplexity by more than 0.1% (neural models only; the n-gram was scored without the 256 cut) or if the scored targets differ between models. | `src/benchmark.py`; `configs/benchmark.json` (`options.perplexity_rel_tolerance`). | Assistant default (shown in the design the user approved, not offered as options). | NEEDS REVIEW |

### My notes

<!-- my-notes:settings:start -->

<!-- my-notes:settings:end -->

## Checks

| Check | Result |
|---|---|
| Untrained perplexity is close to the vocabulary size | **PASS** |
| Test set used only for final evaluation | **FAIL** |
| Padding excluded from the loss | **PASS** |
| All models scored on identical test targets | **FAIL** |
| Generation runs use matched decoding settings and prompts | **NOT YET AVAILABLE** |
| Causality test | **NOT YET AVAILABLE** |
| Frozen embeddings unchanged | **NOT YET AVAILABLE** |

#### Untrained perplexity is close to the vocabulary size: PASS

- untrained validation perplexity 23,926.5; vocabulary size (with special tokens) 20,003
- ratio 1.196; accepted range 0.5 to 2.0 (`configs/update_results.json`, `untrained_ppl_to_vocab_ratio_range`)
- source: `results/sanity_check.json`

#### Test set used only for final evaluation: FAIL

- `src/ngram.py`: `data.test` used at line(s) [333]; `tune_discounts(...)` called at line(s) [321]; single use after the call
- `src/train.py`: `data.test` used at line(s) [220]; `fit(...)` called at line(s) [216]; single use after the call
- other files in `src/` that use `data.test`: `src/benchmark.py` lines [152]
- method: static scan of the source (Python `ast`), not a run-time trace

#### Padding excluded from the loss: PASS

- `src/train.py`: `CrossEntropyLoss(ignore_index=...)` at line(s) [150]; without `ignore_index` at none
- `src/sanity_check.py`: `CrossEntropyLoss(ignore_index=...)` at line(s) [78]; without `ignore_index` at none
- `test_padding_is_ignored` in `results/test_evaluate.json`: passed

#### All models scored on identical test targets: FAIL

- `ngram`: 1,390,673 targets, checksum `fd778f4c313d...`
- `train_bigram`: 981,353 targets, checksum `4f77927a1cd9...`
- `train_bigram_colab`: 981,353 targets, checksum `4f77927a1cd9...`
- `train_lstm`: 981,353 targets, checksum `4f77927a1cd9...`
- `train_rnn`: 981,353 targets, checksum `4f77927a1cd9...`
- `train_transformer`: 981,353 targets, checksum `4f77927a1cd9...`

#### Generation runs use matched decoding settings and prompts: NOT YET AVAILABLE

- 1 generation run(s); at least two are needed to compare

#### Causality test: NOT YET AVAILABLE

- no recorded test with 'causal' in its name

#### Frozen embeddings unchanged: NOT YET AVAILABLE

- no run with a pretrained `embedding_init` (Part II has no such run yet)

### My notes

<!-- my-notes:checks:start -->

<!-- my-notes:checks:end -->
