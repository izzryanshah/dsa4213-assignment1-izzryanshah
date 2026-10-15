# DSA4213 Assignment 1 — NLP for Data Science

Word-level language models on IMDB movie reviews. This README says how to get the data, set up the environment and re-run every part, with the exact commands, configs and seeds.

**Status (2026-10-15):** all three parts are implemented and have recorded results: Part I (trigram Kneser-Ney, RNN, LSTM, Transformer, benchmark, matched generation), Part II (LSTM with random, own Skip-gram and public GloVe 100d embeddings), Part III (frozen LSTM and Transformer features, logistic-regression classifier). The report is written separately; a draft is in `report/` (see section 8). Neural training, the benchmark, neural generation, feature extraction and the GloVe download run only in Colab on a GPU (section 4); everything else runs locally.

## 1. Data access

This project uses the **IMDB Dataset of 50K Movie Reviews** (50,000 reviews, labelled `positive`/`negative`).

The dataset is **not included in this repository** and is not redistributed here: `data/` is git-ignored, so neither the raw CSV nor the processed files are ever committed. Everyone downloads it themselves from Kaggle under Kaggle's terms.

1. Download `IMDB Dataset.csv` from <https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews> (a free Kaggle account is needed; the download is done manually in the browser, no API key is used or stored).
2. Place the file at `data/IMDB Dataset.csv` in this repository.
3. Check it loaded correctly:

   ```
   python src/inspect_data.py --config configs/inspect_data.json
   ```

   Expected: 50,000 reviews, 25,000 per label.

### Citation

This Kaggle dataset is itself built from the IMDB movie review data originally released by Maas et al. (2011). Cite both the Kaggle page and the original paper:

- Kaggle dataset page: <https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews>
- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis*. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

## 2. Preprocessing (what `src/data.py` does)

Run once; every other script reuses its output in `data/processed/`.

1. Remove HTML tags (e.g. `<br />`, replaced by a space) and lowercase.
2. Word-level tokenization with the regex `\w+|[^\s\w]`: words, and each punctuation character as its own token.
3. Split the 50,000 reviews 80/10/10 into train/validation/test, stratified by label (seed 0), **before** building the vocabulary. The split is saved to `data/processed/split_assignment.csv` and reused from then on.
4. Build one shared vocabulary from the **training split only**: the 20,000 most frequent words plus `<pad>` (id 0), `<unk>` (id 1) and `<eos>` (id 2), 20,003 entries in total. Every other word maps to `<unk>`.
5. Append `<eos>` to every review and save each split as a flat array of token ids (`train.npz`, `val.npz`, `test.npz`), plus `vocab.json` and `label_map.json`.

No subsampling is used (`subsample_size` is `null`): all 50,000 reviews are kept.

The reasoning behind each of these choices is in `DECISIONS.md`.

## 3. Dependencies

Python 3.10. Install with:

```
pip install -r requirements.txt
```

`requirements.txt` lists the packages without version pins: `torch`, `numpy`, `nltk`, `gensim`, `scikit-learn`, `matplotlib`, `pandas`. (`nltk` is listed but no script imports it.) All results in this repository were produced with:

| Package | Version |
|---|---|
| Python | 3.10.11 |
| torch | 2.14.1 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| gensim | 4.4.0 |
| matplotlib | 3.10.9 |

Hardware for the recorded runs:
- **Local, CPU only (Apple M2, macOS):** data pipeline, n-gram model, word2vec, Part III data preparation, classifier, error sampling, tables and figures, tests.
- **Colab, Tesla T4 GPU:** training of the RNN, LSTM, Transformer, the bigram re-run, both frozen-embedding LSTMs, the benchmark, neural text generation, GloVe table building and feature extraction. The Colab session installs `requirements.txt` as it comes (no version pins), so its package versions are the ones Colab provided on the day; each run's `results/<run>.json` records its hardware, date and git commit.

No API keys or credentials are needed anywhere. (`pytest` is not required: the test files run as plain scripts.)

## 4. Reproducing the results

Every script takes `--config configs/<name>.json` and writes a run record to `results/<run>.json`. All seeds are 0. Run from the repository root.

Steps marked **(Colab)** must run in Colab on a GPU (`notebooks/run_colab.ipynb` has one labelled cell for each; set its `REPO_URL` and Drive folder first). They also work on a CPU, but the recorded timings, throughputs and Part I to III numbers come from a Colab Tesla T4, and the project rule is that neural training, the benchmark, neural generation, GloVe and feature extraction are not run on the laptop. Each Colab cell is followed by `save_results()`, which copies `results/`, `checkpoints/` and `RESULTS.md` to Google Drive; copy the result files back into this repository before the local steps. Everything else runs locally in seconds to minutes.

`run_part1_ngram.sh` re-runs the first local steps in one go (`bash run_part1_ngram.sh`): steps 1 to 4, the evaluate and generate tests of step 5, step 6 and `summarise_runs.py`. It does not cover the neural, Part II or Part III steps.

### Part I: language models

```
# 1. Processed dataset (about 10 seconds)
python src/data.py --config configs/data.json

# 2. Raw-data statistics (optional)
python src/inspect_data.py --config configs/inspect_data.json

# 3. Untrained bigram sanity check
python src/sanity_check.py --config configs/sanity_check.json

# 4. Trigram Kneser-Ney model: fit, tune discounts on validation, evaluate (about 30 seconds)
python src/ngram.py --config configs/ngram.json

# 5. Tests
python tests/test_evaluate.py --config configs/test_evaluate.json
python tests/test_generate.py --config configs/test_generate.json
python tests/test_models.py --config configs/test_models.json
python tests/test_train.py --config configs/test_train.json
python tests/test_causality.py --config configs/test_causality.json
python tests/test_windows.py

# 6. N-gram text (needs prompts.txt, see "Generation prompts")
python src/generate.py --config configs/generations_ngram.json

# 7. (Colab) Neural models, one run each, about 37 to 64 minutes per run on a T4
python -u src/train.py --config configs/train_lstm.json --model lstm
python -u src/train.py --config configs/train_rnn.json --model rnn
python -u src/train.py --config configs/train_transformer.json --model transformer
python -u src/train.py --config configs/train_bigram.json --model bigram --run-name train_bigram_colab   # baseline re-run on the GPU

# 8. (Colab) Inference benchmark (needs the three checkpoints and the n-gram result) and matched generation
python src/benchmark.py --config configs/benchmark.json          # writes results/benchmark.json, results/part1_costs.csv
python -u src/generate.py --config configs/generations_rnn.json  # likewise generations_lstm, generations_transformer
python src/combine_generations.py --config configs/combine_generations.json   # results/part1_generations.md
```

### Part II: embedding variants (LSTM)

The random-embedding setting reuses the Part I `train_lstm` run. The two frozen-embedding runs use `configs/train_lstm.json` with only the embedding changed (`configs/p2_lstm_w2v.json`, `configs/p2_lstm_glove.json`).

```
# 9. Own Skip-gram vectors on the training split (local CPU, about 5 minutes); times itself separately
python src/train_word2vec.py --config configs/word2vec_cost.json     # data/embeddings/w2v_100d.npy, results/word2vec_cost.json

# 10. (Colab) Public GloVe 100d table and vocabulary coverage (see "Public GloVe vectors" below)
python -u src/load_glove.py --config configs/glove_coverage.json     # data/embeddings/glove_100d.npy, results/glove_coverage.json

# 11. (Colab) The two LSTMs with the whole embedding table frozen, then their text
python -u src/train.py --config configs/p2_lstm_w2v.json --model lstm --embedding w2v
python -u src/train.py --config configs/p2_lstm_glove.json --model lstm --embedding glove
python -u src/generate.py --config configs/generations_p2_lstm_w2v.json     # likewise generations_p2_lstm_glove
python src/combine_generations.py --config configs/combine_generations_p2.json   # results/part2_generations.md
python tests/test_frozen_embedding.py --config configs/test_frozen_embedding.json
```

### Part III: frozen representations and a classifier (LSTM vs Transformer, mean pooling)

```
# 12. Labelled reviews as padded id arrays (local, seconds)
python src/prepare_classification.py --config configs/p3_prepare_classification.json   # data/classification/

# 13. (Colab) Frozen features from the Part I LSTM and Transformer checkpoints
python -u src/extract_features.py --config configs/p3_feat_lstm_mean.json
python -u src/extract_features.py --config configs/p3_feat_transformer_mean.json       # data/features/<run>/

# 14. Logistic regression on each feature set, then the error-review material (local)
python src/classify.py --config configs/p3_classify.json             # results/part3.csv, results/part3_errors_*.csv
python src/error_sample.py --config configs/p3_error_sample.json     # results/part3_error_sample_*.csv, part3_ranking.csv
python tests/test_classify.py                                        # also: test_prepare_classification.py, test_extract_features.py

# 15. Part III extension (Colab for steps a and b, the rest local)
# a. (Colab) mean-pooled features for the Part I RNN and the two Part II LSTMs (notebook section 13)
python -u src/extract_features.py --config configs/p3_feat_rnn_mean.json
python -u src/extract_features.py --config configs/p3_feat_lstm_w2v_mean.json
python -u src/extract_features.py --config configs/p3_feat_lstm_glove_mean.json
# b. (Colab) test perplexity rescored in non-overlapping 64-token windows (notebook section 14)
python -u src/rescore_windows.py --config configs/p3_rescore_windows.json     # results/part3_window_perplexity.csv
# c. (local, after copying data/features/ and results/ back from Drive) all five representations, then the rank table
python src/classify.py --config configs/p3_classify_five.json        # results/part3_five.csv
python src/rank_table.py --config configs/p3_rank_table.json         # results/part3_lm_vs_classifier.csv, Spearman rho
# d. (local) PCA to 100 dimensions of the LSTM features, and McNemar's test LSTM vs Transformer
python src/classify.py --config configs/p3_classify_pca.json         # results/part3_pca.csv
python src/mcnemar.py --config configs/p3_mcnemar.json
python tests/test_part3_extras.py
```

### Collecting everything

```
python src/summarise_runs.py                                          # results/all_runs.csv
python src/make_figures.py --config configs/make_figures.json         # results/tables/, results/figures/part{1,2}_loss_curves.pdf
python src/update_results.py --config configs/update_results.json     # RESULTS.md, MY_TASKS.md, results/figures/
```

Step 1 comes before everything else. Step 4 needs step 1; step 6 needs step 4 (it reads the discounts tuned in `results/ngram.json`); step 8 needs step 7 and step 4; step 11 needs steps 9 and 10; step 13 needs step 7 and step 12; step 14 needs step 13.

### What to expect

Numbers recorded in `results/` (seed 0), for checking a re-run. Local numbers are deterministic on the same package versions; GPU training can differ slightly between runs and machines, so a Colab re-run may not reproduce the neural numbers digit for digit.

| Run | Result |
|---|---|
| `data` | vocabulary 20,003; train 40,000 reviews / 11,067,213 tokens; val 5,000 / 1,388,412; test 5,000 / 1,395,673 |
| `sanity_check` | untrained validation perplexity 23,926.5 (983,447 scored tokens) |
| `ngram` | own scoring (no truncation): test perplexity 116.54 (1,390,673 targets); on the common 981,353 targets: 111.31 |
| Part I, common 981,353 test targets | RNN 109.37; LSTM 80.45; Transformer 156.94 (parameters 7.36M / 8.03M / 4.23M; train time 3,388 / 3,850 / 2,246 s on a T4) |
| Part II (LSTM) | random learned 80.45; own Skip-gram frozen 81.42; GloVe 100d frozen 90.28; GloVe coverage 99.56% of types, 97.52% of tokens; Skip-gram training 273.8 s on the CPU |
| Part III (logistic regression) | LSTM mean-pooled features (256-d): test accuracy 0.8192; Transformer (100-d): 0.7744 |

The full tables are `results/tables/part{1,2,3}_table.{md,csv,tex}` and `RESULTS.md`.

### Generation prompts

`src/generate.py` reads prompts from `prompts.txt` in the repository root: one prompt per line, blank lines and lines starting with `#` are ignored. The file holds the three prompts used, with the reason for each as a `#` comment under it. Every model uses the same prompts and the same decoding settings (temperature 1.0, top-k 40, 50 new tokens, 1 sample, `<pad>` and `<unk>` banned, stop at `<eos>`, seed 0), stored beside the text in `results/generations_<model>.json`.

## Public GloVe vectors (Part II)

The public embeddings are **not included in this repository** (`data/` is git-ignored) and are not downloaded by any script. The download is done in Colab (see `notebooks/run_colab.ipynb`, section 8).

1. Download `glove.6B.zip` (about 822 MB) from the Stanford NLP page <https://nlp.stanford.edu/projects/glove/>, or directly: `wget https://nlp.stanford.edu/data/glove.6B.zip`.
2. Unzip only the 100-d file into `data/embeddings/`: `unzip -j glove.6B.zip glove.6B.100d.txt -d data/embeddings`. The result must be `data/embeddings/glove.6B.100d.txt`.
3. Build the table aligned to our vocabulary and report coverage:

   ```
   python src/load_glove.py --config configs/glove_coverage.json
   ```

   This writes `data/embeddings/glove_100d.npy` (shape 20,003 x 100, row *i* = vocabulary id *i*) and `results/glove_coverage.json` (coverage as % of vocabulary types and % of training tokens, and the 20 most frequent missing words). Words with no GloVe vector (including `<unk>` and `<eos>`) get small seeded random vectors; `<pad>` is zeros. The same rule is used for the own Skip-gram table.

Citation: Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation*. EMNLP 2014.

## 5. Configurations

Every run reads one JSON file from `configs/` (no design values are hard-coded in scripts). Each config has the same keys: `architecture`, `layers`, `hidden_size`, `embedding_dim`, `embedding_init`, `context_length`, `learning_rate`, `batch_size`, `epochs`, `dropout`, `vocab_size`, `subsample_size`, `seed` (`null` where a key does not apply), plus an `options` block for settings that only one script uses.

| Config | Script | Notable settings |
|---|---|---|
| `data.json` | `src/data.py` | vocab 20,000; split 0.1 / 0.1; no subsampling; seed 0 |
| `inspect_data.json`, `sanity_check.json` | `src/inspect_data.py`, `src/sanity_check.py` | 3 example reviews; bigram, embedding dim 100, batch 32, reviews cut to 256 tokens |
| `ngram.json` | `src/ngram.py` | trigram, discount grid 0.1 to 0.95 for each of three discounts, no truncation |
| `train_rnn.json`, `train_lstm.json` | `src/train.py` | 2 layers, hidden 256, embedding 100 (random), dropout 0.3, context 64, Adam lr 0.001, batch 64, up to 20 epochs, early stopping patience 3, gradient clip 1.0, evaluation on reviews cut to 256 tokens |
| `train_transformer.json` | `src/train.py` | 2 pre-norm causal blocks, 4 heads, feed-forward width 256, positions up to 256; the training settings are those of the LSTM |
| `train_bigram.json` | `src/train.py` | bigram baseline; first run locally on CPU, re-run on the GPU as `train_bigram_colab` |
| `p2_lstm_w2v.json`, `p2_lstm_glove.json` | `src/train.py --embedding` | `train_lstm.json` with `embedding_init` w2v or glove and the whole embedding table frozen |
| `word2vec_cost.json` | `src/train_word2vec.py` | Skip-gram, 100-d, window 5, min_count 1, 5 negative samples, 10 epochs, sub-sampling 0.001, 1 worker, seed 0, training split only |
| `glove_coverage.json` | `src/load_glove.py` | `glove.6B.100d.txt`; missing words as in `DECISIONS.md` R2 |
| `benchmark.json` | `src/benchmark.py` | context 256, batch 64, 2 warm-up passes, 3 repeats, cuda and cpu |
| `generations_<model>.json`, `combine_generations*.json` | `src/generate.py`, `src/combine_generations.py` | temperature 1.0, top-k 40, 50 new tokens, 1 sample; merge refuses mismatched decoding settings |
| `p3_prepare_classification.json` | `src/prepare_classification.py` | max length 256, first 256 tokens kept, `<eos>` dropped |
| `p3_feat_lstm_mean.json`, `p3_feat_transformer_mean.json` | `src/extract_features.py` | frozen model, mean over real (non-pad) token states |
| `p3_classify.json`, `p3_error_sample.json` | `src/classify.py`, `src/error_sample.py` | logistic regression, C in [0.001 to 100] chosen on validation macro-F1, standardised features; 15 random errors per variant |
| `p3_feat_rnn_mean.json`, `p3_feat_lstm_w2v_mean.json`, `p3_feat_lstm_glove_mean.json` | `src/extract_features.py` | same settings as the LSTM / Transformer features (mean pooling) |
| `p3_classify_five.json`, `p3_classify_pca.json` | `src/classify.py` | same procedure as `p3_classify.json`; the PCA run first reduces the LSTM features to 100 dimensions (fit on train only) |
| `p3_rank_table.json`, `p3_mcnemar.json` | `src/rank_table.py`, `src/mcnemar.py` | Spearman rho of LM perplexity vs classifier score; McNemar on the LSTM vs Transformer test predictions |
| `p3_rescore_windows.json` | `src/rescore_windows.py` | same first-256-token test targets, scored in consecutive 64-token windows |
| `make_figures.json`, `update_results.json` | `src/make_figures.py`, `src/update_results.py` | which runs go into the tables, figures, `RESULTS.md` and `MY_TASKS.md` |
| `test_*.json` | `tests/` | small made-up models and vocabularies |

Any setting can be overridden on the command line without editing the file:

```
python src/ngram.py --config configs/ngram.json --seed 1 --run-name ngram_seed1
python src/generate.py --config configs/generations_ngram.json --set options.temperature=0.7 --set options.top_k=10 --run-name generations_ngram_t07
```

`--set KEY=VALUE` works for any key (use `options.name` for the options block); `--run-name` sets the output file name so the original result is not overwritten. `DECISIONS.md` gives the reason and the alternatives for every value.

## 6. Random seeds

Every script sets and prints its seed, and records it in its result file. All configs use **seed 0**.

- `data.py`: the train/validation/test split and any subsampling. Changing the seed changes the split, so all later results change too.
- `train.py`: weight initialisation, dropout and the order of training windows. The rows of a frozen table that have no vector are drawn in the table-building script (`train_word2vec.py`, `load_glove.py`) with its seed.
- `train_word2vec.py`: the gensim seed (one worker, so the run is reproducible).
- `generate.py`: each (prompt, sample number) gets its own generator seeded from the seed, the sample number and the prompt text, so the same settings give the same text.
- `classify.py`, `error_sample.py`: the logistic-regression `random_state` and the choice of the 15 errors per variant.
- `inspect_data.py`, `sanity_check.py`: which example reviews are printed; the random initial weights of the untrained model.
- `ngram.py`: the model is deterministic (counting and a grid search); the seed is set and recorded only.

## 7. Outputs and records

- `results/<run>.json`: one file per run: the full config used, the exact command, seed, git commit (and whether the code had uncommitted changes), hardware, date, per-epoch losses, final metrics and details.
- `results/all_runs.csv`: one row per run (made by `src/summarise_runs.py`).
- `results/tables/`, `results/figures/`: report tables (`part{1,2,3}_table`) and loss-curve figures (`src/make_figures.py`); `RESULTS.md` embeds the figures `src/update_results.py` draws.
- `RESULTS.md`: every result in one document (generated; the *My notes* blocks are the author's). `MY_TASKS.md`: what the author still has to do (generated, with the author's ticks kept).
- `results/RESULTS_LOG.md`: dated log of every result with the exact command and file path.
- `DECISIONS.md`: every design choice: value, where it is set, who chose it, alternatives, reason.
- `TOOL_USE.md`: dated log of what was written or changed in the repository, including what the coding assistant did.
- `ASSIGNMENT.md`: the assignment brief restated; `CLAUDE.md`: the standing rules for the coding assistant.

## 8. Repository layout

```
configs/      one JSON config per run
data/         raw CSV (you add it); processed/, embeddings/, classification/, features/ are generated; git-ignored
notebooks/    run_colab.ipynb, the Colab runner (set REPO_URL and the Drive folder first)
results/      run records, all_runs.csv, RESULTS_LOG.md, tables/, figures/, generated-text files
report/       REPORT_DRAFT.md and VERIFY.md (assistant-drafted report text for the author to verify and rewrite; not the submitted PDF)
src/          data.py, inspect_data.py, sanity_check.py, evaluate.py, ngram.py, models.py, train.py,
              benchmark.py, generate.py, combine_generations.py, embedding_utils.py, train_word2vec.py,
              load_glove.py, prepare_classification.py, extract_features.py, classify.py, error_sample.py,
              make_figures.py, runconfig.py, summarise_runs.py, update_results.py, my_tasks.py
tests/        test_*.py (run as plain scripts)
```

## 9. Notes on sharing

- Nothing secret is stored here: no API keys, tokens or credentials are used, and `.env` files are git-ignored.
- The dataset and everything derived from it (`data/`) are kept out of git; do not add them to a submission unless the dataset's terms allow redistribution.
- Model checkpoints (`*.pt`, `*.pth`, `*.ckpt`) are git-ignored too.

## 10. Citations and tool use

- Data: the Kaggle IMDB dataset and Maas et al. (2011), section 1. Public embeddings: Pennington et al. (2014), the Public GloVe section. Own embeddings: gensim's Skip-gram (Mikolov et al., 2013).
- Coding assistant: the code, tests, configs and generated documents in this repository were written with Claude Code (Anthropic) under the author's direction. `TOOL_USE.md` logs what it did, task by task; `DECISIONS.md` records which design values the author chose and which were assistant defaults. The files in `report/` are assistant-drafted text for the author to verify and rewrite, not the submitted report.
- Course notebooks adapted: none are recorded in `TOOL_USE.md`. Add any that were used.
