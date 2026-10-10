# DSA4213 Assignment 1 — NLP for Data Science

Word-level language models on IMDB movie reviews. This README says how to get the data, set up the environment and re-run everything that exists so far, with the exact commands, configs and seeds.

**Status:** data pipeline, evaluation function, a trigram Kneser-Ney language model, and text generation are implemented. Neural language models (e.g. LSTM, Transformer) and downstream tasks are not written yet, so there is nothing to reproduce for them.

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

`requirements.txt` lists the packages without version pins: `torch`, `numpy`, `nltk`, `gensim`, `scikit-learn`, `matplotlib`, `pandas`. The code so far only uses `torch`, `numpy`, `pandas` and `scikit-learn`. All results in this repository were produced with:

| Package | Version |
|---|---|
| Python | 3.10.11 |
| torch | 2.14.1 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |

Hardware for the recorded runs: CPU only (Apple M2, macOS). No GPU is needed for anything that exists so far. No API keys or credentials are needed anywhere. (`pytest` is not required: the test files run as plain scripts.)

## 4. Reproducing the results

Run from the repository root, in this order. Every script takes `--config configs/<name>.json` and writes a run record to `results/<run>.json`.

```
# 1. Build the processed dataset (about 10 seconds)
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

# 6. Text generation from the n-gram model (needs your own prompts.txt, see below)
python src/generate.py --config configs/generations_ngram.json

# 7. Collect every run into one table
python src/summarise_runs.py
```

Step 4 must come after step 1; step 6 after step 4 (it reads the discounts tuned in `results/ngram.json`).

### What to expect

Numbers recorded in `results/` (CPU, seed 0), for checking your re-run:

| Run | Result |
|---|---|
| `data` | vocabulary 20,003; train 40,000 reviews / 11,067,213 tokens; val 5,000 / 1,388,412; test 5,000 / 1,395,673 |
| `sanity_check` | untrained validation perplexity 23,926.5 (983,447 scored tokens) |
| `ngram` | test cross-entropy 4.7582 nats, perplexity 116.54 (1,390,673 scored tokens); validation cross-entropy 4.7617; discounts (D3, D2, D1) = 0.9, 0.8, 0.1; 6,445,342 stored counts |

The processed data and these numbers are deterministic: re-running reproduces them exactly on the same package versions. Floating-point results may differ in the last digits on other platforms.

### Generation prompts

`src/generate.py` reads prompts from `prompts.txt` in the repository root (not committed; create it yourself): one prompt per line, blank lines and lines starting with `#` are ignored. `prompt.txt` holds the three prompts used so far, with the reason for each; copy it to `prompts.txt` to use them:

```
cp prompt.txt prompts.txt
```

Output goes to `results/generations_ngram.json` with the decoding settings stored beside the text.

## 5. Configurations

Every run reads one JSON file from `configs/` (no design values are hard-coded in scripts). Each config has the same keys: `architecture`, `layers`, `hidden_size`, `embedding_dim`, `embedding_init`, `context_length`, `learning_rate`, `batch_size`, `epochs`, `dropout`, `vocab_size`, `subsample_size`, `seed` (`null` where a key does not apply), plus an `options` block for settings that only one script uses.

| Config | Script | Notable settings |
|---|---|---|
| `configs/data.json` | `src/data.py` | vocab 20,000; split 0.1 / 0.1; no subsampling; seed 0 |
| `configs/inspect_data.json` | `src/inspect_data.py` | 3 example reviews; 3,000,000-word budget; seed 0 |
| `configs/sanity_check.json` | `src/sanity_check.py` | bigram model, embedding dim 100, batch size 32, reviews cut to 256 tokens; seed 0 |
| `configs/ngram.json` | `src/ngram.py` | trigram, discount grid 0.1–0.95 for each of three discounts, batch size 32 (scoring only), no truncation; seed 0 |
| `configs/generations_ngram.json` | `src/generate.py` | temperature 1.0, top-k 40, 50 new tokens, 1 sample, ban `<pad>`/`<unk>`, stop at `<eos>`; seed 0 |
| `configs/test_evaluate.json`, `configs/test_generate.json` | `tests/` | small made-up vocabularies; seed 0 |

Any setting can be overridden on the command line without editing the file:

```
python src/ngram.py --config configs/ngram.json --seed 1 --run-name ngram_seed1
python src/generate.py --config configs/generations_ngram.json --set options.temperature=0.7 --set options.top_k=10 --run-name generations_ngram_t07
```

`--set KEY=VALUE` works for any key (use `options.name` for the options block); `--run-name` sets the output file name so the original result is not overwritten.

## 6. Random seeds

Every script sets and prints its seed, and records it in its result file. All current configs use **seed 0**.

- `data.py`: the train/validation/test split and any subsampling. Changing the seed changes the split, so all later results change too.
- `inspect_data.py`: which example reviews are printed.
- `sanity_check.py`: the random initial weights of the untrained model.
- `ngram.py`: the model is deterministic (counting and a grid search); the seed is set and recorded only.
- `generate.py`: each (prompt, sample number) gets its own generator seeded from the seed, the sample number and the prompt text, so the same settings give the same text.

## 7. Outputs and records

- `results/<run>.json` — one file per run: the full config used, seed, git commit (and whether the code had uncommitted changes), hardware, date, per-epoch losses, final metrics and details.
- `results/all_runs.csv` — one row per run (made by `src/summarise_runs.py`).
- `results/RESULTS_LOG.md` — dated log of every result with the exact command and file path.
- `DECISIONS.md` — every design choice: value, where it is set, who chose it, alternatives, reason.
- `TOOL_USE.md` — dated log of what was written or changed in the repository.

## 8. Repository layout

```
configs/      one JSON config per run
data/         raw CSV (you add it) and data/processed/ (generated); git-ignored
notebooks/    run_colab.ipynb, a runner for Google Colab (set its REPO_URL first)
results/      run records, all_runs.csv, RESULTS_LOG.md
src/          data.py, inspect_data.py, sanity_check.py, evaluate.py, ngram.py,
              generate.py, runconfig.py, summarise_runs.py
tests/        test_evaluate.py, test_generate.py
```

## 9. Notes on sharing

- Nothing secret is stored here: no API keys, tokens or credentials are used, and `.env` files are git-ignored.
- The dataset and everything derived from it (`data/`) are kept out of git; do not add them to a submission unless the dataset's terms allow redistribution.
- Model checkpoints (`*.pt`, `*.pth`, `*.ckpt`) are git-ignored too.
