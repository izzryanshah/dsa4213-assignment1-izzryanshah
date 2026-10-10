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

## Note, 2026-10-11 — no generation result saved yet
`src/generate.py` was run end to end twice (n-gram, `configs/generations_ngram.json` with `options.prompts_path=prompt.txt`, output to a scratch directory) to check it works and is reproducible (identical text both times). Those outputs were not saved to `results/`. The real run needs `prompts.txt`; its entry will be added here after that run.
