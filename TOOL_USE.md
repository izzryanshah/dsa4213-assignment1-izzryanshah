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
