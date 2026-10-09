# Project rules for DSA4213 Assignment 1 (NLP for Data Science)

These rules apply to every task in this repository, not just the one that created them.

- The user is learning PyTorch from scratch. Explain every new file and any non-obvious line in plain language. Prefer simple, readable code over clever code.
- Use word-level tokenization everywhere, with one shared vocabulary that includes `<unk>` and `<pad>`.
- Train/validation/test splits are created once, saved to disk, and reused from then on. The test set is only used for final evaluation, never for tuning.
- Set and record a random seed in every script.
- All neural models use embedding dimension 100, to match GloVe 100d.
- Language-model losses must ignore padding tokens.
- Language models may only use preceding tokens to predict each target (no leakage from future tokens).
- Every script takes its settings as command-line arguments and saves its results to `results/` as JSON or CSV.
- Never write report text or interpret results for the user — that is their job.
- After each task, append a short dated entry to `TOOL_USE.md` describing what was written or changed.
