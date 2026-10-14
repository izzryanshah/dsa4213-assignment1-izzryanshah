"""Checks prepare_classification on a tiny made-up split, then (if the saved
files exist) checks the real classification files against the language-model splits.

Run:  python3 tests/test_prepare_classification.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import SplitData, load_processed_data  # noqa: E402
from prepare_classification import prepare_split  # noqa: E402

SEED = 0  # nothing here is random, but every script records a seed
PAD, UNK, EOS = 0, 1, 2

# Three reviews: short (3 tokens, one <unk>), exactly 4 tokens, long (6 tokens). Each ends with <eos>.
reviews = [[5, UNK, 7, EOS], [5, 6, 7, 8, EOS], [9, 9, 9, 9, 9, 9, EOS]]
split = SplitData(ids=np.array(sum(reviews, []), dtype=np.int32),
                  review_lengths=np.array([len(r) for r in reviews], dtype=np.int32),
                  labels=np.array([0, 1, 1]), original_indices=np.array([10, 11, 12]))
arrays, stats = prepare_split(split, 4, PAD, UNK, EOS, "head")

assert arrays["ids"].tolist() == [[5, UNK, 7, PAD], [5, 6, 7, 8], [9, 9, 9, 9]]
assert arrays["lengths"].tolist() == [3, 4, 4]
assert stats["num_truncated"] == 1                      # only the 6-token review
assert stats["num_negative"] == 1 and stats["num_positive"] == 2
assert abs(stats["unk_rate_full"] - 1 / 13) < 1e-12     # 1 <unk> in 3 + 4 + 6 real tokens
assert abs(stats["unk_rate_kept"] - 1 / 11) < 1e-12     # 3 + 4 + 4 kept tokens
print("tiny-split checks passed")

# Real files: same reviews and labels as the language-model splits, nothing re-split.
if os.path.exists("data/classification/test.npz"):
    data = load_processed_data("data/processed")
    csv = pd.read_csv("data/IMDB Dataset.csv")
    for name in ("train", "val", "test"):
        lm = getattr(data, name)
        with np.load(f"data/classification/{name}.npz") as z:
            assert (z["original_indices"] == lm.original_indices).all(), name
            assert (z["labels"] == lm.labels).all(), name
            raw = (csv["sentiment"].iloc[z["original_indices"]] == "positive").astype(int).to_numpy()
            assert (raw == z["labels"]).all(), f"{name}: labels differ from the raw CSV"
            assert (z["ids"][np.arange(len(z["ids"])), z["lengths"] - 1] != PAD).all()
            assert not (z["ids"] == EOS).any()
    print("real-file checks passed (same reviews and labels as the language-model splits)")
else:
    print("data/classification not built yet; skipped real-file checks")
