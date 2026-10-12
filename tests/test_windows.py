"""Checks WindowDataset on a tiny made-up stream (no real data needed).

Run:  python3 tests/test_windows.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import WindowDataset, make_dataloader, SplitData  # noqa: E402

SEED = 0  # nothing here is random, but every script records a seed

ids = np.arange(23, dtype=np.int32)  # tokens 0..22
L = 5
ds = WindowDataset(ids, L)

# 23 tokens -> (23 - 1) // 5 = 4 full windows; the 2-token tail is dropped.
assert len(ds) == 4, len(ds)

for i in range(len(ds)):
    x, y = ds[i]
    assert x.shape == y.shape == (L,)
    assert x.tolist() == list(range(i * L, i * L + L))
    assert y.tolist() == list(range(i * L + 1, i * L + L + 1))  # shifted by one

# Windows do not overlap: the next window starts where this one's input ended.
assert ds[1][0][0].item() == ds[0][0][-1].item() + 1

# Batching: shapes and the seeded shuffle being repeatable.
split = SplitData(ids=ids, review_lengths=np.array([23]), labels=np.array([0]), original_indices=np.array([0]))
a = [x.tolist() for x, _ in make_dataloader(split, L, 2, True, SEED)]
b = [x.tolist() for x, _ in make_dataloader(split, L, 2, True, SEED)]
assert a == b
x, y = next(iter(make_dataloader(split, L, 2, False, SEED)))
assert x.shape == y.shape == (2, L)

print("All window tests passed.")
