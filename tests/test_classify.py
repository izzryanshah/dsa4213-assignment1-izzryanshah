"""Checks select_c on made-up data: it picks C from VALIDATION scores only and prefers the smaller C on a tie.

Run:  python3 tests/test_classify.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from classify import select_c  # noqa: E402

SEED = 0
rng = np.random.default_rng(SEED)
y = rng.integers(0, 2, 400)
x = rng.normal(size=(400, 5)) + y[:, None] * 1.5   # two separable-ish blobs
c, model, table = select_c(x[:200], y[:200], x[200:], y[200:], [0.01, 0.1, 1, 10], "accuracy", 500, SEED)
assert c in (0.01, 0.1, 1, 10) and len(table) == 4
assert max(r["accuracy"] for r in table) == next(r["accuracy"] for r in table if r["C"] == c)
# Tie: labels the model cannot learn from (all features zero) -> every C scores the same -> smallest C wins.
c_tie, _, _ = select_c(np.zeros((50, 3)), y[:50], np.zeros((50, 3)), y[50:100], [0.01, 0.1, 1], "accuracy", 100, SEED)
assert c_tie == 0.01, c_tie
print("select_c checks passed")
