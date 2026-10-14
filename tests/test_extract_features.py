"""Checks that pooling really excludes padding, for all three model types (tiny random models, CPU).

Run:  python3 tests/test_extract_features.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from extract_features import extract_split, pool  # noqa: E402
from models import LSTMLanguageModel, RNNLanguageModel, TransformerLanguageModel  # noqa: E402

SEED = 0
torch.manual_seed(SEED)
PAD, V, MAXLEN = 0, 50, 12

# 1. pool() on hand-made numbers. Row 0 has 2 real tokens, row 1 has 4. The padding positions hold 1000.
h = torch.tensor([[[1., 2.], [3., 4.], [1000., 1000.], [1000., 1000.]],
                  [[1., 1.], [2., 2.], [3., 3.], [4., 4.]]])
n = torch.tensor([2, 4])
assert pool(h, n, "last").tolist() == [[3., 4.], [4., 4.]]
assert pool(h, n, "mean").tolist() == [[2., 3.], [2.5, 2.5]]
print("hand-made pooling checks passed")

# 2. Whole pipeline: the vector of a review must not depend on how much padding follows it.
models = {
    "rnn": RNNLanguageModel(V, PAD, 100, 32, 2, 0.3),
    "lstm": LSTMLanguageModel(V, PAD, 100, 32, 2, 0.3),
    "transformer": TransformerLanguageModel(V, PAD, 100, 64, 2, 0.3, 4, MAXLEN),
}
rng = np.random.default_rng(SEED)
lengths = np.array([3, 7, 12, 1, 9], dtype=np.int32)
ids = np.full((len(lengths), MAXLEN), PAD, dtype=np.int32)
for i, L in enumerate(lengths):
    ids[i, :L] = rng.integers(3, V, size=L)

for name, model in models.items():
    model.eval()
    for pooling in ("last", "mean"):
        padded = extract_split(model, ids, lengths, pooling, 2, torch.device("cpu"))  # 12 columns, padded
        for i, L in enumerate(lengths):  # each review alone, cut to exactly its length: no padding at all
            alone = extract_split(model, ids[i:i + 1, :L], lengths[i:i + 1], pooling, 1, torch.device("cpu"))
            assert np.allclose(padded[i], alone[0], atol=1e-5), (name, pooling, i)
        # Changing the padding ids must not change anything either (here: scribble over the padding).
        noisy = ids.copy()
        for i, L in enumerate(lengths):
            noisy[i, L:] = rng.integers(3, V, size=MAXLEN - L)
        assert np.allclose(padded, extract_split(model, noisy, lengths, pooling, 2, torch.device("cpu")), atol=1e-5), (name, pooling)
    print(f"{name}: padded == unpadded and padding content is ignored (last and mean)")
print("all checks passed")
