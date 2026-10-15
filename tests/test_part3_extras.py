"""Checks the Part III extension code on made-up data: windowed scoring, McNemar, rank table.

Run:  python3 tests/test_part3_extras.py
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import evaluate  # noqa: E402
from mcnemar import mcnemar  # noqa: E402
from models import LSTMLanguageModel, TransformerLanguageModel  # noqa: E402
from rank_table import build_table, spearman  # noqa: E402
from rescore_windows import WindowedScorer  # noqa: E402

SEED = 0
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
PAD, EOS, V = 0, 2, 40

# --- 1. windowed scoring -----------------------------------------------------------------
reviews = [np.append(rng.integers(3, V, size=n), EOS) for n in (5, 30, 70, 150, 300)]
models = {"lstm": LSTMLanguageModel(V, PAD, 100, 16, 1, 0.0),
          "transformer": TransformerLanguageModel(V, PAD, 100, 32, 1, 0.0, 4, 256)}
for name, model in models.items():
    model.eval()
    full = evaluate(model, reviews, PAD, 2, 256)
    same = evaluate(WindowedScorer(model, 256), reviews, PAD, 2, 256)   # window as long as the input: no cutting
    small = evaluate(WindowedScorer(model, 64), reviews, PAD, 2, 256)
    assert abs(same["perplexity"] - full["perplexity"]) < 1e-4, name
    assert (small["num_targets"], small["target_checksum"]) == (full["num_targets"], full["target_checksum"]), name
    assert abs(small["perplexity"] - full["perplexity"]) > 1e-6, f"{name}: 64-windows should change the score"
    print(f"{name}: window>=input equals one piece; 64-windows score the same targets but a different perplexity")

# Windows are independent of each other and causal: changing a token in window 2 cannot change window 1's log-probs.
model = models["lstm"]
a = torch.randint(3, V, (1, 128)); b = a.clone(); b[0, 100] = (a[0, 100] + 1 - 3) % (V - 3) + 3
t = torch.randint(3, V, (1, 128))
sc = WindowedScorer(model, 64)
with torch.no_grad():
    la, lb = sc.target_log_probs(a, t), sc.target_log_probs(b, t)
assert torch.equal(la[:, :64], lb[:, :64]) and not torch.equal(la[:, 64:], lb[:, 64:])
print("windows are independent and causal")

# --- 2. McNemar --------------------------------------------------------------------------
wrong_a = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)   # A wrong on 0,1
wrong_b = np.array([1, 0, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)   # B wrong on 0,2,3,4,5,6
r = mcnemar(wrong_a, wrong_b)
assert (r["both_wrong"], r["only_a_right"], r["only_b_right"], r["both_right"]) == (1, 5, 1, 3)
assert abs(r["statistic"] - (abs(5 - 1) - 1) ** 2 / 6) < 1e-12          # 9/6 = 1.5
assert abs(r["p_value"] - 0.2207) < 1e-3, r["p_value"]                  # chi-square(1) tail at 1.5
assert abs(r["p_value_exact"] - 0.21875) < 1e-9, r["p_value_exact"]     # 2 * P(Bin(6, .5) <= 1) = 14/64
print("McNemar hand-check passed")

# --- 3. rank table / Spearman ----------------------------------------------------------------
clf = pd.DataFrame({"variant": ["a", "b", "c", "d"], "test_accuracy": [0.9, 0.8, 0.7, 0.6],
                    "test_macro_f1": [0.9, 0.8, 0.7, 0.6]})
table = build_table(clf, {"a": 10.0, "b": 20.0, "c": 30.0, "d": 40.0})
assert table["perplexity_rank"].tolist() == [1, 2, 3, 4] and table["accuracy_rank"].tolist() == [1, 2, 3, 4]
assert abs(spearman(table)["test_accuracy"]["rho"] + 1.0) < 1e-12   # same order => rho = -1 (lower ppl, higher acc)
print("rank table hand-check passed")
print("all checks passed")
