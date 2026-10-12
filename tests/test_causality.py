"""Causality test: no model may peek at future tokens.

Run with:   python3 tests/test_causality.py --config configs/test_causality.json

For EVERY language model class in src/models.py (found automatically, so a model
added later is tested too) and for EVERY position t of a random sequence:

  1. copy the sequence and replace the token at position t with a different token;
  2. run both sequences through the model and turn the scores into probability
     distributions over the vocabulary (softmax);
  3. check the distributions at positions 0 .. t-1 are identical (nothing before t
     may change), and, as a sanity check on the test itself, that the distribution
     at position t DOES change (the model really uses its input).

Saves a pass/fail record to results/test_causality.json.
"""

import argparse
import inspect
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import models  # noqa: E402
from runconfig import add_config_args, load_config, save_run  # noqa: E402

PAD_ID, NUM_SPECIAL = 0, 3


def language_model_classes():
    """All nn.Module classes defined in models.py (not imported into it)."""
    return [(name, cls) for name, cls in inspect.getmembers(models, inspect.isclass)
            if issubclass(cls, torch.nn.Module) and cls.__module__ == models.__name__]


def build(cls, config, vocab_total):
    """Build a model, passing the extra Transformer settings only if its __init__ asks for them."""
    args = dict(vocab_size=vocab_total, pad_id=PAD_ID, embedding_dim=config["embedding_dim"],
                hidden_size=config["hidden_size"], num_layers=config["layers"], dropout=config["dropout"],
                num_heads=config["options"]["num_heads"], max_positions=config["options"]["max_positions"])
    wanted = inspect.signature(cls.__init__).parameters
    return cls(**{k: v for k, v in args.items() if k in wanted})


def main():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    config, run = load_config(parser.parse_args())
    seed = config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    vocab_total = config["vocab_size"] + NUM_SPECIAL
    batch, seq = config["batch_size"], config["options"]["seq_len"]
    tol = config["options"]["tolerance"]
    checked = {}

    for name, cls in language_model_classes():
        model = build(cls, config, vocab_total)
        model.eval()  # dropout off, so two runs of the same input give the same output
        x = torch.randint(NUM_SPECIAL, vocab_total, (batch, seq))
        with torch.no_grad():
            base = torch.softmax(model(x), dim=-1)  # (batch, seq, vocab): a distribution per position
            worst_before = 0.0  # biggest change seen at a position BEFORE t (must stay ~0)
            for t in range(seq):
                x2 = x.clone()
                # +1 (wrapping around the real words) guarantees the token is different.
                x2[:, t] = NUM_SPECIAL + (x2[:, t] - NUM_SPECIAL + 1) % config["vocab_size"]
                changed = torch.softmax(model(x2), dim=-1)
                diff = (changed - base).abs().amax(dim=(0, 2))  # per position: max change over batch and vocab
                if t > 0:
                    worst_before = max(worst_before, diff[:t].max().item())
                    assert diff[:t].max().item() <= tol, \
                        f"{name}: changing position {t} changed an earlier position (max diff {diff[:t].max().item():.2e})"
                assert diff[t].item() > tol, f"{name}: position {t} ignores its own input (test would be vacuous)"
        checked[name] = {"positions_tested": seq, "max_change_before_t": worst_before}
        print(f"{name}: OK, {seq} positions, largest change before t = {worst_before:.2e}")

    print("All causality tests passed:", ", ".join(checked))
    save_run("results", run, "tests/test_causality.py", config, metrics={"passed": True, "models": checked})


if __name__ == "__main__":
    main()
