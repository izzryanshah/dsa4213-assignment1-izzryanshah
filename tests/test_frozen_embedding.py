"""Tests for the frozen-embedding option of src/train.py, on tiny made-up data (no dataset needed).

Run with:   python3 tests/test_frozen_embedding.py --config configs/test_frozen_embedding.json

Checks:
  1. With a pretrained table (w2v/glove), the WHOLE table, including the rows
     we deliberately fill with 'missing word' values, is exactly the same after
     training as the table we loaded.
  2. The same tiny model with a RANDOM embedding does change its table when it
     trains (so check 1 would notice if freezing were broken).
  3. The check really fails loudly: nudging a single number in the table, or
     switching the table back to trainable, raises an error.
  4. The frozen table is not counted as trainable (fewer trainable parameters).
Saves a pass/fail record to results/test_frozen_embedding.json.
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import make_dataloader  # noqa: E402
from runconfig import add_config_args, load_config, save_run  # noqa: E402
from test_train import PAD_ID, make_split  # noqa: E402
from train import (MODELS, check_embedding_unchanged, count_parameters, fit,  # noqa: E402
                   freeze_embedding_if_pretrained, set_seed)

NUM_SPECIAL = 3


def train_tiny(config, train, val, vocab_size, table):
    """Build the tiny LSTM from `config`; if `table` is given, load it. Train. Return (model, reference)."""
    class V:  # the builder only needs len(vocab) and vocab.pad_id
        pad_id = PAD_ID
        def __len__(self):
            return vocab_size
    set_seed(config["seed"])
    model = MODELS["lstm"](config, V())
    reference = None
    if table is not None:
        with torch.no_grad():
            model.embedding.weight.copy_(table)
        reference = model.embedding.weight.detach().clone()
    before = model.embedding.weight.detach().clone()
    loader = make_dataloader(train, config["context_length"], config["batch_size"], True, config["seed"])
    with tempfile.TemporaryDirectory() as tmp:
        fit(model, loader, val, config, PAD_ID, torch.device("cpu"), os.path.join(tmp, "m.pt"))
    return model, reference, before


def expect_failure(fn, label):
    try:
        fn()
    except RuntimeError as e:
        assert "FROZEN-EMBEDDING CHECK FAILED" in str(e)
        return
    raise AssertionError(f"the check did not fail for: {label}")


def main():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    config, run = load_config(parser.parse_args())
    vocab_size = config["vocab_size"] + NUM_SPECIAL
    rng = np.random.default_rng(config["seed"])
    train, val = make_split(rng, vocab_size, 60), make_split(rng, vocab_size, 20)

    # A made-up "pretrained" table. Rows 5-9 stand for words with no vector (filled with
    # random values, as build scripts do); they must be frozen like every other row.
    table = torch.from_numpy(rng.normal(0, 0.5, (vocab_size, config["embedding_dim"])).astype(np.float32))
    table[PAD_ID] = 0.0

    # 1. frozen run
    model, reference, _ = train_tiny(config, train, val, vocab_size, table)
    assert torch.equal(model.embedding.weight.detach(), table), "frozen table differs from the loaded table"
    assert check_embedding_unchanged(model, reference, "test") == 0.0
    assert not model.embedding.weight.requires_grad
    n_frozen_trainable = count_parameters(model)

    # 2. the same model with a random trainable embedding does change
    random_cfg = dict(config, embedding_init="random")
    rmodel, _, rbefore = train_tiny(random_cfg, train, val, vocab_size, None)
    assert not torch.equal(rmodel.embedding.weight.detach(), rbefore), "random embedding did not change; check 1 proves nothing"
    assert count_parameters(rmodel) - n_frozen_trainable == vocab_size * config["embedding_dim"], \
        "frozen run should have exactly the embedding table fewer trainable parameters"

    # 3. the check fails loudly
    with torch.no_grad():
        model.embedding.weight[7, 3] += 1e-6  # one tiny nudge to one number (a 'missing word' row)
    expect_failure(lambda: check_embedding_unchanged(model, reference, "nudged"), "one value nudged")
    with torch.no_grad():
        model.embedding.weight[7, 3] -= 1e-6
    model.embedding.weight.requires_grad_(True)
    expect_failure(lambda: check_embedding_unchanged(model, reference, "unfrozen"), "table switched to trainable")

    print("All frozen-embedding tests passed.")
    save_run("results", run, "tests/test_frozen_embedding.py", config,
             metrics={"passed": True, "trainable_parameters_frozen_run": n_frozen_trainable})


if __name__ == "__main__":
    main()
