"""Short tests for src/train.py on tiny made-up data (no dataset needed).

Run with:   python3 tests/test_train.py --config configs/test_train.json

Checks: the loss goes down, the best checkpoint is saved and reloadable,
early stopping triggers, training is repeatable under the same seed, and the
saved weights really are the ones with the lowest validation loss.
Saves a pass/fail record to results/test_train.json.
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import SplitData, make_dataloader  # noqa: E402
from evaluate import evaluate  # noqa: E402
from runconfig import add_config_args, load_config, save_run  # noqa: E402
from sanity_check import BigramLM  # noqa: E402
from train import count_parameters, fit, set_seed  # noqa: E402

PAD_ID, EOS_ID, NUM_SPECIAL = 0, 2, 3


def make_split(rng, vocab_total, n_reviews):
    """Fake reviews with a learnable pattern: word w is usually followed by w+1."""
    reviews = []
    for _ in range(n_reviews):
        start = int(rng.integers(NUM_SPECIAL, vocab_total))
        review = [NUM_SPECIAL + (start - NUM_SPECIAL + k) % (vocab_total - NUM_SPECIAL) for k in range(12)]
        reviews.append(review + [EOS_ID])
    ids = np.concatenate([np.array(r, dtype=np.int32) for r in reviews])
    return SplitData(ids=ids, review_lengths=np.array([len(r) for r in reviews]),
                     labels=np.zeros(n_reviews, dtype=np.int64), original_indices=np.arange(n_reviews))


def run_fit(config, train, val, ckpt):
    set_seed(config["seed"])
    vocab_total = config["vocab_size"] + NUM_SPECIAL
    model = BigramLM(vocab_total, PAD_ID, config["embedding_dim"])
    loader = make_dataloader(train, config["context_length"], config["batch_size"], True, config["seed"])
    return model, fit(model, loader, val, config, PAD_ID, torch.device("cpu"), ckpt)


def main():
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    config, run = load_config(parser.parse_args())
    rng = np.random.default_rng(config["seed"])
    vocab_total = config["vocab_size"] + NUM_SPECIAL
    train, val = make_split(rng, vocab_total, 60), make_split(rng, vocab_total, 20)

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "m.pt")
        model, result = run_fit(config, train, val, ckpt)
        losses = result["epoch_losses"]

        assert losses[-1]["train_loss"] < losses[0]["train_loss"], "training loss did not go down"
        assert result["stopped_early"] and result["epochs_run"] < config["epochs"], "early stopping did not trigger"
        assert result["best_val_loss"] == min(e["val_loss"] for e in losses)
        assert result["best_epoch"] == 1 + int(np.argmin([e["val_loss"] for e in losses]))

        # The saved file must hold the best weights, not the last ones.
        fresh = BigramLM(vocab_total, PAD_ID, config["embedding_dim"])
        fresh.load_state_dict(torch.load(ckpt))
        o = config["options"]
        reloaded = evaluate(fresh, val, PAD_ID, o["eval_batch_size"], o["eval_max_length"])["avg_cross_entropy"]
        assert abs(reloaded - result["best_val_loss"]) < 1e-9, "checkpoint is not the best epoch"

        # Same seed -> same losses.
        _, again = run_fit(config, train, val, os.path.join(tmp, "m2.pt"))
        assert [e["val_loss"] for e in again["epoch_losses"]] == [e["val_loss"] for e in losses]

    assert count_parameters(model) == vocab_total * config["embedding_dim"] + \
        config["embedding_dim"] * vocab_total + vocab_total
    print("All training tests passed.")
    save_run("results", run, "tests/test_train.py", config,
             metrics={"passed": True, "epochs_run": result["epochs_run"], "best_epoch": result["best_epoch"]})


if __name__ == "__main__":
    main()
