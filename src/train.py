"""Train any neural language model in this project: `--model` picks which one.

What this script does, in order:

1.  Reads its settings from a config file (see runconfig.py) and sets the
    random seed.
2.  Loads the saved word-level data (data.py) and builds the model named by
    `--model` from the MODELS table below.
3.  Trains it with Adam. Each training example is a window of
    `context_length` tokens; the target is the same window shifted by one
    token (data.py: WindowDataset). After every batch the gradients are
    "clipped" so one freak batch can't blow the weights up.
4.  After every epoch it measures the training loss and the VALIDATION loss
    (validation goes through evaluate.py, the one scoring function shared by
    all models, including the n-gram). If validation improved, the weights
    are saved as the "best checkpoint".
5.  Stops early if validation has not improved for `patience` epochs in a row.
6.  Loads the best checkpoint and scores the TEST split exactly once.
7.  Saves results/<run>.json (per-epoch losses, training time, parameter
    count, hardware, test result) via runconfig.save_run.

Usage:
    python3 src/train.py --config configs/train_bigram.json --model bigram
    python3 src/train.py --config configs/train_bigram.json --epochs 2 --run-name bigram_quick

    python3 src/train.py --config configs/train_lstm.json --model lstm --hidden-size 128 --layers 1 --dropout 0.1

To add a model: write an nn.Module whose forward(input_ids) returns logits of
shape (batch, seq_len, vocab_size), write a small builder function, and add
one line to MODELS.
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from data import check_config_matches_data, load_processed_data, make_dataloader
from evaluate import evaluate
from runconfig import add_config_args, load_config, save_run
from models import LSTMLanguageModel, RNNLanguageModel
from sanity_check import BigramLM


def build_bigram(config, vocab):
    return BigramLM(len(vocab), vocab.pad_id, config["embedding_dim"])


def build_recurrent(model_class):
    """Make a builder for RNNLanguageModel / LSTMLanguageModel from the config.

    Every size comes from the config (or a command-line override such as
    `--hidden-size 128 --layers 1 --dropout 0.1`); nothing is hard-coded here.
    """
    def build(config, vocab):
        needed = {"embedding_dim": config["embedding_dim"], "hidden_size": config["hidden_size"],
                  "layers": config["layers"], "dropout": config["dropout"]}
        missing = [k for k, v in needed.items() if v is None]
        if missing:
            raise SystemExit(f"{config['architecture']} needs these config values, but they are null: {missing}")
        if config["embedding_init"] != "random":
            raise SystemExit(f"embedding_init must be 'random' for now, got {config['embedding_init']!r}")
        return model_class(len(vocab), vocab.pad_id, config["embedding_dim"], config["hidden_size"],
                           config["layers"], config["dropout"])
    return build


# --model name -> function(config, vocab) that returns an nn.Module.
MODELS = {"bigram": build_bigram, "rnn": build_recurrent(RNNLanguageModel),
          "lstm": build_recurrent(LSTMLanguageModel)}


def set_seed(seed):
    """Make the run repeatable: seed Python, NumPy and PyTorch (CPU and GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # does nothing when there is no GPU


def count_parameters(model):
    """Number of weights the optimiser can change (requires_grad=True)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, criterion, grad_clip, device):
    """One pass over the training windows. Returns the average loss per target token."""
    model.train()  # switches dropout etc. ON (the bigram has none, other models may)
    total_loss, total_targets = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)  # both (batch, seq_len)

        logits = model(x)  # (batch, seq_len, vocab_size)
        # --- the reshape CrossEntropyLoss needs ---------------------------
        # CrossEntropyLoss expects scores of shape (N, vocab_size) and class
        # ids of shape (N,): N independent classification problems. Our model
        # gives (batch, seq_len, vocab_size), so we merge the first two
        # dimensions: every token position of every window becomes one
        # example. Without this, PyTorch would take the seq_len dimension for
        # the class dimension, which is wrong.
        vocab_size = logits.size(-1)
        loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

        optimizer.zero_grad()  # forget the previous batch's gradients
        loss.backward()        # compute new gradients
        # If the total gradient length (norm) exceeds grad_clip, shrink the
        # whole gradient down to that length; the direction is unchanged.
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()       # Adam updates the weights

        # `criterion` averages over real targets, so multiply back by their
        # count to get a sum; dividing by the total at the end gives a true
        # per-token average even if the last batch is smaller.
        n = int((y != criterion.ignore_index).sum().item())
        total_loss += loss.item() * n
        total_targets += n
    return total_loss / total_targets


def fit(model, train_loader, val_split, config, pad_id, device, checkpoint_path):
    """Train with early stopping; leave the best weights saved at `checkpoint_path`.

    Returns a dict: epoch_losses, best_epoch, best_val_loss, stopped_early,
    epochs_run, train_seconds.
    """
    opts = config["options"]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    # ignore_index: positions whose target is <pad> add nothing to the loss.
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    epoch_losses = []
    best_val, best_epoch, epochs_without_gain = float("inf"), None, 0
    stopped_early = False
    start = time.perf_counter()

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, opts["grad_clip"], device)
        val = evaluate(model, val_split, pad_id, opts["eval_batch_size"], opts["eval_max_length"])
        val_loss = val["avg_cross_entropy"]
        epoch_losses.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        improved = val_loss < best_val
        if improved:
            best_val, best_epoch, epochs_without_gain = val_loss, epoch, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_gain += 1
        print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}"
              f"{'  (best, saved)' if improved else ''}")

        if epochs_without_gain >= opts["patience"]:
            stopped_early = True
            print(f"No validation improvement for {opts['patience']} epochs: stopping early.")
            break

    return {"epoch_losses": epoch_losses, "best_epoch": best_epoch, "best_val_loss": best_val,
            "stopped_early": stopped_early, "epochs_run": len(epoch_losses),
            "train_seconds": time.perf_counter() - start}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural language model.")
    add_config_args(parser)
    parser.add_argument("--model", choices=sorted(MODELS), default=None,
                        help="Which model to train. Default: the config's `architecture`.")
    return parser.parse_args()


def main():
    args = parse_args()
    config, run = load_config(args)
    if args.model is not None:
        config["architecture"] = args.model  # so the saved record shows what really ran
    if config["architecture"] not in MODELS:
        raise SystemExit(f"Unknown model '{config['architecture']}'. Choose from {sorted(MODELS)}.")
    opts = config["options"]

    set_seed(config["seed"])
    print(f"Random seed: {config['seed']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_processed_data(opts["processed_dir"])
    check_config_matches_data(config, data)
    vocab = data.vocab

    model = MODELS[config["architecture"]](config, vocab).to(device)
    num_params = count_parameters(model)
    print(f"Model '{config['architecture']}': {num_params:,} trainable parameters, device {device}")

    train_loader = make_dataloader(data.train, config["context_length"], config["batch_size"],
                                   shuffle=True, seed=config["seed"])
    os.makedirs(opts["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(opts["checkpoint_dir"], f"{run}.pt")

    result = fit(model, train_loader, data.val, config, vocab.pad_id, device, checkpoint_path)

    # --- the one and only test evaluation, with the best checkpoint --------
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test = evaluate(model, data.test, vocab.pad_id, opts["eval_batch_size"], opts["eval_max_length"])
    print(f"Best epoch {result['best_epoch']}: val loss {result['best_val_loss']:.4f}; "
          f"TEST cross-entropy {test['avg_cross_entropy']:.4f}, perplexity {test['perplexity']:.1f}")

    metrics = {
        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "test_cross_entropy": test["avg_cross_entropy"],
        "test_perplexity": test["perplexity"],
        "train_seconds": result["train_seconds"],
        "num_parameters": num_params,
        "epochs_run": result["epochs_run"],
        "stopped_early": result["stopped_early"],
    }
    details = {"checkpoint_path": checkpoint_path, "test_num_targets": test["num_targets"],
               "test_target_checksum": test["target_checksum"]}
    path = save_run(args.results_dir, run, "src/train.py", config, metrics=metrics,
                    epoch_losses=result["epoch_losses"], details=details)
    print(f"Saved run record to {path}")


if __name__ == "__main__":
    main()
