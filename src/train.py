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

Part II: input embeddings. `--embedding` chooses how the word-embedding table (the
`embedding` layer of the model) is set up:

    random   (default) random numbers, learned together with the rest of the model.
    w2v      our own Skip-gram table (data/embeddings/w2v_100d.npy), loaded and FROZEN.
    glove    public GloVe 100d table (data/embeddings/glove_100d.npy), loaded and FROZEN.

    python3 src/train.py --config configs/p2_lstm_w2v.json   --model lstm --embedding w2v
    python3 src/train.py --config configs/p2_lstm_glove.json --model lstm --embedding glove

"Frozen" means the WHOLE table, including the rows of words that had no vector
(they hold the random values chosen when the table was built, and never change).
The optimiser never touches it. As a safeguard, a copy of the table is taken
before training and compared with the table after training (and again after
the best checkpoint is reloaded); if a single number differs, the script stops
with an error instead of saving results. Everything else (model sizes, data,
optimiser, epochs, seed) comes from the config and is not affected by `--embedding`.

To add a model: write an nn.Module whose forward(input_ids) returns logits of
shape (batch, seq_len, vocab_size), write a small builder function, and add
one line to MODELS.
"""

import argparse
import hashlib
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from data import check_config_matches_data, load_processed_data, make_dataloader
from evaluate import evaluate
from runconfig import add_config_args, load_config, save_run
from models import LSTMLanguageModel, RNNLanguageModel, TransformerLanguageModel
from sanity_check import BigramLM


EMBEDDING_CHOICES = ("random", "w2v", "glove")


def check_embedding_setting(config):
    """Stop early if the config's `embedding_init` is not one of the known choices."""
    if config["embedding_init"] not in EMBEDDING_CHOICES:
        raise SystemExit(f"embedding_init must be one of {EMBEDDING_CHOICES}, got {config['embedding_init']!r}")


def freeze_embedding_if_pretrained(model, config):
    """For w2v / glove: switch off learning for the whole word-embedding table.

    `requires_grad_(False)` tells PyTorch not to compute gradients for these
    weights, so the optimiser (Adam) never changes them. This is done when the
    model is BUILT (not only during training), so the count of trainable
    parameters is the same whenever the model is rebuilt later (generation,
    benchmark). The values themselves come from `load_embedding_matrix`.
    """
    if config["embedding_init"] != "random":
        model.embedding.weight.requires_grad_(False)


def load_embedding_matrix(config, vocab):
    """Read the saved table for config['embedding_init'] and check it fits our vocabulary.

    Returns (tensor of shape (vocab size, embedding_dim), path, sha256 of the file).
    Row i of the file is the vector for vocabulary id i (built by train_word2vec.py / load_glove.py).
    """
    paths = config["options"].get("embedding_paths") or {}
    path = paths.get(config["embedding_init"])
    if path is None:
        raise SystemExit(f"config options.embedding_paths has no entry for {config['embedding_init']!r}")
    if not os.path.exists(path):
        raise SystemExit(f"Embedding file {path} not found. Build it first (README.md: train_word2vec.py / load_glove.py).")
    matrix = np.load(path)
    expected = (len(vocab), config["embedding_dim"])
    if matrix.shape != expected:
        raise SystemExit(f"{path} has shape {matrix.shape}, but this vocabulary/embedding_dim needs {expected}")
    if not np.isfinite(matrix).all():
        raise SystemExit(f"{path} contains values that are not finite numbers")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return torch.from_numpy(matrix.astype(np.float32)), path, digest


def check_embedding_unchanged(model, reference, when):
    """Fail loudly unless the embedding table is EXACTLY equal to `reference`.

    Returns the largest absolute difference (0.0 when identical). Also fails if
    the table has been switched back to trainable.
    """
    weight = model.embedding.weight
    if weight.requires_grad:
        raise RuntimeError(f"FROZEN-EMBEDDING CHECK FAILED ({when}): the embedding table is trainable again.")
    current = weight.detach().cpu()
    if not torch.equal(current, reference.cpu()):
        change = (current - reference.cpu()).abs().max().item()
        raise RuntimeError(f"FROZEN-EMBEDDING CHECK FAILED ({when}): the embedding table changed "
                           f"(largest absolute change {change:.3e}). Results were NOT saved.")
    return 0.0


def build_bigram(config, vocab):
    if config["embedding_init"] not in (None, "random"):
        raise SystemExit("the bigram model only supports random embeddings")
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
        check_embedding_setting(config)
        model = model_class(len(vocab), vocab.pad_id, config["embedding_dim"], config["hidden_size"],
                            config["layers"], config["dropout"])
        freeze_embedding_if_pretrained(model, config)
        return model
    return build


def build_transformer(config, vocab):
    """Make a TransformerLanguageModel from the config (num_heads, max_positions live in `options`)."""
    opts = config["options"]
    needed = {"embedding_dim": config["embedding_dim"], "hidden_size": config["hidden_size"],
              "layers": config["layers"], "dropout": config["dropout"],
              "options.num_heads": opts.get("num_heads"), "options.max_positions": opts.get("max_positions")}
    missing = [k for k, v in needed.items() if v is None]
    if missing:
        raise SystemExit(f"transformer needs these config values, but they are null or missing: {missing}")
    check_embedding_setting(config)
    if config["embedding_dim"] % opts["num_heads"] != 0:
        raise SystemExit("embedding_dim must be divisible by options.num_heads")
    model = TransformerLanguageModel(len(vocab), vocab.pad_id, config["embedding_dim"], config["hidden_size"],
                                     config["layers"], config["dropout"], opts["num_heads"], opts["max_positions"])
    freeze_embedding_if_pretrained(model, config)  # only the word table; positions stay learned
    return model


# --model name -> function(config, vocab) that returns an nn.Module.
MODELS = {"bigram": build_bigram, "rnn": build_recurrent(RNNLanguageModel),
          "lstm": build_recurrent(LSTMLanguageModel), "transformer": build_transformer}


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
    # Only parameters that can learn (a frozen embedding table is left out; for
    # random-embedding runs this is every parameter, exactly as before).
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"])
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
    parser.add_argument("--embedding", choices=EMBEDDING_CHOICES, default=None,
                        help="Input embeddings: random (learned), w2v (own Skip-gram, frozen) or glove (public, frozen). "
                             "Default: the config's `embedding_init`.")
    return parser.parse_args()


def main():
    args = parse_args()
    config, run = load_config(args)
    if args.model is not None:
        config["architecture"] = args.model  # so the saved record shows what really ran
    if args.embedding is not None:
        config["embedding_init"] = args.embedding  # likewise
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
    # Pretrained table: copy the saved values in, keep a reference copy to compare against later.
    reference, embedding_path, embedding_sha = None, None, None
    if config["embedding_init"] in ("w2v", "glove"):
        table, embedding_path, embedding_sha = load_embedding_matrix(config, vocab)
        with torch.no_grad():
            model.embedding.weight.copy_(table.to(device))
        reference = model.embedding.weight.detach().cpu().clone()
        check_embedding_unchanged(model, reference, "right after loading")
        print(f"Embedding '{config['embedding_init']}': loaded {embedding_path} {tuple(table.shape)}, "
              f"whole table frozen")
    num_params = count_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model '{config['architecture']}': {num_params:,} trainable parameters "
          f"({total_params:,} in total), device {device}")

    train_loader = make_dataloader(data.train, config["context_length"], config["batch_size"],
                                   shuffle=True, seed=config["seed"])
    os.makedirs(opts["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(opts["checkpoint_dir"], f"{run}.pt")

    result = fit(model, train_loader, data.val, config, vocab.pad_id, device, checkpoint_path)

    # --- frozen-embedding safeguard (after the last epoch, then after the best checkpoint is loaded) ---
    if reference is not None:
        check_embedding_unchanged(model, reference, "after training")

    # --- the one and only test evaluation, with the best checkpoint --------
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    frozen_change = None
    if reference is not None:
        frozen_change = check_embedding_unchanged(model, reference, "after loading the best checkpoint")
        print(f"Frozen-embedding check passed: table identical to the loaded file (max abs change {frozen_change}).")
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
               "test_target_checksum": test["target_checksum"],
               "total_parameters": total_params, "embedding_table_trainable": reference is None}
    if reference is not None:
        details.update({"embedding_source": embedding_path, "embedding_source_sha256": embedding_sha,
                        "frozen_embedding_max_abs_change": frozen_change})
    path = save_run(args.results_dir, run, "src/train.py", config, metrics=metrics,
                    epoch_losses=result["epoch_losses"], details=details)
    print(f"Saved run record to {path}")


if __name__ == "__main__":
    main()
