"""Part III step 2: turn each review into ONE fixed-length vector with a frozen language model.

What this file does
-------------------
1.  Loads a trained language model (RNN, LSTM or Transformer) from its checkpoint
    and FREEZES it: eval mode (dropout off), no gradients, weights never change.
2.  Feeds it the padded reviews from data/classification/ (made by
    prepare_classification.py). For every review the model gives one hidden-state
    vector per token position (the vector it uses to predict the next word).
3.  Pools those per-token vectors into ONE vector per review, with
    --pooling last  or  --pooling mean   (explained below).
4.  Saves the vectors and the sentiment labels to data/features/<run>/{train,val,test}.npz
    (features: float32 (num_reviews, vector_size); labels: int64; lengths; original_indices)
    and a results/<run>.json record.

A classifier (a later script) is then trained on those saved vectors only.

How padding is excluded
-----------------------
Every review is a row of 256 ids; reviews shorter than 256 end in <pad> ids. The array
`lengths` says how many real tokens each row has (the real tokens come first, the padding
after them). Two things keep the padding out of the result:

  last   Take the hidden state at position `length - 1`, i.e. the last REAL token, not at
         position 255 (which would be a <pad> position for any short review). In code:
         hidden[row, lengths[row] - 1].

  mean   Build a 0/1 mask that is 1 at positions < length and 0 at the padding, multiply
         the hidden states by it (padding vectors become 0), add up over positions and
         divide by the number of REAL tokens (`length`), not by 256. So the padding is
         neither in the sum nor in the count.

A second protection comes from the models themselves. The padding sits AFTER the real
tokens, and all three models only look backwards (RNN/LSTM read left to right; the
Transformer has a causal mask), so the hidden state at a real position can never depend on
a later <pad>. That is why no extra attention mask is needed, and tests/test_extract_features.py
checks it: a review padded to 256 gives the same vector as the same review with no padding.

Which vector is pooled: the output of the last recurrent layer (RNN/LSTM), or of the final
LayerNorm (Transformer) — the vector that goes into the model's output layer.

Colab only (rule 9: feature extraction runs on the GPU in Colab). Command:
    python src/extract_features.py --config configs/<run>.json --pooling mean
`--pooling` overrides options.pooling in the config.
"""

import argparse
import os
import time

import numpy as np
import torch

from generate import build_neural
from data import load_processed_data
from runconfig import add_config_args, load_config, save_run

POOLINGS = ("last", "mean")


def pool(hidden, lengths, pooling):
    """hidden: (batch, seq_len, size) float; lengths: (batch,) long, each >= 1 and <= seq_len.
    Returns (batch, size), using only the real (non-padding) positions."""
    batch, seq_len, _ = hidden.shape
    if pooling == "last":
        # Row r -> hidden[r, lengths[r] - 1]: advanced indexing picks one position per row.
        return hidden[torch.arange(batch, device=hidden.device), lengths - 1]
    if pooling == "mean":
        # mask[r, t] = True where t < lengths[r] (a real token), False on padding.
        mask = torch.arange(seq_len, device=hidden.device)[None, :] < lengths[:, None]
        summed = (hidden * mask[:, :, None]).sum(dim=1)   # padding positions add 0
        return summed / lengths[:, None].to(hidden.dtype)  # divide by real-token count
    raise ValueError(f"pooling must be one of {POOLINGS}, got '{pooling}'")


@torch.no_grad()  # no gradients: nothing is being trained
def extract_split(model, ids, lengths, pooling, batch_size, device):
    """Run the frozen model over one split and return its pooled vectors (numpy, float32)."""
    chunks = []
    for start in range(0, len(ids), batch_size):
        x = torch.from_numpy(ids[start:start + batch_size].astype(np.int64)).to(device)
        n = torch.from_numpy(lengths[start:start + batch_size].astype(np.int64)).to(device)
        x = x[:, : int(n.max())]  # drop columns that are padding for EVERY review in this batch (speed only)
        chunks.append(pool(model.hidden_states(x), n, pooling).float().cpu().numpy())
    return np.concatenate(chunks)


def main():
    parser = argparse.ArgumentParser(description="Extract pooled frozen-LM features for classification.")
    add_config_args(parser)
    parser.add_argument("--pooling", choices=POOLINGS, default=None,
                        help="last = hidden state at the last non-padding token; mean = average over non-padding tokens.")
    args = parser.parse_args()
    config, run = load_config(args)
    opts = config["options"]
    if args.pooling is not None:
        opts["pooling"] = args.pooling
    if opts.get("pooling") not in POOLINGS:
        raise SystemExit(f"Set options.pooling or pass --pooling ({'/'.join(POOLINGS)}).")
    pooling = opts["pooling"]

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    print(f"Random seed: {config['seed']} (nothing random happens; the model is frozen)")

    data = load_processed_data(opts["processed_dir"])  # only for the vocabulary (sizes, <pad> id)
    model, source = build_neural(opts, data)            # rebuilds the model and loads its best checkpoint
    device = torch.device(source["device"])
    model.eval()                                        # dropout off
    for p in model.parameters():
        p.requires_grad_(False)                         # frozen: no weight can change
    before = sum(float(p.double().abs().sum()) for p in model.parameters())

    os.makedirs(os.path.join(opts["output_dir"], run), exist_ok=True)
    metrics, timings = {}, {}
    for split in ("train", "val", "test"):
        with np.load(os.path.join(opts["classification_dir"], f"{split}.npz")) as z:
            ids, lengths, labels, orig = z["ids"], z["lengths"], z["labels"], z["original_indices"]
        if lengths.min() < 1:
            raise SystemExit(f"{split}: a review has zero real tokens; cannot pool it.")
        if ids.shape[1] > getattr(model, "max_positions", ids.shape[1]):
            raise SystemExit("Reviews are longer than the model's max_positions.")
        t0 = time.time()
        feats = extract_split(model, ids, lengths, pooling, config["batch_size"], device)
        timings[split] = time.time() - t0
        if not np.isfinite(feats).all():
            raise SystemExit(f"{split}: features contain NaN or infinity.")
        np.savez(os.path.join(opts["output_dir"], run, f"{split}.npz"), features=feats.astype(np.float32),
                 labels=labels, lengths=lengths, original_indices=orig)
        metrics[split] = {"num_reviews": len(feats), "feature_size": int(feats.shape[1]),
                          "feature_mean": float(feats.mean()), "feature_std": float(feats.std())}
        print(f"[{split}] {feats.shape} in {timings[split]:.1f}s  mean {feats.mean():.4f}  std {feats.std():.4f}")

    after = sum(float(p.double().abs().sum()) for p in model.parameters())
    if before != after:
        raise SystemExit("Model weights changed during extraction (they must stay frozen).")

    path = save_run(args.results_dir, run, "src/extract_features.py", config,
                    metrics={"pooling": pooling, "splits": metrics},
                    details={**source, "timing_seconds": timings, "output_dir": os.path.join(opts["output_dir"], run),
                             "weights_unchanged": True, "padding": "excluded: see src/extract_features.py docstring"})
    print(f"\nSaved run record to {path}")


if __name__ == "__main__":
    main()
