"""Part III step 1: build the labelled, fixed-length classification data.

What this file does
-------------------
The language models were trained on IMDB reviews with a fixed train / val /
test split saved by data.py. For the Part III sentiment classifier we need the
SAME reviews, but now as (review, label) pairs. So this script does NOT split
anything again: it reads the saved splits, so the classification test reviews
are exactly the language-model test reviews (which no language model trained on).

For every review it then:

1.  Takes the review's token ids. data.py already turned the words into ids with
    the shared vocabulary (words outside it are <unk>), so there is nothing to
    re-tokenize. The <eos> marker data.py appended to each review is dropped:
    it is a document separator for the language models, not part of the review.
2.  Truncates to the FIRST `max_length` tokens ("head" truncation; shorter
    reviews are untouched).
3.  Pads with <pad> on the right up to `max_length`, so every review is a row of
    the same length and reviews can be stacked into one 2D array.
    `lengths` records how many real tokens each row has, so later code can
    ignore the padding (for example, pick the last real position).

It saves data/classification/{train,val,test}.npz with:
    ids      (num_reviews, max_length) int32   token ids, <pad> at the end
    lengths  (num_reviews,) int32              real tokens kept (<= max_length)
    labels   (num_reviews,) int64              0 = negative, 1 = positive
    original_indices (num_reviews,) int64      row in the raw CSV (traceability)

and prints / records, per split: number of reviews, class balance, how many
reviews were truncated, and the <unk> rate (share of real tokens that are <unk>),
both for the whole review and for the part the classifier actually sees.

Run (local, CPU, takes seconds):
    python src/prepare_classification.py --config configs/p3_prepare_classification.json
"""

import argparse
import os

import numpy as np

from data import load_processed_data
from runconfig import add_config_args, load_config, save_run


def prepare_split(split, max_length, pad_id, unk_id, eos_id, truncate):
    """Turn one SplitData into a padded (num_reviews, max_length) array + stats."""
    if truncate != "head":
        raise SystemExit(f"truncate='{truncate}' is not implemented (only 'head').")
    reviews = split.reviews()
    n = len(reviews)
    ids = np.full((n, max_length), pad_id, dtype=np.int32)  # start all-<pad>
    lengths = np.zeros(n, dtype=np.int32)
    full_tokens = full_unk = kept_tokens = kept_unk = 0
    num_truncated = 0

    for i, review in enumerate(reviews):
        review = review[review != eos_id]      # drop the end-of-document marker
        kept = review[:max_length]             # first N tokens
        ids[i, : len(kept)] = kept
        lengths[i] = len(kept)
        num_truncated += int(len(review) > max_length)
        full_tokens += len(review)
        full_unk += int((review == unk_id).sum())
        kept_tokens += len(kept)
        kept_unk += int((kept == unk_id).sum())

    labels = split.labels.astype(np.int64)
    counts = np.bincount(labels, minlength=2)
    stats = {
        "num_reviews": n,
        "num_negative": int(counts[0]),
        "num_positive": int(counts[1]),
        "fraction_positive": float(counts[1] / n),
        "num_truncated": num_truncated,
        "fraction_truncated": num_truncated / n,
        "mean_length_kept": float(lengths.mean()),
        "tokens_full": full_tokens,
        "unk_rate_full": full_unk / full_tokens,
        "tokens_kept": kept_tokens,
        "unk_rate_kept": kept_unk / kept_tokens,
    }
    arrays = {"ids": ids, "lengths": lengths, "labels": labels,
              "original_indices": split.original_indices.astype(np.int64)}
    return arrays, stats


def main():
    parser = argparse.ArgumentParser(description="Build the Part III labelled classification data.")
    add_config_args(parser)
    args = parser.parse_args()
    config, run = load_config(args)
    opts = config["options"]
    print(f"Random seed: {config['seed']} (no randomness is used; the splits are reused as saved)")

    data = load_processed_data(opts["processed_dir"])
    if config["vocab_size"] is not None and len(data.vocab) - 3 != config["vocab_size"]:
        raise SystemExit("Config vocab_size does not match the processed vocabulary.")
    os.makedirs(opts["output_dir"], exist_ok=True)

    all_stats = {}
    for name in ("train", "val", "test"):
        arrays, stats = prepare_split(getattr(data, name), opts["max_length"], data.vocab.pad_id,
                                      data.vocab.unk_id, data.vocab.eos_id, opts["truncate"])
        np.savez(os.path.join(opts["output_dir"], f"{name}.npz"), **arrays)
        all_stats[name] = stats
        print(f"\n[{name}] {stats['num_reviews']} reviews")
        print(f"  class balance: {stats['num_negative']} negative / {stats['num_positive']} positive "
              f"({100 * stats['fraction_positive']:.2f}% positive)")
        print(f"  truncated: {stats['num_truncated']} ({100 * stats['fraction_truncated']:.1f}%); "
              f"mean kept length {stats['mean_length_kept']:.1f}")
        print(f"  <unk> rate: {100 * stats['unk_rate_kept']:.3f}% of kept tokens, "
              f"{100 * stats['unk_rate_full']:.3f}% of full reviews")

    path = save_run(args.results_dir, run, "src/prepare_classification.py", config,
                    metrics={"splits": all_stats},
                    details={"output_dir": opts["output_dir"], "max_length": opts["max_length"],
                             "truncate": opts["truncate"], "eos_dropped": True,
                             "padding": "right, <pad> id 0"})
    print(f"\nSaved run record to {path}")


if __name__ == "__main__":
    main()
