"""Build the word-level IMDB dataset used by every later script in this project.

What this file does, in order:

1.  Clean each raw review (strip HTML tags like "<br />") and tokenize it into
    lowercase words, keeping punctuation characters as their own tokens.
2.  Split the 50,000 reviews into train/validation/test (80/10/10), stratified
    by sentiment label, with a fixed random seed — done ONCE, then saved to
    disk, so every later script (and every later run of this script) sees
    exactly the same three sets of reviews. Optionally subsamples within each
    split afterwards, also with a fixed seed.
3.  Builds one shared vocabulary from the TRAINING split only (see the
    docstring on `build_vocab` for why).
4.  Converts every split into token ids and saves everything to
    data/processed/.
5.  Exposes `load_processed_data()` so any later script can get the exact
    same data with one line (any script run as `python src/<name>.py`, i.e.
    living alongside this file in src/, can just do):

        from data import load_processed_data
        data = load_processed_data()

Run directly to (re)build the processed data:

    python src/data.py --config configs/data_prep.json
    python src/data.py --config configs/data_prep.json --subsample-size 12500   # override one setting

All settings (seed, vocabulary size, subsample size, split fractions, paths)
come from the config file; see runconfig.py for how overrides work.
"""

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from runconfig import add_config_args, load_config, save_run

# --- special tokens -----------------------------------------------------
# Every split uses the same three reserved ids. <eos> (end-of-document) is
# not one of the "most frequent words" from step 3 — it is a control symbol
# we insert ourselves after every review, so it needs a fixed id just like
# <pad> and <unk> do.
PAD_TOKEN, UNK_TOKEN, EOS_TOKEN = "<pad>", "<unk>", "<eos>"
PAD_ID, UNK_ID, EOS_ID = 0, 1, 2

NUM_SPECIAL_TOKENS = 3  # <pad>, <unk>, <eos>

LABEL_MAP = {"negative": 0, "positive": 1}

# Matches any HTML tag, e.g. "<br />", "<br/>", "<i>", "</i>".
HTML_TAG_RE = re.compile(r"<[^>]+>")
# After lowercasing, this matches either a run of "word" characters (a
# word — \w is Unicode-aware in Python 3, so accented letters like "é" stay
# attached to their word instead of being chopped off) OR a single character
# that is neither whitespace nor a word character (one piece of
# punctuation). That is what keeps punctuation as its own token instead of
# being glued onto a word or silently dropped.
TOKEN_RE = re.compile(r"\w+|[^\s\w]")


def clean_and_tokenize(text):
    """Strip HTML tags, lowercase, and split into word/punctuation tokens."""
    text = HTML_TAG_RE.sub(" ", text)
    text = text.lower()
    return TOKEN_RE.findall(text)


@dataclass
class Vocab:
    id_to_token: list
    token_to_id: dict
    pad_id: int = PAD_ID
    unk_id: int = UNK_ID
    eos_id: int = EOS_ID

    def __len__(self):
        return len(self.id_to_token)

    def encode(self, tokens):
        """Map a list of word tokens to ids, falling back to <unk>."""
        return [self.token_to_id.get(t, self.unk_id) for t in tokens]


@dataclass
class SplitData:
    ids: np.ndarray            # 1D array: every review's token ids, back to back
    review_lengths: np.ndarray  # how many ids belong to each review (incl. its <eos>)
    labels: np.ndarray         # one label per review, same order as review_lengths
    original_indices: np.ndarray  # row number in the raw CSV, for traceability

    def reviews(self):
        """Slice the flat `ids` stream back into one array per review.

        `ids` is one long concatenated stream (with an <eos> token between
        reviews), so recovering review i needs the running total of lengths
        of all reviews before it.
        """
        ends = np.cumsum(self.review_lengths)
        starts = ends - self.review_lengths
        return [self.ids[s:e] for s, e in zip(starts, ends)]


@dataclass
class ProcessedData:
    vocab: Vocab
    label_map: dict
    train: SplitData
    val: SplitData
    test: SplitData


def stratified_three_way_split(indices, labels, val_frac, test_frac, seed):
    """Split `indices` into train/val/test, preserving the label proportions
    in each piece (stratified) so e.g. train isn't accidentally 90% positive.
    """
    train_frac = 1.0 - val_frac - test_frac
    if train_frac <= 0:
        raise ValueError("val_frac + test_frac must be less than 1.0")

    # First carve off train vs. "everything else" (val + test combined)...
    train_idx, rest_idx, _, rest_labels = train_test_split(
        indices, labels, test_size=val_frac + test_frac, stratify=labels, random_state=seed
    )
    # ...then split "everything else" into val and test in the right ratio.
    rest_val_share = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=1 - rest_val_share, stratify=rest_labels, random_state=seed
    )
    return train_idx, val_idx, test_idx


def subsample_indices(idx, labels, target_n, seed):
    """Randomly keep `target_n` of `idx`, stratified by label, with a fixed
    seed. If there's nothing to trim, returns `idx` unchanged.
    """
    if target_n is None or target_n >= len(idx):
        return idx
    kept, _ = train_test_split(idx, train_size=target_n, stratify=labels, random_state=seed)
    return np.sort(kept)


def build_vocab(train_token_lists, vocab_size):
    """Build the shared vocabulary from TRAINING reviews only.

    Why training data only: validation and test are meant to estimate how
    well the model handles text it has never seen. If we let their words
    into the vocabulary (or used their word frequencies to help pick which
    words make the cut), the model would effectively get a sneak preview of
    words from the data it's supposed to be evaluated on, making
    validation/test performance look better than it would on truly unseen
    text. Any word that only appears in val/test becomes <unk> to the model,
    exactly as it would for a brand-new review after deployment.
    """
    counts = Counter()
    for tokens in train_token_lists:
        counts.update(tokens)
    most_common = [tok for tok, _ in counts.most_common(vocab_size)]
    id_to_token = [PAD_TOKEN, UNK_TOKEN, EOS_TOKEN] + most_common
    token_to_id = {tok: i for i, tok in enumerate(id_to_token)}
    return Vocab(id_to_token=id_to_token, token_to_id=token_to_id)


def encode_split(tokenized_reviews, idx_array, labels_all, vocab):
    """Turn one split's reviews into the flat (ids, review_lengths, labels,
    original_indices) representation described at the top of this file.
    """
    id_chunks, review_lengths, split_labels = [], [], []
    for original_i in idx_array:
        token_ids = vocab.encode(tokenized_reviews[original_i])
        token_ids.append(vocab.eos_id)  # the end-of-document marker
        id_chunks.append(np.array(token_ids, dtype=np.int32))
        review_lengths.append(len(token_ids))
        split_labels.append(labels_all[original_i])

    ids = np.concatenate(id_chunks) if id_chunks else np.zeros(0, dtype=np.int32)
    return (
        ids,
        np.array(review_lengths, dtype=np.int32),
        np.array(split_labels, dtype=np.int64),
        np.array(idx_array, dtype=np.int64),
    )


def load_processed_data(processed_dir="data/processed"):
    """Load everything `python src/data.py` wrote, ready to use.

    One-line usage in any later script living in src/ alongside this file:
        from data import load_processed_data
        data = load_processed_data()
        data.vocab.pad_id, data.train.ids, data.train.labels, ...
    """
    with open(os.path.join(processed_dir, "vocab.json")) as f:
        vocab_json = json.load(f)
    id_to_token = vocab_json["id_to_token"]
    vocab = Vocab(
        id_to_token=id_to_token,
        token_to_id={tok: i for i, tok in enumerate(id_to_token)},
        pad_id=vocab_json["pad_id"],
        unk_id=vocab_json["unk_id"],
        eos_id=vocab_json["eos_id"],
    )

    with open(os.path.join(processed_dir, "label_map.json")) as f:
        label_map = json.load(f)

    def _load_split(name):
        # np.load on a .npz file returns a dict-like object of the arrays
        # that were saved together with np.savez.
        with np.load(os.path.join(processed_dir, f"{name}.npz")) as npz:
            return SplitData(
                ids=npz["ids"],
                review_lengths=npz["review_lengths"],
                labels=npz["labels"],
                original_indices=npz["original_indices"],
            )

    return ProcessedData(
        vocab=vocab,
        label_map=label_map,
        train=_load_split("train"),
        val=_load_split("val"),
        test=_load_split("test"),
    )


def check_config_matches_data(config, data):
    """Stop early if a run's config disagrees with the processed data on disk.

    Every later script reads the data that `python src/data.py` wrote. If its
    config claims a different vocabulary size than that data really has, the
    saved results would be mislabelled, so fail loudly instead.
    """
    if config["vocab_size"] is not None:
        actual = len(data.vocab) - NUM_SPECIAL_TOKENS
        if actual != config["vocab_size"]:
            raise SystemExit(f"Config says vocab_size={config['vocab_size']} but the processed data "
                             f"has {actual} words (+{NUM_SPECIAL_TOKENS} special tokens). "
                             "Re-run src/data.py or fix the config.")


def parse_args():
    parser = argparse.ArgumentParser(description="Build the word-level IMDB dataset.")
    add_config_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    config, run = load_config(args)
    opts = config["options"]  # data_path, output_dir, val_frac, test_frac
    seed = config["seed"]
    # The assignment fixes the allowed vocabulary range; this is a rule, not a tunable setting.
    if not (10_000 <= config["vocab_size"] <= 20_000):
        raise SystemExit(f"vocab_size must be between 10000 and 20000, got {config['vocab_size']}")

    print(f"Random seed: {seed}")

    df = pd.read_csv(opts["data_path"])
    if not set(df["sentiment"].unique()).issubset(LABEL_MAP):
        raise SystemExit(f"Unexpected sentiment values: {df['sentiment'].unique().tolist()}")

    labels_all = df["sentiment"].map(LABEL_MAP).to_numpy()
    original_indices_all = df.index.to_numpy()

    print(f"Tokenizing {len(df)} reviews...")
    tokenized_reviews = [clean_and_tokenize(text) for text in df["review"]]

    # --- step 2: split BEFORE the vocabulary is built -------------------
    train_idx, val_idx, test_idx = stratified_three_way_split(
        original_indices_all, labels_all, opts["val_frac"], opts["test_frac"], seed
    )

    if config["subsample_size"] is not None:
        total_n = len(df)
        # Scale the subsample target to each split's own size, so the
        # 80/10/10 ratio is preserved rather than e.g. emptying out val/test.
        ratio = config["subsample_size"] / total_n
        train_idx = subsample_indices(train_idx, labels_all[train_idx], round(len(train_idx) * ratio), seed)
        val_idx = subsample_indices(val_idx, labels_all[val_idx], round(len(val_idx) * ratio), seed)
        test_idx = subsample_indices(test_idx, labels_all[test_idx], round(len(test_idx) * ratio), seed)

    # --- save which original review went into which split ---------------
    # This is what lets Part III reuse exactly the same reviews later, even
    # if this script is re-run or the subsampling choice changes.
    kept = {i: True for i in train_idx.tolist() + val_idx.tolist() + test_idx.tolist()}
    split_of = {}
    for i in train_idx:
        split_of[i] = "train"
    for i in val_idx:
        split_of[i] = "val"
    for i in test_idx:
        split_of[i] = "test"
    # Indices not kept after subsampling still had a nominal split from the
    # 80/10/10 division; recompute it the same way, just to record it.
    if config["subsample_size"] is not None:
        full_train_idx, full_val_idx, full_test_idx = stratified_three_way_split(
            original_indices_all, labels_all, opts["val_frac"], opts["test_frac"], seed
        )
        for i in full_train_idx:
            split_of.setdefault(i, "train")
        for i in full_val_idx:
            split_of.setdefault(i, "val")
        for i in full_test_idx:
            split_of.setdefault(i, "test")

    output_dir = opts["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    split_assignment = pd.DataFrame(
        {
            "original_index": list(split_of.keys()),
            "split": list(split_of.values()),
        }
    ).sort_values("original_index")
    split_assignment["included"] = split_assignment["original_index"].map(lambda i: kept.get(i, False))
    split_assignment.to_csv(os.path.join(output_dir, "split_assignment.csv"), index=False)

    # --- step 3: build the vocabulary from the training split only -------
    vocab = build_vocab((tokenized_reviews[i] for i in train_idx), config["vocab_size"])
    print(f"Vocabulary size (incl. <pad>/<unk>/<eos>): {len(vocab)}")

    # --- step 4: encode every split and save it ---------------------------
    split_stats = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        ids, review_lengths, split_labels, orig_idx = encode_split(tokenized_reviews, idx, labels_all, vocab)
        np.savez(
            os.path.join(output_dir, f"{name}.npz"),
            ids=ids,
            review_lengths=review_lengths,
            labels=split_labels,
            original_indices=orig_idx,
        )

        num_tokens = len(ids)
        unk_rate = float((ids == vocab.unk_id).mean()) if num_tokens else 0.0
        print(
            f"{name}: {len(idx)} reviews, {num_tokens} tokens, "
            f"<unk> rate = {unk_rate:.4%}"
        )
        split_stats[name] = {
            "num_reviews": int(len(idx)),
            "num_tokens": int(num_tokens),
            "unk_rate": unk_rate,
        }

    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump(
            {
                "id_to_token": vocab.id_to_token,
                "pad_id": vocab.pad_id,
                "unk_id": vocab.unk_id,
                "eos_id": vocab.eos_id,
            },
            f,
        )
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(LABEL_MAP, f)

    results_path = save_run(
        args.results_dir, run, "src/data.py", config,
        metrics={"vocab_size_with_special_tokens": len(vocab), "splits": split_stats},
    )
    print(f"Saved processed data to {output_dir}/ and run record to {results_path}")


if __name__ == "__main__":
    main()
