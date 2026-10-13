"""Load public GloVe 100d vectors and line them up with OUR vocabulary.

What this is for
----------------
Part II's third setting uses public pretrained word vectors, kept FIXED while
the language model trains. This script turns the GloVe text file into a numpy
table with one row per word of our vocabulary, in our vocabulary's order, so
a later training run can load it straight into nn.Embedding.

Which file
----------
glove.6B.100d.txt (Pennington, Socher & Manning, 2014; trained on Wikipedia +
Gigaword, 6 billion tokens, 400,000 lower-cased words, 100 dimensions). It is
NOT in the repository and NOT downloaded by this script: see README.md,
"Public GloVe vectors" (downloading is a Colab step in this project). Each
line of the file is:  word v1 v2 ... v100   (separated by single spaces).

What the script does
--------------------
1. Reads the file line by line and keeps ONLY the words that are in our
   vocabulary (so memory use stays small; the file has 400,000 lines).
   Words are matched exactly: our tokens are lower-case, and so is GloVe 6B.
2. Puts each found vector in the row of that word's vocabulary id.
3. Measures coverage BEFORE filling anything:
     - % of vocabulary types that have a GloVe vector (<pad> excluded: it
       never occurs in the text);
     - % of training tokens whose word has a GloVe vector (counted on the
       training split's token stream, the text the language models train on).
   Because <unk> and <eos> are symbols we made up, GloVe has no vector for them
   and they count as "missing" in the main figures. The same figures are also
   given excluding those two symbols, so you can see the coverage of real words.
4. Fills rows with no vector using the shared rule in embedding_utils.py
   (small random numbers at the scale of the real vectors, seeded; <pad> zeros),
   the same rule used for the own word2vec table.
5. Saves the table to data/embeddings/glove_100d.npy (float32, shape
   (vocab size, 100), row i = vocabulary id i), and saves the coverage figures
   and the most frequent missing words to results/glove_coverage.json.

Usage (after downloading the file, see README.md):
    python src/load_glove.py --config configs/glove_coverage.json
"""

import argparse
import os
import random

import numpy as np

from data import load_processed_data
from embedding_utils import coverage, fill_missing_rows
from runconfig import add_config_args, load_config, save_run


def read_glove_for_vocab(path, token_to_id, dim):
    """Return {vocab id: vector} for every GloVe word that is in our vocabulary.

    Only the words we need are kept. `rsplit(" ", dim)` splits off the LAST
    `dim` numbers, so even a word that itself contained a space would be read
    correctly (not an issue for 6B, but harmless).
    """
    found = {}
    n_lines = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            parts = line.rstrip("\n").rsplit(" ", dim)
            if len(parts) != dim + 1:
                raise SystemExit(f"{path} line {n_lines}: expected a word plus {dim} numbers, got {len(parts)} fields. "
                                 f"Is this the 100d file?")
            word = parts[0]
            if word in token_to_id and token_to_id[word] not in found:
                found[token_to_id[word]] = np.asarray(parts[1:], dtype=np.float32)
    return found, n_lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    args = parser.parse_args()
    config, run = load_config(args)
    opts = config["options"]
    dim, seed = config["embedding_dim"], config["seed"]
    random.seed(seed)
    np.random.seed(seed)

    if not os.path.exists(opts["glove_path"]):
        raise SystemExit(f"GloVe file not found at {opts['glove_path']}. See README.md, 'Public GloVe vectors'.")

    data = load_processed_data(opts["processed_dir"])
    vocab = data.vocab
    train_ids = data.train.ids

    found, n_lines = read_glove_for_vocab(opts["glove_path"], vocab.token_to_id, dim)
    print(f"read {n_lines:,} GloVe lines; {len(found):,} of them are words in our vocabulary")

    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    has_vector = np.zeros(len(vocab), dtype=bool)
    for i, vec in found.items():
        matrix[i] = vec
        has_vector[i] = True

    # Coverage BEFORE filling.
    type_pct, token_pct, n_types, n_covered = coverage(has_vector, train_ids, vocab.pad_id)
    # The same figures counting only real words (not the made-up <unk>/<eos>).
    special = [vocab.unk_id, vocab.eos_id]
    counts = np.bincount(train_ids, minlength=len(vocab))
    real = np.ones(len(vocab), dtype=bool)
    real[[vocab.pad_id] + special] = False
    real_type_pct = 100.0 * (has_vector & real).sum() / real.sum()
    real_token_pct = 100.0 * counts[has_vector & real].sum() / counts[real].sum()

    # The most frequent missing words (by training count). <unk>/<eos> are listed
    # too, flagged as specials, since they are missing like any other.
    missing_ids = [i for i in np.argsort(-counts) if not has_vector[i] and i != vocab.pad_id]
    total_tokens = int(counts.sum())
    top_missing = [{"word": vocab.id_to_token[i], "train_count": int(counts[i]),
                    "percent_of_train_tokens": 100.0 * counts[i] / total_tokens,
                    "special_symbol": bool(i in special)}
                   for i in missing_ids[: opts["top_missing"]]]

    n_filled, fill_std = fill_missing_rows(matrix, has_vector, vocab.pad_id, seed)
    if not np.isfinite(matrix).all():
        raise SystemExit("some vectors are not finite numbers")
    os.makedirs(os.path.dirname(opts["output_path"]), exist_ok=True)
    np.save(opts["output_path"], matrix)

    print(f"coverage (excluding <pad>): {n_covered:,}/{n_types:,} vocabulary types = {type_pct:.2f}%, "
          f"{token_pct:.2f}% of training tokens")
    print(f"coverage of real words only (also excluding <unk>, <eos>): {real_type_pct:.2f}% of types, "
          f"{real_token_pct:.2f}% of tokens")
    print(f"{n_filled:,} rows filled randomly (std {fill_std:.4f}); <pad> row is zeros")
    print(f"{len(top_missing)} most frequent missing words:")
    for m in top_missing:
        print(f"  {m['word']!r:>14}  {m['train_count']:>9,}  {m['percent_of_train_tokens']:.3f}% of tokens"
              + ("  (special symbol)" if m["special_symbol"] else ""))

    path = save_run(
        args.results_dir, run, "src/load_glove.py", config,
        metrics={"coverage_types_percent": type_pct, "coverage_tokens_percent": token_pct,
                 "coverage_real_words_types_percent": real_type_pct,
                 "coverage_real_words_tokens_percent": real_token_pct,
                 "vocab_types_excluding_pad": n_types, "vocab_types_with_vector": n_covered,
                 "rows_filled_randomly": n_filled},
        details={
            "glove_file": os.path.basename(opts["glove_path"]),
            "glove_lines_read": n_lines,
            "training_tokens": total_tokens,
            "vectors_shape": list(matrix.shape),
            "vectors_path": opts["output_path"],
            "fill_rule": "N(0, std of real vectors), seeded; <pad> row zeros",
            "fill_std": fill_std,
            "pad_row_zero": bool(not matrix[vocab.pad_id].any()),
            "most_frequent_missing_words": top_missing,
        },
    )
    print(f"saved {opts['output_path']} {matrix.shape} and {path}")


if __name__ == "__main__":
    main()
