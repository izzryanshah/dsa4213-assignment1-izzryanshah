"""Train our own Skip-gram word vectors (word2vec, via gensim) on the TRAINING split only.

What this is for
----------------
Part II compares three ways of giving a language model its word embeddings.
This script makes the "own embeddings" one: 100-d vectors learned by word2vec
from our own IMDB training text, which a later run will load and keep FIXED.

What word2vec (Skip-gram) does, in plain words
----------------------------------------------
It slides a window over the text. For each centre word it learns to predict
the words around it ("skip-gram" = centre word -> its neighbours). Words that
appear in similar contexts end up with similar vectors. It is NOT a language
model: it never predicts the next word, it sees words on BOTH sides. That is
fine here because we only use it to initialise embeddings, and it only ever
reads the training split, so nothing from validation or test leaks in.

What goes in, what comes out
----------------------------
- Input: data/processed/train.npz (token ids, one <eos> after every review).
  Each review becomes one "sentence" (a list of word strings). Ids are turned
  back into words with the shared vocabulary, so <unk> is just another token.
- Output: data/embeddings/w2v_100d.npy, a float32 array of shape
  (vocab_size, 100) where ROW i IS THE VECTOR FOR VOCABULARY ID i. That is
  exactly the layout of nn.Embedding's weight matrix, so Part II can do
  `nn.Embedding.from_pretrained(torch.from_numpy(np.load(...)), freeze=True)`.
- <pad> (id 0) never occurs in the text, so word2vec has no vector for it; its
  row is all zeros. (It is masked out of the loss anyway.) Any other vocabulary
  word word2vec did not learn (possible only if min_count > 1) is filled by the
  shared rule in embedding_utils.py: small random numbers on the same scale as
  the real vectors (decision R2); the same rule is used for words missing from
  GloVe. The script reports coverage BEFORE that fill: the % of vocabulary types
  and of training tokens that have a real vector.
- Timing: word2vec training is timed on its own (build-vocabulary step and
  training step separately) and saved with the run record to
  results/word2vec_cost.json.

Main gensim settings (all in configs/word2vec_cost.json, none hard-coded):
  sg=1          Skip-gram (0 would be CBOW).
  vector_size   = embedding_dim = 100, to match GloVe 100d and the models.
  window=5      Look 5 words left and 5 right of the centre word. Small windows
                (2-3) capture syntax ("good" ~ "great"); large ones (10+) capture
                topic. 5 is the usual middle ground and the gensim default.
  min_count=1   Keep every word. Our vocabulary is already cut to the 20,000
                most frequent training words, so every word is common enough
                (coverage 100%); min_count>1 would leave some ids to the fill rule.
  epochs=10     Passes over the training text. The default (5) is tuned for
                huge corpora; ours is ~11M tokens, so a few more passes give
                the rarer words more updates. More epochs = longer, small gains.
  negative=5, sample=1e-3 (gensim defaults, written out so they are recorded):
                negative sampling, 5 random "wrong" words per example, and
                down-sampling of very frequent words ("the", ".").
  workers=1     One thread. With several threads gensim is not reproducible even
                with a seed (the order work is shared out varies). Slower, but
                the same seed then gives the same vectors.
  seed          from the config.

Usage:
    python src/train_word2vec.py --config configs/word2vec_cost.json
"""

import argparse
import os
import random
import time

import numpy as np
from gensim.models import Word2Vec

from data import load_processed_data
from embedding_utils import coverage, fill_missing_rows
from runconfig import add_config_args, load_config, save_run


def reviews_as_word_lists(split, vocab):
    """Turn the split's flat id stream into one list of word strings per review."""
    id_to_token = vocab.id_to_token
    return [[id_to_token[i] for i in review] for review in split.reviews()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    args = parser.parse_args()
    config, run = load_config(args)
    opts = config["options"]

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)

    data = load_processed_data(opts["processed_dir"])
    vocab = data.vocab
    if config["vocab_size"] is not None and len(vocab) != config["vocab_size"] + 3:
        # data.py adds <pad>, <unk>, <eos> on top of vocab_size words.
        print(f"note: vocabulary has {len(vocab)} rows (config vocab_size={config['vocab_size']} + 3 special tokens)")

    sentences = reviews_as_word_lists(data.train, vocab)  # TRAINING split only
    n_tokens = sum(len(s) for s in sentences)
    print(f"training on {len(sentences):,} training reviews, {n_tokens:,} tokens")

    model = Word2Vec(
        vector_size=config["embedding_dim"],
        sg=opts["sg"],
        window=opts["window"],
        min_count=opts["min_count"],
        negative=opts["negative"],
        sample=opts["sample"],
        workers=opts["workers"],
        epochs=config["epochs"],
        seed=seed,
    )

    t0 = time.perf_counter()
    model.build_vocab(sentences)
    t1 = time.perf_counter()
    model.train(sentences, total_examples=len(sentences), epochs=model.epochs)
    t2 = time.perf_counter()
    build_vocab_seconds, train_seconds = t1 - t0, t2 - t1

    # Line the vectors up with OUR vocabulary: row i = vector of vocabulary id i.
    matrix = np.zeros((len(vocab), config["embedding_dim"]), dtype=np.float32)
    has_vector = np.zeros(len(vocab), dtype=bool)
    for i, token in enumerate(vocab.id_to_token):
        if token in model.wv.key_to_index:
            matrix[i] = model.wv[token]
            has_vector[i] = True

    # Coverage is measured BEFORE filling: how much of the vocabulary and of
    # the training text word2vec actually learned a vector for.
    type_pct, token_pct, n_types, n_covered = coverage(has_vector, data.train.ids, vocab.pad_id)
    n_filled, fill_std = fill_missing_rows(matrix, has_vector, vocab.pad_id, seed)
    if not np.isfinite(matrix).all():
        raise SystemExit("some vectors are not finite numbers")
    print(f"coverage: {n_covered:,}/{n_types:,} vocabulary types ({type_pct:.2f}%), "
          f"{token_pct:.2f}% of training tokens; {n_filled} rows filled randomly")

    os.makedirs(os.path.dirname(opts["output_path"]), exist_ok=True)
    np.save(opts["output_path"], matrix)

    path = save_run(
        args.results_dir, run, "src/train_word2vec.py", config,
        metrics={"train_seconds": train_seconds, "build_vocab_seconds": build_vocab_seconds,
                 "total_seconds": build_vocab_seconds + train_seconds},
        details={
            "algorithm": "skipgram" if opts["sg"] == 1 else "cbow",
            "gensim_version": __import__("gensim").__version__,
            "training_reviews": len(sentences),
            "training_tokens": n_tokens,
            "vectors_shape": list(matrix.shape),
            "vectors_path": opts["output_path"],
            "pad_row_zero": bool(not matrix[vocab.pad_id].any()),
            "words_learned": len(model.wv),
            "coverage_types_percent": type_pct,
            "coverage_tokens_percent": token_pct,
            "vocab_types_excluding_pad": n_types,
            "vocab_types_with_vector": n_covered,
            "rows_filled_randomly": n_filled,
            "fill_rule": "N(0, std of real vectors), seeded; <pad> row zeros",
            "fill_std": fill_std,
            "timing_note": "wall-clock, time.perf_counter; covers build_vocab and train only "
                           "(not data loading or saving)",
        },
    )
    print(f"vocab build {build_vocab_seconds:.1f}s, training {train_seconds:.1f}s")
    print(f"saved {opts['output_path']} {matrix.shape} and {path}")


if __name__ == "__main__":
    main()
