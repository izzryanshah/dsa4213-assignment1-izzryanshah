"""Trigram language model with interpolated Kneser-Ney smoothing.

What a trigram model is
-----------------------
It predicts the next word from only the previous TWO words:
    P(w | u, v)      where u, v are the two words just before w.
The simplest way to estimate this is by counting in the training text:
    P(w | u, v) = count(u v w) / count(u v)
The problem: most word triples that could appear at test time NEVER appeared
in training, so their count is 0 and the model would say "impossible". One
such triple makes the whole test-set perplexity infinite. Smoothing fixes
this.

How smoothing handles never-seen sequences (interpolated Kneser-Ney)
---------------------------------------------------------------------
Two ideas work together:

1. ABSOLUTE DISCOUNTING. Every triple that WAS seen gets a small fixed
   amount D (e.g. 0.8) subtracted from its count, so the model is slightly
   less sure about what it saw. All the subtracted probability is collected
   into a "leftover" pot. The more different words were seen after (u, v),
   the bigger the pot (D x number of distinct followers), because a context
   followed by many different words is likely to be followed by yet another
   new word next time.

2. BACKOFF BY INTERPOLATION. The leftover pot is handed out to words in
   proportion to a SHORTER-context model, the bigram P(w | v), which in turn
   saves a pot for the unigram P(w), which in turn saves a pot for a
   uniform distribution over the whole vocabulary:

       P3(w|u,v) = [max(c(uvw) - D, 0) + D * N(uv.) * P2(w|v)] / c(uv.)
       P2(w|v)   = [max(c'(vw) - D, 0) + D * N(v.)  * P1(w)  ] / c'(v.)
       P1(w)     = [max(c'(w)  - D, 0) + D * N(.)   * 1/V    ] / c'(.)

   So an unseen triple (u v w) has c(uvw) = 0 and only gets the pot share
   D * N(uv.) * P2(w|v) / c(uv.). If even (v w) was never seen, the share
   comes from P1(w); if w was never seen after anything, it comes from the
   uniform 1/V term. Because of that last term EVERY word in the vocabulary
   gets a probability above zero in EVERY context, so no sequence ever has
   probability 0 and the perplexity is always finite. (If the context (u, v)
   itself was never seen, the model simply uses P2(w|v) instead.)

What makes it "Kneser-Ney" (the c' in the lower orders)
-------------------------------------------------------
The shorter-context models do NOT use raw counts. They use CONTINUATION
counts: c'(v w) = how many different words u came before "v w", and
c'(w) = how many different words v came before w. Reason: a word like
"francisco" is frequent but almost always follows "san", so it is a poor
guess for a context where "san" is absent; a word that follows MANY
different words (like "the") is a better guess for a never-seen context.
Continuation counts measure exactly that. They are only used when we have
fallen back from a longer context, which is when the lower-order models
matter.

Beginning of a review
---------------------
evaluate.py gives the model the review's tokens and asks it to predict each
next token (the first target is the review's 2nd token). To predict early
tokens the trigram needs two preceding words, so this model pretends each
review is preceded by one invisible start symbol (BOS). BOS is only ever a
context, never predicted, and has an id outside the vocabulary
(id = vocab_size); the predicted vocabulary is exactly the one from data.py
(including <unk>, mapped by data.py). Training uses the same convention, so
training and scoring see identical contexts. Only preceding tokens are ever
used (no leakage of future tokens).

Implementation notes
--------------------
Python dictionaries of 7 million tuples would be slow and memory-hungry, so
every n-gram is packed into ONE integer ("key": (u*B + v)*B + w with
B = vocab_size + 1) and stored in sorted numpy arrays. Counting is then
np.unique, and looking up many n-grams at once is np.searchsorted.

Tuning: the three discounts (D3, D2, D1 for the trigram, bigram, unigram
levels) are chosen by grid search on the VALIDATION split only. The test
split is touched once, at the very end.

Usage:
    python src/ngram.py --config configs/ngram.json
    python src/ngram.py --config configs/ngram.json --set options.max_length=256

Settings (seed, batch size, discount grid, max review length, ...) come from
the config file; see runconfig.py for how overrides work.
"""

import argparse
import itertools
import random
import time

import numpy as np
import torch

from data import check_config_matches_data, load_processed_data
from evaluate import evaluate
from runconfig import add_config_args, load_config, save_run


def build_events(split, bos_id):
    """List every (u, v, w) training event as three aligned int64 arrays.

    For each review [t_0, t_1, ..., t_{L-1}] with a BOS in front, the events
    are the predictions of t_1 ... t_{L-1}: w = t_i, v = t_{i-1}, and
    u = t_{i-2} (or BOS when i = 1). t_0 is never predicted, matching how
    evaluate.py picks its targets.
    """
    ids = split.ids.astype(np.int64)
    lengths = split.review_lengths
    starts = np.cumsum(lengths) - lengths
    # position of every token inside its own review (0 for a review's first token)
    pos_in_review = np.arange(len(ids)) - np.repeat(starts, lengths)

    j = np.nonzero(pos_in_review >= 1)[0]  # flat positions of every predicted token
    w = ids[j]
    v = ids[j - 1]
    # j - 2 is only a valid index (and inside the same review) when pos >= 2.
    u = np.where(pos_in_review[j] >= 2, ids[np.maximum(j - 2, 0)], bos_id)
    return u, v, w


def build_eval_events(reviews, bos_id, max_length):
    """Same as above but for a list of reviews, cut/filtered exactly like evaluate()."""
    us, vs, ws = [], [], []
    for r in reviews:
        if max_length is not None:
            r = r[:max_length]
        if len(r) < 2:
            continue
        r = np.asarray(r, dtype=np.int64)
        vs.append(r[:-1])
        ws.append(r[1:])
        us.append(np.concatenate(([bos_id], r[:-2])))
    return np.concatenate(us), np.concatenate(vs), np.concatenate(ws)


class SortedTable:
    """Maps integer keys to integer values via sorted arrays + binary search."""

    def __init__(self, keys, values):
        self.keys = keys      # sorted, unique
        self.values = values

    def get(self, query):
        """Value for each query key, or 0 where the key is absent."""
        idx = np.searchsorted(self.keys, query)
        idx = np.minimum(idx, len(self.keys) - 1)  # keep index valid for keys beyond the end
        found = self.keys[idx] == query
        return np.where(found, self.values[idx], 0)


class KneserNeyTrigram:
    """Interpolated Kneser-Ney trigram LM. Also a valid `evaluate()` scorer."""

    def __init__(self, vocab_size, discounts):
        self.vocab_size = vocab_size
        self.bos_id = vocab_size   # one id past the vocabulary
        self.base = vocab_size + 1
        self.discounts = discounts  # (D3, D2, D1); set by tune_discounts() before scoring

    # ---------------------------------------------------------------- training
    def fit(self, u, v, w):
        B = self.base
        key3 = (u * B + v) * B + w
        # Trigram types and how often each occurred.
        tri_keys, tri_cnt = np.unique(key3, return_counts=True)
        self.tri = SortedTable(tri_keys, tri_cnt)

        # Context (u, v): total count c(uv.) and number of distinct followers N(uv.).
        # tri_keys is sorted, so equal (u, v) are adjacent -> groups are contiguous.
        uv = tri_keys // B
        uv_keys, first, uv_types = np.unique(uv, return_index=True, return_counts=True)
        self.uv_total = SortedTable(uv_keys, np.add.reduceat(tri_cnt, first))
        self.uv_types = SortedTable(uv_keys, uv_types)

        # Bigram level: continuation count c'(v w) = number of distinct u before "v w"
        # = number of trigram types ending in (v, w).
        bi_keys, bi_cont = np.unique(tri_keys % (B * B), return_counts=True)
        self.bi = SortedTable(bi_keys, bi_cont)
        v_of_bi = bi_keys // B
        v_keys, vfirst, v_types = np.unique(v_of_bi, return_index=True, return_counts=True)
        self.v_total = SortedTable(v_keys, np.add.reduceat(bi_cont, vfirst))  # c'(v.)
        self.v_types = SortedTable(v_keys, v_types)                           # N(v.)

        # Unigram level: c'(w) = number of distinct v before w = bigram types ending in w.
        uni_keys, uni_cont = np.unique(bi_keys % B, return_counts=True)
        self.uni = SortedTable(uni_keys, uni_cont)
        self.uni_total = int(len(bi_keys))   # c'(.) = number of bigram types
        self.uni_types = int(len(uni_keys))  # N(.)  = number of distinct words seen

    def num_stored_counts(self):
        """How many n-gram counts the model stores (its size)."""
        return {
            "trigram_types": int(len(self.tri.keys)),
            "bigram_types": int(len(self.bi.keys)),
            "unigram_types": int(len(self.uni.keys)),
            "total_stored_counts": int(len(self.tri.keys) + len(self.bi.keys) + len(self.uni.keys)),
            # Context summary rows (totals / distinct-follower counts) derived from the above:
            "trigram_context_rows": int(len(self.uv_total.keys)),
            "bigram_context_rows": int(len(self.v_total.keys)),
        }

    # ----------------------------------------------------------------- scoring
    def lookup_counts(self, u, v, w):
        """All count lookups for a batch of (u, v, w). Independent of the discounts,
        so the (slow) searching is done once and reused while tuning."""
        B = self.base
        uv = u * B + v
        vw = v * B + w
        return {
            "c3": self.tri.get(uv * B + w).astype(np.float64),
            "c_uv": self.uv_total.get(uv).astype(np.float64),
            "n_uv": self.uv_types.get(uv).astype(np.float64),
            "c2": self.bi.get(vw).astype(np.float64),
            "c_v": self.v_total.get(v).astype(np.float64),
            "n_v": self.v_types.get(v).astype(np.float64),
            "c1": self.uni.get(w).astype(np.float64),
        }

    def log_probs_from_counts(self, counts, discounts=None):
        """Turn looked-up counts into log P(w | u, v) using the formulas in the docstring."""
        D3, D2, D1 = discounts if discounts is not None else self.discounts
        # Unigram level, with the uniform 1/V floor.
        p1 = (np.maximum(counts["c1"] - D1, 0.0)
              + D1 * self.uni_types / self.vocab_size) / self.uni_total
        # Bigram level. Where the context v was never seen (c_v == 0), use p1 alone.
        c_v = counts["c_v"]
        seen_v = c_v > 0
        p2 = np.where(
            seen_v,
            (np.maximum(counts["c2"] - D2, 0.0) + D2 * counts["n_v"] * p1) / np.where(seen_v, c_v, 1.0),
            p1,
        )
        # Trigram level. Where the context (u, v) was never seen, use p2 alone.
        c_uv = counts["c_uv"]
        seen_uv = c_uv > 0
        p3 = np.where(
            seen_uv,
            (np.maximum(counts["c3"] - D3, 0.0) + D3 * counts["n_uv"] * p2) / np.where(seen_uv, c_uv, 1.0),
            p2,
        )
        return np.log(p3)

    def next_token_log_probs(self, context_ids):
        """Log-probability of EVERY vocabulary word as the next token, given the
        tokens so far (this is what generate.py needs to sample from).

        Only the last two tokens matter for a trigram. If only one token has
        been seen so far, the invisible start symbol stands in for the missing
        one, exactly as in training.
        """
        v = context_ids[-1]
        u = context_ids[-2] if len(context_ids) >= 2 else self.bos_id
        all_words = np.arange(self.vocab_size, dtype=np.int64)
        counts = self.lookup_counts(np.full(self.vocab_size, u, dtype=np.int64),
                                    np.full(self.vocab_size, v, dtype=np.int64), all_words)
        return self.log_probs_from_counts(counts)

    def target_log_probs(self, inputs, targets):
        """The scorer interface evaluate() expects (see evaluate.py)."""
        inputs = inputs.numpy().astype(np.int64)
        targets = targets.numpy().astype(np.int64)
        v = inputs                                    # the token just before each target
        # u = the token before that: inputs shifted right by one, BOS at the start.
        u = np.concatenate([np.full((inputs.shape[0], 1), self.bos_id, dtype=np.int64), inputs[:, :-1]], axis=1)
        counts = self.lookup_counts(u.ravel(), v.ravel(), targets.ravel())
        lp = self.log_probs_from_counts(counts).reshape(targets.shape)
        return torch.from_numpy(lp)


def tune_discounts(model, val_counts, grid):
    """Grid-search (D3, D2, D1) to minimise VALIDATION cross-entropy."""
    rows = []
    for d3, d2, d1 in itertools.product(grid, repeat=3):
        lp = model.log_probs_from_counts(val_counts, (d3, d2, d1))
        rows.append({"d3": d3, "d2": d2, "d1": d1, "val_cross_entropy": float(-lp.mean())})
    best = min(rows, key=lambda r: r["val_cross_entropy"])
    return best, rows


def parse_args():
    parser = argparse.ArgumentParser(description="Train/tune/evaluate a Kneser-Ney trigram LM.")
    add_config_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    config, run = load_config(args)
    opts = config["options"]  # processed_dir, max_length, discount_grid
    # This implementation is a trigram: two tokens of context. Fail loudly
    # rather than record a config that the code does not actually follow.
    if config["context_length"] != 2:
        raise SystemExit(f"ngram.py only implements a trigram (context_length 2), got {config['context_length']}")

    # The model is deterministic, but the seed is set and recorded as a project rule.
    seed = config["seed"]
    print(f"Random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = load_processed_data(opts["processed_dir"])
    check_config_matches_data(config, data)
    vocab_size = len(data.vocab)
    # Discounts are unknown until tuning; start them as None so a forgotten
    # tuning step would crash instead of silently using a made-up value.
    model = KneserNeyTrigram(vocab_size, discounts=None)

    # ---- train (count n-grams on the training split) ----
    t0 = time.perf_counter()
    u, v, w = build_events(data.train, model.bos_id)
    model.fit(u, v, w)
    train_seconds = time.perf_counter() - t0
    size = model.num_stored_counts()
    print(f"Training time: {train_seconds:.1f}s, stored counts: {size['total_stored_counts']:,}")

    # ---- tune discounts on VALIDATION only ----
    t0 = time.perf_counter()
    vu, vv, vw = build_eval_events(data.val.reviews(), model.bos_id, opts["max_length"])
    val_counts = model.lookup_counts(vu, vv, vw)
    best, grid_rows = tune_discounts(model, val_counts, opts["discount_grid"])
    tune_seconds = time.perf_counter() - t0
    model.discounts = (best["d3"], best["d2"], best["d1"])
    print(f"Best discounts (D3, D2, D1) = {model.discounts}, "
          f"val cross-entropy {best['val_cross_entropy']:.4f} ({tune_seconds:.1f}s tuning)")

    # ---- final numbers, all through the shared evaluate() ----
    pad_id, batch_size, max_length = data.vocab.pad_id, config["batch_size"], opts["max_length"]
    val_result = evaluate(model, data.val, pad_id, batch_size, max_length)
    # Consistency check: the fast tuning path and evaluate() must agree.
    assert abs(val_result["avg_cross_entropy"] - best["val_cross_entropy"]) < 1e-9
    # The test split is used ONCE, only now that the discounts are frozen.
    test_result = evaluate(model, data.test, pad_id, batch_size, max_length)
    print(f"Test cross-entropy: {test_result['avg_cross_entropy']:.4f} nats, "
          f"perplexity: {test_result['perplexity']:.1f}")

    results_path = save_run(
        args.results_dir, run, "src/ngram.py", config,
        metrics={"val": val_result, "test": test_result},
        details={
            "vocab_size_with_special_tokens": vocab_size,
            "best_discounts": {"d3": best["d3"], "d2": best["d2"], "d1": best["d1"]},
            "train_seconds": train_seconds,
            "tune_seconds": tune_seconds,
            "model_size": size,
            "tuning_grid": grid_rows,
        },
    )
    print(f"Saved run record to {results_path}")


if __name__ == "__main__":
    main()
