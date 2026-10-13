"""Helpers shared by the scripts that build a fixed embedding table (own word2vec, public GloVe).

Both tables must be built the same way so Part II compares like with like:
row i is the vector for vocabulary id i, and any vocabulary word that has no
vector gets filled by ONE rule, `fill_missing_rows`, below.
"""

import numpy as np


def fill_missing_rows(matrix, has_vector, pad_id, seed):
    """Fill rows with no vector with small random numbers (decision R2).

    - `matrix`      (vocab_size, dim) array; rows without a vector are ignored.
    - `has_vector`  bool array, True where the row holds a real vector.
    - Rows without a vector get draws from a normal distribution with mean 0
      and the standard deviation of ALL the real vectors' values, so they are
      on the same scale as real words and are all different from each other.
    - The <pad> row is always zeros (it is masked out of the loss anyway).
    - The random draws use `seed`, so the same inputs give the same table.

    Returns the number of rows that were filled randomly (not counting <pad>).
    """
    rng = np.random.default_rng(seed)
    std = float(matrix[has_vector].std())
    to_fill = ~has_vector
    to_fill[pad_id] = False
    n_fill = int(to_fill.sum())
    matrix[to_fill] = rng.normal(0.0, std, size=(n_fill, matrix.shape[1])).astype(matrix.dtype)
    matrix[pad_id] = 0.0
    return n_fill, std


def coverage(has_vector, ids, pad_id):
    """Share of vocabulary types and of training tokens that have a real vector.

    `has_vector[i]` says whether vocabulary id i has a vector; `ids` is the
    training token stream. <pad> is excluded from the type count (it never
    occurs in the text). Returns (type_percent, token_percent, n_types, n_with_vector).
    """
    counts = np.bincount(ids, minlength=len(has_vector))
    types = np.ones(len(has_vector), dtype=bool)
    types[pad_id] = False
    type_pct = 100.0 * (has_vector & types).sum() / types.sum()
    token_pct = 100.0 * counts[has_vector].sum() / counts.sum()
    return float(type_pct), float(token_pct), int(types.sum()), int((has_vector & types).sum())
