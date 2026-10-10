"""Short tests for src/evaluate.py, using the untrained BigramLM from sanity_check.py.

Run with:   python tests/test_evaluate.py --config configs/test_evaluate.json

The tests use small made-up reviews (no dataset needed), so they run in a
second or two. Each test is a plain function that takes the run's config and
raises AssertionError if something is wrong; `main()` runs them all and saves
a pass/fail record to results/test_evaluate.json.

The config's `vocab_size` is the number of fake WORDS; the fake vocabulary
also contains the 3 special tokens, so its total size is vocab_size + 3.
"""

import argparse
import math
import os
import random
import sys

import torch

# Make `import evaluate` / `import sanity_check` work when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import evaluate  # noqa: E402
from runconfig import add_config_args, load_config, save_run  # noqa: E402
from sanity_check import BigramLM, compute_perplexity  # noqa: E402

PAD_ID, UNK_ID, EOS_ID = 0, 1, 2  # same reserved ids as data.py
NUM_SPECIAL_TOKENS = 3


def total_vocab(config):
    return config["vocab_size"] + NUM_SPECIAL_TOKENS


def make_reviews(config):
    """Fake reviews: random word ids of varied length, ending in <eos>."""
    rng = random.Random(config["seed"])
    opts = config["options"]
    reviews = []
    for _ in range(opts["num_fake_reviews"]):
        length = rng.randint(1, opts["max_fake_review_length"])  # includes length-1 reviews (no targets)
        reviews.append([rng.randrange(NUM_SPECIAL_TOKENS, total_vocab(config)) for _ in range(length - 1)] + [EOS_ID])
    return reviews


def make_model(config):
    torch.manual_seed(config["seed"])  # fixes the untrained model's random weights
    return BigramLM(total_vocab(config), pad_id=PAD_ID, embedding_dim=config["embedding_dim"])


class RecordingScorer:
    """Stand-in for the n-gram model: a scorer that gives every target
    probability 1/V and records the targets it was asked to score."""

    def __init__(self, vocab_total):
        self.vocab_total = vocab_total
        self.seen = []

    def target_log_probs(self, inputs, targets):
        self.seen.append(targets[targets != PAD_ID].tolist())
        return torch.full(targets.shape, -math.log(self.vocab_total))


def test_untrained_perplexity_is_near_vocab_size(config):
    V = total_vocab(config)
    result = evaluate(make_model(config), make_reviews(config), PAD_ID, config["batch_size"], None)
    assert 0.7 * V < result["perplexity"] < 1.3 * V, result["perplexity"]
    assert math.isclose(result["perplexity"], math.exp(result["avg_cross_entropy"]))


def test_matches_sanity_check_implementation(config):
    V, cut = total_vocab(config), config["options"]["truncate_length"]
    reviews, model = make_reviews(config), make_model(config)
    ppl, nll, n = compute_perplexity(model, reviews, PAD_ID, V, config["batch_size"], "cpu", cut)
    result = evaluate(model, reviews, PAD_ID, config["batch_size"], cut)
    assert result["num_targets"] == n
    assert math.isclose(result["avg_cross_entropy"], nll, rel_tol=1e-5)
    assert math.isclose(result["perplexity"], ppl, rel_tol=1e-4)


def test_result_does_not_depend_on_batch_size(config):
    reviews, model = make_reviews(config), make_model(config)
    a = evaluate(model, reviews, PAD_ID, 1, None)
    b = evaluate(model, reviews, PAD_ID, config["batch_size"], None)
    assert a["num_targets"] == b["num_targets"]
    assert a["target_checksum"] == b["target_checksum"]
    assert math.isclose(a["avg_cross_entropy"], b["avg_cross_entropy"], rel_tol=1e-5)


def test_neural_and_ngram_style_scorer_see_same_targets(config):
    V, cut = total_vocab(config), config["options"]["truncate_length"]
    reviews = make_reviews(config)
    neural = evaluate(make_model(config), reviews, PAD_ID, config["batch_size"], cut)
    scorer = RecordingScorer(V)
    other = evaluate(scorer, reviews, PAD_ID, config["batch_size"], cut)

    assert neural["num_targets"] == other["num_targets"]
    assert neural["target_checksum"] == other["target_checksum"]
    # A scorer that assigns uniform probability must get perplexity V (up to float32 rounding).
    assert math.isclose(other["perplexity"], V, rel_tol=1e-5)
    # No padding id was ever scored, and every real target token was.
    flat = [t for batch in scorer.seen for t in batch]
    assert PAD_ID not in flat
    assert len(flat) == other["num_targets"]


def test_padding_is_ignored(config):
    # A short review scored alone vs. padded inside a batch with a longer
    # one: the combined average must equal the target-weighted average of
    # the two separate scores.
    model = make_model(config)
    long_one, short_one = [5, 6, 7, 8, 9, EOS_ID], [5, 6, EOS_ID]
    alone = evaluate(model, [short_one], PAD_ID, 1, None)
    both = evaluate(model, [long_one, short_one], PAD_ID, 2, None)
    long_alone = evaluate(model, [long_one], PAD_ID, 1, None)
    n_alone, n_long = alone["num_targets"], long_alone["num_targets"]
    expected = (alone["avg_cross_entropy"] * n_alone + long_alone["avg_cross_entropy"] * n_long) / (n_alone + n_long)
    assert math.isclose(both["avg_cross_entropy"], expected, rel_tol=1e-5)


TESTS = [
    test_untrained_perplexity_is_near_vocab_size,
    test_matches_sanity_check_implementation,
    test_result_does_not_depend_on_batch_size,
    test_neural_and_ngram_style_scorer_see_same_targets,
    test_padding_is_ignored,
]


def main():
    parser = argparse.ArgumentParser(description="Run the evaluate.py tests.")
    add_config_args(parser)
    args = parser.parse_args()
    config, run = load_config(args)
    print(f"Random seed: {config['seed']}")

    outcomes = {}
    for test in TESTS:
        try:
            test(config)
            outcomes[test.__name__] = "passed"
        except AssertionError as e:
            outcomes[test.__name__] = f"FAILED: {e!r}"
        print(f"{test.__name__}: {outcomes[test.__name__]}")

    all_passed = all(v == "passed" for v in outcomes.values())
    save_run(args.results_dir, run, "tests/test_evaluate.py", config,
             metrics={"all_passed": all_passed, "num_passed": sum(v == "passed" for v in outcomes.values()),
                      "num_tests": len(outcomes)},
             details={"tests": outcomes})
    sys.exit(0 if all_passed else 1)


# pytest also works: it supplies the config through this fixture.
try:
    import json

    import pytest

    @pytest.fixture
    def config():
        with open(os.path.join(os.path.dirname(__file__), "..", "configs", "test_evaluate.json")) as f:
            return json.load(f)
except ImportError:
    pass

if __name__ == "__main__":
    main()
