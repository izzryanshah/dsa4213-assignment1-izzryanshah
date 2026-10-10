"""Tests for src/generate.py.

Run with:   python tests/test_generate.py --config configs/test_generate.json

Uses a tiny made-up vocabulary (config `vocab_size` words + the 3 special
tokens), so no dataset is needed. Saves a pass/fail record to
results/test_generate.json.
"""

import argparse
import math
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from generate import NeuralNextToken, generate, make_rng, next_token_probs  # noqa: E402
from ngram import KneserNeyTrigram, build_events  # noqa: E402
from runconfig import add_config_args, load_config, save_run  # noqa: E402
from sanity_check import BigramLM  # noqa: E402

PAD_ID, UNK_ID, EOS_ID = 0, 1, 2
NUM_SPECIAL_TOKENS = 3


class FixedModel:
    """Always predicts the same next-word distribution, whatever the context."""

    def __init__(self, probs):
        self.log_probs = np.log(np.asarray(probs, dtype=np.float64))

    def next_token_log_probs(self, context_ids):
        return self.log_probs


def test_temperature_and_topk_match_the_docstring_example(config):
    p = np.log([0.5, 0.3, 0.15, 0.05])
    assert np.allclose(next_token_probs(p, 1.0, 0, []), [0.5, 0.3, 0.15, 0.05])
    assert np.allclose(next_token_probs(p, 0.5, 0, []), [0.685, 0.247, 0.062, 0.007], atol=1e-3)
    assert np.allclose(next_token_probs(p, 2.0, 0, []), [0.379, 0.294, 0.207, 0.120], atol=1e-3)
    assert np.allclose(next_token_probs(p, 1.0, 2, []), [0.625, 0.375, 0, 0])


def test_banned_words_are_never_sampled(config):
    # Probability mass is mostly on the banned ids 0 and 1.
    model = FixedModel([0.45, 0.45, 0.05, 0.05])
    for i in range(config["options"]["num_draws"] // 10):
        new_ids, _ = generate(model, [3], 20, 1.0, 0, [0, 1], EOS_ID + 100, False, make_rng(0, i, "x"))
        assert all(t in (2, 3) for t in new_ids), new_ids


def test_topk_keeps_only_the_k_likeliest_allowed_words(config):
    model = FixedModel([0.4, 0.3, 0.2, 0.1])
    seen = set()
    for i in range(300):
        new_ids, _ = generate(model, [0], 5, 1.0, 2, [], 99, False, make_rng(0, i, "y"))
        seen.update(new_ids)
    assert seen == {0, 1}, seen


def test_same_seed_same_text_different_seed_different_text(config):
    model = FixedModel(np.full(40, 1 / 40))
    run = lambda seed: generate(model, [3], 30, 1.0, 0, [], 99, False, make_rng(seed, 0, "p"))[0]
    assert run(0) == run(0)
    assert run(0) != run(1)


def test_stops_at_eos_and_does_not_include_it(config):
    model = FixedModel([0.0001, 0.0001, 0.9998])  # almost always <eos> (id 2)
    new_ids, ended = generate(model, [0], 50, 1.0, 0, [0, 1], EOS_ID, True, make_rng(0, 0, "z"))
    assert ended and EOS_ID not in new_ids and len(new_ids) < 50
    new_ids, ended = generate(model, [0], 5, 1.0, 0, [0, 1], EOS_ID, False, make_rng(0, 0, "z"))
    assert not ended and len(new_ids) == 5


def test_neural_wrapper_gives_valid_distribution(config):
    V = config["vocab_size"] + NUM_SPECIAL_TOKENS
    torch.manual_seed(config["seed"])
    wrapped = NeuralNextToken(BigramLM(V, pad_id=PAD_ID, embedding_dim=config["embedding_dim"]))
    lp = wrapped.next_token_log_probs([5, 6, 7])
    assert lp.shape == (V,) and math.isclose(np.exp(lp).sum(), 1.0, rel_tol=1e-5)
    new_ids, _ = generate(wrapped, [5], 10, 1.0, 10, [PAD_ID, UNK_ID], EOS_ID, True, make_rng(0, 0, "n"))
    assert all(0 <= t < V for t in new_ids)


def test_ngram_next_token_matches_scoring_path(config):
    """The n-gram's generation distribution must agree with the one evaluate() uses."""
    V = config["vocab_size"] + NUM_SPECIAL_TOKENS
    rng = random.Random(config["seed"])
    opts = config["options"]
    reviews = [[rng.randrange(NUM_SPECIAL_TOKENS, V) for _ in range(rng.randint(2, opts["max_fake_review_length"]))] + [EOS_ID]
               for _ in range(opts["num_fake_reviews"])]
    split = SimpleNamespace(ids=np.concatenate([np.array(r) for r in reviews]),
                            review_lengths=np.array([len(r) for r in reviews]))
    model = KneserNeyTrigram(V, discounts=(0.8, 0.7, 0.6))
    model.fit(*build_events(split, model.bos_id))
    for r in reviews[:20]:
        inputs, targets = torch.tensor([r[:-1]]), torch.tensor([r[1:]])
        scored = model.target_log_probs(inputs, targets)[0].numpy()
        for t in range(len(r) - 1):
            dist = model.next_token_log_probs(r[: t + 1])
            assert math.isclose(np.exp(dist).sum(), 1.0, rel_tol=1e-9)
            assert math.isclose(dist[r[t + 1]], scored[t], rel_tol=1e-9, abs_tol=1e-9)


TESTS = [
    test_temperature_and_topk_match_the_docstring_example,
    test_banned_words_are_never_sampled,
    test_topk_keeps_only_the_k_likeliest_allowed_words,
    test_same_seed_same_text_different_seed_different_text,
    test_stops_at_eos_and_does_not_include_it,
    test_neural_wrapper_gives_valid_distribution,
    test_ngram_next_token_matches_scoring_path,
]


def main():
    parser = argparse.ArgumentParser(description="Run the generate.py tests.")
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
    save_run(args.results_dir, run, "tests/test_generate.py", config,
             metrics={"all_passed": all_passed, "num_passed": sum(v == "passed" for v in outcomes.values()),
                      "num_tests": len(outcomes)},
             details={"tests": outcomes})
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
