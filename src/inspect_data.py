"""Inspect the raw IMDB reviews CSV before any cleaning or tokenization.

This script does not train anything — it just loads the dataset with pandas
and prints basic facts about it (how many reviews, how long they are, what
the labels look like, and a few raw examples) so you can see what cleaning
will be needed later (e.g. leftover HTML tags like "<br />").

Usage:
    python src/inspect_data.py --config configs/inspect_data.json
    python src/inspect_data.py --config configs/inspect_data.json --seed 1 --set options.n_examples=5

Settings (seed, CSV path, number of examples, word budget) come from the
config file; see runconfig.py for how overrides work.
"""

import argparse
import random

import pandas as pd

from runconfig import add_config_args, load_config, save_run


def parse_args():
    # argparse reads settings from the command line. Here the only required
    # one is --config (the file holding this run's settings); any setting in
    # that file can be overridden by a flag, e.g. --seed 1.
    parser = argparse.ArgumentParser(description="Inspect the raw IMDB reviews CSV.")
    add_config_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    config, run = load_config(args)
    opts = config["options"]  # data_path, n_examples, word_budget, example_chars, round_to

    # Record the seed so this exact run (in particular, which example reviews
    # got printed) can be reproduced later just by re-running with --seed 0.
    print(f"Random seed: {config['seed']}")
    rng = random.Random(config["seed"])

    df = pd.read_csv(opts["data_path"])

    expected_columns = {"review", "sentiment"}
    if not expected_columns.issubset(df.columns):
        raise SystemExit(
            f"Expected columns {sorted(expected_columns)} but found "
            f"{df.columns.tolist()}. Check that options.data_path points at the "
            "Kaggle 'IMDB Dataset.csv' file."
        )

    print("\nColumn names:", df.columns.tolist())

    num_reviews = len(df)
    print(f"Number of reviews: {num_reviews}")

    # value_counts() tallies how many times each distinct value ("positive",
    # "negative", ...) appears in the column, i.e. the class balance.
    label_counts = df["sentiment"].value_counts().to_dict()
    print("Label counts:", label_counts)

    # This is a rough, whitespace-based word count (str.split() with no
    # argument splits on any run of spaces/tabs/newlines) purely for sizing
    # the dataset. It will differ slightly from the word-level tokenizer the
    # project uses later for building the vocabulary, since that tokenizer
    # also splits off punctuation.
    word_counts = df["review"].str.split().apply(len)
    total_words = int(word_counts.sum())
    avg_review_length = total_words / num_reviews
    print(f"Total words (whitespace-split, rough count): {total_words}")
    print(f"Average review length: {avg_review_length:.1f} words")

    # sample() without replacement picks distinct random row positions;
    # rng.sample is seeded above so the same --seed always prints the same
    # examples.
    example_indices = rng.sample(range(num_reviews), min(opts["n_examples"], num_reviews))
    print(f"\n{len(example_indices)} raw example review(s) (truncated to {opts['example_chars']} characters):")
    for i in example_indices:
        raw_text = df["review"].iloc[i]
        label = df["sentiment"].iloc[i]
        print(f"\n--- example (index {i}, label={label}) ---")
        print(raw_text[: opts["example_chars"]])

    # Random examples won't always happen to contain "<br />" in their first
    # 500 characters, so check directly how common that known cleaning issue
    # actually is across the whole dataset, instead of leaving it to chance.
    num_with_br_tag = int(df["review"].str.contains("<br", regex=False).sum())
    pct_with_br_tag = 100 * num_with_br_tag / num_reviews
    print(
        f"\nReviews containing a '<br' tag: {num_with_br_tag} "
        f"({pct_with_br_tag:.1f}% of all reviews) — these will need HTML "
        "cleanup before tokenization."
    )

    subsample_suggestion = None
    if total_words > opts["word_budget"]:
        # Scale down proportionally: if N reviews contain `total_words`
        # words, then (word_budget / total_words) * N reviews should contain
        # roughly `word_budget` words. Rounding down to a multiple of options.round_to just
        # gives a tidier number to put in a --n-reviews flag later.
        raw_suggestion = opts["word_budget"] / avg_review_length
        subsample_suggestion = int(raw_suggestion // opts["round_to"] * opts["round_to"])
        print(
            f"\nTotal word count ({total_words:,}) is well above the "
            f"{opts['word_budget']:,}-word budget. To keep four separate "
            "language models trainable in one Colab GPU session, consider "
            f"subsampling to about {subsample_suggestion:,} reviews "
            f"(~{subsample_suggestion * avg_review_length:,.0f} words)."
        )

    results_path = save_run(
        args.results_dir, run, "src/inspect_data.py", config,
        metrics={
            "column_names": df.columns.tolist(),
            "num_reviews": num_reviews,
            "label_counts": label_counts,
            "total_words": total_words,
            "avg_review_length": avg_review_length,
            "num_reviews_with_br_tag": num_with_br_tag,
            "pct_reviews_with_br_tag": pct_with_br_tag,
            "subsample_suggestion_reviews": subsample_suggestion,
        },
    )
    print(f"\nSaved run record to {results_path}")


if __name__ == "__main__":
    main()
