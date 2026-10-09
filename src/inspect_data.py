"""Inspect the raw IMDB reviews CSV before any cleaning or tokenization.

This script does not train anything — it just loads the dataset with pandas
and prints basic facts about it (how many reviews, how long they are, what
the labels look like, and a few raw examples) so you can see what cleaning
will be needed later (e.g. leftover HTML tags like "<br />").

Usage:
    python src/inspect_data.py
    python src/inspect_data.py --data-path "data/IMDB Dataset.csv" --n-examples 5 --seed 0
"""

import argparse
import json
import os
import random

import pandas as pd


def parse_args():
    # argparse reads settings from the command line (e.g. --seed 0) instead of
    # having them hard-coded, so every run is controlled by flags you can see
    # and change without editing the script.
    parser = argparse.ArgumentParser(description="Inspect the raw IMDB reviews CSV.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/IMDB Dataset.csv",
        help="Path to the raw IMDB CSV file.",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=3,
        help="Number of raw example reviews to print.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to pick which example reviews to print.",
    )
    parser.add_argument(
        "--word-budget",
        type=int,
        default=3_000_000,
        help=(
            "Rough total-word threshold. If the dataset's total word count is "
            "well above this, the script suggests how many reviews to "
            "subsample so the corpus stays trainable."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/inspect_data.json",
        help="Where to save the summary statistics as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Record the seed so this exact run (in particular, which example reviews
    # got printed) can be reproduced later just by re-running with --seed 0.
    print(f"Random seed: {args.seed}")
    rng = random.Random(args.seed)

    df = pd.read_csv(args.data_path)

    expected_columns = {"review", "sentiment"}
    if not expected_columns.issubset(df.columns):
        raise SystemExit(
            f"Expected columns {sorted(expected_columns)} but found "
            f"{df.columns.tolist()}. Check that --data-path points at the "
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
    example_indices = rng.sample(range(num_reviews), min(args.n_examples, num_reviews))
    print(f"\n{len(example_indices)} raw example review(s) (truncated to 500 characters):")
    for i in example_indices:
        raw_text = df["review"].iloc[i]
        label = df["sentiment"].iloc[i]
        print(f"\n--- example (index {i}, label={label}) ---")
        print(raw_text[:500])

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
    if total_words > args.word_budget:
        # Scale down proportionally: if N reviews contain `total_words`
        # words, then (word_budget / total_words) * N reviews should contain
        # roughly `word_budget` words. Rounding down to the nearest 500 just
        # gives a tidier number to put in a --n-reviews flag later.
        raw_suggestion = args.word_budget / avg_review_length
        subsample_suggestion = int(raw_suggestion // 500 * 500)
        print(
            f"\nTotal word count ({total_words:,}) is well above the "
            f"{args.word_budget:,}-word budget. To keep four separate "
            "language models trainable in one Colab GPU session, consider "
            f"subsampling to about {subsample_suggestion:,} reviews "
            f"(~{subsample_suggestion * avg_review_length:,.0f} words)."
        )

    summary = {
        "data_path": args.data_path,
        "seed": args.seed,
        "column_names": df.columns.tolist(),
        "num_reviews": num_reviews,
        "label_counts": label_counts,
        "total_words": total_words,
        "avg_review_length": avg_review_length,
        "num_reviews_with_br_tag": num_with_br_tag,
        "pct_reviews_with_br_tag": pct_with_br_tag,
        "word_budget": args.word_budget,
        "subsample_suggestion_reviews": subsample_suggestion,
    }

    # Make sure the results/ folder exists before writing into it (it does
    # already in this repo, but this keeps the script self-contained).
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary statistics to {args.output}")


if __name__ == "__main__":
    main()
