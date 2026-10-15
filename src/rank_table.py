"""Part III: language-model test perplexity next to classifier test accuracy / macro-F1, with Spearman correlation.

What this file does
-------------------
For every feature set in results/part3_five.csv (made by classify.py) it looks up the test
perplexity of the language model that produced the features (the `test_perplexity` stored in that
model's training run record, `options.variant_to_lm_run`), and writes one table row per feature set
to results/part3_lm_vs_classifier.csv with the perplexity, accuracy, macro-F1 and the rank of each
(rank 1 = lowest perplexity / highest accuracy or macro-F1; ties share the lowest rank).

It then computes the Spearman rank correlation (scipy.stats.spearmanr) between perplexity and
accuracy and between perplexity and macro-F1 across the rows, with its two-sided p-value. Spearman
compares RANKS, not values: +1 means the same order, -1 the opposite order. Because LOWER perplexity
is better but HIGHER accuracy is better, "better language model = better classifier" shows up as a
NEGATIVE correlation here. The sign is reported as computed, not flipped.

Run (laptop CPU), after classify.py has run on all five feature sets:
    python3 src/rank_table.py --config configs/p3_rank_table.json
"""

import argparse
import json

import pandas as pd
from scipy.stats import spearmanr

from runconfig import add_config_args, load_config, save_run


def build_table(classifier, lm_perplexity):
    """classifier: DataFrame from part3_five.csv; lm_perplexity: {variant: perplexity}."""
    table = pd.DataFrame({"variant": classifier["variant"],
                          "lm_test_perplexity": classifier["variant"].map(lm_perplexity),
                          "test_accuracy": classifier["test_accuracy"],
                          "test_macro_f1": classifier["test_macro_f1"]})
    table["perplexity_rank"] = table["lm_test_perplexity"].rank(method="min").astype(int)
    table["accuracy_rank"] = table["test_accuracy"].rank(ascending=False, method="min").astype(int)
    table["macro_f1_rank"] = table["test_macro_f1"].rank(ascending=False, method="min").astype(int)
    return table


def spearman(table):
    out = {}
    for metric in ("test_accuracy", "test_macro_f1"):
        rho, p = spearmanr(table["lm_test_perplexity"], table[metric])
        out[metric] = {"rho": float(rho), "p_value": float(p)}
    return out


def main():
    parser = argparse.ArgumentParser(description="LM perplexity vs classifier accuracy table + Spearman.")
    add_config_args(parser)
    args = parser.parse_args()
    config, run = load_config(args)
    opts = config["options"]
    print(f"Random seed: {config['seed']} (no randomness is used)")

    classifier = pd.read_csv(opts["classifier_csv"])
    lm_perplexity, lm_files = {}, {}
    for variant in classifier["variant"]:
        lm_run = opts["variant_to_lm_run"][variant]
        with open(f"results/{lm_run}.json") as f:
            lm_perplexity[variant] = json.load(f)["metrics"]["test_perplexity"]
        lm_files[variant] = f"results/{lm_run}.json"
    table = build_table(classifier, lm_perplexity)
    table.to_csv(opts["output_csv"], index=False)
    corr = spearman(table)
    print(table.round(4).to_string(index=False))
    print(corr)

    path = save_run(args.results_dir, run, "src/rank_table.py", config,
                    metrics={"n_representations": len(table), "spearman_perplexity_vs": corr,
                             "rows": table.to_dict(orient="records")},
                    details={"lm_records": lm_files, "classifier_csv": opts["classifier_csv"]})
    print(f"Saved {opts['output_csv']} and run record {path}")


if __name__ == "__main__":
    main()
