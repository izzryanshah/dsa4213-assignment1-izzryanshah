"""Part III error review: PREPARE the material, with measurements only (no interpretation).

What this file does
-------------------
For each classifier variant in results/part3.csv it writes:

1.  results/part3_error_sample_<variant>.csv: `n_sample` misclassified test reviews drawn at
    random (fixed seed) from results/part3_errors_<variant>.csv, each with the full review text,
    true and predicted label, and a few COUNTED properties: tokens kept (of the first 256),
    whether the review was truncated, number and share of <unk> tokens, and number of negation
    words. These are for you to read; the script does not describe patterns.
2.  results/part3_error_stats.csv: the same counted properties averaged over ALL test errors and
    over all correctly classified test reviews, per variant (so the sample can be compared with
    the whole).
3.  results/part3_ranking.csv: the Part I test perplexity of each language model next to the
    Part III test accuracy / macro-F1 of its classifier, with the rank of each (1 = best).

"Negation words" = the words in options.negation_words plus any n't contraction (the word
tokenizer splits "don't" into don ' t, so a "t" right after an apostrophe is counted).
Sarcasm cannot be counted; that is left to your reading.

Run (laptop CPU):  python3 src/error_sample.py --config configs/p3_error_sample.json
"""

import argparse
import json

import numpy as np
import pandas as pd

from data import load_processed_data
from runconfig import add_config_args, load_config, save_run


def review_properties(ids, lengths, vocab, negation_ids, apostrophe_id, t_id, unk_id):
    """Counted properties of every review row (ids is (n, max_length), padded)."""
    n_tokens = lengths.astype(int)
    real = np.arange(ids.shape[1])[None, :] < lengths[:, None]
    unk = ((ids == unk_id) & real).sum(axis=1)
    neg_words = (np.isin(ids, negation_ids) & real).sum(axis=1)
    # an n't contraction: token "t" whose previous token is "'"
    nt = ((ids[:, 1:] == t_id) & (ids[:, :-1] == apostrophe_id) & real[:, 1:]).sum(axis=1)
    return pd.DataFrame({"n_tokens": n_tokens, "truncated": n_tokens == ids.shape[1], "unk_count": unk,
                         "unk_rate": unk / n_tokens, "negation_count": neg_words + nt})


def main():
    parser = argparse.ArgumentParser(description="Prepare Part III error-review material (no interpretation).")
    add_config_args(parser)
    args = parser.parse_args()
    config, run = load_config(args)
    opts = config["options"]
    print(f"Random seed: {config['seed']}")

    vocab = load_processed_data(opts["processed_dir"]).vocab
    t2i = vocab.token_to_id
    negation_ids = [t2i[w] for w in opts["negation_words"] if w in t2i]
    apostrophe_id, t_id = t2i["'"], t2i["t"]
    with np.load(f"{opts['classification_dir']}/test.npz") as z:
        ids, lengths, labels, orig = z["ids"], z["lengths"], z["labels"], z["original_indices"]
    props = review_properties(ids, lengths, vocab, negation_ids, apostrophe_id, t_id, vocab.unk_id)
    props.insert(0, "original_index", orig)

    part3 = pd.read_csv(opts["part3_csv"])
    stat_rows, ranking_rows = [], []
    for variant in part3["variant"]:
        errors = pd.read_csv(f"{opts['errors_prefix']}{variant}.csv")
        is_error = props["original_index"].isin(errors["original_index"])
        sample = errors.sample(n=opts["n_sample"], random_state=config["seed"]).merge(
            props, on="original_index", how="left")
        sample = sample[["original_index", "true_label", "predicted_label", "n_tokens", "truncated",
                         "unk_count", "unk_rate", "negation_count", "text"]]
        sample.to_csv(f"{opts['sample_prefix']}{variant}.csv", index=False)
        for group, mask in (("errors", is_error), ("correct", ~is_error)):
            p = props[mask]
            stat_rows.append({"variant": variant, "group": group, "n_reviews": len(p),
                              "mean_tokens": p["n_tokens"].mean(), "share_truncated": p["truncated"].mean(),
                              "mean_unk_rate": p["unk_rate"].mean(),
                              "share_with_negation": (p["negation_count"] > 0).mean(),
                              "mean_negation_count": p["negation_count"].mean()})

        lm = json.load(open(f"results/{opts['variant_to_lm_run'][variant]}.json"))
        row = part3[part3["variant"] == variant].iloc[0]
        ranking_rows.append({"variant": variant, "lm_run": opts["variant_to_lm_run"][variant],
                             "lm_test_perplexity": lm["metrics"]["test_perplexity"],
                             "classifier_test_accuracy": row["test_accuracy"],
                             "classifier_test_macro_f1": row["test_macro_f1"]})

    ranking = pd.DataFrame(ranking_rows)
    ranking["perplexity_rank"] = ranking["lm_test_perplexity"].rank(method="min").astype(int)   # lower is better
    ranking["accuracy_rank"] = ranking["classifier_test_accuracy"].rank(ascending=False, method="min").astype(int)
    ranking.to_csv(opts["ranking_csv"], index=False)
    pd.DataFrame(stat_rows).to_csv(opts["stats_csv"], index=False)
    print(pd.DataFrame(stat_rows).round(4).to_string(index=False))
    print(ranking.round(4).to_string(index=False))

    path = save_run(args.results_dir, run, "src/error_sample.py", config,
                    metrics={"stats": stat_rows, "ranking": ranking_rows},
                    details={"negation_ids_found": len(negation_ids)})
    print(f"\nSaved run record to {path}")


if __name__ == "__main__":
    main()
