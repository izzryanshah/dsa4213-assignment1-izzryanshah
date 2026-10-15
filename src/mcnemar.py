"""Part III: McNemar's test on the test predictions of two classifiers (paired, same 5,000 reviews).

What this file does
-------------------
Both classifiers were scored on the same test reviews, so their results can be paired review by
review. classify.py saved every MISCLASSIFIED test review of each classifier
(results/part3_errors_<variant>.csv), so a review that is not in a classifier's error file was
classified correctly. For each test review we get one of four outcomes:

                         B correct      B wrong
        A correct        both right     n10  (only A right)
        A wrong          n01 (only B right)    both wrong

McNemar's test asks whether n10 and n01 differ more than chance would explain. Only the two
"disagreement" counts matter:

    statistic = (|n10 - n01| - 1)^2 / (n10 + n01)     (chi-square, 1 degree of freedom,
                                                       with the usual continuity correction)
    p-value   = P(chi-square_1 >= statistic)

The exact version is also reported: under "no difference" each disagreement is a coin flip, so
the two-sided exact p-value comes from a Binomial(n10 + n01, 0.5) distribution.

Run (laptop CPU):  python3 src/mcnemar.py --config configs/p3_mcnemar.json
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

from runconfig import add_config_args, load_config, save_run


def mcnemar(wrong_a, wrong_b):
    """wrong_a, wrong_b: boolean arrays over the same reviews (True = misclassified)."""
    n10 = int((~wrong_a & wrong_b).sum())   # A right, B wrong
    n01 = int((wrong_a & ~wrong_b).sum())   # A wrong, B right
    both_right = int((~wrong_a & ~wrong_b).sum())
    both_wrong = int((wrong_a & wrong_b).sum())
    disagree = n10 + n01
    statistic = (abs(n10 - n01) - 1) ** 2 / disagree
    return {"both_right": both_right, "both_wrong": both_wrong, "only_a_right": n10, "only_b_right": n01,
            "statistic": statistic, "p_value": float(chi2.sf(statistic, df=1)),
            "p_value_exact": float(binomtest(min(n10, n01), disagree, 0.5).pvalue)}


def main():
    parser = argparse.ArgumentParser(description="McNemar's test on two classifiers' test predictions.")
    add_config_args(parser)
    args = parser.parse_args()
    config, run = load_config(args)
    opts = config["options"]
    print(f"Random seed: {config['seed']} (no randomness is used)")

    with np.load(f"{opts['classification_dir']}/test.npz") as z:
        test_ids = z["original_indices"]
    wrong = {}
    for name in ("variant_a", "variant_b"):
        errs = pd.read_csv(f"{opts['errors_prefix']}{opts[name]}.csv")["original_index"]
        if not errs.isin(test_ids).all():
            raise SystemExit(f"{opts[name]}: an error row is not a test review.")
        wrong[name] = np.isin(test_ids, errs)
        print(f"{opts[name]}: {int(wrong[name].sum())} errors of {len(test_ids)} test reviews")

    result = mcnemar(wrong["variant_a"], wrong["variant_b"])
    print(result)
    path = save_run(args.results_dir, run, "src/mcnemar.py", config,
                    metrics={"variant_a": opts["variant_a"], "variant_b": opts["variant_b"],
                             "n_test": len(test_ids), **result})
    print(f"Saved run record to {path}")


if __name__ == "__main__":
    main()
