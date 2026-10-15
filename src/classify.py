"""Part III step 3: train a logistic-regression sentiment classifier on each frozen-LM feature set.

What this file does
-------------------
For every feature set listed in `options.feature_sets` (each is a folder in data/features/ made
by extract_features.py: one fixed-length vector per review plus its label) it runs exactly the
same procedure, with the same seed:

0.  Optionally (`options.pca_components`, null = off) reduce the features to that many dimensions
    with PCA. PCA is FIT ON THE TRAIN SPLIT ONLY; validation and test are then projected with the
    train-fitted PCA. The steps below run on the reduced features.
1.  Optionally standardise the features (subtract the TRAINING mean, divide by the TRAINING
    standard deviation of each column; the validation and test sets reuse those numbers).
2.  For each regularisation strength C in `options.c_grid`, fit scikit-learn's LogisticRegression
    on the TRAIN split and score it on the VALIDATION split. (C is the INVERSE regularisation
    strength: small C = strong regularisation.)
3.  Pick the C with the best validation `selection_metric` (ties go to the smaller C). The test
    split is not touched until this choice is made.
4.  Use the model trained at that C to predict the TEST split ONCE. Report test accuracy and
    macro-F1 (the average of the F1 scores of the two classes).
5.  Save every misclassified test review (its original text, true label, predicted label) to
    results/part3_errors_<variant>.csv. The errors are only saved, not analysed here.

Outputs: results/part3.csv (one row per feature set), results/part3_errors_<variant>.csv, and the
usual results/p3_classify.json run record (including the validation score for every C).

Runs on a laptop CPU (not Colab-only). Command:
    python3 src/classify.py --config configs/p3_classify.json
"""

import argparse
import csv
import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from runconfig import add_config_args, load_config, save_run

LABEL_NAMES = {0: "negative", 1: "positive"}


def load_features(features_dir, variant, split):
    with np.load(os.path.join(features_dir, variant, f"{split}.npz")) as z:
        return z["features"].astype(np.float64), z["labels"], z["original_indices"]


def score(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


def select_c(x_train, y_train, x_val, y_val, c_grid, metric, max_iter, seed):
    """Train one model per C on TRAIN, score on VALIDATION; return the best C, its model and all scores.
    Uses no test data (the function does not even receive any)."""
    results, best = [], None
    for c in c_grid:
        model = LogisticRegression(C=c, max_iter=max_iter, random_state=seed)
        model.fit(x_train, y_train)
        val = score(y_val, model.predict(x_val))
        results.append({"C": c, **val, "converged": bool(model.n_iter_[0] < max_iter)})
        # '>' (not '>=') keeps the earlier, i.e. smaller, C on a tie because the grid is ascending.
        if best is None or val[metric] > best[0]:
            best = (val[metric], c, model)
    return best[1], best[2], results


def main():
    parser = argparse.ArgumentParser(description="Logistic regression on frozen LM features (Part III).")
    add_config_args(parser)
    args = parser.parse_args()
    config, run = load_config(args)
    opts = config["options"]
    c_grid = sorted(opts["c_grid"])
    print(f"Random seed: {config['seed']}")
    raw = pd.read_csv(opts["raw_csv"])  # the original review texts, for the error files

    rows, details = [], {}
    k = opts.get("pca_components")  # None = no dimensionality reduction
    for source in opts["feature_sets"]:
        t0 = time.time()
        variant = f"{source}_pca{k}" if k else source  # name used in the output files
        x_tr, y_tr, _ = load_features(opts["features_dir"], source, "train")
        x_va, y_va, _ = load_features(opts["features_dir"], source, "val")
        pca = None
        if k:
            pca = PCA(n_components=k, random_state=config["seed"]).fit(x_tr)  # train split only
            x_tr, x_va = pca.transform(x_tr), pca.transform(x_va)
        if opts["standardise"]:
            mean, std = x_tr.mean(axis=0), x_tr.std(axis=0)
            std[std == 0] = 1.0  # a constant column would otherwise divide by zero
            x_tr, x_va = (x_tr - mean) / std, (x_va - mean) / std

        best_c, model, val_table = select_c(x_tr, y_tr, x_va, y_va, c_grid, opts["selection_metric"],
                                            opts["max_iter"], config["seed"])
        val_best = next(r for r in val_table if r["C"] == best_c)

        # ---- the ONE test evaluation, after C has been fixed on validation ----
        x_te, y_te, idx_te = load_features(opts["features_dir"], source, "test")
        if pca is not None:
            x_te = pca.transform(x_te)
        if opts["standardise"]:
            x_te = (x_te - mean) / std
        pred = model.predict(x_te)
        test = score(y_te, pred)

        wrong = np.flatnonzero(pred != y_te)
        errors = pd.DataFrame({"original_index": idx_te[wrong],
                               "text": raw["review"].iloc[idx_te[wrong]].to_numpy(),
                               "true_label": [LABEL_NAMES[int(v)] for v in y_te[wrong]],
                               "predicted_label": [LABEL_NAMES[int(v)] for v in pred[wrong]]})
        errors_path = f"{opts['errors_prefix']}{variant}.csv"
        errors.to_csv(errors_path, index=False)

        rows.append({"variant": variant, "feature_size": x_tr.shape[1], "best_C": best_c,
                     "val_accuracy": val_best["accuracy"], "val_macro_f1": val_best["macro_f1"],
                     "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"],
                     "n_test": len(y_te), "n_test_errors": len(wrong), "seed": config["seed"]})
        if k:
            rows[-1]["pca_components"] = k
        details[variant] = {"validation_by_C": val_table, "errors_file": errors_path,
                            "seconds": time.time() - t0}
        if k:
            details[variant]["pca_explained_variance_ratio_sum"] = float(pca.explained_variance_ratio_.sum())
        print(f"[{variant}] best C {best_c} (val {opts['selection_metric']} {val_best[opts['selection_metric']]:.4f}); "
              f"test accuracy {test['accuracy']:.4f}, macro-F1 {test['macro_f1']:.4f}; {len(wrong)} errors")

    with open(opts["summary_csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    path = save_run(args.results_dir, run, "src/classify.py", config,
                    metrics={"variants": rows}, details=details)
    print(f"\nSaved {opts['summary_csv']} and run record {path}")


if __name__ == "__main__":
    main()
