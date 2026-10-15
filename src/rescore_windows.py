"""Part III diagnostic: rescore test perplexity with NON-OVERLAPPING 64-token windows.

Why this exists
---------------
The Part I test perplexities (results/train_<model>.json) were scored by evaluate.py with each
test review cut to its FIRST 256 tokens and fed to the model in one piece. But the models were
trained on 64-token windows (and, for the Transformer, positions 64-255 were never updated in
training). This script asks: what is the test perplexity when the SAME first-256-token targets are
scored in 64-token windows, i.e. in the length the models were trained on?

How it works
------------
It reuses evaluate.evaluate() unchanged, so the reviews, the cut at 256 tokens and the list of
target tokens are exactly the same as in Part I. The only difference is the scorer: WindowedScorer
cuts the model's input (up to 255 tokens per review) into consecutive pieces of `window` tokens
(0-63, 64-127, ...) and runs the model on each piece SEPARATELY, so the recurrent state starts fresh
and the Transformer sees positions 0-63 only, in every window. Each target is still predicted only
from the tokens before it INSIDE its window (nothing from the future). Because every window predicts
the same targets that the one-piece run predicts, num_targets and target_checksum must match the
one-piece run; the script stops if they do not.

For each model it saves: the stored Part I value, the one-piece 256 value recomputed here (a check
that this script reproduces Part I; it can differ by a tiny amount because of GPU arithmetic), and the
64-token-window value.

Colab only (CLAUDE.md rule 9: neural models are scored on the Colab GPU). Command:
    python src/rescore_windows.py --config configs/p3_rescore_windows.json
`options.max_reviews` (null = the whole test split) exists only for CPU smoke tests of the code.
"""

import argparse
import csv
import json
import time

import torch

from data import load_processed_data
from evaluate import NeuralScorer, evaluate
from generate import build_neural
from runconfig import add_config_args, load_config, save_run


class WindowedScorer:
    """Scorer (see evaluate.py) that runs the model on consecutive `window`-token pieces separately."""

    def __init__(self, model, window):
        self.inner = NeuralScorer(model)
        self.window = window

    def target_log_probs(self, inputs, targets):
        pieces = [self.inner.target_log_probs(inputs[:, s:s + self.window], targets[:, s:s + self.window])
                  for s in range(0, inputs.size(1), self.window)]
        return torch.cat(pieces, dim=1)  # back to (batch, seq_len), same layout as one piece


def main():
    parser = argparse.ArgumentParser(description="Rescore test perplexity in non-overlapping windows.")
    add_config_args(parser)
    args = parser.parse_args()
    config, run = load_config(args)
    opts = config["options"]
    torch.manual_seed(config["seed"])
    print(f"Random seed: {config['seed']} (scoring is deterministic up to GPU arithmetic)")

    data = load_processed_data(opts["processed_dir"])
    reviews = data.test.reviews()
    if opts.get("max_reviews"):
        reviews = reviews[:opts["max_reviews"]]  # smoke tests only
    rows, details = [], {}
    for entry in opts["models"]:
        t0 = time.time()
        model, source = build_neural({"model": entry["model"], "train_result": entry["train_result"],
                                      "checkpoint": entry["checkpoint"]}, data)
        model.eval()  # dropout off (a custom scorer does not switch it off for us)
        with open(entry["train_result"]) as f:
            stored = json.load(f)["metrics"]["test_perplexity"]
        args_eval = (data.vocab.pad_id, config["batch_size"], opts["max_length"])
        full = evaluate(model, reviews, *args_eval)
        windowed = evaluate(WindowedScorer(model, opts["window"]), reviews, *args_eval)
        if (full["num_targets"], full["target_checksum"]) != (windowed["num_targets"], windowed["target_checksum"]):
            raise SystemExit(f"{entry['name']}: the windowed run scored different targets than the one-piece run.")
        rows.append({"model": entry["name"], "stored_test_ppl_256": stored,
                     "recomputed_test_ppl_256": full["perplexity"],
                     f"test_ppl_{opts['window']}_windows": windowed["perplexity"],
                     "num_targets": windowed["num_targets"], "target_checksum": windowed["target_checksum"]})
        details[entry["name"]] = {**source, "seconds": time.time() - t0, "window": opts["window"],
                                  "max_length": opts["max_length"], "reviews_scored": len(reviews),
                                  "avg_cross_entropy_256": full["avg_cross_entropy"],
                                  "avg_cross_entropy_windows": windowed["avg_cross_entropy"]}
        print(f"[{entry['name']}] stored {stored:.4f}  recomputed 256 {full['perplexity']:.4f}  "
              f"{opts['window']}-token windows {windowed['perplexity']:.4f}  ({windowed['num_targets']} targets)")

    with open(opts["output_csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path = save_run(args.results_dir, run, "src/rescore_windows.py", config,
                    metrics={"window": opts["window"], "max_length": opts["max_length"], "models": rows},
                    details=details)
    print(f"\nSaved {opts['output_csv']} and run record {path}")


if __name__ == "__main__":
    main()
