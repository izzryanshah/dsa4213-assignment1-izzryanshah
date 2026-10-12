"""Part I cost comparison: inference throughput, parameter counts and training time.

What this script does
---------------------
1. Loads the BEST checkpoint of the RNN, LSTM and Transformer, and rebuilds the
   n-gram model (it has no checkpoint: counting takes about a second, so we
   refit it from the training split using the discounts it saved).
2. Scores the WHOLE test set with each model, using the same function
   (`evaluate()` from evaluate.py), the same batch size and the same context
   length (reviews cut to their first `context_length` tokens). Because it is
   the very function that produced the reported perplexities, the workload is
   identical for every model. Each model's `target_checksum` is compared to
   prove it.
3. Throughput = number of target tokens scored / seconds taken (tokens/second).
   Every model is timed on each device in `options.devices` (GPU and CPU). The
   n-gram is NumPy code, so it only runs on the CPU.
4. Merges the throughput with the parameter counts and training times that are
   already saved in results/ into results/part1_costs.csv, and saves the full
   record to results/benchmark.json.

Run (in Colab, see notebooks/run_colab.ipynb; every model must be timed on the
same machine):

    python3 src/benchmark.py --config configs/benchmark.json

Why warm-up and GPU synchronisation matter for FAIR timing
----------------------------------------------------------
Warm-up. The first time a model runs on a GPU, a lot of one-off work happens
that has nothing to do with how fast the model is: CUDA starts up, GPU code
("kernels") is loaded, cuDNN tries several algorithms and picks one, PyTorch's
memory allocator grows to the size it needs, and the CPU caches are cold. If we
timed that first pass, whichever model happened to run first (or each model's
first run) would be charged for all of it, and the numbers would reflect the
ORDER we ran things in rather than the models. So each model first does
`warmup_passes` full untimed passes; only afterwards do we start the clock.

Synchronisation. GPU work is ASYNCHRONOUS: when Python calls the model, PyTorch
only puts the work in a queue for the GPU and immediately carries on. A clock
read right after the call would measure how fast the work was QUEUED, not how
fast it was DONE, so the model would look much faster than it is, and the
unfinished work would spill into the next measurement. `torch.cuda.synchronize()`
makes Python wait until the GPU has finished everything queued. We call it
before starting the clock (so earlier leftover work is not counted) and again
before stopping it (so all of this run's work is counted). On the CPU there is
no queue, so no synchronisation is needed.

Averaging over several repeats (`repeats`, 3 here) smooths out run-to-run noise;
we report the mean and the standard deviation.
"""

import argparse
import csv
import json
import os
import statistics
import time

import torch

from data import check_config_matches_data, load_processed_data
from evaluate import evaluate
from ngram import KneserNeyTrigram, build_events
from runconfig import add_config_args, load_config, save_run
from train import MODELS, count_parameters, set_seed

PARAMETER_MEANING_NEURAL = "trainable weights"
PARAMETER_MEANING_NGRAM = "stored n-gram counts (not comparable with weights)"


def sync(device):
    """Wait for all queued GPU work to finish (does nothing on the CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def time_scoring(model, device, test, pad_id, batch_size, max_length, warmup_passes, repeats):
    """Score the test set `warmup_passes` times untimed, then `repeats` times timed.

    Returns (list of seconds per timed repeat, the evaluate() result of the last pass).
    """
    for _ in range(warmup_passes):
        evaluate(model, test, pad_id, batch_size, max_length)  # result thrown away
    seconds, result = [], None
    for _ in range(repeats):
        sync(device)                      # nothing from before may still be running
        start = time.perf_counter()
        result = evaluate(model, test, pad_id, batch_size, max_length)
        sync(device)                      # wait until this pass has really finished
        seconds.append(time.perf_counter() - start)
    return seconds, result


def summarise(seconds, num_targets):
    """Mean and standard deviation of tokens/second over the repeats."""
    rates = [num_targets / s for s in seconds]
    return {"tokens_per_second_mean": statistics.mean(rates),
            "tokens_per_second_std": statistics.stdev(rates) if len(rates) > 1 else 0.0,
            "seconds_per_repeat": seconds}


def load_neural(entry, data):
    """Rebuild a neural model from the config saved in its training record, then load its best weights."""
    record = read_json(entry["train_result"])
    train_config = record["config"]  # the config that was actually used for training
    model = MODELS[train_config["architecture"]](train_config, data.vocab)
    state = torch.load(entry["checkpoint"], map_location="cpu")
    model.load_state_dict(state)
    n = count_parameters(model)
    if n != record["metrics"]["num_parameters"]:
        raise SystemExit(f"{entry['name']}: rebuilt model has {n:,} parameters but the training record says "
                         f"{record['metrics']['num_parameters']:,}; wrong checkpoint or config?")
    costs = {"parameters": n, "parameter_meaning": PARAMETER_MEANING_NEURAL,
             "train_seconds": record["metrics"]["train_seconds"], "tune_seconds": "",
             "train_hardware": record["hardware"], "saved_test_perplexity": record["metrics"]["test_perplexity"]}
    return model, costs


def load_ngram(entry, data):
    """Refit the trigram on the training split with its saved (validation-tuned) discounts."""
    record = read_json(entry["train_result"])
    d = record["details"]["best_discounts"]
    model = KneserNeyTrigram(len(data.vocab), discounts=(d["d3"], d["d2"], d["d1"]))
    model.fit(*build_events(data.train, model.bos_id))
    costs = {"parameters": record["details"]["model_size"]["total_stored_counts"],
             "parameter_meaning": PARAMETER_MEANING_NGRAM,
             "train_seconds": record["details"]["train_seconds"],
             "tune_seconds": record["details"]["tune_seconds"],
             "train_hardware": record["hardware"], "saved_test_perplexity": ""}
    return model, costs


def main():
    parser = argparse.ArgumentParser(description="Time inference for every Part I model on one workload.")
    add_config_args(parser)
    args = parser.parse_args()
    config, run = load_config(args)
    opts = config["options"]
    batch_size, max_length = config["batch_size"], config["context_length"]
    if opts["repeats"] < 2:
        raise SystemExit("options.repeats must be at least 2 to report a standard deviation")

    set_seed(config["seed"])
    print(f"Random seed: {config['seed']}")
    data = load_processed_data(opts["processed_dir"])
    check_config_matches_data(config, data)
    pad_id = data.vocab.pad_id
    test = data.test
    if opts["max_reviews"] is not None:  # smoke tests only: a handful of reviews, never a real result
        test = test.reviews()[: opts["max_reviews"]]
        print(f"SMOKE TEST: only the first {opts['max_reviews']} test reviews. Do not use these numbers.")

    devices = [torch.device(d) for d in opts["devices"] if d != "cuda" or torch.cuda.is_available()]
    skipped = [d for d in opts["devices"] if d == "cuda" and not torch.cuda.is_available()]
    if skipped:
        print("No GPU available: skipping the cuda timings.")
    print(f"Workload: test set, batch size {batch_size}, context length {max_length}; "
          f"{opts['warmup_passes']} warm-up pass(es), {opts['repeats']} timed repeats; devices {[str(d) for d in devices]}")

    rows, details = [], {}
    reference = None  # (num_targets, checksum) of the first model; all others must match it
    for entry in opts["models"]:
        name = entry["name"]
        if name == "ngram":
            model, costs = load_ngram(entry, data)
            model_devices = [torch.device("cpu")]  # NumPy only
        else:
            model, costs = load_neural(entry, data)
            model_devices = devices
        row = {"model": name, **costs}
        details[name] = {}
        for device in model_devices:
            if isinstance(model, torch.nn.Module):
                model.to(device)
            seconds, result = time_scoring(model, device, test, pad_id, batch_size, max_length,
                                           opts["warmup_passes"], opts["repeats"])
            stats = summarise(seconds, result["num_targets"])
            print(f"{name:12s} {str(device):5s} {stats['tokens_per_second_mean']:>12,.0f} tokens/s "
                  f"(std {stats['tokens_per_second_std']:,.0f}); perplexity {result['perplexity']:.2f}")
            row[f"tokens_per_second_{device.type}_mean"] = stats["tokens_per_second_mean"]
            row[f"tokens_per_second_{device.type}_std"] = stats["tokens_per_second_std"]
            row["benchmark_perplexity"] = result["perplexity"]
            row["num_targets"], row["target_checksum"] = result["num_targets"], result["target_checksum"]
            details[name][str(device)] = {**stats, "perplexity": result["perplexity"]}
            this = (result["num_targets"], result["target_checksum"])
            if reference is None:
                reference = this
            elif this != reference:
                raise SystemExit(f"{name} on {device} scored different targets than the first model: "
                                 f"{this} vs {reference}. The workloads are not identical.")
        if isinstance(model, torch.nn.Module):
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # free the GPU memory before the next model
        saved = costs["saved_test_perplexity"]
        if saved != "" and opts["max_reviews"] is None and abs(row["benchmark_perplexity"] / saved - 1) > opts["perplexity_rel_tolerance"]:
            raise SystemExit(f"{name}: perplexity {row['benchmark_perplexity']:.3f} differs from the saved "
                             f"{saved:.3f}; the checkpoint is probably not the one from that run.")
        rows.append(row)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    for row in rows:
        row["benchmark_gpu"] = gpu
    csv_columns = ["model", "parameters", "parameter_meaning", "train_seconds", "tune_seconds", "train_hardware",
                   "tokens_per_second_cuda_mean", "tokens_per_second_cuda_std",
                   "tokens_per_second_cpu_mean", "tokens_per_second_cpu_std", "benchmark_gpu",
                   "benchmark_perplexity", "num_targets", "target_checksum"]
    os.makedirs(os.path.dirname(opts["costs_csv"]) or ".", exist_ok=True)
    with open(opts["costs_csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {opts['costs_csv']}")

    metrics = {name: {dev: {"tokens_per_second_mean": v["tokens_per_second_mean"],
                            "tokens_per_second_std": v["tokens_per_second_std"]}
                      for dev, v in per_dev.items()} for name, per_dev in details.items()}
    path = save_run(args.results_dir, run, "src/benchmark.py", config, metrics=metrics,
                    details={"num_targets": reference[0], "target_checksum": reference[1],
                             "costs_csv": opts["costs_csv"], "per_model": details,
                             "torch_cpu_threads": torch.get_num_threads(), "smoke_test": opts["max_reviews"] is not None})
    print(f"Saved run record to {path}")


if __name__ == "__main__":
    main()
