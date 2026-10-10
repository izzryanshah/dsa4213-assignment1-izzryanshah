"""Collect every results/*.json run record into one table, results/all_runs.csv.

Each run (see runconfig.save_run) leaves one JSON file. This script reads them
all and writes ONE ROW PER RUN, so runs can be compared side by side in a
spreadsheet or with pandas.

Columns, in order:
  run, script, date, git_commit, git_dirty, gpu, hardware, seed   who/when/what code
  <every shared config key>                             architecture, layers, ...
  opt_<name>                                            each script-specific option
  epochs_logged, final_train_loss, final_val_loss       from the per-epoch losses
  metric_<name>                                         each final metric; nested
                                                        metrics get dotted names,
                                                        e.g. metric_test.perplexity

Values that are lists (e.g. the n-gram discount grid) are stored as JSON text
inside the cell. The bulky "details" part of each record (tuning grids,
timings, ...) is not included; it stays in the run's own JSON file.

Files in results/ that are not run records (no "config" field) are skipped
with a message.

Usage:
    python src/summarise_runs.py
    python src/summarise_runs.py --results-dir results --output results/all_runs.csv
"""

import argparse
import csv
import glob
import json
import os

from runconfig import SHARED_KEYS

IDENTITY_COLUMNS = ["run", "script", "date", "git_commit", "git_dirty", "gpu", "hardware", "seed"]


def flatten(prefix, value, out):
    """Put value into `out`; nested dicts become dotted column names."""
    if isinstance(value, dict):
        for key, sub in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, sub, out)
    elif isinstance(value, list):
        out[prefix] = json.dumps(value)
    else:
        out[prefix] = value


def record_to_row(record):
    row = {col: record.get(col) for col in IDENTITY_COLUMNS}
    for key in SHARED_KEYS:
        row[key] = record["config"].get(key)
    for key, value in record["config"].get("options", {}).items():
        flatten(f"opt_{key}", value, row)

    losses = record.get("epoch_losses") or []
    row["epochs_logged"] = len(losses)
    row["final_train_loss"] = losses[-1].get("train_loss") if losses else None
    row["final_val_loss"] = losses[-1].get("val_loss") if losses else None

    for key, value in record.get("metrics", {}).items():
        flatten(f"metric_{key}", value, row)
    return row


def main():
    parser = argparse.ArgumentParser(description="Collect results/*.json into one CSV.")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/all_runs.csv")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        with open(path) as f:
            record = json.load(f)
        if not (isinstance(record, dict) and "config" in record and "run" in record):
            print(f"Skipping {path}: not a run record")
            continue
        rows.append(record_to_row(record))

    # Different runs have different options/metrics, so the table's columns
    # are the union of everything seen: identity, shared config, options,
    # epoch summary, then metrics (each group in first-seen order).
    fixed = IDENTITY_COLUMNS + [k for k in SHARED_KEYS if k not in IDENTITY_COLUMNS]  # seed is in both lists
    tail = ["epochs_logged", "final_train_loss", "final_val_loss"]
    seen = {}
    for row in rows:
        for col in row:
            seen.setdefault(col, None)
    options = [c for c in seen if c.startswith("opt_")]
    metrics = [c for c in seen if c.startswith("metric_")]
    columns = fixed + options + tail + metrics

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} runs x {len(columns)} columns to {args.output}")


if __name__ == "__main__":
    main()
