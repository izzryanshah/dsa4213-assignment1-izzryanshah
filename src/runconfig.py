"""Shared record-keeping helpers: read a run's config file, and save its results.

Why this file exists
--------------------
The project rules say two things about every run:

  1. Its settings (architecture, learning rate, seed, ...) live in ONE config
     file under configs/, never hard-coded inside a script.
  2. Everything needed to reproduce and compare the run is saved with the
     results: the config that was actually used, the seed, which version of
     the code ran (git commit), what hardware, when, the loss after every
     epoch and the final metrics.

Putting that logic here, once, means every script does it the same way and
`summarise_runs.py` can later read all the result files with one loop.

How a script uses it (the pattern is identical in every script):

    parser = argparse.ArgumentParser(...)
    add_config_args(parser)            # adds --config, --set, --seed, ...
    args = parser.parse_args()
    config, run = load_config(args)    # config = file contents + overrides
    ...do the work, reading ONLY values from `config`...
    save_run(args.results_dir, run, "src/myscript.py", config,
             metrics={...}, epoch_losses=[...], details={...})

The config file
---------------
A config is a JSON file with every key in SHARED_KEYS (use null when a key
does not apply to that run, e.g. `layers` for an n-gram model) plus an
"options" object holding any extra settings that only one script needs (file
paths, the n-gram discount grid, ...). The run's name is the config file's
name without ".json" (configs/ngram.json -> run "ngram" -> results/ngram.json).

Overriding from the command line (the config file itself is never modified):

    python src/ngram.py --config configs/ngram.json --seed 1
    python src/ngram.py --config configs/ngram.json --batch-size 64 --run-name ngram_bs64
    python src/ngram.py --config configs/ngram.json --set options.max_length=256

`--set key=value` works for any key (use `options.name` for the options
block); the value is read as JSON when possible (so 256, 0.5, null and
[0.1, 0.2] work) and otherwise as plain text.
"""

import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime

# The settings every config must define (null where not applicable).
SHARED_KEYS = [
    "architecture",
    "layers",
    "hidden_size",
    "embedding_dim",
    "embedding_init",
    "context_length",
    "learning_rate",
    "batch_size",
    "epochs",
    "dropout",
    "vocab_size",
    "subsample_size",
    "seed",
]
CONFIG_KEYS = SHARED_KEYS + ["options"]

_NOT_GIVEN = object()  # marks "this flag was not typed on the command line"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _parse_value(text):
    """'256' -> 256, 'null' -> None, '[0.1, 0.2]' -> list, 'abc' -> 'abc'."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def add_config_args(parser):
    """Add --config plus the override flags to a script's argparse parser."""
    parser.add_argument("--config", type=str, required=True, help="Path to this run's config file (configs/<run>.json).")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run's results file. Default: the config file's name.")
    parser.add_argument("--results-dir", type=str, default="results", help="Where results/<run>.json is saved.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override any config key, e.g. --set options.max_length=256. Can be repeated.")
    # One convenience flag per shared key, e.g. --batch-size, --learning-rate.
    for key in SHARED_KEYS:
        parser.add_argument("--" + key.replace("_", "-"), dest="override_" + key,
                            type=_parse_value, default=_NOT_GIVEN, help=f"Override config key '{key}'.")


def load_config(args):
    """Read the config file named by --config, apply command-line overrides.

    Returns (config, run_name). `config` is the full set of settings that will
    actually be used, which is also what gets saved to the results file.
    """
    with open(args.config) as f:
        config = json.load(f)

    missing = [k for k in CONFIG_KEYS if k not in config]
    unknown = [k for k in config if k not in CONFIG_KEYS]
    if missing or unknown:
        raise SystemExit(f"{args.config}: missing keys {missing}, unknown keys {unknown}. "
                         f"Every config needs exactly: {CONFIG_KEYS}")

    for key in SHARED_KEYS:
        value = getattr(args, "override_" + key)
        if value is not _NOT_GIVEN:
            config[key] = value

    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got '{item}'")
        key, text = item.split("=", 1)
        target = config
        *parents, last = key.split(".")
        for part in parents:
            if not isinstance(target.get(part), dict):
                raise SystemExit(f"--set {key}: '{part}' is not a settings block in the config")
            target = target[part]
        if last not in target:  # catches typos instead of silently adding a new key
            raise SystemExit(f"--set {key}: no such key in the config")
        target[last] = _parse_value(text)

    run = args.run_name or os.path.splitext(os.path.basename(args.config))[0]
    return config, run


# ------------------------------------------------------------------ provenance
def _git(*git_args):
    try:
        out = subprocess.run(["git", *git_args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_commit():
    """(commit hash, whether code has uncommitted changes).

    The "uncommitted changes" check ignores results/ and data/, because
    running a script itself writes into them. If it is True, the hash alone
    does NOT fully identify the code that ran.
    """
    commit = _git("rev-parse", "HEAD")
    changes = _git("status", "--porcelain", "--", ".", ":!results", ":!data")
    return commit, (bool(changes) if changes is not None else None)


def gpu_name():
    """Name of the CUDA GPU in use, or None if the run is on CPU."""
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        return None


def hardware_name():
    """What the run executed on: the GPU's name if CUDA is used, else the CPU's name."""
    gpu = gpu_name()
    if gpu:
        return f"GPU: {gpu}"
    cpu = None
    if platform.system() == "Darwin":  # macOS: platform.processor() is just 'i386'/'arm'
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return f"CPU: {cpu or platform.processor() or platform.machine()} ({platform.system()} {platform.machine()})"


def _json_default(obj):
    """Let json.dump handle numpy numbers/arrays that sneak into results."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def save_run(results_dir, run, script, config, metrics, epoch_losses=None, details=None, extra=None):
    """Write results/<run>.json with the full record of one run.

    Fields:
      run, script         which run / which file produced it
      date                when it finished (ISO 8601, with timezone)
      git_commit          code version (+ git_dirty: uncommitted code changes?)
      gpu                 GPU name, or null on CPU
      hardware            "GPU: <name>" or "CPU: <name> (<OS> <arch>)"
      command             the exact command line that was run (interpreter + arguments)
      seed                copied out of the config for easy filtering
      config              the full config actually used (after overrides)
      epoch_losses        list of {"epoch", "train_loss", "val_loss"}; empty
                          for runs that do not train over epochs
      metrics             the final headline numbers
      details             everything else worth keeping (timings, tables, ...)
      extra               optional dict of additional top-level fields. Generation
                          runs use it for "decoding" (settings) and "generations"
                          (the text); "decoding" must contain temperature, top_k,
                          max_tokens and seed, or the save is refused.
    """
    if extra and "decoding" in extra:
        missing = [k for k in ("temperature", "top_k", "max_tokens", "seed") if k not in extra["decoding"]]
        if missing:
            raise ValueError(f"decoding settings missing {missing}")
    commit, dirty = git_commit()
    record = {
        "run": run,
        "script": script,
        "command": shlex.join([sys.executable] + sys.argv),
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "gpu": gpu_name(),
        "hardware": hardware_name(),
        "seed": config["seed"],
        "config": config,
        "epoch_losses": epoch_losses or [],
        "metrics": metrics,
        "details": details or {},
    }
    record.update(extra or {})
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{run}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=_json_default)
    return path
