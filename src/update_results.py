"""Rebuild RESULTS.md, MY_TASKS.md and the figures in results/figures/ from the saved run records.

What this script is for
-----------------------
Every run leaves a file results/<run>.json (see runconfig.save_run). This script
reads all of them, plus DECISIONS.md, the config files and results/RESULTS_LOG.md,
and writes ONE readable document, RESULTS.md, at the repo root:

    Data | Part I | Part II | Part III | Settings | Checks

It states facts only: numbers, tables, figures, generated text, pass/fail with
evidence. It never interprets anything.

Your notes are safe
-------------------
Each section ends with a "My notes" heading. Whatever you write between the two
marker lines under it

    <!-- my-notes:part1:start -->
    ...your text...
    <!-- my-notes:part1:end -->

is copied back, character for character, every time this script rebuilds
RESULTS.md. Do not delete the marker lines. If the old file has lost its
markers, the script stops instead of overwriting (use --force to override).

Usage (from the repo root):
    python3 src/update_results.py --config configs/update_results.json

Everything that is a design value (thresholds, which run belongs to which part,
colours, paths) is in configs/update_results.json.
"""

import argparse
import ast
import glob
import html
import json
import math
import os
import re
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import matplotlib.ticker

from my_tasks import write_my_tasks
from runconfig import REPO_ROOT, add_config_args, load_config

SECTIONS = [("data", "Data"), ("part1", "Part I"), ("part2", "Part II"), ("part3", "Part III"),
            ("settings", "Settings"), ("checks", "Checks")]
INK, GRID = "#0b0b0b", "#e5e4e0"


# ----------------------------------------------------------------- small helpers
def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def fmt(value):
    """One value -> text for a table cell."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.4f}" if abs(value) < 1000 else f"{value:,.2f}"
    if isinstance(value, (list, dict)):
        return "`" + json.dumps(value) + "`"
    return str(value)


def cell(text):
    """Make text safe inside a markdown table cell."""
    text = html.escape(str(text), quote=False).replace("|", "\\|")
    return text.replace("\r", " ").replace("\n", " ")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def flatten(prefix, value, out, max_list=10):
    """Nested dict -> {'a.b': value}. Long lists (e.g. a 1,000-row grid) are skipped."""
    if isinstance(value, dict):
        for key, sub in value.items():
            flatten(f"{prefix}.{key}" if prefix else key, sub, out, max_list)
    elif isinstance(value, list) and len(value) > max_list:
        out[prefix] = f"({len(value)} entries; see the run's JSON file)"
    else:
        out[prefix] = value


def kv_table(mapping, header=("Name", "Value")):
    return table(list(header), [[f"`{cell(k)}`", cell(fmt(v)) if not isinstance(v, str) or v.startswith("(")
                                 else cell(v)] for k, v in mapping.items()])


def short_commit(rec):
    commit = (rec.get("git_commit") or "")[:7] or "—"
    return commit + (" (uncommitted changes at run time)" if rec.get("git_dirty") else "")


# ------------------------------------------------------------------ loading runs
def load_runs(results_dir):
    """All run records: JSON files directly in results/ that have a 'config' field."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            rec = read_json(path)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and "config" in rec and "run" in rec:
            runs[rec["run"]] = rec
    return runs


def commands_from_log(log_path):
    """Fallback for runs saved before records had a 'command' field: read RESULTS_LOG.md."""
    if not os.path.exists(log_path):
        return {}
    blocks = re.split(r"\n(?=### )", read_text(log_path))
    return {"blocks": blocks}


def find_command(run, rec, log):
    if rec.get("command"):
        return rec["command"], "stored in the run record"
    for block in log.get("blocks", []):
        if f"results/{run}.json" not in block:
            continue
        for span in re.findall(r"`([^`]+)`", block):
            if "python" in span and (f"configs/{run}.json" in span or "--config" not in span):
                return span, "copied from results/RESULTS_LOG.md"
    return None, None


def part_of(run, rules):
    for pattern, part in rules:
        if re.search(pattern, run):
            return part
    return None


# --------------------------------------------------------------------- figures
def style_axes(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))  # whole epochs only


def new_figure():
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=130)
    fig.patch.set_facecolor("white")
    style_axes(ax)
    return fig, ax


def save_figure(fig, figures_dir, name):
    fig.tight_layout()
    path = os.path.join(figures_dir, name)
    fig.savefig(path)
    plt.close(fig)
    return os.path.relpath(path, REPO_ROOT)


def fig_split_tokens(data_rec, figures_dir, colours):
    splits = data_rec["metrics"]["splits"]
    fig, ax = new_figure()
    names = list(splits)
    values = [splits[n]["num_tokens"] for n in names]
    ax.bar(names, values, color=colours[0], width=0.55)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_ylabel("tokens", color=INK)
    ax.set_title("Tokens per split (run: data)", color=INK, fontsize=10, loc="left")
    return save_figure(fig, figures_dir, "data_split_tokens.png")


def fig_ngram_tuning(rec, figures_dir, colours):
    grid = rec["details"].get("tuning_grid")
    best = rec["details"].get("best_discounts")
    if not grid or not best:
        return None
    # D3 on the x axis, with D2 and D1 held at their chosen values.
    points = sorted((g["d3"], g["val_cross_entropy"]) for g in grid
                    if g["d2"] == best["d2"] and g["d1"] == best["d1"])
    fig, ax = new_figure()
    ax.plot([p[0] for p in points], [p[1] for p in points], color=colours[0], linewidth=2, marker="o", markersize=5)
    ax.set_xlabel(f"D3 (D2 = {best['d2']}, D1 = {best['d1']})", color=INK)
    ax.set_ylabel("validation cross-entropy (nats)", color=INK)
    ax.set_title(f"Discount tuning, run: {rec['run']}", color=INK, fontsize=10, loc="left")
    return save_figure(fig, figures_dir, f"{rec['run']}_tuning.png")


def fig_loss_curves(rec, figures_dir, colours):
    losses = rec["epoch_losses"]
    epochs = [e["epoch"] for e in losses]
    fig, ax = new_figure()
    ax.plot(epochs, [e["train_loss"] for e in losses], color=colours[0], linewidth=2, marker="o", markersize=4, label="train")
    ax.plot(epochs, [e["val_loss"] for e in losses], color=colours[1], linewidth=2, marker="o", markersize=4, label="validation")
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("cross-entropy (nats)", color=INK)
    ax.set_title(f"Training and validation loss, run: {rec['run']}", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    return save_figure(fig, figures_dir, f"{rec['run']}_loss.png")


def fig_val_curves_all(recs, figures_dir, colours):
    fig, ax = new_figure()
    for i, rec in enumerate(recs):
        losses = rec["epoch_losses"]
        ax.plot([e["epoch"] for e in losses], [e["val_loss"] for e in losses],
                color=colours[i % len(colours)], linewidth=2, marker="o", markersize=4, label=rec["run"])
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("validation cross-entropy (nats)", color=INK)
    ax.set_title("Validation loss, all trained runs", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    return save_figure(fig, figures_dir, "val_loss_all_runs.png")


# ----------------------------------------------------------- one run's markdown
def render_run(run, rec, command, command_source, figure_paths):
    out = [f"#### `{run}`\n"]
    out.append(table(["Field", "Value"], [
        ["Command", f"`{cell(command)}`" + (f" ({command_source})" if command_source != "stored in the run record" else "")
         if command else "not recorded (run predates command logging)"],
        ["Script", f"`{rec.get('script')}`"],
        ["Date", cell(rec.get("date"))],
        ["Git commit", f"`{short_commit(rec)}`"],
        ["Hardware", cell(rec.get("hardware"))],
        ["Seed", cell(rec.get("seed"))],
        ["Result file", f"`results/{run}.json`"],
    ]))

    settings = {}
    flatten("", rec["config"], settings, max_list=50)
    out.append("**Settings used**\n")
    out.append(kv_table(settings, ("Setting", "Value")))

    if rec.get("metrics"):
        metrics = {}
        flatten("", rec["metrics"], metrics, max_list=50)
        out.append("**Key numbers**\n")
        out.append(kv_table(metrics, ("Metric", "Value")))

    tests = (rec.get("details") or {}).get("tests")
    if tests:
        out.append("**Individual tests**\n")
        out.append(table(["Test", "Result"], [[f"`{k}`", cell(v)] for k, v in tests.items()]))
    elif rec.get("details"):
        details = {}
        flatten("", {k: v for k, v in rec["details"].items() if k != "tests"}, details)
        if details:
            out.append("**Details**\n")
            out.append(kv_table(details))

    if rec.get("decoding"):
        out.append("**Decoding settings**\n")
        out.append(kv_table(rec["decoding"], ("Setting", "Value")))

    if rec.get("epoch_losses"):
        out.append("**Loss per epoch**\n")
        out.append(table(["Epoch", "Train loss", "Validation loss"],
                         [[str(e["epoch"]), fmt(e["train_loss"]), fmt(e["val_loss"])] for e in rec["epoch_losses"]]))

    for path in figure_paths:
        out.append(f"![{os.path.basename(path)}]({path})\n")
    return "\n".join(out)


# ------------------------------------------------------------- generated text
def render_generation_comparison(gen_runs):
    """One table per prompt; one column per model (run, sample)."""
    if not gen_runs:
        return "No generation runs yet.\n"
    columns = []  # (label, {prompt: generation})
    prompts = []
    for run, rec in gen_runs.items():
        samples = {}
        for g in rec["generations"]:
            samples.setdefault(g["sample"], {})[g["prompt"]] = g
            if g["prompt"] not in prompts:
                prompts.append(g["prompt"])
        for sample, by_prompt in sorted(samples.items()):
            label = run if len(samples) == 1 else f"{run} (sample {sample})"
            columns.append((label, by_prompt))
    out = ["Decoding settings for each column are in the run records above (`results/<run>.json`, field `decoding`).\n"]
    for prompt in prompts:
        out.append(f"**Prompt:** `{cell(prompt)}`\n")
        header = [""] + [f"`{label}`" for label, _ in columns]
        def pick(by_prompt, key):
            g = by_prompt.get(prompt)
            return "—" if g is None else cell(fmt(g[key]) if key != "continuation" else g[key])
        rows = [["Continuation"] + [pick(b, "continuation") for _, b in columns],
                ["New tokens"] + [pick(b, "num_new_tokens") for _, b in columns],
                ["Ended with `<eos>`"] + [pick(b, "ended_with_eos") for _, b in columns]]
        out.append(table(header, rows))
    return "\n".join(out)


def render_part2_facts(runs, opts):
    """Part II set-up facts, read from the run records (no interpretation).

    1. Config differences between the reused Part I run and the Part II runs.
    2. The frozen-embedding check values.
    3. The embedding tables: how they were built (own word2vec, public GloVe) and their coverage.
    4. Per-epoch loss facts for the three runs.
    """
    base_name = opts["part2_baseline_run"]
    p2 = {r: rec for r, rec in runs.items()
          if part_of(r, opts["part_rules"]) == "Part II" and rec.get("script") in opts["frozen_check_scripts"]}
    if base_name not in runs or not p2:
        return ""
    base = runs[base_name]
    compared = {base_name: base, **p2}
    out = ["\n### Part II set-up facts\n",
           "Read from the run records; nothing here is interpreted.\n"]

    # 1. config differences ---------------------------------------------------------
    flat = {}
    for run, rec in compared.items():
        f = {}
        flatten("", rec["config"], f)
        flat[run] = f
    keys = sorted(set().union(*[f.keys() for f in flat.values()]))
    differing = [k for k in keys if len({json.dumps(flat[r].get(k, "<absent>"), sort_keys=True) for r in compared}) > 1]
    out.append(f"**Settings compared** (`config` saved in each record; {len(keys)} keys, {len(keys) - len(differing)} identical in all "
               f"{len(compared)} runs). Keys that differ:\n")
    rows = [[f"`{k}`"] + [cell(fmt(flat[r].get(k, "<absent>"))) for r in compared] for k in differing]
    out.append(table(["Setting"] + [f"`{r}`" for r in compared], rows))
    prov = [[f"`{r}`", f"`{rec['git_commit'][:7]}`", fmt(rec.get("git_dirty")), cell(rec.get("hardware")), cell(rec["date"]),
             fmt(rec["seed"])] for r, rec in compared.items()]
    out.append("\n**Provenance**\n")
    out.append(table(["Run", "Git commit", "Uncommitted changes at run time", "Hardware", "Date", "Seed"], prov))

    # 2. frozen check -----------------------------------------------------------------
    key = opts["frozen_embedding_detail_key"]
    rows = []
    for r, rec in p2.items():
        d, m = rec.get("details") or {}, rec.get("metrics") or {}
        rows.append([f"`{r}`", fmt(d.get(key)), cell(d.get("embedding_source")), f"`{str(d.get('embedding_source_sha256'))[:16]}...`",
                     fmt(m.get("num_parameters")), fmt(d.get("total_parameters")), fmt(d.get("embedding_table_trainable"))])
    out.append("\n**Frozen-embedding check** (the script stops before saving a record if the table changed)\n")
    out.append(table(["Run", "Max abs change of the table", "Table file", "File SHA-256 (first 16)", "Trainable parameters",
                      "Total parameters", "Table trainable"], rows))

    # 3. embedding tables -------------------------------------------------------------
    own = runs.get(opts["part2_own_embedding_run"])
    pub = runs.get(opts["part2_public_embedding_run"])
    data_rec = runs.get("data")
    rows = []
    if own:
        d, m, c = own.get("details") or {}, own.get("metrics") or {}, own["config"]
        train_info = (data_rec["metrics"]["splits"]["train"] if data_rec else {})
        match = ("yes" if train_info and d.get("training_tokens") == train_info.get("num_tokens")
                 and d.get("training_reviews") == train_info.get("num_reviews") else "not checked")
        rows.append([f"`{opts['part2_own_embedding_run']}`", f"{d.get('algorithm')} (gensim {d.get('gensim_version')})",
                     f"window {c['options'].get('window')}, min_count {c['options'].get('min_count')}, epochs {c.get('epochs')}",
                     f"{fmt(d.get('training_reviews'))} reviews, {fmt(d.get('training_tokens'))} tokens (subsample_size {fmt(c.get('subsample_size'))}); "
                     f"same count as the language models' training split in `results/data.json`: {match}",
                     f"{fmt(d.get('coverage_types_percent'))}% / {fmt(d.get('coverage_tokens_percent'))}%",
                     fmt(d.get("rows_filled_randomly")), f"{fmt(m.get('total_seconds'))} s on {cell(own.get('hardware'))}"])
    if pub:
        d, m, c = pub.get("details") or {}, pub.get("metrics") or {}, pub["config"]
        rows.append([f"`{opts['part2_public_embedding_run']}`", cell(d.get("glove_file")), "not trained here (public file)",
                     f"{fmt(d.get('glove_lines_read'))} GloVe lines read; vocabulary matched on the training split's "
                     f"{fmt(d.get('training_tokens'))} tokens",
                     f"{fmt(m.get('coverage_types_percent'))}% / {fmt(m.get('coverage_tokens_percent'))}% "
                     f"(real words only: {fmt(m.get('coverage_real_words_types_percent'))}% / {fmt(m.get('coverage_real_words_tokens_percent'))}%)",
                     fmt(m.get("rows_filled_randomly")), "not timed (download only)"])
    if rows:
        out.append("\n**Embedding tables** (coverage = vocabulary types / training tokens with a real vector, `<pad>` excluded; "
                   "rows without a vector are filled by the rule in decision R2)\n")
        out.append(table(["Record", "Source", "Settings", "Text it was built from", "Coverage (types / tokens)",
                          "Rows filled randomly", "Cost"], rows))

    # 4. per-epoch loss facts ---------------------------------------------------------
    rows = []
    for r, rec in compared.items():
        e = rec.get("epoch_losses") or []
        if not e:
            continue
        tr, va = [x["train_loss"] for x in e], [x["val_loss"] for x in e]
        rose_tr = [i + 2 for i in range(len(e) - 1) if tr[i + 1] > tr[i]]
        rose_va = [i + 2 for i in range(len(e) - 1) if va[i + 1] > va[i]]
        nan = any(x != x for x in tr + va)
        steps = [va[i + 1] - va[i] for i in range(len(va) - 1)]
        m = rec.get("metrics") or {}
        rows.append([f"`{r}`", fmt(len(e)), fmt(m.get("best_epoch")), fmt(m.get("stopped_early")),
                     ", ".join(map(str, rose_tr)) or "none", ", ".join(map(str, rose_va)) or "none", fmt(nan),
                     fmt(va[0]), fmt(va[-1]), fmt(tr[-1]), fmt(min(steps)), fmt(steps[-1])])
    if rows:
        out.append("\n**Per-epoch losses** (the only training-stability record kept: one train and one validation loss per epoch; "
                   "per-batch losses and gradient norms are not recorded)\n")
        out.append(table(["Run", "Epochs run", "Best epoch", "Stopped early", "Epochs where train loss rose",
                          "Epochs where validation loss rose", "NaN", "Val loss epoch 1", "Val loss last epoch",
                          "Train loss last epoch", "Largest one-epoch val change", "Last-epoch val change"], rows))
    return "\n".join(out)


def render_model_summary(runs, test_prefix, keep=lambda run: True):
    """One row per language model: perplexity and cost numbers, as recorded in the run records."""
    rows = []
    for run, rec in runs.items():
        if run.startswith(test_prefix) or rec.get("generations") or not keep(run):
            continue
        m, d = rec.get("metrics") or {}, rec.get("details") or {}
        arch = cell(rec["config"].get("architecture"))
        throughput = fmt(m.get("inference_tokens_per_second") or d.get("inference_tokens_per_second"))
        if "test" in m and isinstance(m["test"], dict) and "perplexity" in m["test"]:
            size = f"{d['model_size']['total_stored_counts']:,} stored counts" if "model_size" in d else "—"
            rows.append([f"`{run}`", arch, fmt(m["val"]["perplexity"]), fmt(m["test"]["perplexity"]), size,
                         fmt(d.get("train_seconds")), throughput, cell(rec.get("hardware"))])
        elif "test_perplexity" in m:
            rows.append([f"`{run}`", arch, fmt(math.exp(m["best_val_loss"])), fmt(m["test_perplexity"]),
                         (f"{m['num_parameters']:,} trainable ({d['total_parameters']:,} in total)" if "total_parameters" in d
                          and d["total_parameters"] != m["num_parameters"]
                          else f"{m['num_parameters']:,} parameters") if "num_parameters" in m else "—",
                         fmt(m.get("train_seconds")), throughput, cell(rec.get("hardware"))])
    if not rows:
        return "No trained language models yet.\n"
    return ("Validation perplexity of a neural run is exp(best validation loss); the n-gram figure is as recorded. "
            "Inference throughput shows — until a run records `inference_tokens_per_second`.\n\n"
            + table(["Run", "Architecture", "Validation perplexity", "Test perplexity", "Model size", "Training seconds",
                     "Inference tokens/s", "Hardware"], rows))


# ------------------------------------------------------------------ DECISIONS.md
def parse_decisions(path):
    text = read_text(path)
    decisions = []
    for m_head in re.finditer(r"^### (.*)$", text, re.M):
        title = m_head.group(1)
        body = re.split(r"\n(?:## |### |---)", text[m_head.end():])[0]
        did, _, name = title.partition(". ")
        fields = {}
        for key, label in (("Value", "Value"), ("Where", "Where"), ("Chosen by", "Chosen by"),
                           ("Status", "Status"), ("Alternatives", r"Alternatives(?: considered)?")):
            m = re.search(rf"\*\*{label}:\*\*\s*(.*?)(?=\s*\*\*[A-Za-z][^*\n]*:\*\*|\n- \*\*|\n\n|\Z)", body, re.S)
            fields[key] = " ".join(m.group(1).split()) if m else "not stated"
        line = text.count("\n", 0, m_head.start()) + 1
        decisions.append({"id": did, "title": name.strip(), "line": line, **fields})
    return decisions


def render_settings(decisions):
    flagged = [d for d in decisions if "NEEDS REVIEW" in d["Status"]]
    out = ["Source: `DECISIONS.md` (copied as written there).\n"]
    out.append(f"### NEEDS REVIEW ({len(flagged)})\n")
    if flagged:
        out.append(table(["ID", "Decision", "Status"], [[d["id"], cell(d["title"]), cell(d["Status"])] for d in flagged]))
    else:
        out.append("None.\n")
    out.append(f"### All design choices ({len(decisions)})\n")
    out.append(table(["ID", "Decision", "Value", "Where", "Chosen by", "Status"],
                     [[d["id"], cell(d["title"]), cell(d["Value"]), cell(d["Where"]), cell(d["Chosen by"]),
                       cell(d["Status"])] for d in decisions]))
    return "\n".join(out)


# ------------------------------------------------------------------------ checks
def data_attribute_lines(path, attr_owner, attr):
    """Line numbers where `<owner>.<attr>` is used, and where each function is called (via ast)."""
    tree = ast.parse(read_text(path))
    uses = sorted(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == attr
                  and isinstance(n.value, ast.Name) and n.value.id == attr_owner)
    calls = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            name = n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None)
            calls.setdefault(name, []).append(n.lineno)
    return uses, calls, tree


def check_untrained(runs, opts):
    run = runs.get(opts["sanity_check_run"])
    if not run:
        return "NOT YET AVAILABLE", [f"results/{opts['sanity_check_run']}.json does not exist"]
    ppl = run["metrics"]["val_perplexity_untrained"]
    vocab = run["details"]["vocab_size_with_special_tokens"]
    lo, hi = opts["untrained_ppl_to_vocab_ratio_range"]
    ratio = ppl / vocab
    status = "PASS" if lo <= ratio <= hi else "FAIL"
    return status, [f"untrained validation perplexity {ppl:,.1f}; vocabulary size (with special tokens) {vocab:,}",
                    f"ratio {ratio:.3f}; accepted range {lo} to {hi} (`configs/update_results.json`, `untrained_ppl_to_vocab_ratio_range`)",
                    f"source: `results/{opts['sanity_check_run']}.json`"]


def check_test_set(opts):
    evidence, ok = [], True
    for rel, marker in opts["test_set_markers"].items():
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            evidence.append(f"`{rel}` not found")
            ok = False
            continue
        uses, calls, _ = data_attribute_lines(path, "data", "test")
        marker_lines = calls.get(marker, [])
        good = len(uses) == 1 and marker_lines and uses[0] > max(marker_lines)
        ok = ok and bool(good)
        evidence.append(f"`{rel}`: `data.test` used at line(s) {uses or 'none'}; `{marker}(...)` called at line(s) {marker_lines or 'none'}; "
                        f"{'single use after the call' if good else 'does not meet: exactly one use, after the call'}")
    others = []
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, "src", "*.py"))):
        rel = os.path.relpath(path, REPO_ROOT)
        if rel in opts["test_set_markers"]:
            continue
        uses, _, _ = data_attribute_lines(path, "data", "test")
        if uses:
            others.append(f"`{rel}` lines {uses}")
    evidence.append("other files in `src/` that use `data.test`: " + (", ".join(others) if others else "none"))
    ok = ok and not others
    evidence.append("method: static scan of the source (Python `ast`), not a run-time trace")
    return ("PASS" if ok else "FAIL"), evidence


def check_padding(runs, opts):
    evidence, ok = [], True
    for rel in opts["padding_loss_files"]:
        path = os.path.join(REPO_ROOT, rel)
        tree = ast.parse(read_text(path))
        lines = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "CrossEntropyLoss"
                 and any(k.arg == "ignore_index" for k in n.keywords)]
        bare = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "CrossEntropyLoss"
                and not any(k.arg == "ignore_index" for k in n.keywords)]
        ok = ok and bool(lines) and not bare
        evidence.append(f"`{rel}`: `CrossEntropyLoss(ignore_index=...)` at line(s) {lines or 'none'}; without `ignore_index` at {bare or 'none'}")
    rec = runs.get(opts["padding_test_run"])
    results = (rec or {}).get("details", {}).get("tests", {})
    for name in opts["padding_test_names"]:
        outcome = results.get(name, "not run")
        ok = ok and outcome == "passed"
        evidence.append(f"`{name}` in `results/{opts['padding_test_run']}.json`: {outcome}")
    return ("PASS" if ok else "FAIL"), evidence


def check_causality(runs, opts):
    found = []
    for run, rec in runs.items():
        for name, outcome in ((rec.get("details") or {}).get("tests") or {}).items():
            if opts["causality_test_keyword"] in name:
                found.append((run, name, outcome))
    if not found:
        return "NOT YET AVAILABLE", [f"no recorded test with '{opts['causality_test_keyword']}' in its name"]
    ok = all(o == "passed" for _, _, o in found)
    return ("PASS" if ok else "FAIL"), [f"`{n}` in `results/{r}.json`: {o}" for r, n, o in found]


def check_frozen(runs, opts):
    key = opts["frozen_embedding_detail_key"]
    frozen = [(r, rec) for r, rec in runs.items() if rec["config"].get("embedding_init") not in opts["not_frozen_embedding_inits"]
              and not r.startswith(opts["test_run_prefix"])
              and rec.get("script") in opts["frozen_check_scripts"]]  # embedding-building scripts are not LM runs
    if not frozen:
        return "NOT YET AVAILABLE", ["no run with a pretrained `embedding_init` (Part II has no such run yet)"]
    evidence, ok, missing = [], True, False
    for run, rec in frozen:
        value = (rec.get("details") or {}).get(key)
        if value is None:
            missing = True
            evidence.append(f"`results/{run}.json`: details.{key} not recorded")
        else:
            ok = ok and value == 0
            evidence.append(f"`results/{run}.json`: details.{key} = {value}")
    if missing:
        return "NOT YET AVAILABLE", evidence
    return ("PASS" if ok else "FAIL"), evidence


def check_same_test_targets(runs, opts):
    """Every model's test score must come from the same targets (same count and checksum)."""
    seen = {}
    for run, rec in runs.items():
        if run.startswith(opts["test_run_prefix"]) or rec.get("generations"):
            continue
        m, d = rec.get("metrics") or {}, rec.get("details") or {}
        if isinstance(m.get("test"), dict) and "target_checksum" in m["test"]:
            seen[run] = (m["test"]["num_targets"], m["test"]["target_checksum"])
        elif "test_target_checksum" in d:
            seen[run] = (d["test_num_targets"], d["test_target_checksum"])
    if len(seen) < 2:
        return "NOT YET AVAILABLE", ["fewer than two models with a recorded test-target checksum"]
    ok = len(set(seen.values())) == 1
    return ("PASS" if ok else "FAIL"), [f"`{r}`: {n:,} targets, checksum `{c[:12]}...`" for r, (n, c) in seen.items()]


def check_matched_decoding(runs, opts):
    """Generation runs compared within one part must use the same decoding settings and prompts."""
    keys = opts["matched_decoding_keys"]
    gens = {r: rec for r, rec in runs.items() if rec.get("generations") and not r.startswith(opts["test_run_prefix"])}
    if len(gens) < 2:
        return "NOT YET AVAILABLE", [f"{len(gens)} generation run(s); at least two are needed to compare"]
    table_ = {r: tuple(rec["decoding"].get(k) for k in keys) + (tuple(sorted({g["prompt"] for g in rec["generations"]})),)
              for r, rec in gens.items()}
    ok = len(set(table_.values())) == 1
    return ("PASS" if ok else "FAIL"), [f"`{r}`: " + ", ".join(f"{k}={v}" for k, v in zip(keys, vals)) + f"; {len(vals[-1])} prompts"
                                         for r, vals in table_.items()]


def render_checks(runs, opts):
    checks = [("Untrained perplexity is close to the vocabulary size", check_untrained(runs, opts)),
              ("Test set used only for final evaluation", check_test_set(opts)),
              ("Padding excluded from the loss", check_padding(runs, opts)),
              ("All models scored on identical test targets", check_same_test_targets(runs, opts)),
              ("Generation runs use matched decoding settings and prompts", check_matched_decoding(runs, opts)),
              ("Causality test", check_causality(runs, opts)),
              ("Frozen embeddings unchanged", check_frozen(runs, opts))]
    out = [table(["Check", "Result"], [[name, f"**{res[0]}**"] for name, res in checks])]
    for name, (status, evidence) in checks:
        out.append(f"#### {name}: {status}\n")
        out.append("\n".join(f"- {e}" for e in evidence) + "\n")
    return "\n".join(out), checks


# ------------------------------------------------------------------- notes
def extract_notes(old_text):
    """{section key: text between its my-notes markers}; None if markers are missing."""
    notes = {}
    for key, _ in SECTIONS:
        m = re.search(rf"<!-- my-notes:{key}:start -->\n(.*?)<!-- my-notes:{key}:end -->", old_text, re.S)
        if not m:
            return None
        notes[key] = m.group(1)
    return notes


def notes_block(key, notes):
    return (f"### My notes\n\n<!-- my-notes:{key}:start -->\n{notes.get(key, chr(10))}<!-- my-notes:{key}:end -->\n")


# --------------------------------------------------------------------- main
def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild RESULTS.md and results/figures/.")
    add_config_args(parser)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite RESULTS.md even if the old file lost its My-notes markers.")
    return parser.parse_args()


def main():
    args = parse_args()
    config, _ = load_config(args)
    opts = config["options"]
    os.chdir(REPO_ROOT)  # all paths in the config are relative to the repo root

    runs = load_runs(opts["results_dir"])
    log = commands_from_log(opts["results_log_path"])
    decisions = parse_decisions(opts["decisions_path"])
    colours = opts["figure_colours"]
    test_prefix = opts["test_run_prefix"]

    # Keep the user's notes.
    notes = {}
    if os.path.exists(opts["output_path"]):
        old = read_text(opts["output_path"])
        notes = extract_notes(old)
        if notes is None:
            if not args.force:
                raise SystemExit(f"{opts['output_path']} no longer has all the my-notes marker lines, so your notes might be "
                                 "lost. Restore the markers (or run with --force to discard the old notes).")
            notes = {}

    # Figures are always rebuilt from scratch.
    figures_dir = opts["figures_dir"]
    os.makedirs(figures_dir, exist_ok=True)
    for old_fig in glob.glob(os.path.join(figures_dir, "*.png")):
        os.remove(old_fig)

    figures = {run: [] for run in runs}
    if "data" in runs:
        figures["data"].append(fig_split_tokens(runs["data"], figures_dir, colours))
    if "ngram" in runs:
        path = fig_ngram_tuning(runs["ngram"], figures_dir, colours)
        if path:
            figures["ngram"].append(path)
    trained = [rec for run, rec in runs.items() if rec["epoch_losses"] and not run.startswith(test_prefix)]
    for rec in trained:
        figures[rec["run"]].append(fig_loss_curves(rec, figures_dir, colours))
    all_curves = fig_val_curves_all(trained, figures_dir, colours) if len(trained) >= 2 else None

    # Group runs by section.
    groups = {"Data": {}, "Part I": {}, "Part II": {}, "Part III": {}}
    unassigned = {}
    for run, rec in runs.items():
        part = part_of(run, opts["part_rules"])
        (groups[part] if part in groups else unassigned)[run] = rec

    def render_group(group):
        if not group:
            return "No runs yet.\n"
        blocks = []
        for run, rec in group.items():
            command, source = find_command(run, rec, log)
            blocks.append(render_run(run, rec, command, source, figures.get(run, [])))
        return "\n".join(blocks)

    gen_runs = {r: rec for r, rec in runs.items() if rec.get("generations")}
    gen_by_part = {p: {r: rec for r, rec in gen_runs.items() if part_of(r, opts["part_rules"]) == p}
                   for p in ("Part I", "Part II", "Part III")}
    newest = max((rec["date"] for rec in runs.values()), default="—")

    def part_comparison(part):
        """Perplexity/cost table and side-by-side generated text for the runs of one part."""
        extra = opts["part_comparison_extra_runs"].get(part, [])
        keep = lambda run: part_of(run, opts["part_rules"]) == part or run in extra
        gens = {r: rec for r, rec in gen_runs.items() if part_of(r, opts["part_rules"]) == part or r in extra}
        return ("\n### Language-model comparison\n\n" + render_model_summary(runs, test_prefix, keep)
                + "\n### Generated text by prompt\n\n" + render_generation_comparison(gens))

    body = {}
    body["data"] = render_group(groups["Data"])
    body["part1"] = render_group(groups["Part I"]) + part_comparison("Part I")
    body["part2"] = render_group(groups["Part II"]) + (f"\n![val_loss_all_runs.png]({all_curves})\n" if all_curves else "") + part_comparison("Part II") + render_part2_facts(runs, opts)
    body["part3"] = render_group(groups["Part III"])
    if unassigned:
        body["part3"] += ("\n### Runs not assigned to a section\n\n"
                          "These runs match no rule in `part_rules` (`configs/update_results.json`).\n\n" + render_group(unassigned))
    body["settings"] = render_settings(decisions)
    body["checks"], checks = render_checks(runs, opts)

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    doc = ["# Results\n",
           f"Rebuilt by `src/update_results.py` on {now} from {len(runs)} run records in `results/`, "
           f"`{opts['decisions_path']}` and `{opts['results_log_path']}`. Newest run record: {newest}. "
           "Do not edit outside the *My notes* blocks; they are rebuilt each time.\n",
           "Contents: " + " · ".join(f"[{title}](#{title.lower().replace(' ', '-')})" for _, title in SECTIONS) + "\n"]
    for key, title in SECTIONS:
        doc.append(f"## {title}\n")
        doc.append(body[key])
        doc.append(notes_block(key, notes))
    text = "\n".join(doc)
    with open(opts["output_path"], "w", encoding="utf-8") as f:
        f.write(text)
    n_figs = len(glob.glob(os.path.join(figures_dir, "*.png")))
    sections_with_runs = {"data": list(groups["Data"]), "part1": list(groups["Part I"]),
                          "part2": list(groups["Part II"]), "part3": list(groups["Part III"])}
    tasks_path = write_my_tasks(opts, runs, decisions, checks, notes, sections_with_runs, force=args.force)
    print(f"Wrote {tasks_path}")
    print(f"Wrote {opts['output_path']} ({len(runs)} runs, {n_figs} figures in {figures_dir}/)")


if __name__ == "__main__":
    main()
