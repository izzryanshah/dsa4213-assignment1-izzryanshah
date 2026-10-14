"""Build the report tables (CSV, LaTeX, Markdown) and loss-curve figures (PDF) from results/.

What this script does, in plain words
-------------------------------------
Every training run already saved a JSON file in results/ (see runconfig.save_run).
This script only READS those files and re-arranges the numbers; it trains nothing and
needs no GPU, so it is safe to run on a laptop. It writes:

  Part I    results/tables/part1_table.{csv,tex,md}   test perplexity, parameters,
                                                      training time, throughput
            results/figures/part1_loss_curves.pdf     train + validation loss, RNN / LSTM / Transformer
  Part II   results/tables/part2_table.{csv,tex,md}   the three embedding settings, GloVe
                                                      coverage, word2vec training time
            results/figures/part2_loss_curves.pdf     train + validation loss, three settings
  Part III  results/tables/part3_table.{csv,tex,md}   accuracy and macro-F1
  Record    results/make_figures.json                 the usual run record

Rules for this file: numbers only (no interpretation), and nothing is typed in by hand:
every value comes from a results/*.json or results/*.csv file named in the config.

Notes on where numbers come from
--------------------------------
* Part I "test perplexity" is the `benchmark_perplexity` column of results/part1_costs.csv:
  all four models scored on the SAME 981,353 target tokens (so they are comparable).
  The CSV also keeps `train_time_test_perplexity` (the number printed at the end of
  training) for reference; for the n-gram model that one uses a different token count.
* "Parameters" for the n-gram model are stored counts, not weights (marked with * in
  the table), so they are not comparable with the neural models.
* Training time comes from the machine that trained the model (n-gram: local CPU;
  neural models: Colab T4 GPU). Throughput comes from the benchmark run (same GPU for
  every model; CPU throughput measured on one thread).
* CSV files hold the full-precision numbers. The LaTeX/Markdown files hold rounded text.

Usage:
    python3 src/make_figures.py --config configs/make_figures.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")   # draw to files only; no window needed
import matplotlib.pyplot as plt
import pandas as pd

from runconfig import add_config_args, load_config, save_run


# ----------------------------------------------------------------- reading
def read_json(results_dir, run):
    """Load results/<run>.json, with a clear message if the run does not exist yet."""
    path = os.path.join(results_dir, f"{run}.json")
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run that step first (see MY_TASKS.md).")
    with open(path) as f:
        return json.load(f)


# -------------------------------------------------------------- formatting
def fmt_int(x):
    return "" if pd.isna(x) else f"{int(round(x)):,}"


def fmt_float(x, digits):
    return "" if pd.isna(x) else f"{x:.{digits}f}"


def tex_escape(text):
    """Make a plain string safe inside LaTeX (underscores, percent signs, ...)."""
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def write_table(df_raw, df_text, name, tables_dir, caption, notes=()):
    """Save one table three ways: CSV (raw numbers), LaTeX (booktabs) and Markdown.

    df_raw  - full-precision numbers (goes to the CSV)
    df_text - the same table as rounded strings (goes to .tex and .md)
    notes   - plain factual lines printed under the table (e.g. "* stored counts")
    """
    os.makedirs(tables_dir, exist_ok=True)
    base = os.path.join(tables_dir, name)
    df_raw.to_csv(base + ".csv", index=False)

    cols = list(df_text.columns)
    # LaTeX: first column left-aligned, the rest right-aligned.
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df_text.iterrows():
        lines.append(" & ".join(tex_escape(str(v)) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    for note in notes:
        lines.append(r"\par\vspace{2pt}{\footnotesize " + tex_escape(note) + "}")
    lines += [r"\caption{" + tex_escape(caption) + "}", r"\label{tab:" + name + "}", r"\end{table}"]
    with open(base + ".tex", "w") as f:
        f.write("\n".join(lines) + "\n")

    md = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|"]
    for _, row in df_text.iterrows():
        md.append("| " + " | ".join(str(v) for v in row) + " |")
    md_text = f"**{caption}**\n\n" + "\n".join(md) + "\n"
    if notes:
        md_text += "\n" + "\n".join(f"- {n}" for n in notes) + "\n"
    with open(base + ".md", "w") as f:
        f.write(md_text)
    return [base + ext for ext in (".csv", ".tex", ".md")]


# ------------------------------------------------------------------ tables
def part1_table(opts):
    costs = pd.read_csv(opts["part1_costs_csv"]).set_index("model")
    rows_raw, rows_text = [], []
    for model, run in opts["part1_runs"].items():
        c = costs.loc[model]
        rec = read_json(opts["results_dir"], run)
        # the n-gram record keeps its test perplexity under metrics.test, neural runs at metrics root
        train_time_ppl = (rec["metrics"]["test"]["perplexity"] if model == "ngram"
                          else rec["metrics"]["test_perplexity"])
        rows_raw.append({
            "model": model, "test_perplexity": c["benchmark_perplexity"],
            "train_time_test_perplexity": train_time_ppl,
            "parameters": c["parameters"], "parameter_meaning": c["parameter_meaning"],
            "train_seconds": c["train_seconds"], "train_hardware": c["train_hardware"],
            "tokens_per_second_cuda_mean": c["tokens_per_second_cuda_mean"],
            "tokens_per_second_cuda_std": c["tokens_per_second_cuda_std"],
            "tokens_per_second_cpu_mean": c["tokens_per_second_cpu_mean"],
            "tokens_per_second_cpu_std": c["tokens_per_second_cpu_std"],
            "benchmark_gpu": c["benchmark_gpu"], "num_targets": c["num_targets"]})
        is_counts = "not comparable" in str(c["parameter_meaning"])
        rows_text.append({
            "Model": model,
            "Test perplexity": fmt_float(c["benchmark_perplexity"], 2),
            "Parameters": fmt_int(c["parameters"]) + ("*" if is_counts else ""),
            "Train time (s)": fmt_float(c["train_seconds"], 2 if c["train_seconds"] < 10 else 0),
            "GPU tokens/s": fmt_int(c["tokens_per_second_cuda_mean"]) or "--",
            "CPU tokens/s": fmt_int(c["tokens_per_second_cpu_mean"])})
    notes = ["* stored n-gram counts, not weights (not comparable with the neural models).",
             f"Test perplexity: all models on the same {int(costs['num_targets'].iloc[0]):,} test target tokens. "
             "Throughput: mean of repeated passes, tokens per second (GPU: "
             f"{costs['benchmark_gpu'].iloc[0]}; CPU: one thread). Train time: n-gram on local CPU, "
             "neural models on the Colab GPU."]
    return pd.DataFrame(rows_raw), pd.DataFrame(rows_text), notes


def part2_table(opts):
    rd = opts["results_dir"]
    rows_raw, rows_text = [], []
    for setting, run in opts["part2_runs"].items():
        rec = read_json(rd, run)
        m, d = rec["metrics"], rec["details"]
        total = d.get("total_parameters", m["num_parameters"])   # runs with a frozen table record both counts
        cov = opts["part2_coverage_runs"].get(setting)
        cov_m = read_json(rd, cov)["metrics"] if cov else {}
        cov_det = read_json(rd, cov)["details"] if cov else {}
        # GloVe coverage JSON has the numbers in metrics; word2vec's are in details
        types = cov_m.get("coverage_types_percent", cov_det.get("coverage_types_percent"))
        tokens = cov_m.get("coverage_tokens_percent", cov_det.get("coverage_tokens_percent"))
        prep_run = opts["part2_prep_time_runs"].get(setting)
        prep = read_json(rd, prep_run) if prep_run else None
        prep_s = prep["metrics"]["total_seconds"] if prep else None
        rows_raw.append({
            "embedding_setting": setting, "run": run,
            "test_perplexity": m["test_perplexity"], "best_val_loss": m["best_val_loss"],
            "best_epoch": m["best_epoch"], "trainable_parameters": m["num_parameters"],
            "total_parameters": total, "train_seconds": m["train_seconds"], "train_hardware": rec["hardware"],
            "coverage_types_percent": types, "coverage_tokens_percent": tokens,
            "word2vec_train_seconds": prep["metrics"]["train_seconds"] if prep else None,
            "word2vec_total_seconds": prep_s, "word2vec_hardware": prep["hardware"] if prep else None})
        rows_text.append({
            "Embedding": setting,
            "Test perplexity": fmt_float(m["test_perplexity"], 2),
            "Best val. loss": fmt_float(m["best_val_loss"], 3),
            "Trainable params": fmt_int(m["num_parameters"]),
            "Train time (s)": fmt_float(m["train_seconds"], 0),
            "Coverage, types (%)": fmt_float(types, 2) or "--",
            "Coverage, tokens (%)": fmt_float(tokens, 2) or "--",
            "word2vec time (s)": fmt_float(prep["metrics"]["train_seconds"] if prep else None, 1) or "--"})
    notes = ["Same LSTM, data, seed and settings in all rows; only the embedding table differs.",
             "Coverage: share of vocabulary types / training tokens that have a real vector before "
             "missing rows are filled. word2vec time: training on the local CPU (Apple M2), "
             "not the GPU used for the language-model runs.",
             "Total parameters including the frozen table: "
             + "; ".join(f"{r['embedding_setting']}: {int(r['total_parameters']):,}" for r in rows_raw) + "."]
    return pd.DataFrame(rows_raw), pd.DataFrame(rows_text), notes


def part3_table(opts):
    rec = read_json(opts["results_dir"], opts["part3_run"])
    rows_raw = [dict(v) for v in rec["metrics"]["variants"]]
    rows_text = [{"Representation": v["variant"], "Features": str(v["feature_size"]),
                  "Best C": f"{v['best_C']:g}",
                  "Val. accuracy": fmt_float(v["val_accuracy"], 4),
                  "Test accuracy": fmt_float(v["test_accuracy"], 4),
                  "Test macro-F1": fmt_float(v["test_macro_f1"], 4)} for v in rows_raw]
    notes = [f"Logistic regression on frozen features; C chosen on validation macro-F1; "
             f"{rows_raw[0]['n_test']:,} test reviews."]
    return pd.DataFrame(rows_raw), pd.DataFrame(rows_text), notes


# ----------------------------------------------------------------- figures
def loss_figure(curves, colors, path, opts, title):
    """One figure, one panel per model/setting; solid = training loss, dashed = validation loss.

    Same y-axis in every panel so the panels can be compared by eye. `curves` maps a
    label to the run's list of {"epoch", "train_loss", "val_loss"} records.
    """
    plt.rcParams.update({"font.size": opts["font_size"], "pdf.fonttype": 42, "axes.titlesize": opts["font_size"]})
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(opts["figure_width_in"], opts["figure_height_in"]),
                             sharey=True, constrained_layout=True)
    for ax, (label, epochs) in zip(axes, curves.items()):
        x = [e["epoch"] for e in epochs]
        ax.plot(x, [e["train_loss"] for e in epochs], color=colors[label], linestyle="-", linewidth=1.6, label="train")
        ax.plot(x, [e["val_loss"] for e in epochs], color=colors[label], linestyle="--", linewidth=1.6, label="validation")
        ax.set_title(label)
        ax.set_xlabel("epoch")
        ax.set_xticks(range(0, max(x) + 1, 5) if max(x) >= 5 else x)
        ax.grid(True, color="#dddddd", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("cross-entropy (nats)")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(title, fontsize=opts["font_size"] + 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Build report tables and loss-curve figures from results/.")
    add_config_args(parser)
    args = parser.parse_args()
    config, run = load_config(args)
    opts = config["options"]
    rd = opts["results_dir"]

    written = []
    tables = (("part1_table", part1_table, "Part I: language models on the IMDB test set."),
              ("part2_table", part2_table, "Part II: LSTM with three embedding settings."),
              ("part3_table", part3_table, "Part III: classifier on frozen language-model features."))
    for name, builder, caption in tables:
        raw, text, notes = builder(opts)
        written += write_table(raw, text, name, opts["tables_dir"], caption, notes)
        print(f"\n{caption}\n{text.to_string(index=False)}")

    p1 = {m: read_json(rd, opts["part1_runs"][m])["epoch_losses"] for m in opts["part1_curve_models"]}
    written.append(loss_figure(p1, opts["colors"], os.path.join(opts["figures_dir"], "part1_loss_curves.pdf"),
                               opts, "Part I: training and validation loss"))
    p2 = {s: read_json(rd, r)["epoch_losses"] for s, r in opts["part2_runs"].items()}
    written.append(loss_figure(p2, opts["colors"], os.path.join(opts["figures_dir"], "part2_loss_curves.pdf"),
                               opts, "Part II (LSTM): training and validation loss"))

    path = save_run(rd, run, "src/make_figures.py", config, metrics={}, details={"files_written": written})
    print("\nWrote:\n  " + "\n  ".join(written) + f"\nSaved run record to {path}")


if __name__ == "__main__":
    main()
