#!/usr/bin/env bash
# Re-run everything produced so far (data pipeline, sanity check, trigram
# Kneser-Ney model, tests, n-gram text generation, summary table), in order.
# Stops at the first command that fails (set -e).
#
# Usage, from anywhere:   bash run_part1_ngram.sh
# Use another Python:     PYTHON=/path/to/python bash run_part1_ngram.sh
#
# Needs data/IMDB Dataset.csv (see README.md, section 1) and prompts.txt.
# Overwrites the matching files in results/ (same names, same seeds).

set -euo pipefail

cd "$(dirname "$0")"          # run from the repository root
PYTHON="${PYTHON:-python3}"   # must have torch, numpy, pandas, scikit-learn installed

[ -f "data/IMDB Dataset.csv" ] || { echo "Missing data/IMDB Dataset.csv (see README.md section 1)"; exit 1; }
[ -f "prompts.txt" ] || { echo "Missing prompts.txt (see README.md, 'Generation prompts')"; exit 1; }

run() {
    echo
    echo "=== $* ==="
    "$@"
}

run "$PYTHON" src/data.py          --config configs/data.json          # -> data/processed/*, results/data.json
run "$PYTHON" src/inspect_data.py  --config configs/inspect_data.json  # -> results/inspect_data.json
run "$PYTHON" src/sanity_check.py  --config configs/sanity_check.json  # -> results/sanity_check.json
run "$PYTHON" src/ngram.py         --config configs/ngram.json         # -> results/ngram.json (needs data.py first)
run "$PYTHON" tests/test_evaluate.py --config configs/test_evaluate.json   # -> results/test_evaluate.json
run "$PYTHON" tests/test_generate.py --config configs/test_generate.json   # -> results/test_generate.json
run "$PYTHON" src/generate.py      --config configs/generations_ngram.json # -> results/generations_ngram.json (needs ngram.py first)
run "$PYTHON" src/summarise_runs.py                                    # -> results/all_runs.csv

echo
echo "All steps finished. Outputs are in results/."
