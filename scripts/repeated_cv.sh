#!/bin/sh
# Repeated ten-fold cross-validation of the KG experiment: the same 644-repository corpus and
# readme_8k_desc20 CNBN corpus as the main run, re-split with several fold seeds.
#
# The pipeline is deterministic (identical metrics under another PYTHONHASHSEED), so the only
# random choice in an experiment is which repositories share a fold. Re-running with other
# fold seeds measures how much the A/B/C comparison depends on that choice. Seed 42 is the
# main experiment's own split; its fold tree and metrics are checked against it.
#
# Per seed, everything goes under OUT/seed_<s>/ (the main experiment's files are not touched):
#   topfilter/       a copy of outpath/TopFilter, projects.txt re-ordered by make_folds.py
#   folds.json       scripts/make_folds.py --seed <s>
#   cnbn_folds/      cnbn/generate_datatset.py over repo_readme_8k_desc20 with those folds
#   experiments/     scripts/run_experiment.sh: CNBN once, then TopFilter runs A, B and C
#   artifacts/       scripts/collect_results.py
#
# Usage: scripts/repeated_cv.sh OUT SEED [SEED ...]     e.g. outpath/repeated_cv 42 1 2 3 4
# Summarise with scripts/summarize_repeated_cv.py OUT.
set -eu
OUT="$1"; shift
PY=.venv/bin/python
export PYTHONHASHSEED=0
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_sha.txt"
git status --porcelain --untracked-files=no > "$OUT/git_status.txt"

for s in "$@"; do
    W="$OUT/seed_$s"
    rm -rf "$W"; mkdir -p "$W/topfilter"
    find outpath/TopFilter -maxdepth 1 \( -name 'dicth_*' -o -name 'graph_*' \) -exec cp -t "$W/topfilter/" {} +
    cp outpath/TopFilter/projects.txt "$W/topfilter/"
    $PY scripts/make_folds.py --src "$W/topfilter" --out "$W/folds.json" --seed "$s"
    $PY cnbn/generate_datatset.py --csv repositories_test.csv --readme-dir repo_readme_8k_desc20 \
        --output-dir "$W/cnbn_folds" --folds-json "$W/folds.json" > "$W/generate.log" 2>&1
    EXP_SRC="$W/topfilter" EXP_VARIANTS="readme_8k_desc20:$W/cnbn_folds" \
        sh scripts/run_experiment.sh "$W/experiments"
    EXP_FOLDS_JSON="$W/folds.json" EXP_INPUT_CSV=repositories_test.csv EXP_PRIMARY=readme_8k_desc20 \
        $PY scripts/collect_results.py "$W/experiments" "$W/artifacts" > "$W/collect.log" 2>&1
    rm -rf "$W/cnbn_folds"   # README copies per topic (~640 MB); its sha256 is in artifacts/config.json
    echo "seed $s done"
done
