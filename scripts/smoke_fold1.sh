#!/bin/sh
# Fold-1 smoke tests: the effect of each Phase-4 fix, measured by reverting it alone on top
# of the current code, on the primary dataset's CNBN seeds (readme_8k_desc20, fold 1).
#
# Usage: scripts/smoke_fold1.sh [OUT_CSV] [WORK]
#   OUT_CSV  default paper_artifacts/smoke/smoke_fold1.csv
#   WORK     default outpath/smoke
set -eu
OUT="${1:-paper_artifacts/smoke/smoke_fold1.csv}"
W="${2:-outpath/smoke}"
PY=.venv/bin/python
export PYTHONHASHSEED=0
rm -rf "$W"; mkdir -p "$W" "$(dirname "$OUT")"

# CNBN fold 1 only, KG off, into WORK.
env -u KG_ENABLED -u CNBN_DOUBLE_TFIDF CNBN_FOLDS_DIR=custom_dataset/CNBN/folds_desc20 \
    CNBN_RESULTS_DIR="$W/cnbn" CNBN_EASE_OUT="$W/seeds.csv" \
    $PY -c "import sys; sys.path.insert(0, 'cnbn'); import main; main.run_experiment(1, 'CNB', False)" \
    > "$W/cnbn.log" 2>&1
sed 's/;/,/g' "$W/seeds.csv" > "$W/seeds_comma.csv"
git show d0f7f9c:topfilter/metrics.py > "$W/metrics_d0f7f9c.py"

case_run() {  # case_run <name> <args...>
    name="$1"; shift
    $PY scripts/smoke_fold1_case.py "$W/work_$name" "$@" > "$W/$name.json" 2> "$W/$name.log"
}
case_run current        --seeds "$W/seeds.csv" &
case_run delimiter      --seeds "$W/seeds_comma.csv" &
case_run prefix         --seeds "$W/seeds.csv" --seed-prefix ' ' &
case_run floor          --seeds "$W/seeds.csv" --score-step floor &
case_run rank_offset    --seeds "$W/seeds.csv" --metrics-module "$W/metrics_d0f7f9c.py" &
wait
PYTHONHASHSEED=12345 $PY scripts/smoke_fold1_case.py "$W/work_hashseed" --seeds "$W/seeds.csv" \
    > "$W/hashseed.json" 2> "$W/hashseed.log"

$PY - "$W" "$OUT" <<'EOF'
import csv, json, sys
W, OUT = sys.argv[1], sys.argv[2]
cases = [("current (all fixes)", "current"),
         ("revert fix 1a: comma-separated bridge", "delimiter"),
         ("revert fix 1b: '#DEP# ' seed prefix", "prefix"),
         ("revert fix 3: floor fold window in validator", "floor"),
         ("revert rank-offset fix: metrics.py from d0f7f9c", "rank_offset"),
         ("current, PYTHONHASHSEED=12345", "hashseed")]
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "projects_scored", "projects_with_5_seeds", "metric", "N", "value"])
    for label, name in cases:
        d = json.load(open(f"{W}/{name}.json"))
        for metric, key in [("success_rate", "SR"), ("precision", "P"), ("recall", "R"),
                            ("catalog_coverage", "Cov")]:
            for n, v in d[key].items():
                w.writerow([label, d["projects_scored"], d["projects_with_5_seeds"], metric, n, v])
print("wrote", OUT)
EOF
