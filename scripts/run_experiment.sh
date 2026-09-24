#!/bin/sh
# KG experiment: runs A (KG off), B (KG_MODE=append), C (KG_MODE=rerank) on every CNBN
# README-corpus variant, full ten-fold, identical folds and seeds within a variant.
#
# Per variant, CNBN runs once (KG disabled) and its seed file is shared by A, B and C, so the
# three runs differ only in the Topics Merger's KG step. Every TopFilter run gets its own copy
# of the dataset directory (TOPFILTER_SRC_DIR), so runs cannot overwrite each other.
#
# Usage: scripts/run_experiment.sh [OUT_DIR]      (default outpath/experiments)
# Run from the repository root. Wall-clock seconds are written to OUT_DIR/timings.tsv and the
# commit being run to OUT_DIR/git_sha.txt.
#
# Overridable for another corpus (defaults reproduce the 644-repository experiment):
#   EXP_SRC       TopFilter dataset directory (dicth_/graph_/projects.txt)
#   EXP_VARIANTS  space-separated name:cnbn_fold_tree pairs
set -eu
OUT="${1:-outpath/experiments}"
PY=.venv/bin/python
SRC="${EXP_SRC:-outpath/TopFilter}"
export PYTHONHASHSEED=0

VARIANTS="${EXP_VARIANTS:-readme_8k_desc20:custom_dataset/CNBN/folds_desc20 readme_8k:custom_dataset/CNBN/folds_sync meta_only:custom_dataset/CNBN/folds_meta}"

mkdir -p "$OUT"
: > "$OUT/timings.tsv"
git rev-parse HEAD > "$OUT/git_sha.txt"
git status --porcelain --untracked-files=no > "$OUT/git_status.txt"
printf '%s\n' $VARIANTS > "$OUT/variants.txt"
printf '%s\n' "$SRC" > "$OUT/src.txt"

timed() {  # timed <label> <logfile> <command...>
    label="$1"; log="$2"; shift 2
    start=$(date +%s.%N)
    status=0
    "$@" > "$log" 2>&1 || status=$?
    end=$(date +%s.%N)
    printf '%s\t%s\t%s\n' "$label" "$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.1f", b-a}')" "$status" >> "$OUT/timings.tsv"
}

# --- CNBN, one per variant, in parallel -------------------------------------------------
for spec in $VARIANTS; do
    v="${spec%%:*}"; folds="${spec#*:}"
    mkdir -p "$OUT/$v"
    timed "cnbn/$v" "$OUT/$v/cnbn.log" env -u KG_ENABLED -u CNBN_DOUBLE_TFIDF \
        CNBN_FOLDS_DIR="$folds" CNBN_RESULTS_DIR="$OUT/$v/cnbn" CNBN_EASE_OUT="$OUT/$v/seeds.csv" \
        "$PY" cnbn/main.py &
done
wait

# --- TopFilter, three runs per variant, in parallel -------------------------------------
for spec in $VARIANTS; do
    v="${spec%%:*}"
    for run in A B C; do
        d="$OUT/$v/$run"
        rm -rf "$d"; mkdir -p "$d"
        find "$SRC" -maxdepth 1 \( -name 'dicth_*' -o -name 'graph_*' \) -exec cp -t "$d/" {} +
        cp "$SRC/projects.txt" "$d/"
        cp "$OUT/$v/seeds.csv" "$d/training_data.csv"
        # Clear every KG/ablation variable first, then set only what this run needs.
        case "$run" in
            A) kg="" ;;
            B) kg="KG_ENABLED=1 KG_MODE=append" ;;
            C) kg="KG_ENABLED=1 KG_MODE=rerank KG_LAMBDA=0.5 KG_SOURCES=10" ;;
        esac
        timed "topfilter/$v/$run" "$d.log" env -u KG_ENABLED -u KG_MODE -u KG_ORIGINALS \
            -u KG_LAMBDA -u KG_SOURCES -u KG_NORMALISE -u NO_SEED_PREPEND \
            $kg TOPFILTER_SRC_DIR="$d" "$PY" topfilter/runner.py &
    done
done
wait
echo "done -> $OUT"
