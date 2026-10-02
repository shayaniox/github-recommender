#!/bin/sh
# CF alone on a corpus: TopFilter in half mode (TOPFILTER_HALF=1), no CNBN, KG disabled.
# Each test repository shows the first half of its own topics and the second half is ground truth.
#
# Repositories without a README are dropped by preprocessing exactly as in the hybrid pipeline,
# so the CF-only universe is the one scripts/run_corpus_pipeline.sh would use.
#
# Usage: scripts/run_cf_only.sh SELECTION_CSV RAW_README_DIR WORK
#   or:  scripts/run_cf_only.sh --dataset TOPFILTER_DIR WORK     (skip preprocessing and folds)
# Writes WORK/timings.tsv (label, wall-clock s, exit status, peak RSS kB), WORK/corpus.txt,
# WORK/git_sha.txt, WORK/cf_only_summary.csv (scripts/collect_cf_only.py) and WORK/METRICS.md,
# every metric of the run (scripts/metrics_report.py).
set -eu
PY=.venv/bin/python
export PYTHONHASHSEED=0

if [ "$1" = "--dataset" ]; then
    DATASET="$2"; WORK="$3"
else
    SEL="$1"; RAW="$2"; WORK="$3"
fi
mkdir -p "$WORK"
: > "$WORK/timings.tsv"
git rev-parse HEAD > "$WORK/git_sha.txt"
git status --porcelain --untracked-files=no > "$WORK/git_status.txt"
nproc > "$WORK/host.txt"; grep -m1 "model name" /proc/cpuinfo >> "$WORK/host.txt" || true
free -g | sed -n 2p >> "$WORK/host.txt"

timed() {  # timed <label> <logfile> <command...>
    label="$1"; log="$2"; shift 2
    status=0
    /usr/bin/time -f "%e %M" -o "$WORK/.time" "$@" > "$log" 2>&1 || status=$?
    set -- $(tail -1 "$WORK/.time")
    printf '%s\t%s\t%s\t%s\n' "$label" "$1" "$status" "$2" >> "$WORK/timings.tsv"
    rm -f "$WORK/.time"
    return "$status"
}

if [ -z "${DATASET:-}" ]; then
    rm -rf "$WORK/topfilter_src" "$WORK/cnbn_export"
    mkdir -p "$WORK/topfilter_src" "$WORK/cnbn_export"
    timed preprocessing "$WORK/preprocessing.log" $PY preprocessing/main.py --input "$SEL" \
        --readme-dir "$RAW" --topfilter-out "$WORK/topfilter_src" --cnbn-out "$WORK/cnbn_export/"
    timed make_folds "$WORK/make_folds.log" \
        $PY scripts/make_folds.py --src "$WORK/topfilter_src" --out "$WORK/folds.json"
    rm -rf "$WORK/cnbn_export"   # CNBN's per-topic README copies; not used here
    DATASET="$WORK/topfilter_src"
    grep -E "^#Repos|^#Topics" "$WORK/preprocessing.log" > "$WORK/corpus.txt"
fi

d="$WORK/topfilter"
rm -rf "$d"; mkdir -p "$d"
find "$DATASET" -maxdepth 1 \( -name 'dicth_*' -o -name 'graph_*' \) -exec cp -t "$d/" {} +
cp "$DATASET/projects.txt" "$d/"
echo "projects $(wc -l < "$d/projects.txt")" >> "$WORK/corpus.txt"

timed topfilter_half "$WORK/topfilter.log" env -u KG_ENABLED -u KG_MODE -u KG_ORIGINALS \
    -u KG_LAMBDA -u KG_SOURCES -u KG_NORMALISE -u NO_SEED_PREPEND \
    TOPFILTER_HALF=1 TOPFILTER_SRC_DIR="$d" $PY topfilter/runner.py

$PY scripts/collect_cf_only.py "$WORK"
$PY scripts/metrics_report.py --out "$WORK/METRICS.md" --title "CF-only run: all metrics" \
    "CF-only $(basename "$WORK")=$d"
echo "done -> $WORK"
