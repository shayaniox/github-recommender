#!/bin/sh
# Runtime cost of the knowledge graph in the Topics Merger: TopFilter runs A (KG off),
# B (KG_MODE=append) and C (KG_MODE=rerank) on identical inputs, one at a time.
#
# run_experiment.sh starts all its TopFilter runs in parallel, so its timings measure contention
# as much as KG. Here each run has the machine to itself, the A/B/C order rotates every
# repetition so slow drift (thermal, page cache) falls on all three equally, and one extra
# profiled run per mode measures the time spent inside the KG functions themselves.
#
# Usage: scripts/bench_kg.sh [REPS] [OUT] [SRC] [SEEDS]
#   defaults: 5  outpath/bench_kg  outpath/TopFilter  outpath/experiments/readme_8k_desc20/seeds.csv
# Writes OUT/timings.tsv (rep, run, wall-clock s, exit, peak RSS kB), OUT/profile_<run>.prof,
# OUT/git_sha.txt, OUT/host.txt; summarise with scripts/bench_kg_summary.py OUT.
set -eu
REPS="${1:-5}"; OUT="${2:-outpath/bench_kg}"
SRC="${3:-outpath/TopFilter}"; SEEDS="${4:-outpath/experiments/readme_8k_desc20/seeds.csv}"
PY=.venv/bin/python
export PYTHONHASHSEED=0

mkdir -p "$OUT"
: > "$OUT/timings.tsv"
git rev-parse HEAD > "$OUT/git_sha.txt"
git status --porcelain --untracked-files=no > "$OUT/git_status.txt"
nproc > "$OUT/host.txt"; grep -m1 "model name" /proc/cpuinfo >> "$OUT/host.txt" || true
printf '%s\n%s\n' "$SRC" "$SEEDS" > "$OUT/inputs.txt"

prepare() {  # prepare <dir>: a fresh copy of the dataset with the CNBN seeds, as run_experiment.sh
    rm -rf "$1"; mkdir -p "$1"
    find "$SRC" -maxdepth 1 \( -name 'dicth_*' -o -name 'graph_*' \) -exec cp -t "$1/" {} +
    cp "$SRC/projects.txt" "$1/"
    cp "$SEEDS" "$1/training_data.csv"
}

kg_env() {  # the exact KG settings of runs A, B and C in run_experiment.sh
    case "$1" in
        A) echo "" ;;
        B) echo "KG_ENABLED=1 KG_MODE=append" ;;
        C) echo "KG_ENABLED=1 KG_MODE=rerank KG_LAMBDA=0.5 KG_SOURCES=10" ;;
    esac
}

clean_env="-u KG_ENABLED -u KG_MODE -u KG_ORIGINALS -u KG_LAMBDA -u KG_SOURCES -u KG_NORMALISE -u NO_SEED_PREPEND"

for rep in $(seq 1 "$REPS"); do
    case $((rep % 3)) in 1) order="A B C" ;; 2) order="B C A" ;; 0) order="C A B" ;; esac
    for run in $order; do
        d="$OUT/work/$run"
        prepare "$d"
        status=0
        /usr/bin/time -f "%e %M" -o "$OUT/.time" env $clean_env $(kg_env "$run") \
            TOPFILTER_SRC_DIR="$d" $PY topfilter/runner.py > "$OUT/$run.rep$rep.log" 2>&1 || status=$?
        set -- $(tail -1 "$OUT/.time")
        printf '%s\t%s\t%s\t%s\t%s\n' "$rep" "$run" "$1" "$status" "$2" >> "$OUT/timings.tsv"
        echo "rep $rep $run: $1 s"
    done
done

for run in A B C; do
    d="$OUT/work/$run"
    prepare "$d"
    env $clean_env $(kg_env "$run") TOPFILTER_SRC_DIR="$d" \
        $PY -m cProfile -o "$OUT/profile_$run.prof" topfilter/runner.py > "$OUT/$run.profile.log" 2>&1
    echo "profiled $run"
done
rm -rf "$OUT/work" "$OUT/.time"
$PY scripts/bench_kg_summary.py "$OUT"
