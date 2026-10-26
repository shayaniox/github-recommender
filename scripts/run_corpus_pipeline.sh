#!/bin/sh
# End-to-end pipeline for a corpus: raw READMEs -> readme_8k_desc20 corpus -> preprocessing ->
# folds -> CNBN fold trees -> CNBN + TopFilter runs A/B/C -> collected results and tables.
#
# Everything is written under WORK; the 644-repository experiment's files are not touched.
#
# Usage: scripts/run_corpus_pipeline.sh SELECTION_CSV RAW_README_DIR WORK [DESCRIPTIONS_CSV]
#   SELECTION_CSV     repositories_test.csv format (repo,stars,forks,issues,topics...)
#   RAW_README_DIR    one '<owner>,<name>.txt' README per repository
#   DESCRIPTIONS_CSV  repo,description,languages; extracted from the Kaggle dump when omitted
#
# Optional: ORIG_SRC=repo_readme ORIG_FOLDS=custom_dataset/folds.json keeps the originally mined
# README text for the repositories listed in ORIG_FOLDS and takes every other README from
# RAW_README_DIR, so a corpus-size comparison against that experiment is not confounded by
# upstream README edits made since it was mined.
set -eu
SEL="$1"; RAW="$2"; WORK="$3"; DESC="${4:-}"
PY=.venv/bin/python
export PYTHONHASHSEED=0
mkdir -p "$WORK"

if [ -z "$DESC" ]; then
    DESC="$WORK/descriptions.csv"
    $PY scripts/extract_descriptions.py --only-mined --readme-dir "$RAW" --out "$DESC"
fi

CORPUS_SRC="--src $RAW"
if [ -n "${ORIG_SRC:-}" ]; then
    # rebuild_readme_corpus.py keeps the first copy of each file name, --src before --extra-src.
    rm -rf "$WORK/readme_orig"
    mkdir -p "$WORK/readme_orig"
    $PY -c 'import json, shutil, sys
for repo in json.load(open(sys.argv[1]))["order"]:
    shutil.copy2(sys.argv[2] + "/" + repo.replace("___", ",", 1) + ".txt", sys.argv[3])' \
        "$ORIG_FOLDS" "$ORIG_SRC" "$WORK/readme_orig"
    CORPUS_SRC="--src $WORK/readme_orig --extra-src $RAW"
fi

# The primary dataset's recipe (verified byte-identical on the 1,000-repository corpus).
rm -rf "$WORK/readme_8k_desc20"
$PY scripts/rebuild_readme_corpus.py $CORPUS_SRC --out "$WORK/readme_8k_desc20" \
    --max-bytes 8192 --descriptions "$DESC" --description-repeat 20 --with-languages

rm -rf "$WORK/topfilter" "$WORK/cnbn_export"
mkdir -p "$WORK/topfilter" "$WORK/cnbn_export"
$PY preprocessing/main.py --input "$SEL" --readme-dir "$RAW" \
    --topfilter-out "$WORK/topfilter" --cnbn-out "$WORK/cnbn_export/"

$PY scripts/make_folds.py --src "$WORK/topfilter" --out "$WORK/folds.json"
$PY cnbn/generate_datatset.py --csv "$SEL" --readme-dir "$WORK/readme_8k_desc20" \
    --output-dir "$WORK/cnbn_folds" --folds-json "$WORK/folds.json"

EXP_SRC="$WORK/topfilter" EXP_VARIANTS="readme_8k_desc20:$WORK/cnbn_folds" \
    sh scripts/run_experiment.sh "$WORK/experiments"

EXP_FOLDS_JSON="$WORK/folds.json" EXP_INPUT_CSV="$SEL" EXP_PRIMARY=readme_8k_desc20 \
    $PY scripts/collect_results.py "$WORK/experiments" "$WORK/artifacts"
