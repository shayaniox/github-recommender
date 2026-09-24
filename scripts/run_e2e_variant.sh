#!/bin/sh
# End-to-end ten-fold run for one CNBN seed file.
#
# The collaborative stage reads a single training_data.csv, so a variant is evaluated by
# installing its seeds as the bridge and re-running TopFilter. Fold assignment, projects.txt,
# dicth_ and graph_ are untouched, so every variant is scored on exactly the same split.
#
# Usage: scripts/run_e2e_variant.sh <seeds.csv> <archive-dir>
set -e
SEEDS="$1"
OUT="$2"
PY=.venv/bin/python

[ -f "$SEEDS" ] || { echo "no such seed file: $SEEDS" >&2; exit 1; }

echo "=== $OUT: installing seeds from $SEEDS"
cp "$SEEDS" outpath/TopFilter/training_data.csv

$PY - <<'EOF'
import shutil
from pathlib import Path
for d in sorted(Path('outpath/TopFilter').glob('Round*')):
    shutil.rmtree(d)
print('cleared stale round directories')
EOF

$PY topfilter/runner.py
$PY - "$OUT" <<'EOF'
import shutil, sys
shutil.copytree('outpath/TopFilter/Results', sys.argv[1], dirs_exist_ok=True)
print('archived ->', sys.argv[1])
EOF
