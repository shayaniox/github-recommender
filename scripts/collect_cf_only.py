"""Collect a CF-only run (scripts/run_cf_only.sh) into WORK/cf_only_summary.csv.

Reads the validator's half-mode per-fold files under WORK/topfilter/Results:

    SR_Round<k>    N <TAB> success rate@N
    PRC_Round<k>   N <TAB> recall@N <TAB> precision@N
    Catalog<k>     N <TAB> catalog coverage@N

and reports, per N and metric, the unweighted mean and sample std (ddof=1) over the ten folds,
like scripts/collect_results.py does for the hybrid. Fails if any fold scored a different number
of repositories than projects.txt assigns to it.

Usage: python scripts/collect_cf_only.py WORK
"""

import csv
import math
import statistics
import sys
from pathlib import Path

WORK = Path(sys.argv[1])
TF = WORK / "topfilter"
NS = range(1, 11)


def rows(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        v = line.split("\t")
        out[int(v[0])] = [float(x) for x in v[1:]]
    return out


n_projects = sum(1 for line in (TF / "projects.txt").read_text().splitlines() if line.strip())
step = math.ceil(n_projects / 10)
problems, raw = [], []
for k in range(1, 11):
    expected = min(step, n_projects - (k - 1) * step)
    rec = TF / f"Round{k}" / "Recommendations"
    n_rec = len(list(rec.iterdir())) if rec.exists() else 0
    if n_rec != expected:
        problems.append(f"fold {k}: {n_rec} recommendation files, expected {expected}")
    sr = rows(TF / "Results" / f"SR_Round{k}")
    pr = rows(TF / "Results" / f"PRC_Round{k}")
    cat = rows(TF / "Results" / f"Catalog{k}")
    for n in NS:
        for m, v in (("success_rate", sr[n][0]), ("precision", pr[n][1]),
                     ("recall", pr[n][0]), ("catalog_coverage", cat[n][0])):
            raw.append((k, n, m, v))

with open(WORK / "cf_only_raw.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["fold", "N", "metric", "value"])
    w.writerows(raw)

summary = []
for m in ("success_rate", "precision", "recall", "catalog_coverage"):
    for n in NS:
        vals = [v for _, nn, mm, v in raw if nn == n and mm == m]
        summary.append([m, n, len(vals), statistics.mean(vals), statistics.stdev(vals)])
with open(WORK / "cf_only_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "N", "n_folds", "mean", "std"])
    w.writerows(summary)

for p in problems:
    print("PROBLEM:", p)
for m, n, _, mean, std in summary:
    if n in (1, 5, 10):
        print(f"{m}@{n}\t{mean:.4f} ({std:.4f})")
sys.exit(1 if problems else 0)
