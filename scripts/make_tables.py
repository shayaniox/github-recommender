"""Regenerate every metrics table in paper_artifacts/REPORT.md from the pipeline's outputs.

Tables are written between markers in the report, so they are never edited by hand:

    <!-- table:NAME -->
    ...generated...
    <!-- /table -->

NAME is one of: smoke, run_a:<dataset>, c_minus_a, timings. Sources: results_summary.csv,
significance.csv, config.json and smoke/smoke_fold1.csv in the artifact directory. The
thesis values are not used anywhere.

Usage: python scripts/make_tables.py [ARTIFACT_DIR]      (default paper_artifacts)
"""

import csv
import json
import re
import sys
from pathlib import Path

ART = Path(sys.argv[1] if len(sys.argv) > 1 else "paper_artifacts")
METRICS = ("success_rate", "precision", "recall", "catalog_coverage")
HEAD = {"success_rate": "success rate", "precision": "precision", "recall": "recall",
        "catalog_coverage": "catalog coverage"}


def read(name):
    with open(ART / name, newline="") as f:
        return list(csv.DictReader(f))


summary = {(r["run"], r["dataset"], int(r["N"]), r["metric"]): r for r in read("results_summary.csv")}
sig = {(r["comparison"], r["dataset"], int(r["N"]), r["metric"]): r for r in read("significance.csv")}
config = json.loads((ART / "config.json").read_text())
primary = config.get("primary_dataset")
datasets = sorted({k[1] for k in summary}, key=lambda d: (d != primary, d))


def run_a(ds):
    identical = all(summary[("A", ds, n, m)]["mean"] == summary[("B", ds, n, m)]["mean"]
                    for n in range(1, 11) for m in METRICS if ("B", ds, n, m) in summary)
    lines = [f"Run A{' (= run B at every N and metric)' if identical else ''}, "
             f"mean ± std over {summary[('A', ds, 1, METRICS[0])]['n_folds']} folds", "",
             "| N | " + " | ".join(HEAD[m] for m in METRICS) + " |",
             "|---|" + "---|" * len(METRICS)]
    for n in range(1, 11):
        cells = [f"{float(summary[('A', ds, n, m)]['mean']):.4f} ± "
                 f"{float(summary[('A', ds, n, m)]['std']):.4f}" for m in METRICS]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def c_minus_a():
    lines = ["| dataset | N | " + " | ".join(f"Δ{HEAD[m]} (p)" for m in METRICS) + " |",
             "|---|---|" + "---|" * len(METRICS)]
    for ds in datasets:
        for n in range(1, 11):
            cells = []
            for m in METRICS:
                r = sig[("A_vs_C", ds, n, m)]
                cells.append("0 (undefined)" if int(r["n_nonzero_diff"]) == 0 else
                             f"{float(r['mean_diff']):+.4f} ({float(r['p_value']):.4f})")
            lines.append(f"| {ds} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def smoke():
    rows = read("smoke/smoke_fold1.csv")
    cols = [("success_rate", 1), ("success_rate", 5), ("success_rate", 10),
            ("precision", 5), ("precision", 10), ("recall", 10)]
    lines = ["| Fold 1 (65 test projects) | projects scored | SR@1 | SR@5 | SR@10 | P@5 | P@10 | R@10 |",
             "|---|---|---|---|---|---|---|---|"]
    for case in dict.fromkeys(r["case"] for r in rows):
        t = {(r["metric"], int(r["N"])): r for r in rows if r["case"] == case}
        first = next(iter(t.values()))
        lines.append(f"| {case} | {first['projects_scored']} | " +
                     " | ".join(f"{float(t[c]['value']):.4f}" for c in cols) + " |")
    return "\n".join(lines)


def timings():
    lines = ["| job | wall-clock (s) |", "|---|---|"]
    for k, v in sorted(config["timings"].items()):
        lines.append(f"| {k} | {v['wall_clock_seconds']} |")
    return "\n".join(lines)


def render(name):
    if name.startswith("run_a:"):
        return run_a(name.split(":", 1)[1])
    return {"smoke": smoke, "c_minus_a": c_minus_a, "timings": timings}[name]()


report = ART / "REPORT.md"
text = report.read_text()
text, n = re.subn(r"<!-- table:(\S+) -->\n.*?<!-- /table -->",
                  lambda m: f"<!-- table:{m.group(1)} -->\n{render(m.group(1))}\n<!-- /table -->",
                  text, flags=re.S)
report.write_text(text)
print(f"regenerated {n} tables in {report}")
