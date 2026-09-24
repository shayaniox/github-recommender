"""Collect the KG experiment (scripts/run_experiment.sh) into paper_artifacts/.

Reads, for every <dataset>/<run> under OUT_DIR, the validator's per-fold result files:

    Results/SRB_Round<k>    N <TAB> success rate@N          (hybrid path, >=1 hit in top N)
    Results/PRCB_Round<k>   N <TAB> recall@N <TAB> precision@N
    Results/Catalog<k>      N <TAB> catalog coverage@N      (written last by catalog_coverage)

Each file already holds that fold's mean over its test projects. Across folds this script
reports the UNWEIGHTED mean and the sample standard deviation (ddof=1) of the ten fold values.

Paired test: Wilcoxon signed-rank over the ten fold values, two-sided, zero differences dropped
(zero_method='wilcox'). 'exact' when there are no zero and no tied |differences|, otherwise the
normal approximation; the method used is recorded per row. All-zero differences make the test
undefined, recorded as such with an empty p-value.

Usage: python scripts/collect_results.py [OUT_DIR] [ARTIFACT_DIR]
"""

import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import numpy
import pandas
import scipy
import sklearn
import nltk
from scipy.stats import wilcoxon

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outpath/experiments")
ART = Path(sys.argv[2] if len(sys.argv) > 2 else "paper_artifacts")
# The variants, dataset directory and commit are read from what run_experiment.sh recorded in
# OUT_DIR, so the collector always describes the run it is collecting.
_variants = OUT / "variants.txt"
DATASETS = (dict(line.split(":", 1) for line in _variants.read_text().split())
            if _variants.exists() else {
                "readme_8k_desc20": "custom_dataset/CNBN/folds_desc20",
                "readme_8k": "custom_dataset/CNBN/folds_sync",
                "meta_only": "custom_dataset/CNBN/folds_meta"})
SRC = ((OUT / "src.txt").read_text().strip() if (OUT / "src.txt").exists()
       else "outpath/TopFilter")
FOLDS_JSON = Path(os.environ.get("EXP_FOLDS_JSON", "custom_dataset/folds.json"))
PRIMARY = os.environ.get("EXP_PRIMARY", "readme_8k_desc20")
# How each CNBN text corpus is built from the raw READMEs; each was verified byte-identical
# (1000/1000 files) against the corpus on disk.
_REBUILD = ".venv/bin/python scripts/rebuild_readme_corpus.py --src repo_readme --out {} "
CORPUS_RECIPES = {
    "readme_8k_desc20": _REBUILD.format("repo_readme_8k_desc20") + "--max-bytes 8192 "
        "--descriptions repo_descriptions.csv --description-repeat 20 --with-languages",
    "readme_8k": _REBUILD.format("repo_readme_8k") + "--max-bytes 8192",
    "meta_only": _REBUILD.format("repo_meta_only") + "--no-readme --descriptions "
        "repo_descriptions.csv --description-repeat 1 --with-languages --with-name",
}
RUNS = {
    "A": {"description": "baseline, KG disabled everywhere"},
    "B": {"description": "KG in Topics Merger, append", "KG_ENABLED": "1", "KG_MODE": "append"},
    "C": {"description": "KG in Topics Merger, rerank", "KG_ENABLED": "1", "KG_MODE": "rerank",
          "KG_LAMBDA": "0.5", "KG_SOURCES": "10"},
}
METRICS = ("success_rate", "precision", "recall", "catalog_coverage")
NS = range(1, 11)


def rows(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        v = line.split("\t")
        out[int(v[0])] = [float(x) for x in v[1:]]
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(root):
    """Hash of every (relative path, file hash) under root: identifies a fold tree."""
    h = hashlib.sha256()
    n = 0
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(sha256(p).encode())
            n += 1
    return h.hexdigest(), n


folds = json.loads(FOLDS_JSON.read_text())
fold_sizes = [len(f) for f in folds["folds"]]

raw, problems = [], []
for ds in DATASETS:
    clog = OUT / ds / "cnbn.log"
    ctext = clog.read_text() if clog.exists() else ""
    n_cerr = sum(1 for line in ctext.splitlines() if "- ERROR -" in line or ": ERROR -" in line)
    if n_cerr or "Completed round 10" not in ctext:
        problems.append(f"{ds}: CNBN log has {n_cerr} ERROR lines; "
                        f"round 10 completed: {'Completed round 10' in ctext}")
for ds in DATASETS:
    for run in RUNS:
        d = OUT / ds / run
        log = Path(f"{d}.log")
        text = log.read_text() if log.exists() else ""
        if "Error in main execution" in text:
            problems.append(f"{ds}/{run}: runner reported a fatal error")
        n_err = sum(1 for line in text.splitlines() if line.startswith("ERROR"))
        if n_err:
            problems.append(f"{ds}/{run}: {n_err} ERROR lines in {log}")
        for k in range(1, 11):
            rec = d / f"Round{k}" / "Recommendations"
            n_rec = len(list(rec.iterdir())) if rec.exists() else 0
            if n_rec != fold_sizes[k - 1]:
                problems.append(f"{ds}/{run} fold {k}: {n_rec} recommendation files, "
                                f"expected {fold_sizes[k - 1]}")
            try:
                sr = rows(d / "Results" / f"SRB_Round{k}")
                pr = rows(d / "Results" / f"PRCB_Round{k}")
                cat = rows(d / "Results" / f"Catalog{k}")
            except FileNotFoundError as e:
                problems.append(f"{ds}/{run} fold {k}: missing {e.filename}")
                continue
            for n in NS:
                vals = {"success_rate": sr[n][0], "precision": pr[n][1],
                        "recall": pr[n][0], "catalog_coverage": cat[n][0]}
                for m in METRICS:
                    raw.append({"run": run, "dataset": ds, "fold": k, "N": n,
                                "metric": m, "value": vals[m]})

ART.mkdir(parents=True, exist_ok=True)
with open(ART / "results_raw.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run", "dataset", "fold", "N", "metric", "value"])
    w.writeheader()
    w.writerows(raw)

index = {}
for r in raw:
    index.setdefault((r["run"], r["dataset"], r["N"], r["metric"]), {})[r["fold"]] = r["value"]

summary = []
for (run, ds, n, m), byfold in sorted(index.items()):
    vals = [byfold[k] for k in sorted(byfold)]
    summary.append({"run": run, "dataset": ds, "N": n, "metric": m, "n_folds": len(vals),
                    "mean": statistics.mean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else float("nan")})
with open(ART / "results_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run", "dataset", "N", "metric", "n_folds", "mean", "std"])
    w.writeheader()
    w.writerows(summary)

sig = []
for other in ("B", "C"):
    for ds in DATASETS:
        for n in NS:
            for m in METRICS:
                a, b = index.get(("A", ds, n, m)), index.get((other, ds, n, m))
                if not a or not b or sorted(a) != sorted(b):
                    continue
                ks = sorted(a)
                diff = [b[k] - a[k] for k in ks]
                # Fold values are means of k/65-type fractions written as decimals; treat
                # differences below 1e-12 as exact zeros rather than float noise.
                diff = [0.0 if abs(x) < 1e-12 else x for x in diff]
                nz = [x for x in diff if x != 0.0]
                row = {"comparison": f"A_vs_{other}", "dataset": ds, "N": n, "metric": m,
                       "n_folds": len(ks), "n_nonzero_diff": len(nz),
                       "mean_diff": statistics.mean(diff), "statistic": "", "p_value": "",
                       "method": "", "note": ""}
                if not nz:
                    row["note"] = "undefined: all paired differences are zero"
                else:
                    ties = len({round(abs(x), 12) for x in nz}) != len(nz)
                    method = "exact" if (len(nz) == len(diff) and not ties) else "approx"
                    res = wilcoxon([b[k] for k in ks], [a[k] for k in ks],
                                   zero_method="wilcox", alternative="two-sided", method=method)
                    row.update(statistic=res.statistic, p_value=res.pvalue, method=method)
                    if len(nz) < 6:
                        row["note"] = (f"only {len(nz)} non-zero differences; smallest attainable "
                                       f"two-sided exact p is {2 / 2 ** len(nz):.4f}")
                sig.append(row)
with open(ART / "significance.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["comparison", "dataset", "N", "metric", "n_folds",
                                      "n_nonzero_diff", "mean_diff", "statistic", "p_value",
                                      "method", "note"])
    w.writeheader()
    w.writerows(sig)

timings = {}
tfile = OUT / "timings.tsv"
if tfile.exists():
    for line in tfile.read_text().splitlines():
        label, secs, status = line.split("\t")
        timings[label] = {"wall_clock_seconds": float(secs), "exit_status": int(status)}

if (OUT / "git_sha.txt").exists():
    sha = (OUT / "git_sha.txt").read_text().strip()
    dirty = (OUT / "git_status.txt").read_text().split("\n")
else:  # runs made before run_experiment.sh recorded the commit
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.split("\n")
config = {
    "git_sha": sha,
    "primary_dataset": PRIMARY,
    "primary_dataset_note": "chosen by the project owner as the variant with the highest metrics",
    "git_modified_tracked_files": [l for l in dirty if l.strip()],
    "command": f"PYTHONHASHSEED=0 sh scripts/run_experiment.sh {OUT} && "
               f".venv/bin/python scripts/collect_results.py {OUT} {ART}",
    "seeds": {"fold_assignment_seed (scripts/make_folds.py)": folds["seed"],
              "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
              "other_randomness": "none: ComplementNB, TF-IDF and user-based CF are deterministic"},
    "tau_cnbn_seeds_per_test_repo": 5,
    "real_topics_shown_per_test_repo": 0,
    "k_cf_neighbours": 20,
    "cnbn_topics_written_per_repo": 20,
    "N_range_reported": [1, 10],
    "N_range_computed": [1, 20],
    "ground_truth": "each test repository's full normalised topic set (dicth_ file); no topic split",
    "catalog_coverage_denominator": "distinct topics across the fold's training repositories",
    "fold_mean": "unweighted mean of the 10 per-fold values; std = sample std (ddof=1)",
    "folds": {"n_splits": folds["n_splits"], "step": folds["step"],
              "num_projects": folds["num_projects"], "fold_sizes": fold_sizes},
    "datasets": {},
    "runs": {r: {**v, "cleared": ["KG_ENABLED", "KG_MODE", "KG_ORIGINALS", "KG_LAMBDA",
                                  "KG_SOURCES", "KG_NORMALISE", "NO_SEED_PREPEND"]}
             for r, v in RUNS.items()},
    "cnbn_env": {"KG_ENABLED": "unset", "CNBN_DOUBLE_TFIDF": "unset (single TF-IDF)",
                 "model": "ComplementNB"},
    "inputs_sha256": {p: sha256(p) for p in (
        os.environ.get("EXP_INPUT_CSV", "repositories_test.csv"), "repo_topics.csv", "sedkgraph.csv",
        str(FOLDS_JSON), f"{SRC}/projects.txt")},
    "timings": timings,
    "problems": problems,
    "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                 "nltk": nltk.__version__, "numpy": numpy.__version__,
                 "pandas": pandas.__version__, "scipy": scipy.__version__},
}
for ds, tree in DATASETS.items():
    digest, nfiles = tree_digest(tree)
    seeds = OUT / ds / "seeds.csv"
    config["datasets"][ds] = {
        "cnbn_fold_tree": tree, "fold_tree_sha256": digest, "fold_tree_files": nfiles,
        "corpus_recipe": CORPUS_RECIPES.get(ds),
        "seed_file_sha256": sha256(seeds) if seeds.exists() else None,
        "seed_file_rows": sum(1 for _ in open(seeds)) if seeds.exists() else None,
        "topfilter_projects": folds["num_projects"],
    }
(ART / "config.json").write_text(json.dumps(config, indent=2, default=str))

print(f"{len(raw)} raw rows, {len(summary)} summary rows, {len(sig)} tests")
print("problems:" if problems else "no problems detected")
for p in problems:
    print("  " + p)
