"""Write every metric of one or more TopFilter runs to a Markdown file, from the run's own files.

A run is a TopFilter dataset directory after runner.py has finished (it holds projects.txt,
Round<k>/ and Results/). For each run the report gives:

    - provenance: mode (hybrid if SRB_ files exist, else CF-only half mode), projects, fold
      sizes, commit and dirty files (git_sha.txt / git_status.txt), wall-clock and peak RSS
      (timings.tsv), all looked up in the run directory and up to two levels above it
    - mean (sample std, ddof=1) over the ten folds of every metric the validator writes, at
      every cut-off it writes (N = 1..20; catalog coverage N = 1..10)
    - the per-fold values at N = 1, 5, 10 and 20

Nothing is typed in by hand; re-running regenerates the file.

Usage: python scripts/metrics_report.py --out FILE [--title T] LABEL=RUN_DIR [LABEL=RUN_DIR ...]
"""

import argparse
import math
import statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="TopFilter runs: all metrics")
ap.add_argument("runs", nargs="+", metavar="LABEL=RUN_DIR")
a = ap.parse_args()

FOLDS = range(1, 11)
DEFINITIONS = """\
## Metric definitions

Every value is a mean over the test repositories of a fold, and then the unweighted mean over the
ten folds. A *hit* is a recommended topic that is in the repository's held-back ground truth.

| column | file | definition |
|---|---|---|
| SR@N | `SR_Round<k>` | share of test repos with ≥1 hit in the top N |
| SR≥2/3/4@N | `SR_STAR_Round<k>` | share with at least 2 / 3 / 4 hits in the top N (column 1 of the file is the ≥1 case, identical to SR@N) |
| P@N, R@N | `PRC_Round<k>` | precision = hits / N; recall = hits / \\|ground truth\\| |
| nDCG@N | `nDCG_Round<k>` | binary-relevance DCG over the top N divided by the ideal DCG |
| Entropy@N | `Entropy_Round<k>` | Shannon entropy (natural log) of how often each topic appears across all top-N lists of the fold; higher = recommendations spread over more topics |
| EPC@N | `EPC_Round<k>` | novelty: rank-discounted mean of (1 − popularity) over the hits, popularity = how many of the fold's lists contain the topic ÷ the maximum |
| Cov@N | `Catalog<k>` | distinct topics recommended in the top N across the fold ÷ all topics (N = 1..10) |
| HR@20 | `Recall_Round<k>` | share of test repos with ≥1 hit in the top 20 (equals SR@20) |
| SRb@N, Pb@N, Rb@N | `SRB_`/`PRCB_Round<k>` | hybrid runs only: the same, scored on the merged list with the 5 CNBN seeds counted once (the numbers `paper_artifacts/REPORT.md` cites) |

`EPC@20` in `Results/` is not reported: it is an unlabelled aggregate (EPC and entropy values
interleaved per fold, `validator.py`), not a metric.

**Worked example** (the toy repo of `docs/KG-MODES.md`). `acme/chatbot-server` has 8 ground-truth
topics, and its top 10 holds hits at ranks 1, 2, 3, 4, 7, 8:

- SR@10 = 1, because there is at least one hit. SR≥4@10 = 1, because there are 6 hits.
- P@10 = 6/10 = 0.60 and R@10 = 6/8 = 0.75.
- P@5 = 4/5 = 0.80 and R@5 = 4/8 = 0.50.
- nDCG@10 = (1/log₂2 + 1/log₂3 + 1/log₂4 + 1/log₂5 + 1/log₂8 + 1/log₂9) ÷ (sum of 1/log₂(i+1) for i = 1..8) = 3.210 / 3.953 = 0.812.

A fold's value is this averaged over its ~595 (10k) or ~65 (644) test repos.

**CF-only versus hybrid ground truth.** In CF-only (half mode) a repo shows the first half of its
own topics and the second half is the ground truth, so |ground truth| ≈ half its topics. In the
hybrid the repo shows 5 CNBN-predicted topics and *all* its own topics are the ground truth. The
two protocols answer different questions; their numbers are not directly comparable.
"""


def series(path):
    """N -> list of values. Accepts 'N<TAB>v...' files and one-value-per-line files (line i = N=i)."""
    out = {}
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        v = line.split("\t")
        if len(v) == 1:
            out[i] = [float(v[0])]
        else:
            out[int(v[0])] = [float(x) for x in v[1:]]
    return out


def find_up(d, name, levels=2):
    for p in [d, *list(d.parents)[:levels]]:
        if (p / name).exists():
            return p / name
    return None


def timing(d):
    t = find_up(d, "timings.tsv")
    if not t:
        return None
    # run_cf_only.sh labels its run 'topfilter_half'; run_experiment.sh 'topfilter/<variant>/<run>'.
    rel = str(d.relative_to(t.parent))
    want = "topfilter_half" if rel == "topfilter" else f"topfilter/{rel}"
    for line in t.read_text().splitlines():
        v = line.split("\t")
        if v[0] == want:
            return v[1:]
    return None


def fmt(xs):
    if not xs:
        return ""
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else float("nan")
    return f"{m:.4f} ({s:.4f})"


def run_section(label, d):
    res = d / "Results"
    hybrid = (res / "SRB_Round1").exists()
    n_proj = sum(1 for x in (d / "projects.txt").read_text().splitlines() if x.strip())
    step = math.ceil(n_proj / 10)
    sizes = [min(step, n_proj - (k - 1) * step) for k in FOLDS]
    scored = [len(list((d / f"Round{k}" / "Recommendations").iterdir())) for k in FOLDS]

    # metric name -> N -> [value per fold]
    cols = {}

    def add(name, fname, col, fold):
        p = res / fname
        if not p.exists():
            return
        for n, vals in series(p).items():
            cols.setdefault(name, {}).setdefault(n, []).append(vals[col])

    for k in FOLDS:
        add("SR", f"SR_Round{k}", 0, k)
        for j, name in ((1, "SR≥2"), (2, "SR≥3"), (3, "SR≥4")):
            add(name, f"SR_STAR_Round{k}", j, k)
        add("P", f"PRC_Round{k}", 1, k)
        add("R", f"PRC_Round{k}", 0, k)
        add("nDCG", f"nDCG_Round{k}", 0, k)
        add("Entropy", f"Entropy_Round{k}", 0, k)
        add("EPC", f"EPC_Round{k}", 0, k)
        add("Cov", f"Catalog{k}", 0, k)
        add("HR@20", f"Recall_Round{k}", 0, k)
        if hybrid:
            add("SRb", f"SRB_Round{k}", 0, k)
            add("Pb", f"PRCB_Round{k}", 1, k)
            add("Rb", f"PRCB_Round{k}", 0, k)

    out = [f"## {label}", ""]
    sha = find_up(d, "git_sha.txt")
    status = find_up(d, "git_status.txt")
    dirty = status.read_text().split("\n") if status else []
    dirty = [x.strip() for x in dirty if x.strip()]
    t = timing(d)
    out += [
        f"- directory: `{d}`",
        f"- mode: {'hybrid (CNBN seeds → CF)' if hybrid else 'CF-only (TopFilter half mode, no CNBN)'}",
        f"- projects: {n_proj:,}; fold sizes {', '.join(map(str, sizes))}; "
        f"scored {'all' if scored == sizes else scored}",
        f"- commit: `{sha.read_text().strip()[:7]}`" if sha else "- commit: not recorded",
    ]
    if dirty:
        out.append(f"- uncommitted tracked changes at run time: {', '.join(f'`{x}`' for x in dirty)}")
    if t:
        mem = f", peak RSS {int(t[2]) / 1024:.0f} MB" if len(t) > 2 else ""
        out.append(f"- TopFilter wall-clock: {float(t[0]):,.1f} s{mem} (exit {t[1]})")
    out.append("")

    curve = [c for c in ("SR", "SR≥2", "SR≥3", "SR≥4", "P", "R", "nDCG", "Entropy", "EPC", "Cov",
                         "SRb", "Pb", "Rb") if c in cols]
    out.append("Mean (std) over the 10 folds:")
    out.append("")
    out.append("| N | " + " | ".join(curve) + " |")
    out.append("|---|" + "---|" * len(curve))
    for n in range(1, 21):
        out.append(f"| {n} | " + " | ".join(fmt(cols[c].get(n, [])) for c in curve) + " |")
    if "HR@20" in cols:
        out += ["", f"HR@20: {fmt(cols['HR@20'][1])}"]

    out += ["", "<details><summary>Per-fold values at N = 1, 5, 10, 20</summary>", ""]
    for n in (1, 5, 10, 20):
        show = [c for c in curve if n in cols[c]]
        out.append(f"N = {n}")
        out.append("")
        out.append("| fold | " + " | ".join(show) + " |")
        out.append("|---|" + "---|" * len(show))
        for i, k in enumerate(FOLDS):
            out.append(f"| {k} | " + " | ".join(f"{cols[c][n][i]:.4f}" for c in show) + " |")
        out.append("")
    out += ["</details>", ""]
    return out


lines = [f"# {a.title}", "",
         "Generated by `scripts/metrics_report.py` from each run's `Results/` files; "
         "do not edit by hand.", "", "Runs:", ""]
specs = [s.split("=", 1) for s in a.runs]
lines += [f"- [{lab}](#{lab.lower().replace(' ', '-')})" for lab, _ in specs]
lines += ["", DEFINITIONS, ""]
for lab, d in specs:
    lines += run_section(lab, Path(d))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{len(specs)} runs -> {a.out}")
