"""Runtime table for the thesis, from what the runs recorded. No value is typed in by hand.

    CF-only runs     <WORK>/timings.tsv (label, wall-clock s, exit, peak RSS kB),
                     <WORK>/corpus.txt (projects N), <WORK>/git_sha.txt, <WORK>/host.txt
    hybrid runs      <EXP>/timings.tsv from scripts/run_experiment.sh (label, wall-clock s, exit)
    README fetches   <readme_dir>/manifest.tsv: rows split into sessions wherever two consecutive
                     fetched_at_utc stamps are more than 10 min apart

TopFilter's cost is modelled as t = a * N^b. b is fitted on the two smallest CF-only runs and
then checked against every larger run, where the projection made from the smaller two is printed
next to the measured value.

Usage: python scripts/runtime_table.py --cf outpath/cfonly_644_run outpath/cfonly_10k \
           --hybrid outpath/experiments --manifest enrich/readme_raw/manifest.tsv
"""

import argparse
import datetime
import math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--cf", nargs="*", default=[])
ap.add_argument("--hybrid", nargs="*", default=[])
ap.add_argument("--manifest", nargs="*", default=[])
a = ap.parse_args()


def timings(work):
    out = {}
    for line in (Path(work) / "timings.tsv").read_text().splitlines():
        v = line.split("\t")
        out[v[0]] = v[1:]
    return out


def projects(work):
    for line in (Path(work) / "corpus.txt").read_text().splitlines():
        if line.startswith("projects "):
            return int(line.split()[1])


def sha(work):
    p = Path(work) / "git_sha.txt"
    return p.read_text().strip()[:7] if p.exists() else "?"


print("| job | N repos | wall-clock | peak RSS | commit |")
print("|---|---|---|---|---|")
cf = []
for w in a.cf:
    t, n = timings(w), projects(w)
    for label, (sec, status, *rss) in t.items():
        if status != "0":
            continue
        mem = f"{int(rss[0]) / 1024:.0f} MB" if rss else ""
        print(f"| CF-only {label} | {n:,} | {float(sec):,.1f} s | {mem} | {sha(w)} |")
        if label == "topfilter_half":
            cf.append((n, float(sec)))
for e in a.hybrid:
    for label, (sec, status, *_) in sorted(timings(e).items()):
        if status == "0":
            print(f"| hybrid {label} | 644 | {float(sec):,.1f} s | | {sha(e)} |")
for m in a.manifest:
    stamps = []
    for line in Path(m).read_text().splitlines()[1:]:
        ts = line.split("\t")[5]
        stamps.append(datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")))
    sessions, start = [], 0
    for i in range(1, len(stamps) + 1):
        if i == len(stamps) or (stamps[i] - stamps[i - 1]).total_seconds() > 600:
            sessions.append((start, i))
            start = i
    for s, e in sessions:
        n, sec = e - s, (stamps[e - 1] - stamps[s]).total_seconds()
        if n < 100:
            continue
        print(f"| README fetch {stamps[s]:%Y-%m-%d} | {n:,} | {sec:,.0f} s "
              f"({sec / max(n - 1, 1):.2f} s/repo, {3600 * (n - 1) / sec:,.0f}/h) | | |")

cf.sort()
if len(cf) >= 2:
    (n0, t0), (n1, t1) = cf[0], cf[1]
    b = math.log(t1 / t0) / math.log(n1 / n0)
    print(f"\nTopFilter scaling: t = {t0:.1f} s * (N / {n0})^{b:.2f}  "
          f"(fitted on N={n0:,} and N={n1:,}; the N^2 model predicts "
          f"{t0 * (n1 / n0) ** 2:,.0f} s at N={n1:,}, measured {t1:,.0f} s)")
    for n, t in cf[2:]:
        print(f"  N={n:,}: projected {t0 * (n / n0) ** b:,.0f} s, measured {t:,.0f} s")
    for n in (10000, 15000, 25000):
        print(f"  projection N={n:,}: {t0 * (n / n0) ** b / 3600:.1f} h")
