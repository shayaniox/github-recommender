"""Freeze the repository selection for an enlarged corpus, from the UNFILTERED Kaggle dump.

Selection rule (decided before any results are seen; do not change it after looking at metrics):
    - source: dataset/pretty-dataset.json (Kaggle "github-repository-metadata-with-5-stars")
    - not a fork (a fork carries its parent's README, which would leak across folds)
    - at least MIN_TOPICS topics
    - the top N by stars, ties broken by nameWithOwner ascending

repositories.csv is deliberately NOT used: cmd/convert_dataset only writes repositories that
have at least one topic in all_topics.csv (the SED-KGraph vocabulary), so sampling from it
would bias the corpus towards the knowledge graph whose effect is being measured.

Output, in the repositories_test.csv format the pipeline reads (repo,stars,forks,issues,topics...):
    <out>.csv            the selection
    <out>.rule.json      the rule, the source file's size/mtime, the counts and the CSV's sha256

The CSV is data and is not committed; the sha256 in the rule file identifies it.

Usage: python scripts/select_repos.py [--n 10000] [--min-topics 3] [--out enrich/top10k]
"""

import argparse
import csv
import hashlib
import heapq
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="dataset/pretty-dataset.json")
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--min-topics", type=int, default=3)
ap.add_argument("--out", default="enrich/top10k")
a = ap.parse_args()


def records(path):
    """Stream the pretty-printed top-level JSON array one object at a time."""
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("  {"):
                buf = [line]
            elif buf:
                buf.append(line)
                if line.startswith("  }"):
                    yield json.loads("".join(buf).rstrip().rstrip(","))
                    buf = []


seen = total = forks = few_topics = 0
heap = []  # min-heap of (stars, negated name key, row) keeping the top n
for r in records(a.src):
    total += 1
    if r.get("isFork"):
        forks += 1
        continue
    topics = [t["name"] for t in (r.get("topics") or [])]
    if len(topics) < a.min_topics:
        few_topics += 1
        continue
    seen += 1
    name = r["nameWithOwner"]
    # heap ordered so the element popped first is the one to drop: fewest stars, then the
    # alphabetically LAST name (ties keep the alphabetically first names).
    key = (r["stars"], [-ord(c) for c in name] + [1])  # +[1]: a prefix name ranks first
    row = [name, r["stars"], r["forks"], r["issues"], *topics]
    if len(heap) < a.n:
        heapq.heappush(heap, (key, row))
    elif key > heap[0][0]:
        heapq.heapreplace(heap, (key, row))

rows = [row for _, row in sorted(heap, key=lambda x: (-x[1][1], x[1][0]))]
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
with open(f"{out}.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(rows)
st = os.stat(a.src)
Path(f"{out}.rule.json").write_text(json.dumps({
    "source": a.src, "source_bytes": st.st_size, "source_mtime": st.st_mtime,
    "rule": {"exclude_forks": True, "min_topics": a.min_topics, "top_n_by_stars": a.n,
             "tie_break": "nameWithOwner ascending"},
    "counts": {"records": total, "forks_excluded": forks,
               f"fewer_than_{a.min_topics}_topics": few_topics, "eligible": seen,
               "selected": len(rows)},
    "min_stars_selected": rows[-1][1] if rows else None,
    "selection_sha256": hashlib.sha256(Path(f"{out}.csv").read_bytes()).hexdigest(),
}, indent=2))
print(f"{total} records, {seen} eligible, {len(rows)} selected -> {out}.csv")
