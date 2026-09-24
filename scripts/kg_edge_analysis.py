"""What would have to change for SED-KGraph to help?

One-hop expansion from the CNBN seeds yields ~3,700 candidates containing ~27 correct
topics. This script decomposes that failure along the three axes that suggest a fix, and
compares the graph against a corpus co-occurrence graph built from the same training folds:

  A. vocabulary coverage      - can the graph even name the ground truth?
  B. relation type            - `kg.build_adjacency` discards `rel`; is-a parents are exactly
                                the topics developers omit as redundant, while works-with
                                neighbours are the ones they co-tag.
  C. neighbour degree         - hubs (`programming-language`, deg 55) are adjacent to
                                everything and carry no information.
  D. alternative graph        - PMI co-occurrence over the fold's TRAINING repos only,
                                which encodes tagging behaviour rather than semantics.

Usage:
    python scripts/kg_edge_analysis.py [--src outpath/TopFilter] [--graph sedkgraph.csv]
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

DEP = "#DEP#"


def load_triples(path):
    triples = []
    with Path(path).open(encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header and [c.strip().lower() for c in header[:3]] != ["lhs", "rel", "rhs"]:
            handle.seek(0)
            reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            lhs, rel, rhs = row[0].strip(), row[1].strip(), row[2].strip()
            if lhs and rhs and lhs != rhs:
                triples.append((lhs, rel, rhs))
    return triples


def adjacency(triples, keep=None):
    adj = defaultdict(set)
    for lhs, rel, rhs in triples:
        if keep and rel not in keep:
            continue
        adj[lhs].add(rhs)
        adj[rhs].add(lhs)
    return adj


def repo_topics(src):
    """Every repo's full topic set, from the dicth_ files TopFilter reads."""
    topics = {}
    for path in src.glob("dicth_*"):
        name = path.name[len("dicth_"):].replace("___", "/")
        found = set()
        for line in path.read_text(errors="ignore").splitlines():
            parts = line.split("\t")
            if len(parts) > 1 and parts[1].strip().startswith(DEP):
                found.add(parts[1].strip()[len(DEP):])
        if found:
            topics[name] = found
    return topics


def load_seeds(src, n=5):
    seeds = {}
    path = src / "training_data.csv"
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.strip().split(";")]
        if len(parts) > 1:
            seeds[parts[0].replace(".txt", "")] = [t for t in parts[1:] if t][:n]
    return seeds


def score(pairs, adj, degrees=None, max_degree=None):
    """Candidate precision/recall of one-hop expansion over (seeds, ground truth) pairs."""
    candidates = correct = relevant = 0
    reachable_projects = 0
    for seeds, truth in pairs:
        held_out = truth - set(seeds)
        if not held_out:
            continue
        cand = set()
        for seed in seeds:
            for nb in adj.get(seed, ()):
                if nb in seeds:
                    continue
                if max_degree is not None and degrees.get(nb, 0) > max_degree:
                    continue
                cand.add(nb)
        hit = cand & held_out
        candidates += len(cand)
        correct += len(hit)
        relevant += len(held_out)
        reachable_projects += 1 if hit else 0
    return {
        "candidates": candidates,
        "correct": correct,
        "precision": correct / candidates if candidates else 0.0,
        "recall": correct / relevant if relevant else 0.0,
        "projects_with_a_hit": reachable_projects,
    }


def row(label, s, projects):
    print(f"{label:<34} {s['candidates']:>8} {s['correct']:>8} "
          f"{s['precision']*100:>9.2f}% {s['recall']*100:>8.2f}% "
          f"{s['projects_with_a_hit']:>6}/{projects}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="outpath/TopFilter")
    parser.add_argument("--graph", default="sedkgraph.csv")
    parser.add_argument("--folds", default="custom_dataset/folds.json")
    args = parser.parse_args()

    src = Path(args.src)
    triples = load_triples(args.graph)
    full_adj = adjacency(triples)
    degrees = {t: len(n) for t, n in full_adj.items()}
    truth = repo_topics(src)
    seeds = load_seeds(src)

    pairs = [(seeds[r], truth[r]) for r in seeds if r in truth and seeds[r]]
    projects = len(pairs)
    if not projects:
        raise SystemExit("no (seed, ground-truth) pairs - run the pipeline first")

    all_truth = set()
    for _, t in pairs:
        all_truth |= t
    in_graph = sum(1 for t in all_truth if t in full_adj)

    print(f"projects: {projects}   distinct ground-truth topics: {len(all_truth)}\n")
    print("=== A. vocabulary coverage ===")
    print(f"ground-truth topics present in the graph: {in_graph}/{len(all_truth)} "
          f"({in_graph/len(all_truth)*100:.0f}%)")
    print(f"graph vocabulary: {len(full_adj)} topics, {len(triples)} triples\n")

    header = f"{'condition':<34} {'cands':>8} {'correct':>8} {'precision':>10} {'recall':>9} {'hit':>10}"
    print("=== B. relation type (rel is currently discarded) ===")
    print(header)
    print("-" * len(header))
    row("all relations (current)", score(pairs, full_adj), projects)
    by_rel = Counter(rel for _, rel, _ in triples)
    for rel, count in by_rel.most_common():
        adj = adjacency(triples, keep={rel})
        s = score(pairs, adj)
        if s["candidates"]:
            row(f"  {rel} ({count})", s, projects)

    print("\n=== C. hub suppression (neighbour degree cap) ===")
    print(header)
    print("-" * len(header))
    for cap in (None, 50, 20, 10, 5):
        label = "no cap (current)" if cap is None else f"degree <= {cap}"
        row(label, score(pairs, full_adj, degrees, cap), projects)

    hubs = sorted(degrees.items(), key=lambda kv: -kv[1])[:8]
    print("\n  highest-degree topics: " + ", ".join(f"{t} ({d})" for t, d in hubs))

    print("\n=== D. curated relation subset ===")
    print(header)
    print("-" * len(header))
    curated = {"provides-functionality", "is-used-in-field", "is-subset-of"}
    row("precision-positive relations", score(pairs, adjacency(triples, keep=curated)), projects)
    row("  + degree <= 20", score(pairs, adjacency(triples, keep=curated),
                                  degrees, 20), projects)

    print("\n=== E. corpus co-occurrence graph (PMI, training folds only) ===")
    folds_path = Path(args.folds)
    if not folds_path.exists():
        print("  (folds.json missing - skipped)")
        return 0
    assignment = json.loads(folds_path.read_text())

    print(header)
    print("-" * len(header))
    pmi_adjacencies = {}
    per_fold_pmi = {}
    for min_pmi, top_k in ((0.0, 10), (1.0, 10), (1.0, 5)):
        last_adj = {}
        cand_total = correct_total = relevant_total = hit_projects = 0
        scored = 0
        for fold_idx, fold in enumerate(assignment["folds"]):
            test_names = set(fold)
            train_topics = [truth[r] for r in truth if r not in test_names]
            if not train_topics:
                continue
            counts = Counter()
            pair_counts = Counter()
            for topics in train_topics:
                for t in topics:
                    counts[t] += 1
                for a in topics:
                    for b in topics:
                        if a < b:
                            pair_counts[(a, b)] += 1
            n = len(train_topics)
            co = defaultdict(list)
            for (a, b), c in pair_counts.items():
                if c < 2:
                    continue
                pmi = math.log((c / n) / ((counts[a] / n) * (counts[b] / n)))
                if pmi < min_pmi:
                    continue
                co[a].append((b, pmi))
                co[b].append((a, pmi))
            adj = {t: {nb for nb, _ in sorted(v, key=lambda kv: -kv[1])[:top_k]}
                   for t, v in co.items()}

            last_adj = adj
            per_fold_pmi[(min_pmi, top_k, fold_idx)] = adj
            fold_pairs = [(seeds[r], truth[r]) for r in test_names
                          if r in seeds and r in truth and seeds[r]]
            s = score(fold_pairs, adj)
            cand_total += s["candidates"]
            correct_total += s["correct"]
            relevant_total += sum(len(t - set(sd)) for sd, t in fold_pairs)
            hit_projects += s["projects_with_a_hit"]
            scored += len(fold_pairs)

        pmi_adjacencies[(min_pmi, top_k)] = last_adj
        print(f"{'PMI>=' + str(min_pmi) + ', top-' + str(top_k):<34} {cand_total:>8} "
              f"{correct_total:>8} {correct_total/cand_total*100 if cand_total else 0:>9.2f}% "
              f"{correct_total/relevant_total*100 if relevant_total else 0:>8.2f}% "
              f"{hit_projects:>6}/{scored}")

    # F. combining the two graphs, per fold so the co-occurrence side stays leakage-free.
    print("\n=== F. SED-KGraph combined with corpus co-occurrence ===")
    print(header)
    print("-" * len(header))
    for mode in ("union", "intersection", "sed-confirmed-by-corpus"):
        cand_total = correct_total = relevant_total = hit_projects = scored = 0
        for fold_idx, fold in enumerate(assignment["folds"]):
            pmi_adj = per_fold_pmi.get((0.0, 10, fold_idx))
            if pmi_adj is None:
                continue
            combined = {}
            keys = set(full_adj) | set(pmi_adj)
            for t in keys:
                a, b = full_adj.get(t, set()), pmi_adj.get(t, set())
                if mode == "union":
                    combined[t] = a | b
                elif mode == "intersection":
                    combined[t] = a & b
                else:  # SED edges that the corpus also supports, plus corpus edges
                    combined[t] = (a & b) | b
            test_names = set(fold)
            fold_pairs = [(seeds[r], truth[r]) for r in test_names
                          if r in seeds and r in truth and seeds[r]]
            s2 = score(fold_pairs, combined)
            cand_total += s2["candidates"]
            correct_total += s2["correct"]
            relevant_total += sum(len(t - set(sd)) for sd, t in fold_pairs)
            hit_projects += s2["projects_with_a_hit"]
            scored += len(fold_pairs)
        print(f"{mode:<34} {cand_total:>8} {correct_total:>8} "
              f"{correct_total/cand_total*100 if cand_total else 0:>9.2f}% "
              f"{correct_total/relevant_total*100 if relevant_total else 0:>8.2f}% "
              f"{hit_projects:>6}/{scored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
