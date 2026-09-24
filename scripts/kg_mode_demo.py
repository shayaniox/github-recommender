"""Reproduce the worked example in docs/KG-MODES.md.

Runs the Topics Merger's two knowledge-graph strategies over one fixed merged list and prints
the top 10 of each, with precision and recall against a fixed ground truth. The merged list and
its CF scores are the invented example from docs/CF-WALKTHROUGH.md; the graph edges and every
expansion come from the real sedkgraph.csv.

    python scripts/kg_mode_demo.py

The two functions below mirror RecommendationEngine._expand_with_kg and ._kg_rerank. They are
duplicated rather than imported because the engine's methods need a constructed engine, which
needs a populated outpath/TopFilter tree; keep them in step if that code changes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kg import build_adjacency, expand as kg_expand

DEP = "#DEP#"

MERGED = [
    # the 5 CNBN seeds, pinned by the Topics Merger at a flat 2.0
    ("#DEP#python", 2.0), ("#DEP#api", 2.0), ("#DEP#docker", 2.0),
    ("#DEP#rest-api", 2.0), ("#DEP#server", 2.0),
    # then the CF output, mean-centered user-based scores
    ("#DEP#async", 1.200), ("#DEP#bot", 1.166), ("#DEP#chat", 1.166),
    ("#DEP#llm", 0.983), ("#DEP#nlp", 0.907),
    ("#DEP#django", 0.799), ("#DEP#web", 0.799), ("#DEP#html", 0.799),
    ("#DEP#machine-learning", 0.729), ("#DEP#jupyter", 0.729), ("#DEP#pandas", 0.729),
    ("#DEP#cli", 0.723), ("#DEP#rust", 0.723), ("#DEP#tool", 0.723),
]

GROUND_TRUTH = {"python", "fast", "api", "rest-api", "machine-learning", "chat", "bot", "docker"}


def bare(label):
    return label[len(DEP):] if label.startswith(DEP) else label


def dedup(merged):
    seen, out = set(), []
    for label, score in merged:
        if label not in seen:
            seen.add(label)
            out.append((label, score))
    return out


def expand_with_kg(merged, adjacency, cap=None):
    """KG_MODE=append. Neighbours are appended BEHIND the originals, scored strictly below."""
    originals = dedup(merged)
    if cap is not None:
        originals = originals[:cap]          # <- the other originals are discarded outright
    seen = {label for label, _ in originals}
    neighbours = kg_expand([bare(l) for l, _ in originals], adjacency)
    if not neighbours:
        return originals
    floor = min(score for _, score in originals)
    top = neighbours[0][1]
    expanded = list(originals)
    for topic, encounters in neighbours:
        label = DEP + topic
        if label in seen:
            continue
        seen.add(label)
        expanded.append((label, floor - 1.0 + encounters / (top + 1.0)))
    return expanded


def kg_rerank(merged, adjacency, lam=0.5, num_sources=10):
    """KG_MODE=rerank. No topic is introduced; existing candidates are boosted by KG agreement."""
    candidates = dedup(merged)
    head = candidates[:num_sources]
    encounters = dict(kg_expand([bare(l) for l, _ in head], adjacency))
    if not encounters:
        return candidates
    top = max(encounters.values())
    boosted = [(l, s + lam * (encounters.get(bare(l), 0) / top)) for l, s in candidates]
    boosted.sort(key=lambda item: item[1], reverse=True)   # stable: ties keep prior order
    return boosted


def report(name, ranked, k=10):
    print(f"\n=== {name} ===")
    hits = 0
    for rank, (label, score) in enumerate(ranked[:k], 1):
        hit = bare(label) in GROUND_TRUTH
        hits += hit
        print(f"  {rank:2}. {bare(label):22} {score:+.3f}  {'HIT' if hit else ''}")
    print(f"  -> P@{k} = {hits / k:.2f}   R@{k} = {hits}/{len(GROUND_TRUTH)} = "
          f"{hits / len(GROUND_TRUTH):.3f}")


def main():
    adjacency = build_adjacency("sedkgraph.csv")
    report("baseline (KG disabled)", MERGED)
    report("append, no KG_ORIGINALS cap", expand_with_kg(MERGED, adjacency))
    report("append, KG_ORIGINALS=5", expand_with_kg(MERGED, adjacency, cap=5))
    report("rerank (KG_LAMBDA=0.5, KG_SOURCES=10)", kg_rerank(MERGED, adjacency))


if __name__ == "__main__":
    main()
