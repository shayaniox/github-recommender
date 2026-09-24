"""SED-KGraph access layer.

The graph is the artifact published with KGRec (Izadi, Nejati & Heydarnoori, EMSE 2023):
864 topics, 2,234 triples, 13 relation types, stored as `lhs,rel,rhs` rows.

Following that paper (p. 10), relation types and edge direction are ignored: the graph is
treated as untyped, undirected topic adjacency. Expansion is depth one — direct neighbours
only, no multi-hop traversal.

This module replaces the three near-identical copies that previously lived at the repo root,
in `cnbn/` and in `topfilter/`. Two of those parsed the CSV and printed the whole mapping as
an import-time side effect; nothing here runs on import.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
import csv

DEFAULT_GRAPH = "sedkgraph.csv"


def build_adjacency(csv_path=DEFAULT_GRAPH) -> Dict[str, Set[str]]:
    """Undirected adjacency over the topic graph.

    Both directions of every triple are unioned into the same set. The previous
    implementation built two separate dicts and merged them by assignment, which
    silently dropped the forward edges of any topic appearing on both sides of a
    triple (629 of 4,372 links).
    """
    adjacency: Dict[str, Set[str]] = {}
    with Path(csv_path).open(encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header and [column.strip().lower() for column in header[:3]] != ["lhs", "rel", "rhs"]:
            handle.seek(0)  # no header after all; do not lose the first triple
            reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            lhs, rhs = row[0].strip(), row[2].strip()
            if not lhs or not rhs or lhs == rhs:
                continue
            adjacency.setdefault(lhs, set()).add(rhs)
            adjacency.setdefault(rhs, set()).add(lhs)
    return adjacency


def expand(sources: Iterable[str], adjacency: Dict[str, Set[str]]) -> List[Tuple[str, int]]:
    """Depth-one neighbours of `sources`, ranked by encounter count.

    A candidate's score is the number of distinct source topics it is adjacent to, so a
    neighbour reached from several sources (e.g. from both a CNBN topic and a CF topic)
    outranks one reached from a single source.

    Topics already present in `sources` are never returned. Ordering is deterministic:
    encounter count desc, then degree desc, then name asc.
    """
    seeds = {topic for topic in sources if topic}
    encounters: Counter = Counter()
    for seed in seeds:
        for neighbour in adjacency.get(seed, ()):
            if neighbour not in seeds:
                encounters[neighbour] += 1
    return sorted(
        encounters.items(),
        key=lambda item: (-item[1], -len(adjacency.get(item[0], ())), item[0]),
    )


def csv_bidirectional_map(csv_file=DEFAULT_GRAPH, **_kwargs) -> Dict[str, List[str]]:
    """Backwards-compatible view of `build_adjacency` returning lists."""
    return {topic: sorted(neighbours) for topic, neighbours in build_adjacency(csv_file).items()}
