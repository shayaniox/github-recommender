"""Canonicalise corpus topics onto the SED-KGraph vocabulary.

Motivation: the recommender's topic space (1,314 topics) is larger and more fragmented than
SED-KGraph's (864). Where a corpus topic is a surface variant of a graph topic
(`front-end` / `frontend`, `c-sharp` / `csharp`), collapsing the two densifies the
project-topic matrix and lets the graph's vocabulary act as a canonical form.

Only meaning-preserving rewrites are applied — case, separators, plural, and a small set of
suffix aliases. Token-splitting rules were deliberately rejected: they produce semantically
wrong merges such as `recurrent-neural-network` -> `network` and `game-streaming-server` ->
`game`, which corrupt the label space rather than normalise it.

Measured headroom on this corpus is small (8 of 1,314 topics merge), because
`preprocessing.preprocess()` already applies 765 hand-built alias rules upstream. Kept
configurable and off by default; see docs/METRICS.md.
"""

import re
from typing import Dict, Iterable, Optional, Set

DEFAULT_GRAPH = "sedkgraph.csv"


def _variants(topic: str) -> Iterable[str]:
    """Surface variants of `topic` that preserve its meaning."""
    base = topic.strip().lower()
    yield base
    yield base.replace("_", "-")
    yield base.replace("-", "")
    yield base.replace("_", "")
    yield base.replace("-", " ")
    if len(base) > 3 and base.endswith("s"):
        yield base[:-1]
    if len(base) > 4 and base.endswith("es"):
        yield base[:-2]
    stripped = re.sub(r"-(js|lang)$", "", base)
    if stripped != base:
        yield stripped


def canonical(topic: str, vocabulary: Set[str]) -> Optional[str]:
    """The canonical form of `topic`, or None if it has no counterpart in `vocabulary`.

    A topic already in the vocabulary is its own canonical form.
    """
    for variant in _variants(topic):
        if variant in vocabulary:
            return variant
    return None


def build_canonical_map(topics: Iterable[str], vocabulary: Set[str]) -> Dict[str, str]:
    """Map each topic to its canonical form, leaving unmatched topics untouched.

    Unmatched topics are deliberately kept rather than dropped: over half of this corpus's
    topics are absent from SED-KGraph and are still legitimate labels.
    """
    mapping = {}
    for topic in topics:
        mapping[topic] = canonical(topic, vocabulary) or topic
    return mapping


def load_vocabulary(csv_path=DEFAULT_GRAPH) -> Set[str]:
    from kg import build_adjacency
    return set(build_adjacency(csv_path))


def normalise_topic_list(topics: Iterable[str], vocabulary: Set[str]) -> list:
    """Canonicalise a topic list, dropping duplicates created by the merge, order preserved."""
    out = []
    seen = set()
    for topic in topics:
        canon = canonical(topic, vocabulary) or topic
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out
