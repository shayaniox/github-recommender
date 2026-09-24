# `append` vs `rerank` — what the knowledge graph does in the Topics Merger

Stage F can expand or re-order its output using `sedkgraph.csv`, behind `KG_ENABLED=1`. Two
strategies are available via `KG_MODE`, and they differ in one way that decides everything
else:

| | candidate set | can delete a correct topic? |
|---|---|---|
| `append` (default) | grows; **truncated first** when `KG_ORIGINALS` is set | **yes, by construction** |
| `rerank` | unchanged | no — only re-ordered |

This page works both through on one example. A companion to
[`CF-WALKTHROUGH.md`](CF-WALKTHROUGH.md), whose six-repository toy fold it continues, and to
[`ARCHITECTURE.md`](ARCHITECTURE.md) Stage F.

The repository and the CF scores are invented; **the graph edges and every expansion below are
real**, read from `sedkgraph.csv`. Reproduce with `scripts/kg_mode_demo.py`.

---

## The input

`acme/chatbot-server`, merged list as the Topics Merger leaves it — 5 CNBN seeds pinned at 2.0,
then 14 CF-scored candidates:

```
python 2.0  api 2.0  docker 2.0  rest-api 2.0  server 2.0
async 1.200  bot 1.166  chat 1.166  llm 0.983  nlp 0.907
django 0.799  web 0.799  html 0.799  machine-learning 0.729  jupyter 0.729  pandas 0.729
cli 0.723  rust 0.723  tool 0.723
```

Ground truth: `python, fast, api, rest-api, machine-learning, chat, bot, docker`.

## Baseline — KG disabled

```
 1. python   2.000  HIT      2. api      2.000  HIT
 3. docker   2.000  HIT      4. rest-api 2.000  HIT
 5. server   2.000           6. async    1.200
 7. bot      1.166  HIT      8. chat     1.166  HIT
 9. llm      0.983          10. nlp      0.907

P@10 = 0.60      R@10 = 6/8 = 0.750
```

## `append`, no `KG_ORIGINALS` cap — invisible

`_expand_with_kg` keeps every original, then appends one-hop neighbours at a score below the
minimum original score (`floor - 1.0 + encounters / (top_encounters + 1)`), so that a consumer
sorting by score sees the same order as one reading positionally.

There are 19 originals, so the first neighbour lands at **rank 20**. The evaluated window is 10.

```
identical to baseline        P@10 = 0.60      R@10 = 0.750
```

This is section 4, item 7: with default settings the graph contributes nothing to the reported
numbers. Not wrong — just never reached.

## `append` with `KG_ORIGINALS=5` — visible, and it costs

The only way to get neighbours inside the window is to cap the originals. That cap is a
truncation, not a re-ordering:

```python
originals = originals[:self.kg_originals]      # the other 14 are DISCARDED
```

`bot` and `chat` - both correct, both found by the collaborative filter - are deleted, and their
slots go to one-hop neighbours of the five CNBN seeds:

```
 1. python   2.000  HIT      2. api      2.000  HIT
 3. docker   2.000  HIT      4. rest-api 2.000  HIT
 5. server   2.000
 6. communication         1.667    <- neighbour of api and rest-api
 7. ubuntu                1.667
 8. programming-language  1.333
 9. software-development  1.333
10. platform              1.333

P@10 = 0.40      R@10 = 4/8 = 0.500
```

Precision down a third, recall down a third, zero hits contributed by the graph. This is what
"append can only reach the evaluated window by pushing correct CF predictions out of it" means
in practice.

## `rerank` — order changes, set does not

`_kg_rerank` introduces no topic. It takes the top `KG_SOURCES` (default 10) entries as sources,
asks the graph which *existing* candidates neighbour them, and adds a bonus:

```
score'(t) = score(t) + lambda * encounters(t) / max_encounters        lambda = KG_LAMBDA = 0.5
```

```
 1. python   2.000  HIT      2. api      2.000  HIT
 3. docker   2.000  HIT      4. rest-api 2.000  HIT
 5. server   2.000
 6. web      1.299           <- 0.799 + 0.5*(2/2); neighbours both bot and rest-api
 7. async    1.200
 8. bot      1.166  HIT      <- pushed down one rank, still inside the window
 9. chat     1.166  HIT
10. django   1.049           <- 0.799 + 0.5*(1/2); neighbour of python

P@10 = 0.60      R@10 = 6/8 = 0.750
```

Nothing was lost. `web` and `django` were promoted and are both wrong, so the re-ordering did not
help either - but `bot` and `chat` shifted down one rank instead of disappearing.

One quirk: `kg.expand` excludes its own sources, so the top `KG_SOURCES` candidates all receive a
bonus of exactly 0. `rerank` is strictly a promotion mechanism for the tail.

Its guarantee is also bounded. It cannot remove a correct topic from the candidate list, but by
promoting a wrong one it can still push a correct one past cut-off *k*. It just cannot do the
catastrophic version.

## Summary

| configuration | candidate set | P@10 | R@10 |
|---|---|---|---|
| baseline, KG off | 19 | 0.60 | 0.750 |
| `append`, uncapped | 19 + neighbours, past the window | 0.60 | 0.750 |
| `append`, `KG_ORIGINALS=5` | 5 + neighbours, **14 discarded** | **0.40** | **0.500** |
| `rerank` | 19, re-ordered | 0.60 | 0.750 |

## Why neither strategy helps

Look at what the graph actually returns for these sources: `communication`, `ubuntu`,
`programming-language`, `software-development`, `platform`.

SED-KGraph's relation types include `is-a` and `is-used-in-field` - the first rows of the file are
`2d,is-a,field` and `2d,is-used-in-field,graphics`. A one-hop neighbour is therefore usually an
**abstraction** of the source topic, not a sibling of it. Repository topics are siblings.

That is the structural reason [`METRICS.md`](METRICS.md) reports the KG never improving the hybrid
at any cut-off, even at 100% seed coverage. It is a property of the graph's semantics, not a
tuning problem that `KG_LAMBDA` can fix. A graph of topic *co-occurrence* rather than topic
*taxonomy* is what this merge step would need.
