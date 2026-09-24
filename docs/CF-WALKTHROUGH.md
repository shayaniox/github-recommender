# TopFilter's collaborative filter — worked end to end

Stage F of the pipeline, traced with numbers you can check by hand. A companion to
[`WALKTHROUGH.md`](WALKTHROUGH.md), which covers all seven stages at summary level, and
[`ARCHITECTURE.md`](ARCHITECTURE.md), which describes the components. [`FOLDS.md`](FOLDS.md)
continues this same toy fold to show how its membership was decided in the first place.

To make the arithmetic fit on a page the fold here is shrunk to **6 training repositories** — a real
fold has ~579, and `num_of_neighbours = 20` rather than "all of them". The repository, its topics,
and the scores are **invented**; every formula, file path, and ordering rule is what the code does.

---

## Setup

**Test repo:** `acme/chatbot-server`

**Its real topics** (held out as ground truth): `python, fast, api, rest-api, machine-learning, chat,
bot, docker`

**CNBN's top-5 from `training_data.csv`:** `python, api, docker, rest-api, server`

**Training repositories**, owner-defined topics read from their `dicth_` files:

| | repo | topics |
|---|---|---|
| T1 | `bravo/llm-gateway` | python, api, docker, llm, chat, bot |
| T2 | `delta/slack-assistant` | python, bot, chat, rest-api, nlp |
| T3 | `echo/fastapi-starter` | python, api, rest-api, docker, async |
| T4 | `foxtrot/django-blog` | python, django, web, html, docker |
| T5 | `golf/rust-cli` | rust, cli, tool |
| T6 | `hotel/data-notebook` | python, jupyter, machine-learning, pandas |

---

## 1. Build the test repo's visible profile

`extract_EASE_dictionary` (`data_reader.py:154`) returns

```
1 → git://github.com/acme/chatbot-server
2 → #DEP#python    3 → #DEP#api    4 → #DEP#docker    5 → #DEP#rest-api    6 → #DEP#server
```

and writes all 8 real topics to `Round3/GroundTruth/acme___chatbot-server`.

There is a quiet trick here. `Graph(graph_file, testing_dict)` re-reads the repository's original
`graph_` file — `1#2, 1#3, … 1#9`, built from the *real* topics — but `_init_with_dictionary`
(`graph.py:39`) keeps only edges whose **both** endpoints are in the dictionary's key set, now
`{1…6}`. Edges `1#7, 1#8, 1#9` are dropped, and the surviving star `1#2…1#6` is relabelled through
the new dictionary onto the CNBN topics. The topology is recycled; the labels are swapped.

A consequence worth knowing: if CNBN emitted fewer than 5 topics, or the repository had fewer real
topics than there are seeds, edges silently vanish.

## 2. Merge into one global graph, compute IDF

`Graph.combine` (`graph.py:74`) folds all 6 training `graph_` files plus the relabelled test star
into one bipartite graph, interning topic strings to integer IDs via `_extract_key`. Seven repo
nodes have outlinks, so `number_of_projects = len(graph_edges) = 7`.

`idf = log(7 / freq)`:

| topic | in repos | freq | IDF |
|---|---|---|---|
| python | T1,T2,T3,T4,T6,+test | 6 | 0.154 |
| docker | T1,T3,T4,+test | 4 | 0.560 |
| api | T1,T3,+test | 3 | 0.847 |
| rest-api | T2,T3,+test | 3 | 0.847 |
| chat, bot | T1,T2 | 2 | 1.253 |
| server, llm, nlp, async, django, … | one each | 1 | 1.946 |

`python` is nearly worthless at 0.154; `server` — which only the test repo carries — gets maximum
weight.

## 3. Similarity against every training repository

For each pair, two vectors over the union of their topic sets, entry = that topic's IDF if the repo
has it and 0 otherwise, then `cosine_similarity`.

Test vector: `python 0.154, api 0.847, docker 0.560, rest-api 0.847, server 1.946`, ‖v₁‖ = 2.358.

| | shared with test | dot | ‖v₂‖ | **score as coded** | true cosine |
|---|---|---|---|---|---|
| T3 | python, api, rest-api, docker | 1.772 | 2.358 | **0.752** | 0.319 |
| T1 | python, api, docker | 1.055 | 2.825 | **0.409** | 0.158 |
| T2 | python, rest-api | 0.741 | 2.769 | **0.290** | 0.113 |
| T4 | python, docker | 0.337 | 3.420 | **0.119** | 0.042 |
| T6 | python | 0.024 | 3.374 | **0.008** | 0.003 |
| T5 | — | 0 | 2.999 | **0.000** | 0.000 |

The gap between the last two columns is divergence ⊗6 (`topic_similarity_calculator.py:165`, §4 item
6): the denominator is `sqrt(norm1*norm2)` where `norm1` and `norm2` are *already* magnitudes, so
every score is the true cosine multiplied by `sqrt(‖v₁‖·‖v₂‖)`. That factor differs per pair — 2.36
for T3, 2.84 for T4 — so it is not a monotone rescaling. Here the ranking happens to survive; a
near-tie would not. `similarity_calculator.py:183` has the correct form and is dead code.

Results are written to `Round3/Similarities/acme___chatbot-server`, sorted descending.
`get_most_similar_projects` then takes the **first 20 lines** — in this toy, all 6; rows 6–19 of the
matrix below stay all-zero and contribute nothing to either sum.

## 4. Build the 21 × |topics| user-item matrix

`lib_set` is the **sorted** union of all neighbour topics and the visible profile — sorted, not raw
set order, because set iteration varies with `PYTHONHASHSEED` and was reordering columns between
runs, breaking score ties differently each time. Nineteen columns:

```
api async bot chat cli django docker html jupyter llm machine-learning nlp pandas python rest-api rust server tool web
```

Rows 0–19 are the neighbours **in similarity rank order** — row 0 is T3, the most similar, not the
lowest project ID — with 1.0 wherever that neighbour has the topic. Row 20 is the active repository:
**1.0** for the 5 visible seeds and **−1.0** for the other 14, where −1 is the marker meaning *score
this one*.

```
row  neighbour  sim     |topics|  mean = Σrow/19
 0   T3         0.752      5        0.2632
 1   T1         0.409      6        0.3158
 2   T2         0.290      5        0.2632
 3   T4         0.119      5        0.2632
 4   T6         0.008      4        0.2105
 5   T5         0.000      3        0.1579
```

## 5. Mean-centered CF scoring

```
score(j) = 1.0 + Σₖ (r[k][j] − mean(row k)) · sim[k] / Σₖ sim[k]        Σ sim = 1.578
```

Worked for `chat`, which T1 and T2 carry:

```
k=0 T3  (0 − 0.2632) × 0.752 = −0.1978
k=1 T1  (1 − 0.3158) × 0.409 = +0.2798
k=2 T2  (1 − 0.2632) × 0.290 = +0.2137
k=3 T4  (0 − 0.2632) × 0.119 = −0.0313
k=4 T6  (0 − 0.2105) × 0.008 = −0.0018
k=5 T5  (0 − 0.1579) × 0.000 =  0.0000
                        Σ   = +0.2626   →  1 + 0.2626/1.578 = 1.166
```

All 14 candidates:

```
async             1.200     ← only in T3, but T3 is the top neighbour
bot               1.166
chat              1.166
llm               0.983
nlp               0.907
django web html   0.799
machine-learning  0.729
jupyter pandas    0.729
cli rust tool     0.723
```

Two things to read off this list:

- `async` outranks `chat` and `bot` on the strength of a single very similar neighbour. Mean-centering
  rewards a topic held by one high-`sim` row more than one held by two mid-`sim` rows.
- `machine-learning`, a genuine ground-truth topic, lands *below* `django`, because its only carrier
  is T6 at sim 0.008.

The `bot`/`chat` tie is broken by column order: Python's sort is stable and `recommendations` was
filled by ascending `j`, so `bot` — earlier alphabetically in `lib_set` — wins.

## 6. Topics Merger

The 5 CNBN topics are prepended at a **flat score of 2.0** (`recommendation_engine.py:133-147`), so
they hold ranks 1–5 regardless of what the CF thought. `Round3/Recommendations/acme___chatbot-server`:

```
#DEP#python    2.000      ← CNBN seeds, fixed score
#DEP#api       2.000
#DEP#docker    2.000
#DEP#rest-api  2.000
#DEP#server    2.000
#DEP#async     1.200      ← CF output
#DEP#bot       1.166
#DEP#chat      1.166
#DEP#llm       0.983
#DEP#nlp       0.907
...
```

With `KG_ENABLED=1` one further pass runs here: `append` tacks one-hop `sedkgraph.csv` neighbours on
behind everything, which with no `KG_ORIGINALS` cap puts them past the evaluated window (§4 item 7);
`rerank` instead boosts existing entries by KG agreement without introducing anything new.

[`KG-MODES.md`](KG-MODES.md) continues this exact list through both strategies, against the real
graph: `append` with `KG_ORIGINALS=5` drops P@10 from 0.60 to 0.40 by discarding `bot` and `chat`,
while `rerank` leaves the candidate set intact.

## 7. What the metrics see

Ground truth is the full 8-topic set, because `extract_EASE_dictionary` copied every `#DEP#` line
from `dicth_` into `GroundTruth/` — so correct seeds count as hits. At k=10, `python, api, docker,
rest-api, bot, chat` hit and `server, async, llm, nlp` miss:

```
P@10 = 0.60        R@10 = 6/8 = 0.75
```

Still missing: `fast`, which no neighbour carries, and `machine-learning`, at rank 12 and buried.

The collaborative stage earned its place — `bot` and `chat` were invisible to the text classifier and
came back from the neighbours. It also injected `async` and `llm`, which is the cost.
