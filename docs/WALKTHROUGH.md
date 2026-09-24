# Pipeline walkthrough — one repository, end to end

A companion to [`ARCHITECTURE.md`](ARCHITECTURE.md). That document describes each stage in the
abstract; this one follows a **single repository through all seven stages**, showing what it looks
like on disk and in memory at every step.

The repository and its topics below are **invented** — no such repo exists in the dataset, and the
scores are illustrative rather than measured. The file paths, formats, rule files, and formulas are
real. For measured numbers from an actual run, see [`METRICS.md`](METRICS.md); for how the
ten-fold split is decided and picked up by each stage, see [`FOLDS.md`](FOLDS.md).

---

## The shape of the system

A **hybrid topic recommender**: given a repository's README, predict which GitHub topic labels it
should carry. Two recommenders chained together, not two competing baselines:

```
README ──► CNBN (content) ──► top-5 topics ──► TopFilter (collaborative) ──► Topics Merger ──► ranking
                                                    ▲                              ▲
                                            20 neighbour repos            sedkgraph.csv (optional)
```

The bridge is one file: `training_data.csv`, written by `cnbn/model.py:208` into
`outpath/TopFilter/` and read back by `data_reader.load_EASE_output`. Both branches take their
train/test split from one shared `custom_dataset/folds.json`.

A point worth fixing before the walkthrough starts: TopFilter reads **owner-defined** topics for
every *training* repository. The CNBN seeds substitute for the owner's topics only on the *test*
repository, whose real topics are held out as ground truth — which is the case our example follows.

---

## The repository

**`acme/chatbot-server`** — a README describing an async Python HTTP service that serves an LLM.

**Real GitHub topics, as scraped:**

```
python3, fast-api, restful-apis, ml, chat-bots, docker, to
```

---

## Stage A — ingestion (Go)

`cmd/export_repos` streams the 3.7 GB Kaggle dump with `json.Decoder` and keeps a repository if
*any* of its topics appears in the 863-entry allow-list `all_topics.csv` (`hasAllowedTopic`).

`ml` is in the allow-list → kept. `owner/name` is appended to `repos_list.txt`.

## Stage B — mining (`preprocessing/miner.py`)

PyGithub fetches the README → `repo_readme/acme,chatbot-server.txt`, and the repository lands in the
~1,000-row `repositories_test.csv`. That mined subset — not the 882k-row `repositories.csv` — is the
universe for everything downstream, which is what keeps the two branches over the same repositories.

## Stage C — normalization (`preprocessing/preprocessing.py:124`)

13 rule-driven rewriting passes, each backed by a headerless CSV in `preprocessing/rules/`. Applied
to our topics:

| Pass | Rule file | Effect on this repo |
|---|---|---|
| version strip | `topics_contains_version.csv` | `python3` → `python` |
| dash split | `split_dash_topics.csv` | `fast-api` → `fast`, `api`; `chat-bots` → `chat`, `bots` |
| singularize | `remove_plural_topics.csv` | `restful-apis` → `restful-api`; `bots` → `bot` |
| stopwords | `remove_stopwords_topic.csv` | `to` deleted |
| alias 1:1 | `replace.csv` | `restful-api` → `rest-api` |
| contractions | `contractions.csv` | `ml` → `machine-learning` |

Two sentinel values drive the rewriting primitive (`apply_csv`, `preprocessing.py:33`): `-1` deletes
the topic, `-2` keeps it unchanged, anything else expands one topic into N. The passes are bracketed
by a `stars >= 10` filter before and a **more than 4 topics** filter after.

**Result — 8 topics, survives the >4 filter:**

```
[python, fast, api, rest-api, machine-learning, chat, bot, docker]
```

With `KG_NORMALISE=1` the topics are additionally collapsed onto the SED-KGraph vocabulary.
`KG_NORMALISE_STAGE=pre` is the only setting where that measurably does anything, since the 765
hand-written alias rules have already merged most surface variants by the time `post` runs.

## Stage D — fan-out to two formats

The same normalized DataFrame is exported twice, in two incompatible shapes.

**CNBN tree** (`export_to_CNBN.py`) — the README is copied once per topic, directory name = label:

```
dataset_j/CNBN/python/acme,chatbot-server.txt
dataset_j/CNBN/rest-api/acme,chatbot-server.txt
dataset_j/CNBN/docker/acme,chatbot-server.txt        ... ×8
```

**TopFilter triplet** (`export_to_TopFilter.py`) — the CrossRec graph format, flat in
`outpath/TopFilter/`:

```
projects.txt                     acme___chatbot-server        ← line number = project ID

dicth_acme___chatbot-server      1<TAB>acme___chatbot-server
                                 2<TAB>#DEP#python
                                 3<TAB>#DEP#rest-api  ...

graph_acme___chatbot-server      1#2   1#3   1#4  ...         ← star: repo node → each topic node
```

The `#DEP#` prefix is inherited from CrossRec, where the items were software *dependencies* rather
than topics.

## Shared fold assignment

`scripts/make_folds.py` shuffles `projects.txt` deterministically, writes `custom_dataset/folds.json`,
and rewrites `projects.txt` in that order so TopFilter's existing window arithmetic selects the same
fold. ~579 train / 65 test per fold. **Re-run it after every run of `preprocessing/main.py`**, which
regenerates `projects.txt`.

Say `acme/chatbot-server` lands in the **test** set of fold 3.

## Stage E — CNBN, the content-based stage

`TfidfVectorizer(stop_words='english')` is fit over the 579 training READMEs, then `ComplementNB`.
For our test README, `predict_proba` paired with `clf.classes_` and sorted gives the **top 20**:

```
1. python  0.31    2. api    0.22    3. docker 0.14    4. rest-api 0.09
5. server  0.06    6. flask  0.04    7. machine-learning 0.03   ... (to 20)
```

Note the failure modes: `flask` is a false positive picked up from generic web-service vocabulary,
and `chat` / `bot` are missed because the README's wording overlaps heavily with plain API servers.
Recovering that kind of miss is exactly what the collaborative stage is for.

The **top 5** — `python, api, docker, rest-api, server` — are written to `training_data.csv`
(`;`-separated), copied both to `results/` for reporting and to `outpath/TopFilter/` for the next
stage.

KG expansion at this point is behind `KG_ENABLED=1` and exists only for the **CNBN-alone ablation**;
in the hybrid it would bury CNBN's own predictions under graph neighbours before the CF ever saw them.

## Stage F — TopFilter, the collaborative stage

Classic user-based collaborative filtering where **repositories are "users" and topics are "items"**.

> [`CF-WALKTHROUGH.md`](CF-WALKTHROUGH.md) works this stage through in full arithmetic — IDF table,
> similarity vectors, the user-item matrix, and the scoring sum — on a six-repository toy fold.

**1. Visible input vs. ground truth.** This applies to `acme/chatbot-server` because it is in the
fold's *test* set. The other ~579 repositories of the fold are training repos, and their
owner-defined topics are read whole from `dicth_` (`get_libraries`) — CNBN never touches them.

For our test repo, its 8 real topics are exactly what we are trying to predict, so they are held out
into `Round3/GroundTruth/acme___chatbot-server` and something else stands in:

- *Bayesian mode* (the hybrid path, and the only one a default run executes since `runner.main()`
  hardcodes `run(True)`): the visible profile is the 5 CNBN topics above and **nothing else**.
  `extract_EASE_dictionary` (`data_reader.py:154`) returns `{1: acme/chatbot-server}` plus those 5,
  and opens `dicth_` only to write the ground-truth file. So `python, fast, api, rest-api,
  machine-learning, chat, bot, docker` never enter the similarity vector or row 20 of the user-item
  matrix — the model is working from `python, api, docker, rest-api, server`.
- *Half mode*: the first 4 real topics stay visible and the last 4 become ground truth. This is the
  one path where a test repo's owner-defined topics are read.

**2. One global bipartite graph** built by merging every training repo's `graph_` file
(`Graph.combine`, `graph.py:74`), interning topic strings to integer IDs.

**3. IDF per topic**: `log(num_repos / num_repos_containing_topic)`. With 579 training repos,
`python` in 300 of them weighs ≈ 0.66; `langchain` in 6 weighs ≈ 4.57. Rare topics dominate the
similarity.

**4. Cosine similarity** between our IDF vector and each training repo's, over the union of their
topics, written to `Round3/Similarities/<repo>`:

```
bravo/llm-gateway       0.72    [python, api, docker, llm, chat, bot]
delta/slack-assistant   0.68    [python, bot, chat, rest-api, nlp]
echo/fastapi-starter    0.61    [python, api, rest-api, docker, async]
...                             top 20 kept
```

**5. Mean-centered user-based CF** (`recommendation_engine.py:73`): a `21 × |topics|` matrix where
rows 0–19 are the neighbours (1 = has topic) and row 20 is the active repo (1 = visible topic,
−1 = candidate to score), then

```
score(j) = 1.0 + Σₖ (r[k][j] − mean(r[k])) · sim[k] / Σₖ sim[k]
```

```
chat 1.41   bot 1.38   llm 1.22   nlp 1.05   async 0.94   machine-learning 0.88
```

**6. Topics Merger** (`recommendation_engine.py:133-147`): in bayesian mode the 5 CNBN topics are
**prepended at a fixed score of 2**, so they occupy ranks 1–5 above any CF output. That prepend *is*
the hybrid mechanism.

```
outpath/TopFilter/Round3/Recommendations/acme___chatbot-server

python 2.0 | api 2.0 | docker 2.0 | rest-api 2.0 | server 2.0
chat 1.41 | bot 1.38 | llm 1.22 | nlp 1.05 | async 0.94 | machine-learning 0.88
```

The collaborative stage recovered `chat`, `bot`, and `machine-learning`, which the text classifier
missed.

**Knowledge graph in the merger** (`KG_ENABLED=1`). This is where expansion belongs — it is the only
point that sees both branches' output at once. `KG_MODE` picks the strategy:

| Mode | Behaviour on this example |
|---|---|
| `append` (default) | One-hop neighbours of every listed topic are appended *behind* the originals, scored strictly below them: `rest-api` → `http`, `json`; `bot` → `nlp`, `automation`. `KG_ORIGINALS=<n>` decides whether they reach the evaluated cut-off at all — and they can only get there by pushing correct CF predictions out of it, so precision tends to drop wherever they become visible. |
| `rerank` | Introduces nothing new; boosts candidates the recommender already produced by their KG agreement with the head of the list: `score'(t) = score(t) + λ·encounters(t)/max_encounters`. Here `llm` neighbours both `machine-learning` and `chat` → 2 encounters → rises. Tuned by `KG_LAMBDA` (0.5) and `KG_SOURCES` (10). |

`kg.py:build_adjacency` discards the `rel` column and edge direction, treating the 864-topic,
2,234-triple SED-KGraph as untyped undirected adjacency with depth-one expansion, per the KGRec
paper.

## Stage G — evaluation

`validator.py` orchestrates `metrics.py`, writing twelve metric families into `Round3/Results/`:
success rate / SR@N, cumulative precision and recall swept over cut-offs 1–20, nDCG, EPC, entropy,
catalog coverage, and long tail. The `_b` variants (`SRB`, `PRCB`) score the hybrid path against the
held-out topics in `Round3/GroundTruth/`.

Against our 8-topic ground truth at k=10: 6 hits (`python, api, docker, rest-api, chat, bot`), with
`machine-learning` just outside at rank 11; `server`, `llm`, and `nlp` are false positives.

`scripts/compare_metrics.py <results_dir> [...]` diffs the run against the published HybridRec
Table 5 numbers and the TopFilter+ baseline.

---

## The two couplings that decide whether any of this means anything

Both are recorded as fixed bugs in [`ARCHITECTURE.md` §4](ARCHITECTURE.md#4-known-divergences-between-design-and-running-code):

1. **The shared `folds.json`.** Before it existed, the two branches split independently and the
   hybrid was largely measuring fold mismatch rather than recommendation quality.
2. **`_b` metrics scoring against `dicth_`** — the repository's *full* topic set, including the seed
   topics handed to the model — which inflated every hybrid number.
