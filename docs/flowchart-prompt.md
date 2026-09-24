# Claude Design prompt — github-recommender pipeline flowchart

Copy everything below the line into Claude Design. It is self-contained: you do not need the
repository open for it to work.

---

## PROMPT

Create a single-page architecture flowchart for a research codebase called **github-recommender** — a
hybrid recommender system that predicts GitHub repository topic labels from README text.

Target use: a **figure in a master's thesis**. It must stay legible printed at A4 portrait, and read
clearly in both light and dark. Landscape orientation, left-to-right flow.

### The story the diagram must tell

Two recommenders chained together, not two competing baselines. A content-based model (CNBN) reads
README text and predicts topics; those predictions are written to a file that a collaborative filter
(TopFilter) picks up and uses as seed input. `training_data.csv` is the bridge between them — **this
edge is the single most important thing in the diagram and should be visually emphasised.**

A second, quieter coupling matters almost as much: both recommenders read their train/test split from
**one shared fold assignment**, `custom_dataset/folds.json`. Draw it as a second cross-lane element so
it is clear the two stages are evaluated on the same repositories.

The knowledge graph (SED-KGraph) enters in two places, both optional and both governed by environment
flags: as a topic-vocabulary canonicaliser during normalization, and inside TopFilter's **Topics
Merger**, where it either appends neighbours or re-ranks existing candidates.

### Layout: seven labelled lanes, left to right

Lanes are stages, each tagged with its runtime:

1. **A · Ingestion** — `Go + PostgreSQL`
2. **B · Mining** — `Python + GitHub API`
3. **C · Normalization** — `Python + NLTK`
4. **D · Export** — `Python`
5. **E · CNBN — content-based** — `Python + scikit-learn`
6. **F · TopFilter — collaborative** — `Python`
7. **G · Evaluation** — `Python`

### Node inventory

Use **three distinct shapes**, consistently:

- **Rounded rectangle** = process (a script that runs)
- **Sharp rectangle / document shape** = data artifact (a file or directory on disk)
- **Cylinder** = database
- **Dashed outline** = external source outside the repo

**Lane A — Ingestion**
- `Kaggle dump — GitHub repos ≥5 stars` *(external, dashed)*
- `dataset/pretty-dataset.json` *(data, 3.7 GB — render this node visibly larger to convey scale)*
- `all_topics.csv` *(data, 863-topic allow-list)*
- `cmd/export_repos` *(process, Go)*
- `cmd/convert_dataset` *(process, Go)*
- `repos_list.txt` *(data, 22 MB)*
- `repositories.csv` *(data, 79 MB — caption: "882k rows; only the mined subset is used downstream")*
- `cmd/fetch_content` *(process, Go)*
- `PostgreSQL · repo_docs` *(cylinder — draw greyed out / at reduced opacity with the caption
  "vestigial: only `repo_docs` is written, and nothing downstream reads it")*

**Lane B — Mining**
- `preprocessing/miner.py` *(process)*
- `repo_readme/` *(data, one README per repo)*
- `repositories_test.csv` *(data, caption: "mined subset — the repo universe both branches share")*

**Lane C — Normalization**
- `preprocessing/main.py` *(process)*
- `stars ≥ 10 filter` *(process)*
- `preprocess() — 13 rewriting passes` *(process; add a small caption listing a few:
  "version-strip · de-hyphenate · singularize · lemmatize · alias-merge · expand contractions")*
- `preprocessing/rules/*.csv` *(data, 18 rule files — draw as a side-input feeding the passes)*
- `kg_normalise.py — KG vocabulary canonicalisation` *(process, dashed/optional outline; caption:
  "`KG_NORMALISE=1`; `KG_NORMALISE_STAGE=pre|post` places it before or after the rewriting rules")*

**Lane D — Export**
- `export_to_CNBN.py` *(process)*
- `export_to_TopFilter.py` *(process)*
- `dataset_j/CNBN/<topic>/*.txt` *(data, caption: "directory-per-topic README tree")*
- `outpath/TopFilter/` *(data, caption: "projects.txt · dicth_<repo> · graph_<repo>")*

**Fold assignment — straddling lanes E and F**
- `scripts/make_folds.py` *(process; caption: "one shared split — `step = ceil(N/10)` over a
  deterministically shuffled `projects.txt`, which it rewrites in fold order")*
- `custom_dataset/folds.json` *(data — **the second cross-lane node**; feeds both
  `generate_datatset.py` and TopFilter's fold windows)*

**Lane E — CNBN**
- `generate_datatset.py` *(process)*
- `custom_dataset/CNBN/folds_sync/{train,test}<k>/` *(data, caption: "all ten folds; 579/65 per fold,
  trees cleared and rebuilt on every run")*
- `TF-IDF vectorizer` *(process)*
- `ComplementNB classifier` *(process)*
- `top-20 predicted topics` *(data)*
- `knowledge-graph expansion (ablation only)` *(process, dashed/optional outline; caption:
  "`KG_ENABLED=1`; off by default — expansion belongs to the Topics Merger in lane F")*
- `training_data.csv` *(data — **the bridge node**; caption: "written twice: `results/` and
  `outpath/TopFilter/`, where TopFilter reads it")*
- `results/results_{1,7,9}_topics.csv` *(data)*

**Lane F — TopFilter**
- `runner.py` *(process)*
- `IDF-weighted cosine similarity` *(process, caption: "weight = log(N / repos-with-topic)")*
- `top-20 nearest-neighbour repos` *(data)*
- `user-based CF scoring` *(process, caption: "mean-centered; 5 seed topics pinned at score 2")*
- `Topics Merger` *(process; caption: "CNBN seeds → CF-ranked topics → optional KG step:
  `KG_MODE=append` adds one-hop neighbours, `KG_MODE=rerank` boosts existing candidates by
  `score' = score + λ·encounters/max`")*
- `sedkgraph.csv` *(data, caption: "SE knowledge graph · 864 topics · 2,234 `lhs,rel,rhs` triples,
  read as untyped undirected adjacency" — side-input, feeding the Topics Merger and, optionally,
  `kg_normalise.py` in lane C)*
- `Round<k>/Recommendations/` *(data, k = 1…10)*
- `Round<k>/GroundTruth/` *(data, k = 1…10)*

**Lane G — Evaluation**
- `validator.py + metrics.py` *(process)*
- `Results/*_Round{1..10}` *(data, caption: "Success rate · Precision/Recall · nDCG · EPC · Entropy ·
  Catalog coverage · Long tail")*
- `scripts/compare_metrics.py` *(process, caption: "diffs a run against HybridRec Table 5")*

### Edges

Main spine, left to right:

```
Kaggle dump → pretty-dataset.json → cmd/export_repos → repos_list.txt → miner.py
    → {repo_readme/, repositories_test.csv}
repositories_test.csv → preprocessing/main.py → stars ≥ 10 → preprocess()
    → {export_to_CNBN.py, export_to_TopFilter.py}
repo_readme/ → export_to_CNBN.py
export_to_CNBN.py → dataset_j/CNBN/
export_to_TopFilter.py → outpath/TopFilter/
repositories_test.csv → generate_datatset.py → folds_sync/ → TF-IDF → ComplementNB
    → top-20 topics → training_data.csv
outpath/TopFilter/ → runner.py → cosine similarity → top-20 neighbours → CF scoring
    → Topics Merger → Round<k>/Recommendations/ → validator.py → Results/*_Round<k>
Results/*_Round<k> → compare_metrics.py
```

Side-inputs, drawn as **thinner lines entering from above or below** so they don't compete with the
spine:

- `all_topics.csv` → `cmd/export_repos` and `cmd/convert_dataset`
- `preprocessing/rules/*.csv` → `preprocess()`
- `sedkgraph.csv` → `Topics Merger` (primary) and → `kg_normalise.py` (dashed, optional)
- `sedkgraph.csv` → `knowledge-graph expansion (ablation only)` (dashed, optional)

**The bridge edge — emphasise this one:**

```
training_data.csv ══► Topics Merger    (label: "top-5 CNBN topics seed the collaborative filter")
```

Draw it **thicker, in the accent colour, spanning the gap between lane E and lane F**. It is the edge
that makes the system hybrid. Use a soft background band behind lanes E+F labelled "hybrid
recommender" to group them.

**The shared-fold edges — emphasise these too, but one step below the bridge:**

```
outpath/TopFilter/projects.txt → scripts/make_folds.py → custom_dataset/folds.json
custom_dataset/folds.json → generate_datatset.py        (lane E)
custom_dataset/folds.json → runner.py fold windows       (lane F)
```

Draw the two outgoing edges in the accent colour but thinner than the bridge, with the label
"one split, both branches". `make_folds.py` also rewrites `projects.txt` in fold order — show this as
a short feedback arrow back onto `outpath/TopFilter/`.

Note that `cmd/convert_dataset → repositories.csv` no longer feeds lane E: CNBN reads
`repositories_test.csv`, the mined subset, so both branches see the same repositories. Draw
`repositories.csv` as a terminating branch in lane A rather than a long bottom edge into lane E.

### Annotations — numbered warning badges

The committed code diverges from this intended design in two places. Attach a **small numbered badge
(①–②) to the relevant node or edge**, and place a legend in the lower-right. Badges are the only
saturated/red element in the diagram — everything else stays muted, so the eye reads the architecture
first and the problems second.

| Badge | Attach to | Legend text |
|---|---|---|
| ① | `IDF-weighted cosine similarity` | Denominator is `sqrt(n1·n2)`, not `n1·n2` → unbounded values |
| ② | `Topics Merger` | With `KG_MODE=append` and no `KG_ORIGINALS` cap, neighbours land past the evaluated cut-off → KG has no effect on the reported metrics |

Badge ② deserves slightly more weight — it is the one that decides whether the knowledge graph
contributes anything measurable.

### Visual direction

- **Palette:** muted and desaturated — soft slate/indigo for process nodes, warm grey for data nodes,
  a single accent colour for the bridge and shared-fold edges, and red reserved *exclusively* for the
  warning badges. No rainbow lane colours; differentiate lanes with subtle background tint bands
  instead.
- **Optional stages:** nodes governed by an environment flag (`kg_normalise.py`, the CNBN-side KG
  ablation) get a dashed outline and slightly reduced opacity, so the default run path reads as a
  solid line through the figure.
- **Typography:** one sans-serif family. File paths, code identifiers and environment variables in
  monospace at a smaller size. Lane titles are the only large text.
- **Density:** this is an information-dense figure — prefer tight, deliberate spacing over airiness,
  but keep at least 16 px of clear space around every label. Never let an edge cross a node.
- **Edge routing:** orthogonal (right-angle) routing, not curves. Minimise crossings; where a crossing
  is unavoidable, use a small jump arc.
- **Legend:** lower-right, containing the shape key (process / data / database / external / optional)
  and the four numbered badges.
- **Title:** "github-recommender — hybrid topic recommendation pipeline", with a subtitle:
  "intended architecture; ⊗ marks where the committed code diverges".

Evaluation runs all ten folds; `Results/*_Round{1..10}` rather than a single round.

Do not invent nodes, files, or stages beyond the inventory above. Every label should match the
strings given here exactly, since they are real paths in the repository.
