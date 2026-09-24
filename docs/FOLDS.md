# The fold assignment — where the split happens, and who reads it

One question answered in full: **the ten-fold split is decided once, before either recommender
runs, and long before the knowledge graph is touched.** This page traces where that decision is
made, how each of the two stages picks it up — by two completely different mechanisms — and where
KG expansion sits relative to it.

A companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) (Stages C–F), to
[`CF-WALKTHROUGH.md`](CF-WALKTHROUGH.md), whose six-repository toy fold is continued here, and to
[`KG-MODES.md`](KG-MODES.md), which covers what expansion does once it fires.

---

## 1. Ordering: folds first, KG expansion much later

```
Stage C/D  preprocessing/main.py     → outpath/TopFilter/projects.txt   (644 repos)
  ↓
FOLD SPLIT scripts/make_folds.py     → custom_dataset/folds.json        ← once, up front
  ↓
Stage E    cnbn/main.py              → per-fold NB train/predict        ┐ KG expansion
Stage F    topfilter/runner.py       → per-fold CF + Topics Merger      ┘ lives in here
```

`scripts/make_folds.py` is a standalone pre-step that runs straight after preprocessing and looks
at nothing but repository **names**. It hashes each `owner___name` into one of ten buckets
(`make_folds.py:49-51`):

```python
def fold_of(name):
    digest = hashlib.sha1(f"{args.seed}:{name}".encode()).hexdigest()
    return int(digest[:8], 16) % N_SPLITS
```

Hashing rather than shuffling is deliberate: a list shuffle re-permutes everything when the corpus
gains or loses one project, silently changing the test set and making two runs incomparable.

KG expansion never happens at dataset-construction time. It happens **inside a fold, at
recommendation time, per test repository**:

| Call site | Stage | Role |
|---|---|---|
| `cnbn/model.py:229` | E | CNBN-alone ablation only (`KG_ENABLED=1`) |
| `topfilter/recommendation_engine.py:283` | F | Topics Merger, `append` mode |
| `topfilter/recommendation_engine.py:244` | F | Topics Merger, `rerank` mode |

It is post-processing on an already-produced candidate list. In the hybrid path CNBN deliberately
does *not* expand, since doing so would bury its own predictions under neighbours before the CF
ever sees them (`ARCHITECTURE.md` Stage E, step 5).

### Why the ordering is leak-free regardless

`sedkgraph.csv` is a **static external artifact** — the 864-topic / 2,234-triple SED-KGraph
published with KGRec (Izadi, Nejati & Heydarnoori, EMSE 2023) — not something derived from this
corpus. It carries no information about which repositories are in which fold, so its position
relative to the split is a pipeline-design question, not a validity one.

What genuinely must come first, and does, is the split itself. Before `make_folds.py` existed the
two stages split independently — CNBN with a shuffled sklearn `KFold` over the raw CSV, TopFilter
with contiguous index windows over its own project list — and only **8 of 65** fold-1 test projects
had a CNBN seed, so the hybrid was measuring fold mismatch (`make_folds.py:3-6`).

---

## 2. What `make_folds.py` writes

Two artifacts, describing the same partition in two spellings:

```jsonc
// custom_dataset/folds.json
{
  "seed": 42, "n_splits": 10, "step": 65, "num_projects": 644,
  "order": ["freeCodeCamp___freeCodeCamp", ...],   // the projects.txt ordering
  "folds": [["a/b", "c/d", ...], ...]              // ten lists, owner/name
}
```

plus a rewrite of `outpath/TopFilter/projects.txt` into that same `order`
(`make_folds.py:79-80`). Note the spelling split: `projects.txt` carries `owner___name`,
`folds.json` carries `owner/name` (`make_folds.py:72`).

The `folds` lists are **contiguous windows of `order`**, sliced with
`step = math.ceil(len(ordered) / N_SPLITS)` (`make_folds.py:62`) — the same arithmetic TopFilter
uses — not the uneven hash buckets themselves. That is what makes the next two sections agree.

---

## 3. CNBN: reads `folds.json` directly

**One call site in the whole codebase: `cnbn/generate_datatset.py`.** `cnbn/main.py` and
`cnbn/model.py` never open it; they only read the directory tree it lays down.

The path is a default parameter (`generate_datatset.py:32`):

```python
def prepare_datasets(
    csv_path: Path,
    readme_dir: Path = Path("./repo_readme"),
    output_dir: Path = Path("custom_dataset/CNBN/folds_sync"),
    folds_path: Path = Path("custom_dataset/folds.json"),   # ← here
    folds=None,
):
```

Three uses inside the function:

| Line | What it takes from the JSON |
|---|---|
| `:45-49` | Hard-fails if absent — *"run `python scripts/make_folds.py` first so CNBN and TopFilter share one fold assignment"* |
| `:53-54` | `universe` = every name in every fold; used at `:59` to filter the mined CSV down to the ~644 repos TopFilter also has |
| `:70` | `test_names = set(assignment["folds"][fold - 1])` — the per-fold split itself |

Only the `folds` key is read. `order`, `step`, `seed` and `num_projects` are for TopFilter's window
arithmetic; CNBN ignores them and uses the explicit name lists.

`test_names` is then used exactly twice, and that is the whole of CNBN's split:

```python
# :90   the ground-truth list model.py reads back
test_file = output_dir / f"test_{fold}.txt"
...
# :108  the routing decision, one repository at a time
target_dir = test_dir if repo_name in test_names else train_dir
```

So `folds.json` is converted into **filesystem layout** — `folds_sync/train{k}/` vs
`folds_sync/test{k}/` plus `test_{k}.txt` — and from there the split exists only as directory
membership. `cnbn/main.py:38` points at `folds_sync`; `model.py` classifies whatever READMEs it
finds under those trees.

> **Stale trees were a real bug.** `train{k}`/`test{k}` are `shutil.rmtree`'d and rebuilt on every
> run (`generate_datatset.py:79-85`). They used to be reused with `mkdir(exist_ok=True)`, so when
> the fold assignment changed between runs, README copies from earlier folds stayed behind — that
> put 63 of 65 test repositories into the training set and inflated every downstream metric.

---

## 4. TopFilter: never reads `folds.json` at all

`grep -rn "folds.json" topfilter/` returns nothing. The assignment reaches Stage F **through the
line order of `projects.txt`**.

That was the design choice: rather than modify TopFilter's CV code to consume a JSON file,
`make_folds.py` rewrites `projects.txt` so the pre-existing contiguous-window arithmetic lands on
exactly the same folds.

**Write side** — the last thing `make_folds.py` does (`:79-80`):

```python
# Rewrite projects.txt so TopFilter's contiguous windows == these folds.
projects_file.write_text("\n".join(ordered) + "\n")
```

**Read side** — `runner.ten_fold_cross_validation` (`runner.py:74-85`), pure index arithmetic, no
file of names:

```python
step = math.ceil(num_of_projects / 10)
for i in range(10):
    testing_start_pos  = 1 + i * step
    testing_end_pos    = (i + 1) * step
    training_start_pos1, training_end_pos1 = 1, i * step
    training_start_pos2, training_end_pos2 = (i + 1) * step + 1, num_of_projects
```

`num_of_projects` is just a line count, `dr.get_number_of_projects(projects.txt)`
(`runner.py:53-54`). Those four numbers pass down to `TopicSimilarityCalculator` and
`RecommendationEngine`, which resolve them through `DataReader.read_project_list(filename,
start_pos, end_pos)` (`data_reader.py:47`) — a 1-based inclusive line-window read whose dict key
*is* the line number:

```
start_pos=131, end_pos=195  ->  {131: 'freeCodeCamp___freeCodeCamp', 132: ...}
```

Call sites of that window read: `similarity_calculator.py:44-56`,
`recommendation_engine.py:139/300/361`, `metrics.py:77/733`.

### The fragility this creates

The coupling is positional and unchecked. Nothing in `topfilter/` validates that `projects.txt` is
still in `make_folds.py` order, or that its line count matches `folds.json`'s `num_projects`. Two
ways it desyncs silently:

- **`preprocessing/main.py` regenerates `projects.txt`** in its own order. Re-run TopFilter without
  re-running `make_folds.py` and the windows select an arbitrary partition while CNBN's seeds still
  follow the old one — exactly the 8-of-65 mismatch the script was written to fix.
- **The `___` vs `/` spelling** means a mismatch never surfaces as a key error anywhere. It shows up
  only as bad numbers.

> **Re-run `python scripts/make_folds.py` after every run of `preprocessing/main.py`.**

---

## 5. The toy fold, end to end

Continuing the six training repositories of [`CF-WALKTHROUGH.md`](CF-WALKTHROUGH.md) — test repo
`acme/chatbot-server`, training T1–T6 — with 7 projects total, so `step = ceil(7/10) = 1`. The
membership below is invented to make fold 3 land on the test repo; every path and rule is what the
code does.

**Step 1 — split, once, for all seven repositories.** `make_folds.py` hashes names only; no topic,
README or graph edge is consulted:

```
custom_dataset/folds.json
  "order":  ["bravo___llm-gateway", "delta___slack-assistant", "acme___chatbot-server",
             "echo___fastapi-starter", "foxtrot___django-blog", "golf___rust-cli",
             "hotel___data-notebook"]
  "folds":  [["bravo/llm-gateway"], ["delta/slack-assistant"], ["acme/chatbot-server"], ...]
                                                                ^^^^^^^^^^^^^^^^^^^^ fold 3

outpath/TopFilter/projects.txt   (rewritten into "order")
  line 1  bravo___llm-gateway
  line 2  delta___slack-assistant
  line 3  acme___chatbot-server        ← fold 3 window
  line 4  echo___fastapi-starter
  ...
```

**Step 2 — CNBN picks it up from the JSON** (`generate_datatset.py:70,90,108`):

```
folds[2] ⊇ {"acme/chatbot-server"}  →  test_names
  acme/chatbot-server  → folds_sync/test3/<topic>/acme,chatbot-server.txt   (one copy per topic)
  bravo/llm-gateway    → folds_sync/train3/<topic>/bravo,llm-gateway.txt
  delta, echo, foxtrot, golf, hotel                                  → train3/
  folds_sync/test_3.txt  ← "acme/chatbot-server"
```

**Step 3 — TopFilter picks up the same split from the line order.** At `i=2`:
`testing_start_pos=3, testing_end_pos=3`, so `read_project_list` returns
`{3: 'acme___chatbot-server'}`, training = lines `1..2` and `4..7`. **Identical membership to
CNBN's, reached by a completely different mechanism.**

**Step 4 — only now does the knowledge graph appear.** Stage F builds the candidate list for that
one test repository: CNBN's five seeds `python, api, docker, rest-api, server` at score 2.0, then
the CF output `#DEP#async 1.200, #DEP#bot 1.166, #DEP#chat 1.166, …`. *Then*
`_expand_with_kg` fires (`recommendation_engine.py:283`), stripping `#DEP#` and feeding those labels
into `kg.expand` as sources. The fold boundary was fixed for all seven repositories before a single
neighbour was looked up, and the expansion is recomputed fresh for every test repo in every fold.

---

## 6. Summary

| | CNBN (Stage E) | TopFilter (Stage F) |
|---|---|---|
| reads `folds.json` | **yes** — `generate_datatset.py:32,45,53,70` | **no** — never opens it |
| carrier of the split | `folds_sync/train{k}/` vs `test{k}/` directories | line ranges in `projects.txt` |
| key spelling | `owner/name` | `owner___name` |
| fails loudly if stale | yes (`:45-49` raises) | no — silently scores the wrong window |
| KG expansion | `model.py:229`, ablation only | `recommendation_engine.py:244/283`, at merge time |
