> **Status of this file (2026-09-24).** This is a chronological log from earlier sessions and
> none of its numbers were produced from committed code or with recorded configurations; do not
> cite them. Two claims in it are wrong: (1) removing the `get_libraries` ground-truth overwrite
> in the `_b` metrics had **no** numeric effect - that set is identical to the GroundTruth file for
> 644/644 projects; (2) in bayesian mode the ground truth is the repository's **full** topic set,
> not a held-out split. The audited results are in `paper_artifacts/REPORT.md`.

# Pipeline run — 2026-09-17

First end-to-end execution of the full pipeline. Records the configuration, the metrics produced,
and what the numbers mean. Raw outputs are snapshotted in [`results/run_2026-09-17/`](../results/run_2026-09-17).

## Run configuration

| Setting | Value |
|---|---|
| Repo universe | `repositories_test.csv` — 1,000 mined repos |
| READMEs on disk | 1,000 (`repo_readme/`) |
| After normalization | 644 repos, 1,289 topics |
| CNBN classifier | `ComplementNB` over TF-IDF, **2,964 raw topic classes** |
| CNBN test fold | 100 repos (`KFold(n_splits=10, shuffle=True, random_state=42)`, fold 1) |
| TopFilter projects | 644; fold-1 test window = 65 projects |
| TopFilter neighbours | 20 |
| CNBN training time | 19.9 s (`timestamp.csv`) |
| Python | 3.8.10 · pandas 2.0.3 · scikit-learn 1.3.2 · nltk 3.9.1 |

Folds executed: **1 of 10**. Every runner still has the unconditional `break` (divergence #4),
so these are single-fold numbers, not cross-validated ones.

## Fixes applied before this run

| # | Fix |
|---|---|
| 1 | Implemented the undefined `get_topic_frequency_map_topics_as_list` in `preprocessing.py` |
| 2 | `cnbn/model.py` now writes `;`-separated EASE output and mirrors it to TopFilter's source dir |
| 3 | Generated the missing `repo_topics.csv` (ground truth was the literal `topic1,topic2,topic3`) |
| 4 | `topfilter/data_reader.py:127` — `#DEP# ` → `#DEP#`, matching every other producer |

Fix 4 was only discoverable *after* fix 2: while the bridge carried nothing, the prefix mismatch
was invisible.

---

## CNBN — content-based stage

100 test repos. Source: `results/run_2026-09-17/cnbn_results_{1,7,9}_topics.csv`.

| cut-off | precision | recall | top-1 correct | ≥1 hit | ≥3 hits |
|---|---|---|---|---|---|
| @1 | 41.00% | 7.62% | 0.410 | 41% | — |
| @7 | 18.86% | 19.62% | 0.410 | 78% | 14% |
| @9 | 16.22% | 21.58% | 0.410 | 80% | 18% |

The precision/recall crossover between @7 and @9 is the expected shape. `success@n` here means
"at least *n* correct predictions", not "a hit within the top *n*" (`cnbn/metrics.py:48`).

## TopFilter — collaborative stage

65 test projects. Source: `results/run_2026-09-17/topfilter_results_postfix/`.

`PRC`/`SR` = collaborative filtering alone. `PRCB`/`SRB` = bayesian mode, seeded with the top-5
CNBN predictions — the hybrid path.

| k | PRC prec | PRC rec | PRCB prec | PRCB rec | SR | SRB |
|---|---|---|---|---|---|---|
| 1 | 0.0045 | 0.0312 | 0.0045 | 0.0312 | 0.0312 | 0.0312 |
| 5 | 0.0353 | 0.0500 | 0.0353 | 0.0375 | 0.2031 | 0.2031 |
| 10 | 0.0790 | 0.0516 | 0.0635 | 0.0323 | 0.3438 | 0.3281 |
| 15 | 0.0804 | 0.0354 | 0.0790 | 0.0294 | 0.3438 | 0.3438 |
| 20 | **0.1183** | 0.0391 | **0.0346** | 0.0117 | **0.4219** | **0.0938** |

Other aggregates: recall-rate **0.4219** · catalog coverage@10 **0.0250** · nDCG max **0.0334** ·
EPC max **0.9261**.

### Effect of the `#DEP#` prefix fix

The pre-fix run's `Results/` was overwritten by the re-run, so these figures are **recorded here
from the session transcript and are not re-derivable from disk**. Re-run with
`data_reader.py:127` reverted to reproduce them.

| metric | before fix 4 | after fix 4 |
|---|---|---|
| PRC precision@20 | 0.0573 | 0.1183 |
| SR@20 | 0.3438 | 0.4219 |
| recall-rate | 0.3438 | 0.4219 |
| nDCG | **all zero** | nonzero (max 0.0334) |
| EPC | **all zero** | nonzero (max 0.9261) |

nDCG and EPC were structurally zero, not merely poor: recommendations held `#DEP# react` while
ground truth held `#DEP#react`, so exact-string comparison could never match.

---

## Interpretation

**The hybrid currently underperforms plain collaborative filtering.** At k=20 precision falls from
0.1183 to 0.0346 and success rate from 0.4219 to 0.0938. Two causes, both structural:

1. **The folds are unsynchronized.** CNBN splits 1,000 raw repos with shuffled `KFold`; TopFilter
   splits its own 644-project list by contiguous index range. Only **8 of 65** fold-1 test projects
   (12%) receive a CNBN seed. For the other 57, bayesian mode still pins five top slots at a fixed
   score of 2, displacing genuine CF results below the cut-off. The hybrid is paying the cost of
   seeding without receiving the benefit.

2. **The label spaces differ.** CNBN trains on 2,964 *raw* topic classes read straight from
   `repositories_test.csv`; TopFilter operates in the 1,289-topic *normalized* vocabulary. The
   `dataset_j/CNBN/` tree that `export_to_CNBN.py` produces is orphaned — nothing reads it. So even
   a matched seed may carry a topic string that no longer exists downstream.

The `PRCB`/`SRB` collapse specifically at k=20 (fine through k=15) is consistent with the rank
`+5` shift in the `_b` metric path (`metrics.py:148-196`) running past the available data at the
last cut-off — a metric-layer artifact on top of the two real problems.

**Highest-value next change:** drive both stages from one shared repo list and one shared fold
assignment. Until seed coverage is near 100%, the hybrid cannot be fairly compared against the CF
baseline, and no conclusion about CNBN's contribution is supportable.

## Caveats before citing any of this

- **Single fold, not 10-fold.** Divergence #4 is unfixed.
- **`results/results_*.csv` accumulate.** They are opened in append mode with the header written
  only when absent; the live files held 176,650 rows across 88,358 distinct repos from a 100-repo
  test set. The snapshot in `results/run_2026-09-17/` is the isolated 100 rows. Rotate these files
  before any future run.
- **The `stars >= 10` filter filters on issue count** (`preprocessing/main.py:18-21` mislabels the
  column order), so the 644-repo universe is not the one the code intends.
- **`_b` metrics score against ground truth that includes the model's own inputs**
  (divergence #5), inflating the bayesian numbers relative to the CF ones.


---

# KG ablation — 2026-09-17 (second session)

One-hop SED-KGraph expansion of the merged CNBN + CF topic list, neighbours ranked by
encounter count (how many source topics each is adjacent to). Raw outputs:
[`results/ablation/`](../results/ablation).

## Conditions

| Condition | Meaning |
|---|---|
| `kg_off` | Baseline. Topics Merger emits CNBN seeds + CF ranking, no KG. |
| `kg_all` | KG on; every original kept ahead of the neighbours. |
| `kg_top10` | KG on; originals truncated to 10, neighbours follow. |
| `kg_top5` | KG on; originals truncated to 5, neighbours follow. |

## Results vs the HybridRec paper (Table 5, p. 9719)

| N | TopFilter⁺ | HybridRec | kg_off | kg_all | kg_top10 | kg_top5 |
|---|---|---|---|---|---|---|
| **success rate@1** | 0.171 | 0.153 | 0.109 | 0.109 | 0.109 | 0.109 |
| **success rate@5** | 0.290 | 0.335 | 0.203 | 0.203 | 0.203 | 0.203 |
| **success rate@10** | 0.479 | 0.614 | **0.375** | 0.375 | 0.375 | 0.203 |
| **precision@10** | 0.095 | 0.127 | **0.063** | 0.063 | 0.063 | 0.037 |
| **recall@10** | 0.108 | 0.191 | **0.032** | 0.032 | 0.032 | 0.021 |

## Findings

**1. The KG does not improve the hybrid on this data.**
`kg_all` and `kg_top10` are **identical to the baseline at every cut-off** — the neighbours
land past rank 10, so they never enter the evaluated window. `kg_top5`, the only condition
where neighbours are actually visible (65/65 projects gain a new top-10 topic), is
**strictly worse** from N=6 on: success rate plateaus at 0.203 against the baseline's 0.375
because truncating the CF list to 5 discards correct predictions, and the KG neighbours
replacing them do not match ground truth.

**2. This is the evaluation problem KGRec itself documents, not necessarily a failure of the idea.**
KG neighbours are *semantically related* topics (`django` → `backend`), which a repository
owner plausibly should have tagged but did not. Scored against the owner's actual tags they
count as wrong. KGRec hit exactly this and reported that automated evaluation understates
KG-based recommendation badly — 30.0% automated vs 50.8% manual ASR@5 (`kgrec.pdf` Table 1,
p. 13) — which is why their headline results come from **manual expert evaluation of 50
sampled projects** and from FCR (failed-case ratio), not from ground-truth precision.
**Reproducing their claim requires their evaluation protocol, not just their dataset.**

**3. Absolute numbers sit well below the published HybridRec figures** (SR@10 0.375 vs 0.614).
Expected: 65 test projects on a single fold versus thousands averaged over three datasets,
and only 8 of 65 test projects receive a CNBN seed (see divergence #11), so what is measured
here is largely CF alone.

## Correction

An earlier run of this ablation appeared to show the KG improving every metric
(SR@10 0.234 → 0.344). **That was run-to-run noise, not a KG effect.** `lib_set` was built by
iterating a Python `set`, so column order — and therefore tie-breaking — varied with
`PYTHONHASHSEED` between processes: two *identical* baseline runs disagreed on the top-10 of
58/364 projects, the same magnitude as the apparent effect. Fixed by sorting
(`recommendation_engine.py`, `build_user_item_matrix`); two identical runs now differ on 0/364.
All numbers above are from the deterministic build.

## Reproducing

```bash
# baseline
KG_ENABLED=0 PYTHONPATH=.:topfilter python topfilter/runner.py
# KG on, neighbours visible from rank 6
KG_ENABLED=1 KG_ORIGINALS=5 PYTHONPATH=.:topfilter python topfilter/runner.py
# compare against the paper
python scripts/compare_metrics.py results/ablation/kg_off results/ablation/kg_top5 --labels off,top5
```


---

# Synced-fold ablation — 2026-09-17 (final)

Both stages now read one fold assignment (`custom_dataset/folds.json`, built by
`scripts/make_folds.py`). Seed coverage: **65/65 (was 8/65)**. Raw outputs in
[`results/ablation/`](../results/ablation); the pre-sync run is preserved in
[`results/ablation_unsynced/`](../results/ablation_unsynced).

## Two corrections to earlier numbers in this file

1. **Precision and recall were swapped** in every comparison table I produced before this
   section. `metrics.py` writes rows as `rank <TAB> recall <TAB> precision`; the comparison
   script read them as `rank, precision, recall`. Fixed in `scripts/compare_metrics.py`.
2. **An apparent KG improvement was run-to-run noise** (see the previous section):
   `lib_set` iterated a Python `set`, so tie-breaking varied with `PYTHONHASHSEED`. Fixed by
   sorting; two identical runs now differ on 0/364 projects.

## Baseline vs the HybridRec paper (Table 5, p. 9719)

| N | HybridRec | synced (this work) | unsynced (before) |
|---|---|---|---|
| success rate@1 | 0.153 | **0.312** | 0.109 |
| success rate@5 | 0.335 | **0.594** | 0.203 |
| success rate@10 | 0.614 | 0.594 | 0.375 |
| precision@1 | 0.212 | **0.312** | 0.044 |
| precision@5 | 0.117 | **0.175** | 0.038 |
| precision@10 | 0.127 | 0.088 | 0.032 |
| recall@5 | 0.090 | **0.131** | 0.037 |
| recall@10 | 0.191 | 0.131 | 0.051 |

Fold synchronization is the dominant effect: it roughly triples every metric. The synced
hybrid **beats the published HybridRec at N≤5** and falls behind by N=10 — consistent with
a 65-project single fold versus their thousands averaged over three datasets.

## KG effect (bayesian/hybrid path, precision@N)

| N | kg_off | kg_all | kg_top10 | kg_top5 |
|---|---|---|---|---|
| 5 | 0.1750 | 0.1750 | 0.1750 | 0.1750 |
| 10 | 0.0875 | 0.0875 | 0.0875 | 0.0875 |
| 15 | 0.1052 | 0.1052 | 0.1052 | 0.0594 |
| 20 | 0.1039 | 0.1039 | 0.0789 | 0.0461 |

Success rate@20: 0.750 / 0.750 / 0.703 / 0.609.

**The KG never improves the hybrid, at any cut-off, even with 100% seed coverage.**
`kg_all` is bit-identical to the baseline (neighbours land past rank 20). `kg_top10` and
`kg_top5` are the conditions where neighbours become visible, and both are strictly worse
from the rank where they enter — the neighbours displace correct CF predictions without
matching ground truth themselves.

This reproduces the pre-sync conclusion under much stronger conditions, so it is not an
artifact of missing seeds. See the previous section for why this most likely reflects
KGRec's own evaluation problem (semantically-correct topics absent from owner-assigned
ground truth) rather than a failure of the idea; KGRec reported 30.0% automated vs 50.8%
manual ASR@5 for exactly this reason.

## Additional fix applied

`topfilter/metrics.py` — the `_b` metric paths read the ground truth and then immediately
**overwrote it** with `get_libraries(dicth_<repo>)`, the repo's full topic set *including the
topics fed to the model as input*, making every pinned seed a guaranteed hit. Both
occurrences removed. (Numbers above are post-fix.)


---

# Why the KG fails — diagnostic measurements

Three measurements, run after both integration strategies degraded, explain the result.
Seeds are the real CNBN predictions from `training_data.csv` (top-5, as the engine uses).

| Measurement | Value |
|---|---|
| Ground-truth topics absent from SED-KGraph | **54%** (252/465) — ceiling on KG-derived recall is 46% |
| Ground truth at graph distance 1 from seeds | **5%** (24/465); distance 2 adds 15% |
| One-hop candidate precision | **0.6%** — 4,362 candidates contain 24 correct topics |
| **CF candidate correct, IS a KG neighbour** | **3.34%** (13/389) |
| **CF candidate correct, NOT a KG neighbour** | **6.22%** (178/2861) |
| **Lift from KG adjacency** | **0.54x — anti-correlated** |

The lift is measured on CF's own top-50 candidates, so it is independent of how the KG is
integrated: any adjacency-based scheme inherits it, including KGRec's full spreading-activation
model.

**Mechanism.** SED-KGraph encodes what topics *mean* (`django is-a framework`,
`django works-with python`). Developers tag for information, not completeness — a repo tagged
`django` does not also tag `framework` or `backend`, because that is redundant. Semantic
neighbours are therefore precisely the topics owners omit. The most-promoted candidates were
generic hubs: `programming-language` (degree 55, promoted for 30/65 projects), `web-development`
(77), `open-source` (48).

## Second integration strategy: rerank

Added after `append` failed. The KG introduces no new topics; it boosts candidates the
recommender already produced: `score'(t) = score(t) + λ · encounters(t)/max`.
Enable with `KG_MODE=rerank KG_LAMBDA=<λ> KG_SOURCES=<head size>`.

precision@20 vs λ: **0.104** (off) → 0.094 (λ=0.25) → 0.086 (λ=0.5) → 0.078 (λ=1.0).
Monotonically worse, consistent with the 0.54x lift.

## Correction recorded

An earlier version of this file reported the KG improving every metric. That was run-to-run
noise from `PYTHONHASHSEED`-dependent tie-breaking, since fixed. See the synced-fold section.


---

# KG normalisation — 2026-09-17

Third integration strategy: use SED-KGraph's 864 topics as a **canonical vocabulary** and
collapse corpus surface variants onto it (`front-end` -> `frontend`, `c-sharp` -> `csharp`),
rather than using the graph to add topics. Module: `kg_normalise.py`; enabled with
`KG_NORMALISE=1`; applied in `preprocessing.preprocess()` (after the hand-built rules) and in
`data_reader.load_EASE_output()` (so seeds share the space).

Only meaning-preserving rewrites are used — case, separators, plural, `-js`/`-lang` suffixes.
Token-splitting rules were rejected after measurement: they produced
`recurrent-neural-network` -> `network` and `game-streaming-server` -> `game`.

Fold assignment verified identical to the baseline run (65/65, seed 42), so the comparison is
like-for-like.

## Results

| metric | baseline | normalised | normalised + KG append | normalised + KG rerank |
|---|---|---|---|---|
| precision@3 | 0.2188 | **0.2240** | 0.2240 | 0.2240 |
| precision@5 | 0.1750 | **0.1781** | 0.1781 | 0.1781 |
| precision@10 | 0.0875 | **0.0891** | 0.0891 | 0.0891 |
| precision@20 | **0.1039** | 0.1031 | 0.0461 | 0.0867 |
| recall@10 | 0.1305 | **0.1331** | 0.1331 | 0.1331 |
| recall@20 | **0.2951** | 0.2925 | 0.1380 | 0.2480 |
| success rate@5 | 0.5938 | **0.6094** | 0.6094 | 0.6094 |
| success rate@20 | 0.7500 | 0.7500 | 0.6250 | 0.6719 |

## Reading these numbers honestly

**Normalisation is the first KG-derived change that improves anything** — roughly +2% relative
on precision, recall and success rate at N<=10, and neutral-to-slightly-negative at N=20.

**But the effect is at the resolution limit of the experiment.** Success rate is computed over
64 projects, so the smallest resolvable difference is 1/64 = 0.0156 — and the observed SR@5 gain
is exactly 0.0156, i.e. **one additional project**. With n=65 this is not a significant result
and must not be reported as one.

The mechanism is correspondingly small: the corpus vocabulary went from **1,289 to 1,288
distinct topics — a single merge**. The 765 hand-built alias rules already in
`preprocessing.preprocess()` (449 `replace`, 266 `abbr`, 50 `contractions`) have done this work
already; SED-KGraph adds almost nothing on top. Of the corpus's topics, 1,062 are absent from
the graph entirely, and they are not surface variants — they are concepts the graph lacks
(`uefi`, `iso`, `boot`, `usb`).

**KG expansion still hurts on top of normalisation**, exactly as before: append drops
precision@20 from 0.1031 to 0.0461, rerank to 0.0867. Normalising the vocabulary does not
rescue the expansion strategies, which is consistent with the 0.54x adjacency lift being the
binding constraint rather than vocabulary mismatch.

## Conclusion across all three strategies

| Strategy | Effect |
|---|---|
| **append** (add one-hop neighbours) | Harmful wherever visible |
| **rerank** (boost existing candidates by KG agreement) | Harmful, monotonically in lambda |
| **normalise** (canonical vocabulary) | +1 project at N<=10; below the resolution of this experiment |

The graph's value on this task is bounded by two measured facts: its adjacency is
anti-correlated with topic co-assignment (0.54x), and its vocabulary covers less than half of
the ground truth. Normalisation avoids the first problem — it never asserts that a neighbour is
a topic — which is why it is the only strategy that does not lose. It cannot yet demonstrate a
gain, because the corpus is already normalised by hand.


---

# CORRECTION: train/test leakage invalidated the normalisation results

`cnbn/generate_datatset.py` created its fold directories with `mkdir(exist_ok=True)` and copied
READMEs in without ever clearing them. Within a single fold assignment this is harmless — the
same files are rewritten. But **every time the fold assignment changed between runs, copies from
the previous assignment stayed behind.** By the time the normalisation experiments ran,
`train1/` held 643 repositories and `test1/` held 151, where the fold defines 580 and 65.

**63 of the 65 test repositories were also in the training set.** CNBN was being trained on the
READMEs it was tested on.

Scale of the inflation, same configuration before and after the fix:

| | leaked | leakage-free |
|---|---|---|
| success rate@1 | 0.7656 | **0.2344** |
| precision@5 | 0.5313 | **0.1500** |
| CNBN seed hits (of 5) | 2.68 | — |

Fixed: the fold directories are now removed and recreated on every build.

**Invalidated and quarantined** in `results/CONTAMINATED_do_not_cite/`: `norm_off`,
`norm_append5`, `norm_rerank`, `pre_off`, `pre_rerank`, and the numbers reported for them in the
"KG normalisation" section above.

**Unaffected**: the synced KG ablation (`kg_off`, `kg_all`, `kg_top10`, `kg_top5`, `rr_*`) ran
before any fold assignment changed, so its fold directories were built once. The core finding —
KG expansion never helps, with a 0.54x adjacency lift behind it — stands.

## Valid normalisation result (leakage-free, same 65 test projects)

Fold membership is now a stable hash of the project name (`scripts/make_folds.py`), so a corpus
that gains or loses a project no longer reshuffles the split. Baseline (644 repos) and pre-rules
normalisation (645 repos) share **65/65 test projects**.

| N | HybridRec | baseline | pre-rules normalised | normalisation effect |
|---|---|---|---|---|
| success rate@1 | 0.153 | 0.2344 | 0.2344 | **+0.0000** |
| success rate@5 | 0.335 | 0.5469 | 0.5469 | **+0.0000** |
| success rate@10 | 0.614 | 0.5469 | 0.5469 | **+0.0000** |
| precision@3 | 0.153 | 0.1823 | 0.1875 | +0.0052 |
| precision@5 | 0.117 | 0.1500 | 0.1500 | +0.0000 |
| precision@10 | 0.127 | 0.0750 | 0.0750 | +0.0000 |
| recall@5 | 0.090 | 0.1120 | 0.1124 | +0.0003 |
| recall@10 | 0.191 | 0.1120 | 0.1124 | +0.0003 |

**Normalisation before the hand-built rules has no effect either.** It moves precision@3 by
0.005 and recall by 0.0003; success rate is identical at every cut-off. The 30 extra topic merges
it achieves (3,128 -> 3,098) do not change any recommendation that the metrics can see.

**Versus HybridRec**, the implementation (with or without normalisation) is **better at N<=5**
— success rate +46% to +63%, precision +9% to +28%, recall +7% to +25% — and **worse from N=6
on**, down to -41% at N=10. That profile belongs to the baseline, not to the knowledge graph.

## Final position across all four strategies

| Strategy | Effect on the metrics |
|---|---|
| append one-hop neighbours | Harmful wherever visible |
| rerank by KG agreement | Harmful, monotone in lambda |
| normalise after the rules | +1 project; below resolution |
| normalise before the rules | No effect |

No use of SED-KGraph improved this recommender under ground-truth evaluation.


---

# FINAL NUMBERS — leakage-free, stable folds (2026-09-18)

All conditions on the **same 65 test projects**, 580/65 train/test with verified zero overlap,
fold membership by stable hash of project name. Raw outputs: `results/final/`.

## Baseline vs HybridRec

| N | HybridRec | this work | |
|---|---|---|---|
| success rate@1 | 0.153 | **0.2344** | +53% |
| success rate@5 | 0.335 | **0.5469** | +63% |
| success rate@10 | 0.614 | 0.5469 | −11% |
| precision@1 | 0.212 | **0.2344** | +11% |
| precision@5 | 0.117 | **0.1500** | +28% |
| precision@10 | 0.127 | 0.0750 | −41% |
| recall@5 | 0.090 | **0.1120** | +25% |
| recall@10 | 0.191 | 0.1120 | −41% |

Better at N≤5, worse from N=6 — expected for 65 projects on one fold against thousands averaged
over three datasets.

## KG effect — all four strategies

| N | baseline | append | top-10 | top-5 | rerank | normalise |
|---|---|---|---|---|---|---|
| precision@1 | 0.2344 | 0.2344 | 0.2344 | 0.2344 | 0.2344 | 0.2344 |
| precision@5 | 0.1500 | 0.1500 | 0.1500 | 0.1500 | 0.1500 | 0.1500 |
| precision@10 | 0.0750 | 0.0750 | 0.0750 | 0.0750 | 0.0750 | 0.0750 |
| success rate@10 | 0.5469 | 0.5469 | 0.5469 | 0.5469 | 0.5469 | 0.5469 |

**Identical to four decimal places at every evaluable cut-off.**

## Why N>10 is not reportable

Above N=10 the conditions diverge and the KG appears to win (precision@15: 0.017 -> 0.057). It is
an artifact. With the KG off, **52 of 65 recommendation lists are shorter than 15 items (median
5)** — CF frequently produces nothing beyond the seeds — and the baseline's success rate goes
**non-monotonic**: 0.547 at N=10, 0.141 at N=11. A cumulative measure cannot decrease, so the
metric is undefined there. The KG is filling empty slots, not finding correct topics; its own
precision at N=20 is 0.046. HybridRec reports N<=10, and so do we.

## Diagnostics re-measured on clean data

| Measurement | clean | previously reported | replicates? |
|---|---|---|---|
| Ground truth absent from SED-KGraph | **59%** | 54% | yes |
| Ground truth at graph distance 1 | **6%** | 5% | yes |
| One-hop candidate precision | **0.72%** (27/3,774) | 0.6% | yes |
| KG-adjacency lift | **1.06x** (n=41) | 0.54x | **NO** |

**The anti-correlation claim is withdrawn.** It was computed on a contaminated run with a much
larger candidate pool. On clean data the lift is 1.06x on 41 candidates — no effect in either
direction, and far too small a sample to conclude anything. The surviving explanation does not
need it: one-hop expansion yields 3,774 candidates containing 27 correct topics, so surfacing one
correct topic costs roughly 140 wrong ones. Ranking reorders a pool; it cannot enrich one.

## Final position

| Strategy | Effect at N<=10 |
|---|---|
| append one-hop neighbours | none |
| truncated append (top-10, top-5) | none |
| rerank by KG agreement | none |
| normalise (before or after the rules) | none |

No use of SED-KGraph changed this recommender's output within the evaluable range.


---

# TEN-FOLD, RANK-FIX — 2026-09-18 (supersedes everything above)

Full ten-fold cross-validation, leakage-free, with the evaluation rank-offset bug fixed.
All 644 repositories are scored exactly once, as the test fold of one of ten runs. Raw
outputs: `results/tenfold/tenfold_fixed/`.

## Headline: the implementation now beats HybridRec at every cut-off

| N | HybridRec | this work (10-fold) | |
|---|---|---|---|
| success rate@1 | 0.153 | **0.284** | +86% |
| success rate@5 | 0.335 | **0.576** | +72% |
| success rate@10 | 0.614 | **0.710** | +16% |
| precision@1 | 0.212 | **0.284** | +34% |
| precision@5 | 0.117 | **0.164** | +40% |
| precision@10 | 0.127 | **0.145** | +14% |
| recall@5 | 0.090 | **0.122** | +36% |
| recall@10 | 0.191 | **0.211** | +10% |

Per-fold spread (n=10): SR@10 = 0.709 ± 0.019 (se), sd 0.059, range 0.615–0.800.
**Every individual fold meets or exceeds HybridRec's 0.614**, so the margin at N=10 does not
depend on which fold is picked. P@10 = 0.145 ± 0.005, R@10 = 0.211 ± 0.009.

## The N>=6 collapse was an evaluation bug, not a property of the recommender

`metrics.success_rate_b` and `precision_recall_b` scored ranks 1..5 from `get_EASE_topic`
(the five CNBN seeds), then walked the recommendation file **from its first line** and placed
entry *k* at rank `k + 5`. But the Topics Merger writes those same five seeds at the head of
the recommendation file. Measured over the full corpus (`scripts/diagnose_cutoff.py`):

- **644/644** recommendation files begin with exactly the five seeds.
- The first genuinely new (CF) topic was therefore evaluated at **rank 11** in every project
  — one position outside the window HybridRec reports.

Evaluation ranks 6–10 were re-scoring ranks 1–5. That is why every curve froze at N=5 and why
precision@10 was *exactly* precision@5 / 2 in every configuration ever tested here.

Same recommendations, scored both ways:

| N | P@N as-scored | P@N fixed | SR@N as-scored | SR@N fixed |
|---|---|---|---|---|
| 5 | 0.164 | 0.164 | 0.576 | 0.576 |
| 6 | 0.137 | **0.168** | 0.576 | **0.630** |
| 8 | 0.103 | **0.159** | 0.576 | **0.680** |
| 10 | 0.082 | **0.145** | 0.576 | **0.710** |

Fixed in `metrics.py`: a topic already scored no longer consumes a second rank slot.

## Corrections to earlier sections of this file

1. **"52 of 65 recommendation lists are shorter than 15 items (median 5)" is wrong.** It was
   computed with `__` instead of `___` as the project-name separator, so almost every lookup
   missed. True figures over 644 projects: **median list length 79, none shorter than 15.**
2. **"N>10 is not reportable" is withdrawn.** The non-monotonic success rate that motivated it
   came from `compute_average_success_rate_b`, which sums per-project files *by line number*
   and divides by the project count, with no carry-forward for short lists — unlike its
   non-bayesian twin. A short file silently dragged the average down. Per-project files are now
   padded to `num_libs`, so recall holds flat and precision falls as hits/rank.
3. **Fold-1 numbers reported earlier in this file are unverified.** `topic_similarity_calculator`
   guarded the *testing* topic lookup against `KeyError` but not the *training* one; a training
   topic absent from the combined graph raised into the catch-all and abandoned every project in
   the fold. Fold 1 hits this (`#DEP#confluence-alternative`) and produced **no similarity files
   at all** — earlier runs silently read stale files left by an older project ordering. Guarded
   now; the ten-fold run above logs zero errors.
4. `validator.py` used `num_of_projects // 10` where `runner.py` uses `ceil` — 64 vs 65 projects
   per fold. This was §4 item 8 in ARCHITECTURE.md; now aligned.

## The knowledge graph, re-measured on all 644 projects

One-hop expansion from the CNBN seeds (`scripts/kg_edge_analysis.py`):

| Graph | Candidates | Correct | Precision | Recall | Projects with a hit |
|---|---|---|---|---|---|
| SED-KGraph, all relations (current) | 40,433 | 303 | 0.75% | 7.5% | 232/644 |
| SED-KGraph, precision-positive relations only | 5,002 | 60 | 1.20% | 1.5% | 52/644 |
| SED-KGraph, neighbour degree <= 20 | 36,871 | 250 | 0.68% | 6.2% | 195/644 |
| **corpus PMI co-occurrence (training folds only)** | 16,552 | 461 | **2.79%** | **11.4%** | **270/644** |
| union of both | 53,990 | 678 | 1.26% | **16.7%** | 376/644 |
| intersection (SED edge confirmed by corpus) | 2,291 | 55 | 2.40% | 1.4% | 50/644 |

**Only 19% of ground-truth topics (239/1,289) exist in SED-KGraph's vocabulary at all**, which
caps any KG-derived recall at 19% before ranking is even considered. A co-occurrence graph built
from the same training folds — free, no external resource — is 3.7x more precise and reaches 1.5x
more of the ground truth, because it encodes how developers *tag* rather than what topics *mean*.

`works-with` supplies 30,040 of the 40,433 candidates at 0.74% precision: most of the noise, and
the reason discarding the `rel` column is expensive. Hub suppression does not help — precision
falls monotonically as the degree cap tightens.

## Reproducing

```bash
python preprocessing/main.py            # -> outpath/TopFilter/, dataset_j/CNBN/
python scripts/make_folds.py            # -> custom_dataset/folds.json
python cnbn/generate_datatset.py        # all ten fold trees, verified zero overlap
python cnbn/main.py                     # ten folds -> training_data.csv (644 rows)
python topfilter/runner.py              # ten folds + validator
python scripts/compare_metrics.py outpath/TopFilter/Results
python scripts/diagnose_cutoff.py       # the rank-offset evidence
python scripts/kg_edge_analysis.py      # the KG decomposition
```

CNBN classified each README once per topic directory it appeared in (~10x redundant); it now
classifies once per repository, cutting a fold from ~9.5 min to ~72 s. Verified output-identical:
seed *sets* unchanged for all 65 fold-1 repos (mean top-5 overlap 5.00/5).
