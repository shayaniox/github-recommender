# What the knowledge graph costs in HybridRec

The question: how much does adding SED-KGraph to the HybridRec pipeline cost, compared with the
same pipeline without it? This page gives the measured runtime, memory, disk and accuracy cost.
The KG step is implemented in the Topics Merger (`topfilter/recommendation_engine.py`) in two
variants. Every number below comes from a file named next to it.

| run | KG setting | what the KG does to each test repo's merged list |
|---|---|---|
| A | off | nothing |
| B | `KG_MODE=append` | one-hop neighbours of every topic in the list are appended *after* all the original topics |
| C | `KG_MODE=rerank`, λ = 0.5, 10 sources | topics already in the list that neighbour the top-10 entries get a bonus; nothing new is added |

The KG option on the CNBN branch (`KG_ENABLED` for `cnbn/main.py`) is off in every experiment and
is not costed here.

## Summary

| cost | B: append | C: rerank | source |
|---|---|---|---|
| runtime, whole TopFilter run (10 folds, 644 repos) | **+0.52 s (+3.4%)** | **none measurable** (−0.15 s, inside the noise) | §1 |
| time inside the KG code | 0.32 s, 1.2% of the run | 0.19 s, 0.75% of the run | §1 |
| share of a full HybridRec run (CNBN + TopFilter, 644 repos) | ≈ 0.15% | ≈ 0.05% | §1 |
| extra peak memory | +1.2 MB | +0.5 MB | §2 |
| extra disk, recommendation files | ×4.0 (+4.7 MB per run) | none | §3 |
| accuracy at N ≤ 10 | none: identical lists in the top 10 | **lower** in all 5 fold splits: R@10 −0.016 ± 0.002, P@10 −0.011 ± 0.002, SR@10 −0.021 ± 0.008 | §4, §5 |

In short, the KG is cheap to run. Its cost in this framework is in accuracy (rerank) or in having no
effect at all (append), not in compute time.

## Worked example: one test repository

`11ty/eleventy`, fold 1, readme_8k_desc20 seeds (`outpath/experiments/readme_8k_desc20/{A,B,C}/Round1/Recommendations/11ty___eleventy`):

- **A** writes 86 topics: the 5 CNBN seeds `react, javascript, json, yaml, tool` (score 2.0), then
  the CF candidates `csv 1.122, command-line 1.116, library 1.088, python 1.078, nodejs 1.071, …`.
- **B** writes those same 86 topics in the same order, then **274 graph neighbours** at ranks
  87–360. The evaluated window ends at rank 20, so for this repository B computes 274 extra
  entries and writes them to disk, and none of them can change a metric.
- **C** writes the same 86 topics, re-ordered. The graph bonus lifts `redux 1.443, nextjs 1.195,
  html 1.144` into ranks 6–8 above `csv 1.122`. This is the only one of the three that changes
  what is evaluated.

The KG work for this repository is one lookup per topic in an 864-topic adjacency map. That takes
about 0.4 ms for append and 0.2 ms for rerank (§1).

## 1. Runtime

Measured with `scripts/bench_kg.sh` at commit `e8a10a7` on an Intel i7-10700 (16 cores), 2026-10-02.
Inputs: the 644-repository corpus `outpath/TopFilter` with the readme_8k_desc20 CNBN seeds.
Each run had the machine to itself (only a network-bound README download was running). A, B
and C ran 5 times each, with their order rotated every repetition. Generated report:
`outpath/bench_kg/KG_RUNTIME.md`.

| run | median | min | max | overhead vs A |
|---|---|---|---|---|
| A: KG off | 15.39 s | 15.20 s | 15.73 s | — |
| B: append | 15.91 s | 15.88 s | 16.55 s | +0.52 s (+3.4%) |
| C: rerank | 15.24 s | 15.19 s | 15.86 s | −0.15 s (−1.0%) |

Every B run was slower than every A run, so B's overhead is real. C's runs overlap A's, so its
overhead is smaller than the run-to-run noise (0.67 s at most). An extra profiled run per mode
(cProfile) measures the KG code directly:

| run | loading the graph (`build_adjacency`) | KG merge step | KG share of the run |
|---|---|---|---|
| B: append | 0.040 s over 10 calls (4 ms per fold) | 0.283 s over 644 calls (0.44 ms per test repo) | 1.21% |
| C: rerank | 0.040 s over 10 calls | 0.150 s over 644 calls (0.23 ms per test repo) | 0.75% |

Append's end-to-end overhead (0.52 s) is larger than the time inside its KG functions (0.32 s).
The rest goes on writing and then scoring its ~4× longer recommendation files (§3).

**Against the whole HybridRec run.** On the same corpus CNBN took 333.7 s for its 10 folds
(`outpath/experiments/timings.tsv`, commit `c3eb693`). A full hybrid run is therefore about
333.7 + 15.4 ≈ 349 s, and append's +0.52 s is about 0.15% of it. Caveat: the 333.7 s CNBN time
was measured with three CNBN variants running in parallel, so it is an upper estimate of CNBN
alone.

**Larger corpora (projection, not measured).** The KG cost per test repository does not depend
on the training set. It is one adjacency lookup per merged topic, and the merged list's length is
set by the 20 nearest neighbours. CF's similarity step, by contrast, grows with N² (measured
exponent 1.89 between 644 and 5,954 repos, `paper_artifacts/RUNTIMES.md`). At 5,954 repos that
gives roughly 0.44 ms × 5,954 + 0.04 s ≈ 2.7 s for append, against 956 s of CF (measured,
CF-only). That is about 0.3%, and the share keeps shrinking as the corpus grows. A hybrid
TopFilter run at that size could not be measured, because CNBN does not fit in memory there
(`docs/CNBN-MEMORY.md`).

## 2. Memory

| | value | source |
|---|---|---|
| `sedkgraph.csv` | 73,190 bytes | file size |
| adjacency map in memory | 864 topics, 4,372 directed edges, 2.1 MB, loaded in 15 ms | `tracemalloc` around `build_adjacency` |
| TopFilter peak RSS, A | 16.4–16.5 MB | `outpath/bench_kg/timings.tsv`, 5 runs |
| TopFilter peak RSS, B | 17.6–17.7 MB (+1.2 MB) | same |
| TopFilter peak RSS, C | 17.0 MB (+0.5 MB) | same |

The graph is small and loaded once per fold. Its memory cost is negligible next to CNBN's, which
peaks at 2.8 GB on this corpus.

## 3. Disk

Recommendation files written by one 10-fold run on 644 repos
(`outpath/experiments/readme_8k_desc20/{A,B,C}/Round*/Recommendations`):

| run | bytes | lines |
|---|---|---|
| A | 1,603,331 | 51,826 |
| B | 6,348,077 (×3.96) | 190,859 |
| C | 1,603,023 | 51,826 |

Rerank only re-orders, so its output is the same size as A's. Append grows every list by its
graph neighbours. This is linear in the number of test repos, about 60 MB per run at 5,954 repos
(projection).

## 4. Accuracy

Mean ± standard deviation over the 10 folds of the main experiment (`paper_artifacts/results_raw.csv`,
644 repos, readme_8k_desc20, commit `c3eb693`). p is a two-sided Wilcoxon signed-rank test over the
10 folds (`paper_artifacts/significance.csv`).

| N = 10 | A: KG off | C: rerank | C − A, per fold | 95% CI of C − A | p |
|---|---|---|---|---|---|
| success rate | 0.7983 ± 0.0494 | 0.7718 ± 0.0596 | −0.0265 ± 0.0180 | [−0.0393, −0.0136] | 0.012 |
| precision | 0.1820 ± 0.0160 | 0.1683 ± 0.0194 | −0.0136 ± 0.0062 | [−0.0180, −0.0092] | 0.005 |
| recall | 0.2624 ± 0.0191 | 0.2439 ± 0.0235 | −0.0185 ± 0.0089 | [−0.0249, −0.0121] | 0.002 |
| catalog coverage | 0.1734 ± 0.0142 | 0.1682 ± 0.0112 | −0.0051 ± 0.0057 | [−0.0092, −0.0010] | 0.037 |

Append (B) is identical to A in every fold, so its Δ is exactly 0: the originals fill the window
first (86 of them in the example above). Each method's own spread (±0.019 for A's recall) is about
as large as rerank's effect, so those columns alone cannot rule out noise. The paired column can:
A and C rise and fall together from fold to fold, and C is lower in all 10 folds.

## 5. Is it noise? Five fold splits

The pipeline is deterministic (identical metrics under another `PYTHONHASHSEED`), so the only random
choice is which repositories share a fold. `scripts/repeated_cv.sh` re-ran the whole experiment
(CNBN training sets, then A, B and C) on 5 fold splits, `make_folds.py --seed` 42, 1, 2, 3 and 4, at
commit `adafc8d`. Split 42 is the main experiment's: its CNBN fold tree and all 1,200 fold values
reproduced the published ones exactly. Generated report: `outpath/repeated_cv/REPEATED_CV.md`.

| C − A at N = 10 | mean over the 5 splits ± std over splits | range of the split means | splits with p < 0.05 |
|---|---|---|---|
| recall | −0.0156 ± 0.0021 | −0.0185 to −0.0131 | 5 of 5 |
| precision | −0.0111 ± 0.0019 | −0.0136 to −0.0086 | 5 of 5 |
| success rate | −0.0210 ± 0.0084 | −0.0323 to −0.0108 | 3 of 5 |

- Recall and precision drop in every split, and each split's 95% CI excludes zero.
- Each method's mean barely moves between splits: A's recall@10 is 0.2622 ± 0.0028 across splits,
  against ±0.0234 across folds.
- Append is identical to A in all 50 folds. At N = 5 rerank is identical too: the 5 CNBN seeds are
  pinned at score 2.0 and its bonus is at most 0.5, so it only reorders ranks 6 and below.

## Reproduce

```sh
sh scripts/bench_kg.sh 5 outpath/bench_kg       # §1 and the peak RSS in §2, about 6 minutes
sh scripts/repeated_cv.sh outpath/repeated_cv 42 1 2 3 4    # §5, about 35 minutes
.venv/bin/python scripts/summarize_repeated_cv.py outpath/repeated_cv
.venv/bin/python scripts/bench_kg_summary.py outpath/bench_kg
```
