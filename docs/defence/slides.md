---
marp: true
paginate: true
---

# Semantically-Enhanced Topic Recommendation for GitHub Repositories

Integrating the SED-KGraph knowledge graph into the HybridRec recommender

**Master's Thesis Defence**

<!--
Opening line: "This thesis asks whether a software-engineering knowledge graph can improve
automatic topic recommendation for GitHub repositories. The answer turned out to be more
interesting than a yes."
-->

---

## The problem

GitHub repositories are annotated with **topics** — `python`, `machine-learning`, `rest-api`.

Topics drive **discovery, search and categorisation**, but:

- most repositories are under-tagged or untagged
- the vocabulary is enormous and inconsistent (`js` / `javascript`, `oh-my-zsh` / `zsh`)

**Task:** given a repository's README and its existing topics, recommend the topics it should carry.

<!--
Emphasise this is a multi-label recommendation problem with a very long tail, not classification
over a small fixed label set.
-->

---

## Baseline: HybridRec

Di Rocco et al., *Applied Intelligence* 53:9708–9730 (2023)

Two recommenders **chained**, not competing:

```
README text ──► CNBN (content-based) ──► top-N topics ─┐
                                                        ├──► Topics Merger ──► final ranking
repo's existing topics ──► CF (collaborative) ─────────┘
```

- **CNBN** — Complement Naive Bayes over TF-IDF of README text
- **CF** — user-based collaborative filtering; repositories are "users", topics are "items"
- CNBN's predictions **seed** the collaborative stage

**HybridRec uses no knowledge graph.** It appears only in their future work.

<!--
Stress the cascade. The CF stage starts from CNBN's guesses rather than from nothing, which is
how it handles repositories with few existing topics. The KG is therefore genuinely my
contribution, not a reimplementation.
-->

---

## Contribution: adding SED-KGraph

Izadi et al., *Empirical Software Engineering* 28:50 (2023) — the **KGRec** paper

**SED-KGraph**: a software-engineering knowledge graph of *topics*

- **864 topics**, **2,234 verified relationships**, **13 relation types**
- built by 170 contributors across 16 companies and 11 universities
- `django is-a framework` · `django works-with python` · `ajax is-a standard`

**Hypothesis:** if CNBN predicts `django` and CF predicts `python`, the graph knows both relate
to `backend` and `web-development` — topics the recommenders never proposed.

<!--
Note the graph is topic-to-topic, not repo-to-topic. That is what makes it droppable into an
existing recommender without retraining anything.
-->

---

## Method

**Integration point:** the Topics Merger, where both branches' outputs meet.

1. Collect source topics = CNBN predictions ∪ CF predictions
2. Look up each in SED-KGraph, take **one-hop neighbours** (relation types ignored, undirected — as in KGRec)
3. Rank neighbours by **encounter count**: how many source topics each neighbour is adjacent to
4. Merge: originals keep their ranks, neighbours follow

**Two strategies evaluated**

| | |
|---|---|
| **append** | neighbours added after the originals |
| **rerank** | neighbours never added; existing candidates *boosted* by KG agreement |

<!--
Rerank was designed after append failed. Its motivation: append can only enter the evaluated
window by displacing correct predictions. Rerank introduces nothing new, so it cannot lose that
way. Both were evaluated.
-->

---

## Experimental setup

| | |
|---|---|
| Corpus | 1,000 mined repositories → 644 after normalisation |
| Topic vocabulary | 1,289 normalised topics |
| Evaluation | fold 1 of 10, 65 test projects, **identical across all conditions** |
| Fold assignment | stable hash of project name (corpus changes cannot reshuffle it) |
| Train/test | 580 / 65, **verified zero overlap** |
| CF neighbours | k = 20 |
| Metrics | success rate@N, precision@N, recall@N (HybridRec's metrics) |

**Ablation:** `kg_off` · `kg_all` · `kg_top10` · `kg_top5` · `rerank(λ = 0.5 / 1.0)` ·
`normalise(pre-rules / post-rules)`

<!--
Single fold, not ten — be upfront. The original code breaks after fold 1 and I kept that to stay
comparable with my own baseline. Say so before the committee asks.
-->

---

## Result 1 — the reproduction is sound

Baseline (no KG) vs the published HybridRec figures:

| N | HybridRec | This work | |
|---|---|---|---|
| success rate@1 | 0.153 | **0.234** | +53% |
| success rate@5 | 0.335 | **0.547** | +63% |
| success rate@10 | 0.614 | 0.547 | −11% |
| precision@1 | 0.212 | **0.234** | +11% |
| precision@5 | 0.117 | **0.150** | +28% |
| precision@10 | 0.127 | 0.075 | −41% |
| recall@5 | 0.090 | **0.112** | +25% |
| recall@10 | 0.191 | 0.112 | −41% |

**Better at N ≤ 5, worse from N = 6** — 65 projects on one fold versus thousands averaged over
three datasets. Leakage-free and fold-stable (see validity slide).

<!--
This slide exists to establish that the platform is trustworthy before presenting a negative
result. If the baseline were broken, the KG finding would mean nothing.
-->

---

## Result 2 — the knowledge graph never helps

All conditions on the **same 65 test projects**, leakage-free:

| N | baseline | append | top-10 | top-5 | rerank | normalise |
|---|---|---|---|---|---|---|
| precision@1 | 0.2344 | 0.2344 | 0.2344 | 0.2344 | 0.2344 | 0.2344 |
| precision@5 | 0.1500 | 0.1500 | 0.1500 | 0.1500 | 0.1500 | 0.1500 |
| precision@10 | 0.0750 | 0.0750 | 0.0750 | 0.0750 | 0.0750 | 0.0750 |
| success rate@10 | 0.5469 | 0.5469 | 0.5469 | 0.5469 | 0.5469 | 0.5469 |

**Identical to four decimal places at every evaluable cut-off.** Four integration strategies —
two that add topics, two that do not — and none changes a single recommendation the metrics see.

Beyond N = 10 the baseline metric is undefined: 52 of 65 recommendation lists are shorter than
15 items (median 5), and success rate goes *non-monotonic* — 0.547 at N=10 to 0.141 at N=11.
No comparison is valid there.

<!--
The N>10 point matters: an earlier version of this analysis showed the KG "winning" at N=15-20.
It was filling empty slots where CF had produced nothing, not finding correct topics. Precision
there is 0.046. Say this before anyone spots the gap in the table.
-->

---

## Why — the KG cannot pay for itself

**1. Vocabulary gap**
**59%** of ground-truth topics are absent from SED-KGraph. Ceiling on KG-derived recall: 41%.

**2. Reach**
Only **6%** of ground-truth topics lie at graph distance 1 from the seeds.

**3. Signal-to-noise — the decisive one**
One-hop expansion produced **3,774 candidates containing 27 correct topics — 0.72% precision.**

> To surface one correct topic you must accept roughly **140 wrong ones.**

No ranking function repairs a candidate pool that thin; it can only reorder it.

<!--
All three numbers are from the clean, leakage-free run. An earlier version of this analysis
claimed KG adjacency was anti-correlated with correctness (0.54x lift). That did NOT replicate
once the leakage was fixed - 1.06x on a sample of 41 - so I dropped it. If asked whether the
graph is actively misleading: I cannot show that. I can show it is far too imprecise to help.
-->

---

## Interpretation

**Semantic relatedness is not tag co-assignment — the two are in tension.**

SED-KGraph encodes what topics *mean*: `django is-a framework`, `django works-with python`.

Developers tag for **information**, not completeness. A repository tagged `django` does **not**
also tag `framework` or `backend` — that would be redundant.

> Semantic neighbours are precisely the topics an owner has reason to omit.

The most-promoted topics were generic hubs — `programming-language` (degree 55), `web-development`
(77), `open-source` (48) — correct in meaning, absent from every ground truth.

<!--
This reframes the negative result as a finding about the task, not a failed engineering attempt.
It is the strongest thing in the thesis: an explanation with numbers behind it.
-->

---

## This is consistent with KGRec's own evidence

KGRec never evaluated against owner-assigned tags for exactly this reason.

| Evaluation | ASR@5 |
|---|---|
| automated (ground truth) | 30.0% |
| **manual (expert judges)** | **50.8%** |

Their headline gains come from **5 experts scoring 50 sampled projects** and from FCR
(failed-case ratio), not from ground-truth precision.

**Reproducing their claim requires their evaluation protocol, not just their dataset.**

<!--
If challenged with "so the KG paper was wrong" — no. They measured something different.
Their topics are judged plausible by humans; mine are judged against what owners happened to
type. My result is evidence for their methodological argument.
-->

---

## Validity — what I checked and what it cost

| Defect | Effect | How it surfaced |
|---|---|---|
| Train/test leakage | 63/65 test repos in training; SR@1 inflated 0.234 → **0.766** | A baseline I had not changed moved 3.3× |
| Fold instability | Corpus 644→645 reshuffled the whole split | Two "comparable" runs shared few test projects |
| Non-determinism | Two identical runs differed on 15% of top-10s | An apparent KG gain that would not reproduce |
| Metric column swap | precision/recall transposed | recall@1 = 0.31 was impossible |

Each one produced a *positive* result that disappeared under scrutiny.

Contaminated outputs are quarantined, not deleted: `results/CONTAMINATED_do_not_cite/`.

<!--
Own this slide. A committee trusts a student who found their own errors far more than one whose
numbers merely look clean. Each row is a result I reported and then withdrew.
-->

---

## Contributions

1. **A working, verified reproduction** of the HybridRec pipeline, competitive with published figures at low cut-offs
2. **First integration of SED-KGraph into HybridRec**, with two integration strategies and a parameter sweep
3. **A quantified negative result with a mechanism** — 59% vocabulary gap, 6% one-hop reach, 0.72% candidate precision
4. **A methodological finding**: ground-truth evaluation structurally penalises semantically-driven recommendation

<!--
Lead with the reproduction so the committee sees engineering competence, then the negative result
framed as knowledge gained.
-->

---

## Future work

**Follows directly from the measurements**

- **Human evaluation** — the only protocol that can credit a semantically valid topic the owner omitted (KGRec's own approach)
- **KG for normalisation** — *tested both before and after the hand-built rules*: no effect either way. Not a promising direction on this corpus
- **Close the vocabulary gap** — 54% of ground truth is outside the graph; extend it, or restrict evaluation to its 864 topics
- **Full 10-fold cross-validation** and the remaining folds

<!--
End on the normalization idea: it uses the same asset for a task where its bias is a feature
rather than a defect. That is the strongest answer to "what would you do next".
-->

---

# Thank you

Questions?

<!--
Anticipated:
- "Is your baseline trustworthy?" -> yes; reproduction slide, plus the pipeline required repairs
  before it ran end to end, all documented and verified by re-running to an exact match.
- "Did you implement KGRec's actual method?" -> no, deliberately: a simplified one-hop expander.
  But the candidate pool it would rank is the same one measured at 0.72% precision, so it
  inherits the same ceiling.
- "One fold?" -> yes, stated up front; the runners break after fold 1. Not fixed because every
  condition is affected identically, so the comparison holds.
-->
