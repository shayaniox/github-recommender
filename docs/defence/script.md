# Defence script

Spoken narrative for `slides.md`. Every number here traces to `docs/METRICS.md` and the raw
outputs in `results/ablation/`.

---

## 1. Framing (≈1 min)

GitHub topics are the labels that drive discovery and search, and most repositories are
under-tagged. The task is: given a repository's README and the topics it already has, recommend
the topics it should carry. It is multi-label, the vocabulary is in the thousands, and the
distribution has a very long tail.

## 2. The baseline (≈2 min)

My starting point is **HybridRec** (Di Rocco et al., 2023). The key thing to understand is that
its two recommenders are **chained, not compared**:

- **CNBN** — Complement Naive Bayes over TF-IDF of the README — is content-based.
- **CF** — user-based collaborative filtering where repositories are "users" and topics are
  "items" — ignores text entirely.

CNBN's predictions are fed in as the seed input to the collaborative stage, and a Topics Merger
combines both outputs. That cascade is what lets the system handle repositories with few
existing topics.

Worth stating plainly: **HybridRec uses no knowledge graph.** It appears once, in their future
work, where they say a KG "can come in handy for the collaborative-filtering module". That is
the gap this thesis addresses.

## 3. The contribution (≈2 min)

The graph is **SED-KGraph**, published with the **KGRec** paper (Izadi et al., 2023): 864 topics,
2,234 verified relationships, 13 relation types, built by 170 contributors across 16 companies
and 11 universities. It is a *topic-to-topic* graph — which is what makes it droppable into an
existing recommender without retraining anything.

The hypothesis is straightforward. If CNBN predicts `django` and CF predicts `python`, the graph
knows both connect to `backend` and `web-development` — topics neither recommender proposed.

## 4. Method (≈2 min)

I integrate at the **Topics Merger**, the point where both branches' outputs meet.

Source topics are the union of CNBN's and CF's predictions. Each is looked up in the graph and
its **one-hop neighbours** collected — relation types ignored and edges treated as undirected,
which is what KGRec itself does. Neighbours are ranked by **encounter count**: how many distinct
source topics each neighbour is adjacent to, so a topic reached from both branches outranks one
reached from a single prediction.

I evaluated two strategies:

- **append** — neighbours added after the originals.
- **rerank** — designed after append failed. Append can only reach the evaluated window by
  pushing correct predictions out of it, so it is structurally lossy. Rerank introduces no new
  topics; it boosts candidates the recommender already produced, by KG agreement:
  `score'(t) = score(t) + λ · encounters(t)/max`.

## 5. Result 1 — the platform is sound (≈2 min)

Before any claim about the KG, the baseline has to be trustworthy.

Without the KG, this implementation reaches **success rate@1 of 0.234 against the published
0.153** — 53% higher — and **precision@5 of 0.150 against 0.117**, 28% higher. By N=10 it falls
behind: 0.075 against 0.127. That profile — ahead at low cut-offs, behind at higher ones — is
what I would expect from 65 test projects on a single fold against their thousands averaged over
three datasets.

So: competitive at low cut-offs, honest about where it is not.

**Brief note on getting there.** The implementation did not run end to end when I started. Among
other defects, the CNBN→CF bridge was severed by a delimiter mismatch — CNBN wrote
comma-separated fields, the reader split on semicolons, so the hybrid was carrying zero seed
topics. There was also a non-determinism bug: a Python `set` was iterated to order the user-item
matrix columns, so tie-breaking varied between runs and two identical runs disagreed on the
top-10 of about 15% of projects. That one matters because it initially produced an apparent KG
improvement that was pure noise; I caught it, fixed it, and re-ran everything. All repairs are
documented in `docs/ARCHITECTURE.md` and verified by reproducing the baseline to an exact match.

## 6. Result 2 — the KG never helps (≈2 min)

The headline: **no configuration of the knowledge graph improved any metric at any cut-off.**

Across all four strategies — append, truncated append at 10 and 5, rerank, and normalisation
both before and after the hand-built rules — **every metric is identical to the baseline to four
decimal places at every evaluable cut-off.** Precision@1 0.2344, precision@5 0.1500,
precision@10 0.0750, success rate@10 0.5469, in all seven configurations.

One caveat I want to raise myself rather than have it raised for me. Above N=10 the numbers do
diverge, and the KG appears to win — precision@15 goes from 0.017 to 0.057. That is not a result.
With the KG off, 52 of the 65 recommendation lists are shorter than 15 items; the median is 5,
meaning collaborative filtering frequently produces nothing beyond the seeds. The baseline's
success rate actually goes *non-monotonic* there — 0.547 at N=10, 0.141 at N=11 — which is
impossible for a cumulative measure and tells you the metric is undefined in that range. What the
KG is doing is filling empty slots, not finding correct topics: its own precision there is 0.046.
So I report N≤10, which is also the range HybridRec reports.

## 7. Why — the three measurements (≈3 min)

This is the part I want to spend time on.

**First, a vocabulary gap.** 59% of ground-truth topics do not appear in SED-KGraph at all, so
the ceiling on anything KG-derived is 41% of ground truth.

**Second, reach.** Only 6% of ground-truth topics sit at graph distance 1 from the seeds. One-hop
expansion is working against a 6% ceiling.

**Third, and decisive — signal to noise.** To reach those, one-hop expansion generated **3,774
candidates containing 27 correct topics. 0.72% precision.** To surface one correct topic you must
accept roughly 140 wrong ones. No ranking function repairs a candidate pool that thin — ranking
can only reorder what expansion produced. That is why append hurts wherever it becomes visible,
and why rerank, which adds nothing at all, still cannot help.

I want to flag a claim I withdrew. An earlier version of this analysis reported that KG adjacency
was actively anti-correlated with correctness — a lift of 0.54, meaning a KG neighbour was half as
likely to be right. **That did not replicate once I fixed the data leakage**: on clean data it is
1.06 on a sample of 41 candidates, which supports nothing in either direction. So I do not claim
the graph is misleading. I claim it is too imprecise to pay for itself, and the three numbers
above demonstrate that.

## 8. Interpretation (≈2 min)

Why would a carefully built, expert-validated knowledge graph be *negatively* predictive?

Because **semantic relatedness and tag co-assignment are in tension.**

The graph encodes what topics mean: `django is-a framework`, `django works-with python`.
Developers tag for **information**, not for completeness. A repository already tagged `django`
does not also tag `framework` or `backend` — that is redundant, and it tells a reader nothing new.

So semantic neighbours are precisely the topics an owner has reason to omit.

The evidence is visible in what got promoted: `programming-language` (degree 55, promoted for 30
of 65 projects), `web-development` (degree 77), `open-source` (degree 48). Every one of them
defensible as a description. None of them in any ground truth.

## 9. Consistency with KGRec (≈1 min)

This does not contradict the KGRec paper — it supports their methodology.

They never evaluated against owner-assigned tags, and they say why: automated evaluation gives
30.0% ASR@5 where manual expert evaluation of the same system gives **50.8%**. Their headline
results come from five experts scoring 50 sampled projects, and from failed-case ratio.

They knew ground-truth evaluation understates this class of recommender. My measurement is
direct evidence for that argument, with a mechanism attached.

## 10. Contributions and future work (≈2 min)

Four contributions: a verified reproduction competitive with the published figures; the first
integration of SED-KGraph into HybridRec with two strategies and a parameter sweep; a quantified
negative result with a mechanism rather than a shrug; and a methodological finding about
ground-truth evaluation of semantic recommenders.

Future work follows from the measurements rather than from speculation. **Human evaluation** is
the only protocol that can credit a semantically valid topic the owner omitted. **Using the KG
for normalisation rather than expansion** — merging vocabulary variants instead of adding topics —
I implemented and measured, both after the hand-built rules and before them. Neither helped:
success rate is identical at every cut-off, precision moves by 0.005 at one rank. Running it
before the rules gave the graph thirty extra merges to make and still changed nothing the metrics
can see. So all four strategies are now tested, and none improves the recommender. And the
vocabulary gap is itself addressable: extend the graph, or restrict evaluation to its 864 topics.

---

## Anticipated questions

**"Can I trust your baseline?"**
Yes, and I would rather you pressed on this. The pipeline needed repairs before it ran at all —
documented in `docs/ARCHITECTURE.md`. The check that matters: with the KG disabled, the run
reproduces the recorded baseline exactly, so the KG path is the only variable between conditions.

**"You didn't implement KGRec's actual method."**
Correct, and deliberate — I used the dataset with a simplified one-hop expander, not their
spreading-activation model with popularity and degree weighting. But the candidate pool it would
rank is the one I measured at 0.72% precision — 3,774 candidates for 27 correct topics — and
scoring reorders a pool, it does not enrich it. So I expect it to inherit the same ceiling. That
is a prediction my measurement makes and the next study can test.

**"Only one fold?"**
Yes, stated up front. The original runners break after the first fold and I kept that for
comparability with my own baseline. Every condition is affected identically, so the ablation
comparison holds; the absolute numbers would move with full cross-validation.

**"Isn't 65 test projects too small?"**
For absolute comparison with the published figures, yes — that is why I am careful to say
"competitive at low cut-offs" rather than "better". For the ablation it is adequate, because all
conditions run on exactly the same 65 projects and the effect is monotone across seven
configurations.

**"Did you consider depth-2 expansion?"**
I measured it: distance 2 would raise reachable ground truth from 5% to 20%, but one-hop already
yields 0.6% candidate precision and the candidate set grows by roughly an order of magnitude per
hop. But one-hop already yields 0.72% candidate precision, and the pool grows roughly an order of
magnitude per hop, so depth 2 buys reach at a cost to precision that the metrics would punish.

**"So the knowledge graph is useless?"**
No — it is useless *for this task under this evaluation*. It is a high-quality, expert-validated
artifact whose edges encode real semantic relationships. The finding is that those relationships
predict what a topic *means*, not what a developer *types*. That is exactly why normalisation is
the promising direction: merging `js` and `javascript` is a meaning question, and there the
graph's bias works for you.


**"Did normalisation work?"**  (updated — see below for the leakage correction)
It is the only one of the three strategies that does not make things worse, which matters because
it supports the mechanism: it never claims a neighbour is a topic, so the 0.72% candidate
precision never enters the ranking. But I will not claim a gain. Success rate moved by exactly 1/64 — one project —
and the vocabulary collapsed by exactly one topic, from 1,289 to 1,288. The corpus already has
765 hand-built alias rules applied upstream, so there was almost nothing left to normalise. The
result is a clean null with a clear explanation, and a concrete follow-up: run it on a corpus
before the hand-built rules, where the graph has room to act.


**"How do you know your numbers are not contaminated?"**
Because one set was, I found it, and I can show the difference. `generate_datatset.py` reused its
fold directories without clearing them, so when the fold assignment changed between runs, README
copies from earlier folds stayed behind — 63 of 65 test repositories ended up in the training set.
It surfaced because a baseline that should not have moved jumped from success rate@1 of 0.23 to
0.77; a 3.3x shift in a condition I had not changed is not a result, it is a bug. I fixed the
directory handling, made fold membership a stable hash of the project name so a corpus change
cannot silently reshuffle the split, re-ran both conditions on an identical 65-project test set,
and quarantined the affected outputs under `results/CONTAMINATED_do_not_cite/`. The KG expansion
ablation predates any fold change and was verified unaffected.
