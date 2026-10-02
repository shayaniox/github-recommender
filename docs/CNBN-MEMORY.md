# Where CNBN's memory goes

CNBN on the top-10k corpus (5,954 repositories after preprocessing) was killed by the kernel's
out-of-memory killer during `ComplementNB.fit`, at 19.7 GB resident and still growing
(`journalctl`, 2026-10-02 11:35:41). This page explains why and how much memory it would need.
Measured with `scripts/cnbn_memory.py` at commit `e8a10a7`, scikit-learn 1.3.2, numpy 1.24.4.

## The formula

`cnbn/model.py` runs `TfidfVectorizer` and then `ComplementNB().fit`. With S training samples,
C classes and F vocabulary terms, sklearn's `fit` holds these dense arrays at its peak:

| array | dtype, shape | where |
|---|---|---|
| `Y`, the one-hot labels | int64, S × C | `LabelBinarizer`, alive for the whole fit |
| `feature_count_` | float64, C × F | kept on the model |
| `comp_count`, `comp_count / sum`, `logged` | float64, 3 × C × F | `_update_feature_log_prob`, alive together with `feature_count_` |

**Peak ≈ 8 B × (S·C + 4·C·F)**, on top of the texts and the sparse TF-IDF matrix.

Checked by training on configurations that fit in memory. The measured value is the tracemalloc
peak over `fit` alone:

| training set (fold 1) | S | C | F | predicted | measured |
|---|---|---|---|---|---|
| 644 corpus | 5,066 | 2,420 | 34,575 | 2.776 GB | 2.777 GB |
| 10k corpus, 500 largest classes | 20,389 | 500 | 189,650 | 3.116 GB | 3.120 GB |
| 10k corpus, 1,000 largest classes | 24,936 | 1,000 | 192,280 | 6.352 GB | 6.357 GB |

## The 10k corpus

Fold 1 has S = 45,772, C = 14,285 and F = 196,248:

| part | size |
|---|---|
| `Y`: 45,772 × 14,285 × 8 B | 5.2 GB |
| four C × F tables: 4 × 14,285 × 196,248 × 8 B | **89.7 GB** |
| **predicted peak** | **94.9 GB**, on a 31 GB machine |

Two things drive it:

- **The number of classes.** CNBN's classes are the raw topic labels, one directory per label
  (`cnbn/generate_datatset.py`). The 10k fold has 14,285 of them, while TopFilter's normalised
  vocabulary for the same repositories has 5,222. Most labels are rare: only 4,709 classes have
  at least 2 repositories.
- **The size of the vocabulary.** It is 196k terms, and every class gets a dense row over all of them.

**Worked example.** One class row of `feature_count_` is F = 196,248 floats = 1.57 MB. Four
such tables make 6.3 MB per class, and 14,285 classes make 89.7 GB. A label carried by a single
repository costs the same 6.3 MB as `javascript`.

## What each lever buys (fold 1 of the 10k corpus, predicted peak)

| change | S | C | F | peak |
|---|---|---|---|---|
| none (current method) | 45,772 | 14,285 | 196,248 | 94.9 GB |
| classes with ≥ 2 repositories | 36,196 | 4,709 | 195,919 | 30.9 GB |
| classes with ≥ 3 repositories | 32,536 | 2,879 | 195,046 | 18.7 GB |
| classes with ≥ 5 repositories | 28,108 | 1,568 | 193,701 | 10.1 GB |
| classes with ≥ 10 repositories | 22,502 | 696 | 191,169 | 4.4 GB |
| vocabulary `min_df=5` | 45,772 | 14,285 | 193,261 | 93.6 GB |
| ≥ 5 repositories and `min_df=2` | 28,108 | 1,568 | 186,819 | 9.7 GB |

`min_df` barely helps. Each README is copied once per topic it carries, so a term from one
README already occurs in several samples and passes the threshold. Shrinking the vocabulary
would need `max_features`.

About 15–19 GB of the 31 GB is free while the machine runs its usual other processes. As the
pipeline stands, a class threshold of ≥ 5 repositories is the smallest change that fits.

## Options

Each option either changes the method or keeps it:

1. **Keep the method: an exact sparse reformulation (a proposal, not yet tested).** For
   ComplementNB, `log(feature_all_ + α − feature_count_)` equals `log(feature_all_ + α)` wherever
   a class never saw a term, and `feature_count_` has at most nnz(X) = 11.5 M non-zeros. The model
   can therefore be stored as one shared dense vector of length F plus a sparse per-class
   correction. Memory then scales with nnz instead of C × F, and predictions should be the same up
   to floating-point rounding. This could be checked bit-for-bit against sklearn on the 644 corpus.
2. **Change the method: drop rare classes** (≥ 5 repositories → about 10 GB). CNBN can then no
   longer predict those labels.
3. **Change the method: cap the vocabulary** (`max_features`).
4. **Use a machine with ~100 GB of RAM** for 10k. The 25k corpus needs more and has not been
   sized yet, because its README fetch is still running.

## Reproduce

```sh
.venv/bin/python scripts/cnbn_memory.py custom_dataset/CNBN/folds_desc20/train1 --fit
.venv/bin/python scripts/cnbn_memory.py outpath/hybrid_10k/cnbn_folds/train1 --max-classes 1000 --fit
.venv/bin/python scripts/cnbn_memory.py outpath/hybrid_10k/cnbn_folds/train1 --min-repos 5
```
