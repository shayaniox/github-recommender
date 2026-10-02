"""Measure where CNBN's training memory goes, and predict it for corpora too large to train.

CNBN (cnbn/model.py predict_topics) is TfidfVectorizer -> sklearn ComplementNB. Reading
sklearn 1.3.2's naive_bayes.py, fit() holds these dense arrays (S samples, C classes, F
vocabulary terms):

    Y                  LabelBinarizer output, int64, S x C        alive for the whole fit
    feature_count_     float64, C x F                             kept on the model
    Y.T @ X            float64, C x F temporary in _count
    comp_count         float64, C x F  \
    comp_count / sum   float64, C x F   } _update_feature_log_prob: four C x F arrays are
    logged / -logged   float64, C x F  /  alive together at its peak

so the predicted peak is 8 B * (S*C + 4*C*F) on top of the texts and the sparse TF-IDF matrix.

For one training directory (one class per sub-directory, as the CNBN fold tree), this script
reports S, C, F and the TF-IDF nnz, the predicted peak, and with --fit trains ComplementNB
and records the measured peak: tracemalloc (numpy registers its buffers) over the fit alone,
and the process peak RSS. --max-classes keeps only the K largest classes, so the formula can
be checked on configurations of a big corpus that still fit in RAM.

Usage: python scripts/cnbn_memory.py TRAIN_DIR [--max-classes K] [--min-repos M] [--min-df D] [--fit]
Prints one tab-separated line: dir, K, M, D, S, C, F, nnz, predicted_GB, fit_peak_GB, rss_peak_GB, fit_s
"""

import argparse
import resource
import sys
import time
import tracemalloc
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB

ap = argparse.ArgumentParser()
ap.add_argument("train_dir")
ap.add_argument("--max-classes", type=int, default=None, help="keep only the K largest classes")
ap.add_argument("--min-repos", type=int, default=1,
                help="keep only classes with at least M repositories")
ap.add_argument("--min-df", type=int, default=1,
                help="TfidfVectorizer min_df: drop terms in fewer than this many samples (CNBN: 1)")
ap.add_argument("--fit", action="store_true")
a = ap.parse_args()

# The same reading order and options as cnbn/model.py load_data / predict_topics.
classes = {d.name: sorted(d.glob("*.txt")) for d in Path(a.train_dir).iterdir() if d.is_dir()}
size = Counter({c: len(f) for c, f in classes.items()})
keep = [c for c, n in size.most_common(a.max_classes) if n >= a.min_repos]
X, y = [], []
for c in keep:
    for p in classes[c]:
        X.append(p.read_text(encoding="utf-8", errors="ignore"))
        y.append(c)

vec = TfidfVectorizer(input="content", stop_words="english", lowercase=True, analyzer="word",
                      min_df=a.min_df)
T = vec.fit_transform(X)
S, F = T.shape
C = len(set(y))
predicted = 8 * (S * C + 4 * C * F) / 1e9

fit_peak = rss = fit_s = float("nan")
if a.fit:
    tracemalloc.start()
    t0 = time.time()
    ComplementNB().fit(T, y)
    fit_s = time.time() - t0
    fit_peak = tracemalloc.get_traced_memory()[1] / 1e9
    tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9

print("\t".join(str(v) for v in (
    a.train_dir, a.max_classes or "all", a.min_repos, a.min_df, S, C, F, T.nnz, f"{predicted:.3f}",
    f"{fit_peak:.3f}", f"{rss:.3f}", f"{fit_s:.1f}")))
sys.stdout.flush()
