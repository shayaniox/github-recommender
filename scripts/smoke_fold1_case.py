"""One fold-1 smoke case (driven by scripts/smoke_fold1.sh): TopFilter similarity + recommendation + bayesian scoring on one fold.

Uses the committed topfilter/ classes unchanged. Only two things are varied, to reproduce a
single defect at a time:
  --seeds FILE      the training_data.csv installed as the CNBN bridge
  --score-step      'ceil' (current validator) or 'floor' (validator before commit 3)
  --seed-prefix     '' (current) or ' ' (reader before commit 1: '#DEP# topic')

Usage: smoke.py <workdir> --seeds FILE [--score-step ceil|floor] [--seed-prefix ' ']
Prints one JSON line with SR/P/R at N=1..10, catalog coverage, #projects scored, seeds parsed.
"""
import argparse, glob, importlib.util, json, math, os, shutil, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "topfilter"))
os.chdir(REPO)  # sedkgraph.csv is opened relative to the repo root

import data_reader  # noqa: E402
from topic_similarity_calculator import TopicSimilarityCalculator  # noqa: E402
from recommendation_engine import RecommendationEngine  # noqa: E402
from metrics import Metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("workdir")
ap.add_argument("--seeds", required=True)
ap.add_argument("--score-step", default="ceil", choices=["ceil", "floor"])
ap.add_argument("--seed-prefix", default="")
ap.add_argument("--metrics-module", default=None,
                help="score with this metrics.py instead (e.g. the pre-fix one from git)")
a = ap.parse_args()

if a.seed_prefix:
    # Reproduce the pre-fix reader exactly: '#DEP# ' + topic.
    def _load(self, _p=a.seed_prefix):
        fn = os.path.join(self.src_dir, "training_data.csv")
        with open(fn) as r:
            for line in r:
                v = line.split(";")
                for z in v[1:]:
                    self.eASEOutput[v[0].replace(".txt", "")].append("#DEP#" + _p + z.strip())
    data_reader.DataReader.load_EASE_output = _load

src_orig = os.environ.get("SMOKE_SRC", os.path.join(REPO, "outpath", "TopFilter"))
wd = os.path.abspath(a.workdir)
if os.path.exists(wd):
    shutil.rmtree(wd)
os.makedirs(wd)
for f in glob.glob(os.path.join(src_orig, "dicth_*")) + glob.glob(os.path.join(src_orig, "graph_*")):
    shutil.copy(f, wd)
shutil.copy(os.path.join(src_orig, "projects.txt"), wd)
shutil.copy(a.seeds, os.path.join(wd, "training_data.csv"))
src = wd + "/"

n = sum(1 for _ in open(src + "projects.txt"))
step = math.ceil(n / 10)
t0 = time.time()
TopicSimilarityCalculator(src, "Round1", 1, 0, step + 1, n, 1, step, True).compute_weight_cosine_similarity()
RecommendationEngine(src, "Round1", 20, 1, step, True).user_based_recommendation()
t_rec = time.time() - t0

sstep = step if a.score_step == "ceil" else n // 10
if a.metrics_module:
    spec = importlib.util.spec_from_file_location("metrics_alt", a.metrics_module)
    alt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(alt)
    Metrics = alt.Metrics  # noqa: F811
    for sub in ("Results", "Round1/SuccesRateB", "Round1/PrecisionRecallB"):
        os.makedirs(src + sub, exist_ok=True)
m = Metrics(1, 20, src, "Round1", 1, 0, sstep + 1, n, 1, sstep)
m.catalog_coverage_b(); m.success_rate_b(5); m.compute_average_success_rate_b()
m.precision_recall_b(5); m.compute_average_precision_recall_b(); m.catalog_coverage()

def rows(p):
    out = {}
    for line in open(p):
        v = line.split("\t")
        out[int(v[0])] = [float(x) for x in v[1:]]
    return out

sr, pr, cat = rows(src + "Results/SRB_Round1"), rows(src + "Results/PRCB_Round1"), rows(src + "Results/Catalog1")
dr = data_reader.DataReader(src)
proj = [l.strip() for l in open(src + "projects.txt")][:step]
parsed = [len(dr.get_EASE_topic(p, 5)) for p in proj]
print(json.dumps({
    "score_window": [1, sstep], "projects_scored": len(m.testing_projects),
    "projects_with_5_seeds": sum(x == 5 for x in parsed), "projects_with_0_seeds": sum(x == 0 for x in parsed),
    "SR": {k: round(sr[k][0], 4) for k in range(1, 11)},
    "P": {k: round(pr[k][1], 4) for k in range(1, 11)},
    "R": {k: round(pr[k][0], 4) for k in range(1, 11)},
    "Cov": {k: round(cat[k][0], 4) for k in range(1, 11)},
    "rec_seconds": round(t_rec, 1),
}))
