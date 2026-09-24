"""Define one fold assignment shared by CNBN and TopFilter.

Before this, the two stages split independently — CNBN with a shuffled sklearn KFold over
the raw CSV, TopFilter with contiguous index windows over its own project list — so only
8 of 65 fold-1 test projects had a CNBN seed and the hybrid was measuring fold mismatch.

The shared universe is TopFilter's `projects.txt`: those are the repositories that survived
preprocessing and actually have `dicth_`/`graph_` files. Folds are contiguous chunks of a
deterministically shuffled ordering, sized exactly like `runner.ten_fold_cross_validation`
(`step = ceil(N / 10)`), and `projects.txt` is rewritten in that order. TopFilter's window
arithmetic then selects precisely fold i without any change to its CV code, and
`cnbn/generate_datatset.py` reads the same JSON.

Re-run this after any run of `preprocessing/main.py`, which regenerates `projects.txt`.

Usage:
    python scripts/make_folds.py [--src outpath/TopFilter] [--out custom_dataset/folds.json]
"""

import argparse
import json
import math
import hashlib
import random
from pathlib import Path

SEED = 42
N_SPLITS = 10


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="outpath/TopFilter")
    parser.add_argument("--out", default="custom_dataset/folds.json")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    src = Path(args.src)
    projects_file = src / "projects.txt"
    projects = [line.strip() for line in projects_file.read_text().splitlines() if line.strip()]
    if not projects:
        raise SystemExit(f"no projects found in {projects_file}")

    # Fold membership is a stable function of the project NAME, not of its position in a
    # shuffled list. A list shuffle re-permutes everything when the corpus gains or loses a
    # single project, which silently changes the test set and makes two runs incomparable.
    # Hashing the name keeps every project in the same fold across corpus variants, so runs
    # differing only by a preprocessing change remain like-for-like.
    def fold_of(name):
        digest = hashlib.sha1(f"{args.seed}:{name}".encode()).hexdigest()
        return int(digest[:8], 16) % N_SPLITS

    buckets = [[] for _ in range(N_SPLITS)]
    for name in sorted(projects):
        buckets[fold_of(name)].append(name)
    ordered = [name for bucket in buckets for name in bucket]

    # runner.ten_fold_cross_validation slices contiguous windows of ceil(N/10), so the folds
    # recorded here must be those same windows, not the (uneven) hash buckets. Grouping by
    # bucket first means a corpus that gains or loses a project perturbs only a window
    # boundary instead of re-permuting every project, which is what a list shuffle did.
    step = math.ceil(len(ordered) / N_SPLITS)
    folds = [ordered[i * step:(i + 1) * step] for i in range(N_SPLITS)]

    # projects.txt carries "owner___name"; the rest of the pipeline keys on "owner/name".
    payload = {
        "seed": args.seed,
        "n_splits": N_SPLITS,
        "step": step,
        "num_projects": len(ordered),
        "order": ordered,
        "folds": [[name.replace("___", "/") for name in fold] for fold in folds],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    # Rewrite projects.txt so TopFilter's contiguous windows == these folds.
    projects_file.write_text("\n".join(ordered) + "\n")

    print(f"{len(ordered)} projects -> {N_SPLITS} folds (step {step})")
    for i, fold in enumerate(payload["folds"], 1):
        print(f"  fold {i:>2}: {len(fold):>3} projects  (positions {(i-1)*step+1}..{min(i*step, len(ordered))})")
    print(f"\nwrote {out} and reordered {projects_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
