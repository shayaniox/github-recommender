"""Score CNBN's seeds directly, without running the collaborative stage.

The end-to-end metrics are partly insensitive to seed quality: the CF pins the seeds at ranks
1..5 and then adds its own topics, so a better seed set can be masked. This measures the seed
set itself, which is what a change to the README text actually affects.

Ground truth is the repo's normalised topic set from `outpath/TopFilter/dicth_<repo>` -- the
same set the pipeline's own metrics score against.

Usage:
    python scripts/seed_quality.py results/training_data.csv [more.csv ...] [--labels a,b]
"""

import argparse
from pathlib import Path

DEP = "#DEP#"


def ground_truth(src=Path("outpath/TopFilter")):
    truth = {}
    for path in src.glob("dicth_*"):
        repo = path.name[len("dicth_"):].replace("___", "/")
        topics = set()
        for line in path.read_text(errors="ignore").splitlines():
            parts = line.split("\t")
            if len(parts) > 1 and parts[1].strip().startswith(DEP):
                topics.add(parts[1].strip()[len(DEP):])
        if topics:
            truth[repo] = topics
    return truth


def seeds(path, n=5):
    out = {}
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [p.strip() for p in line.strip().split(";")]
        if len(parts) > 1:
            out[parts[0].replace(".txt", "")] = [t for t in parts[1:] if t][:n]
    return out


def score(pred, truth, n):
    hits = total = covered = 0
    any_hit = 0
    for repo, topics in pred.items():
        gt = truth.get(repo)
        if not gt:
            continue
        top = topics[:n]
        correct = len(set(top) & gt)
        hits += correct
        total += len(top)
        covered += 1
        any_hit += 1 if correct else 0
    return {
        "repos": covered,
        "precision": hits / total if total else 0.0,
        "hit_rate": any_hit / covered if covered else 0.0,
        "mean_correct": hits / covered if covered else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--labels", default=None)
    args = parser.parse_args()

    labels = args.labels.split(",") if args.labels else [Path(f).stem for f in args.files]
    truth = ground_truth()
    print(f"ground-truth repos: {len(truth):,}\n")

    header = f"{'run':<22}{'repos':>7}{'seeds@5 prec':>14}{'>=1 correct':>13}{'mean correct/5':>16}"
    print(header)
    print("-" * len(header))
    for label, path in zip(labels, args.files):
        s = score(seeds(path), truth, 5)
        print(f"{label:<22}{s['repos']:>7,}{s['precision']:>13.3f} {s['hit_rate']:>12.3f} "
              f"{s['mean_correct']:>15.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
