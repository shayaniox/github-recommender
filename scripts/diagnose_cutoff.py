"""Why does the hybrid stop improving at N>=6?

Measures, on the recommendation files a TopFilter run actually produced:

1. Seed duplication — how much of the evaluated top-10 window is re-scoring the CNBN
   seeds. `metrics.success_rate_b` / `precision_recall_b` score ranks 1..5 from
   `get_EASE_topic`, then walk the recommendation file *from its first line* and place it
   at rank `key + 5`. The Topics Merger writes those same 5 seeds at the head of the
   recommendation file, so ranks 6..10 re-score ranks 1..5 and the first CF topic is
   evaluated at rank 11 — outside the reported window.

2. The precision ceiling — with |GT| ground-truth topics, precision@N cannot exceed
   min(|GT|, N)/N, which falls as 1/N once N passes |GT|.

3. What the top-10 would contain if duplicates did not consume rank slots.

Usage:
    python scripts/diagnose_cutoff.py [--src outpath/TopFilter] [--folds 10]
"""

import argparse
import statistics
from pathlib import Path


def load_ease(src):
    ease = {}
    path = src / "training_data.csv"
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.strip().split(";")]
        if len(parts) > 1:
            ease[parts[0].replace(".txt", "")] = ["#DEP#" + t for t in parts[1:] if t]
    return ease


def read_list(path):
    return [line.split("\t")[0].strip() for line in path.read_text().splitlines() if line.strip()]


def read_ground_truth(path):
    topics = set()
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) > 1 and "#DEP#" in parts[1]:
            topics.add(parts[1].strip())
    return topics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="outpath/TopFilter")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    src = Path(args.src)
    ease = load_ease(src)

    dup_head = 0
    projects = 0
    first_new_rank = []
    gt_sizes = []
    list_lens = []
    # hits[n] = correct topics within the first n evaluated ranks, summed over projects
    hits_dup = {n: 0 for n in range(1, 21)}
    hits_dedup = {n: 0 for n in range(1, 21)}
    sr_dup = {n: 0 for n in range(1, 21)}
    sr_dedup = {n: 0 for n in range(1, 21)}

    for fold in range(1, args.folds + 1):
        rec_dir = src / f"Round{fold}" / "Recommendations"
        gt_dir = src / f"Round{fold}" / "GroundTruth"
        if not rec_dir.exists():
            continue
        for rec_file in sorted(rec_dir.iterdir()):
            if rec_file.name == ".gitkeep":
                continue
            gt_file = gt_dir / rec_file.name
            if not gt_file.exists():
                continue
            repo = rec_file.name.replace("___", "/")
            seeds = ease.get(repo, [])[:args.seeds]
            if not seeds:
                continue

            rec = read_list(rec_file)
            gt = read_ground_truth(gt_file)
            if not gt:
                continue

            projects += 1
            gt_sizes.append(len(gt))
            list_lens.append(len(rec))
            if rec[:args.seeds] == seeds:
                dup_head += 1

            # The ranking the metrics actually score: seeds, then the rec file from line 1.
            scored_dup = seeds + rec
            # The ranking they would score if duplicates did not consume a slot.
            scored_dedup = list(dict.fromkeys(seeds + rec))

            new_at = next((i + 1 for i, t in enumerate(rec) if t not in set(seeds)), None)
            if new_at is not None:
                first_new_rank.append(new_at + args.seeds)

            for name, scored, hits, srs in (
                ("dup", scored_dup, hits_dup, sr_dup),
                ("dedup", scored_dedup, hits_dedup, sr_dedup),
            ):
                seen = set()
                for n in range(1, 21):
                    if n <= len(scored):
                        seen.add(scored[n - 1])
                    correct = len(seen & gt)
                    hits[n] += correct
                    srs[n] += 1 if correct else 0

    if not projects:
        raise SystemExit("no scored projects found - run topfilter/runner.py first")

    print(f"projects scored: {projects}\n")

    print("=== 1. seed duplication inside the evaluated window ===")
    print(f"recommendation files whose first {args.seeds} entries ARE the {args.seeds} seeds: "
          f"{dup_head}/{projects}")
    if first_new_rank:
        print(f"evaluated rank of the first genuinely new (CF) topic: "
              f"median {statistics.median(first_new_rank):.0f}, mean {statistics.mean(first_new_rank):.1f}")
    print(f"recommendation list length: median {statistics.median(list_lens):.0f}, "
          f"shorter than 15: {sum(l < 15 for l in list_lens)}/{projects}\n")

    print("=== 2. ground-truth size (the precision ceiling) ===")
    print(f"|GT| per project: median {statistics.median(gt_sizes):.0f}, "
          f"mean {statistics.mean(gt_sizes):.1f}, "
          f"<=5: {sum(g <= 5 for g in gt_sizes)}/{projects}")
    print("  N  ceiling=mean(min(|GT|,N))/N")
    for n in (1, 5, 10, 20):
        ceiling = statistics.mean(min(g, n) for g in gt_sizes) / n
        print(f"  {n:>2}  {ceiling:.3f}")
    print()

    print("=== 3. as-scored vs de-duplicated top-10 ===")
    print(f"{'N':>3} | {'P@N dup':>8} {'P@N dedup':>10} | {'SR@N dup':>9} {'SR@N dedup':>11}")
    print("-" * 52)
    for n in range(1, 11):
        print(f"{n:>3} | {hits_dup[n]/projects/n:>8.3f} {hits_dedup[n]/projects/n:>10.3f} | "
              f"{sr_dup[n]/projects:>9.3f} {sr_dedup[n]/projects:>11.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
