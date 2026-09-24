"""Compare a TopFilter run's metrics against the HybridRec paper's published results.

Usage:
    python scripts/compare_metrics.py <results_dir> [<results_dir> ...] [--labels a,b]

Each <results_dir> is a directory holding PRC_Round1 / SR_Round1 / PRCB_Round1 /
SRB_Round1 etc., i.e. outpath/TopFilter/Results or a snapshot of it.

Reference numbers are Table 5 of Di Rocco et al., "HybridRec: A recommender system for
tagging GitHub repositories", Applied Intelligence 53:9708-9730 (2023), p. 9719 --
averaged over datasets D1, D2, D3.
"""

import sys
from pathlib import Path

# HybridRec Table 5 (p. 9719): N -> (SR, precision, recall)
HYBRIDREC = {
    1: (0.153, 0.212, 0.034), 2: (0.236, 0.179, 0.056), 3: (0.291, 0.153, 0.070),
    4: (0.317, 0.133, 0.082), 5: (0.335, 0.117, 0.090), 6: (0.504, 0.133, 0.122),
    7: (0.549, 0.137, 0.145), 8: (0.579, 0.136, 0.164), 9: (0.598, 0.132, 0.179),
    10: (0.614, 0.127, 0.191),
}
# TopFilter+ baseline from the same table, for context.
TOPFILTER_PLUS = {
    1: (0.171, 0.113, 0.014), 2: (0.223, 0.077, 0.018), 3: (0.258, 0.060, 0.021),
    4: (0.276, 0.052, 0.024), 5: (0.290, 0.044, 0.025), 6: (0.382, 0.084, 0.058),
    7: (0.417, 0.099, 0.079), 8: (0.449, 0.101, 0.091), 9: (0.465, 0.099, 0.100),
    10: (0.479, 0.095, 0.108),
}


def _rows(path: Path):
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split("\t")
        if not parts or not parts[0]:
            continue
        try:
            out[int(parts[0])] = [float(v) for v in parts[1:]]
        except ValueError:
            continue
    return out


def fold_sizes(folds_path=Path("custom_dataset/folds.json")):
    """Test-set size per fold, used to weight the cross-fold average."""
    if not folds_path.exists():
        return {}
    import json
    payload = json.loads(folds_path.read_text())
    return {i: len(fold) for i, fold in enumerate(payload["folds"], 1)}


def load(results_dir: Path, bayesian: bool, n_folds: int = 10):
    """Return {N: (success_rate, precision, recall)} averaged over every fold present.

    Each Round file holds that fold's own per-project mean, so the cross-fold figure is a
    size-weighted average: a micro-average over the projects actually evaluated. Folds with
    no output file are skipped and reported, rather than silently counted as zero.
    """
    suffix = "B" if bayesian else ""
    weights = fold_sizes()
    per_fold = []
    for k in range(1, n_folds + 1):
        prc = _rows(results_dir / f"PRC{suffix}_Round{k}")
        sr = _rows(results_dir / f"SR{suffix}_Round{k}")
        if not prc and not sr:
            continue
        per_fold.append((weights.get(k, 1), prc, sr))

    load.folds_found = len(per_fold)
    metrics = {}
    for n in range(1, 11):
        totals = [0.0, 0.0, 0.0]
        seen = [0.0, 0.0, 0.0]
        for weight, prc, sr in per_fold:
            # metrics.py writes "rank\trecall\tprecision" (see precision_recall*), so the
            # first value after the rank is RECALL, not precision.
            row = prc.get(n)
            recall, precision = (row + [None, None])[:2] if row else (None, None)
            success = sr.get(n, [None])[0]
            for idx, value in enumerate((success, precision, recall)):
                if value is not None:
                    totals[idx] += value * weight
                    seen[idx] += weight
        metrics[n] = tuple(
            totals[i] / seen[i] if seen[i] else None for i in range(3)
        )
    return metrics


def fmt(v):
    return "  --  " if v is None else f"{v:.3f}"


def main(argv):
    labels = None
    if "--labels" in argv:
        i = argv.index("--labels")
        labels = argv[i + 1].split(",")
        argv = argv[:i] + argv[i + 2:]
    dirs = [Path(a) for a in argv[1:]]
    if not dirs:
        print(__doc__)
        return 1
    if not labels:
        labels = [d.name for d in dirs]

    runs = []
    for lab, d in zip(labels, dirs):
        metrics = load(d, bayesian=True)
        runs.append((lab, metrics))
        print(f"{lab}: averaged over {load.folds_found} fold(s) in {d}")

    for metric_idx, metric_name in enumerate(("success rate", "precision", "recall")):
        print(f"\n=== {metric_name} @N ===")
        header = f"{'N':>3} | {'TopFilter+':>10} {'HybridRec':>10} |" + "".join(
            f"{lab:>14}" for lab, _ in runs)
        print(header)
        print("-" * len(header))
        for n in range(1, 11):
            row = f"{n:>3} | {TOPFILTER_PLUS[n][metric_idx]:>10.3f} {HYBRIDREC[n][metric_idx]:>10.3f} |"
            for _, m in runs:
                row += f"{fmt(m[n][metric_idx]):>14}"
            print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
