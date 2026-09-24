"""Re-run evaluation only, over recommendation files that already exist.

The similarity and recommendation stages are the expensive part of a TopFilter run and do
not depend on how the metrics are computed, so a scoring change can be evaluated against
the same recommendations instead of re-running the whole pipeline.

Usage:
    python scripts/rescore.py [--src outpath/TopFilter] [--out results/tenfold/<name>]
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "topfilter"))

from validator import Validator  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="outpath/TopFilter")
    parser.add_argument("--out", default=None, help="copy the Results directory here afterwards")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    src = str(Path(args.src).resolve()) + "/"

    Validator(src, bayesian=True).run()

    if args.out:
        out = Path(args.out)
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(Path(args.src) / "Results", out)
        print(f"copied Results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
