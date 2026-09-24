"""Stream `dataset/pretty-dataset.json` into a repo -> description CSV.

The Kaggle dump carries no README text, but it does carry a one-line `description` for every
repository -- 100% coverage on the repos we have mined, median 63 characters. Two uses:

1. Prepended to a README it is a cheap extra feature (small effect; a 63-char line against a
   ~9,400-char document).
2. It is the only text available for the other 880,841 repositories, so if CNBN turns out to be
   usable on descriptions alone, the content stage scales with no GitHub API access at all.

The file is 3.7 GB, so it is parsed incrementally rather than loaded.

Usage:
    python scripts/extract_descriptions.py [--dump dataset/pretty-dataset.json]
                                           [--out repo_descriptions.csv]
                                           [--only-mined]          # restrict to repo_readme/
"""

import argparse
import csv
import json
from pathlib import Path

CHUNK = 1 << 22


def stream_records(path):
    """Yield each top-level object of a pretty-printed JSON array."""
    decoder = json.JSONDecoder()
    buffer = ""
    with open(path, encoding="utf-8", errors="ignore") as handle:
        handle.read(1)  # opening '['
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                break
            buffer += chunk
            while True:
                candidate = buffer.lstrip().lstrip(",").lstrip()
                if not candidate.startswith("{"):
                    break
                try:
                    obj, end = decoder.raw_decode(candidate)
                except ValueError:
                    break  # object straddles the chunk boundary; read more
                buffer = candidate[end:]
                yield obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default="dataset/pretty-dataset.json")
    parser.add_argument("--out", default="repo_descriptions.csv")
    parser.add_argument("--only-mined", action="store_true",
                        help="only emit repositories that have a README on disk")
    parser.add_argument("--readme-dir", default="repo_readme")
    args = parser.parse_args()

    wanted = None
    if args.only_mined:
        wanted = {f.stem.replace(",", "/", 1)
                  for f in Path(args.readme_dir).iterdir() if f.suffix == ".txt"}
        print(f"restricting to {len(wanted):,} mined repositories")

    scanned = written = 0
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for obj in stream_records(args.dump):
            scanned += 1
            name = obj.get("nameWithOwner")
            if not name:
                owner, repo = obj.get("owner"), obj.get("name")
                name = f"{owner}/{repo}" if owner and repo else None
            if not name:
                continue
            if wanted is not None and name not in wanted:
                continue
            description = (obj.get("description") or "").strip()
            # `languages` is a list of {name, size}; the names alone are strong topic evidence
            # (a repo whose languages are TypeScript+CSS is not a Rust CLI) and, like the
            # description, they are available at prediction time for any unseen repository.
            languages = " ".join(
                (lang.get("name") or "").strip()
                for lang in (obj.get("languages") or [])
                if isinstance(lang, dict) and lang.get("name")
            )
            if description or languages:
                writer.writerow([name, description, languages])
                written += 1
            if wanted is not None and written == len(wanted):
                break
            if scanned % 100000 == 0:
                print(f"  scanned {scanned:,}, written {written:,}")

    print(f"scanned {scanned:,} records, wrote {written:,} descriptions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
