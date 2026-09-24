"""Stage E, part 1 - materialise the ten fold trees CNBN trains from.

RUNNING EXAMPLE, carried through from preprocessing/. Row 1 of `repositories_test.csv`:

    repo    'freeCodeCamp/freeCodeCamp'
    topics  learn-to-code nonprofits programming nodejs react d3 careers education teachers javascript

Note these are the RAW mined topics, not Stage C's normalized vocabulary - this module reads
`repositories_test.csv` directly, so CNBN learns classes like 'learn-to-code' and 'careers'
while TopFilter works with 'learn' and 'career'. That mismatch is docs/ARCHITECTURE.md
section 4, item 12, still open.

If the repo lands in fold 3's test set, this module writes:

    custom_dataset/CNBN/folds_sync/test3/learn-to-code/freeCodeCamp,freeCodeCamp.txt
    custom_dataset/CNBN/folds_sync/test3/nonprofits/freeCodeCamp,freeCodeCamp.txt
    ...                                                        one copy per topic, x10
    custom_dataset/CNBN/folds_sync/test_3.txt                  'freeCodeCamp/freeCodeCamp'

and for folds 1, 2, 4..10 the same copies under train<k>/ instead.
"""

import csv
import json
from pathlib import Path
import shutil

def prepare_datasets(
    csv_path: Path,
    readme_dir: Path = Path("./repo_readme"),
    output_dir: Path = Path("custom_dataset/CNBN/folds_sync"),
    folds_path: Path = Path("custom_dataset/folds.json"),
    folds=None,
):
    # Read repository data
    #   row     ['freeCodeCamp/freeCodeCamp', '408888', '38677', '18688', 'learn-to-code', ...]
    #   row[0]  repo name       row[4:]  topics       rows with <5 fields are malformed
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        repos = [row for row in reader if len(row) > 4]  # Skip malformed rows

    # Fold membership comes from the shared assignment written by scripts/make_folds.py,
    # the same file TopFilter's project ordering is built from. Splitting independently
    # here is what left only 8 of 65 TopFilter test projects with a CNBN seed.
    if not folds_path.exists():
        raise SystemExit(
            f"{folds_path} not found - run 'python scripts/make_folds.py' first so CNBN "
            f"and TopFilter share one fold assignment."
        )
    #   assignment  {"folds": [["a/b", "c/d", ...],      <- fold 1 test set, ~65 names
    #                            ["e/f", ...], ...]}       <- ten of them
    #   universe    every name in every fold - the ~644 repos that survived preprocessing
    assignment = json.loads(folds_path.read_text())
    universe = {name for names in assignment["folds"] for name in names}
    if folds is None:
        folds = range(1, len(assignment["folds"]) + 1)

    # Restrict to the shared universe so both stages see the same repositories.
    #   1,000 mined rows -> ~644 rows whose repo also has dicth_/graph_ files
    repos = [row for row in repos if row[0] in universe]
    if not repos:
        raise SystemExit("no repositories in csv_path overlap the shared fold universe")

    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        #   fold=3 -> test_names {'freeCodeCamp/freeCodeCamp', 'pytorch/pytorch', ...}  ~65 names
        #             everything else in `repos` is training data for this fold
        test_names = set(assignment["folds"][fold - 1])
        print(f"fold {fold}: {len(test_names)} test repositories")

        # Create train/test directories, CLEARING any previous contents first.
        # These directories used to be reused across runs with mkdir(exist_ok=True), so when
        # the fold assignment changed between runs, README copies from earlier folds stayed
        # behind. That put 63 of 65 test repositories into the training set and inflated
        # every downstream metric. Always start from an empty tree.
        train_dir = output_dir / f"train{fold}"
        test_dir = output_dir / f"test{fold}"
        for stale in (train_dir, test_dir):
            if stale.exists():
                shutil.rmtree(stale)
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        # Flat list of this fold's test repos, one 'owner/name' per line. model.py reads it to
        # look up ground truth; it does NOT drive which files get classified.
        #   test_3.txt   'freeCodeCamp/freeCodeCamp'
        #                'pytorch/pytorch'   ...
        test_file = output_dir / f"test_{fold}.txt"
        with test_file.open('w', encoding='utf-8') as f:
            for row in repos:
                if row[0] in test_names:
                    f.write(f"{row[0]}\n")

        # Process each repository
        for repo in repos:
            #   repo_name  'freeCodeCamp/freeCodeCamp'
            #   topics     ['learn-to-code', 'nonprofits', 'programming', 'nodejs', 'react',
            #               'd3', 'careers', 'education', 'teachers', 'javascript']
            repo_name = repo[0]
            topics = repo[4:]

            if not topics:  # Skip repos with no topics
                continue

            # Determine if this is train or test repo
            target_dir = test_dir if repo_name in test_names else train_dir
            
            # Use existing README file
            #   'freeCodeCamp/freeCodeCamp' -> 'repo_readme_clean/freeCodeCamp,freeCodeCamp.txt'
            readme_file = readme_dir / f"{repo_name.replace('/', ',')}.txt"
            if not readme_file.exists():
                continue
                
            # One copy of the README per topic; the directory name is the class label. The
            # same text therefore becomes 10 single-label training samples for this repo.
            #   train3/learn-to-code/freeCodeCamp,freeCodeCamp.txt
            #   train3/nodejs/freeCodeCamp,freeCodeCamp.txt        ... x10
            for topic in topics:
                topic_dir = target_dir / topic
                topic_dir.mkdir(exist_ok=True)
                
                # Create symlink to original README (or copy if symlinks not supported)
                repo_file = topic_dir / f"{repo_name.replace('/', ',')}.txt"
                # try:
                #     if not repo_file.exists():
                #         repo_file.symlink_to(readme_file)  # Use symlinks to save space
                #         # Alternative for Windows: shutil.copy2(readme_file, repo_file)
                # except OSError:
                shutil.copy2(readme_file, repo_file)

if __name__ == "__main__":
    # repositories.csv holds 882k rows but only the ~1k repos in repo_readme/ have a
    # README on disk, and the rest are skipped. Use the same mined subset that
    # preprocessing/main.py consumes so CNBN and TopFilter share one repo universe.
    # All ten folds by default; pass --folds to rebuild a subset.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--readme-dir", default="repo_readme_clean",
                        help="README corpus to build the fold trees from")
    parser.add_argument("--output-dir", default="custom_dataset/CNBN/folds_sync")
    parser.add_argument("--csv", default="repositories_test.csv")
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    args = parser.parse_args()

    prepare_datasets(
        csv_path=Path(args.csv),
        readme_dir=Path(args.readme_dir),
        output_dir=Path(args.output_dir),
        folds=args.folds,
    )
