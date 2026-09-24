"""Stage C+D driver: read the mined CSV, normalize topics, export both formats.

Data shapes at each hand-off, following row 1 of `repositories_test.csv`:

    file      'freeCodeCamp/freeCodeCamp,408888,38677,18688,learn-to-code,nonprofits,...'
    df        repo='freeCodeCamp/freeCodeCamp'
              topics='learn-to-code,nonprofits,programming,nodejs,react,d3,careers,...'   (str)
    preprocess   topics=['learn','nodejs','react','d3','education','teacher','javascript']  (list)
    export       dataset_j/CNBN/<topic>/freeCodeCamp,freeCodeCamp.txt        one copy per topic
                 outpath/TopFilter/dicth_freeCodeCamp___freeCodeCamp        one node per topic
"""

import preprocessing
import export_to_TopFilter
import export_to_CNBN
import pandas as pd


def get_raw_dataset(filename):
    """Parse `repositories_test.csv` into a DataFrame.

        line      'freeCodeCamp/freeCodeCamp,408888,38677,18688,learn-to-code,nonprofits,...'
        elements  ['freeCodeCamp/freeCodeCamp', '408888', '38677', '18688', 'learn-to-code', ...]
        repo      'freeCodeCamp/freeCodeCamp'
        topics    'learn-to-code,nonprofits,programming,nodejs,react,d3,careers,education,...'

    Two hazards. The columns are labelled fork/open_issue/star below, but the file is
    written repo,stars,forks,issues - so `stars` in the frame is really the issue count
    (18688 here, not 408888). And the split is a plain `str.split(',')` with no CSV quote
    handling, so a quoted field would leak its quotes into a topic name.
    """
    repos = []
    topics = []
    star = []
    fork = []
    open_issue = []
    with open(filename, "r") as reader:
        for row in reader:
            elements = row.replace("\n", "").split(",")
            repo = elements[0]
            fork_row = int(elements[1])

            open_issue_row = int(elements[2])
            star_row = int(elements[3])

            topics_row = elements[4:]
            repos.append(repo)
            topics.append(','.join(topics_row))
            star.append(star_row)
            fork.append(fork_row)
            open_issue.append(open_issue_row)

    # Repo,fork,open_issue,star,Lib
    df = pd.DataFrame({'repo': repos,
                       'topics': topics,
                       'stars': star,
                       'forks': fork,
                       'open_issues': open_issue})
    df = df.drop_duplicates(subset="repo")
    return df


README_FOLDER = "repo_readme"
INPUT_CSV_PATH = "repositories_test.csv"
TOPFILTER_OUT = "outpath/TopFilter"
CNBN_OUT = "dataset_j/CNBN/"

if __name__ == '__main__':
    # Defaults reproduce the original run; the flags let an enlarged corpus be preprocessed
    # into its own directories without touching outpath/TopFilter.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_CSV_PATH)
    parser.add_argument("--readme-dir", default=README_FOLDER)
    parser.add_argument("--topfilter-out", default=TOPFILTER_OUT)
    parser.add_argument("--cnbn-out", default=CNBN_OUT)
    args = parser.parse_args()
    INPUT_CSV_PATH, README_FOLDER = args.input, args.readme_dir
    TOPFILTER_OUT, CNBN_OUT = args.topfilter_out, args.cnbn_out

    # data preprocessing
    print("Loading Dataset")
    df = get_raw_dataset(INPUT_CSV_PATH)
    export_to_CNBN.remove_repository_without_readme_file(df, README_FOLDER)

    print("Preprocessing dataset")
    # 13 rewriting passes. Per repo, e.g.:
    #   before  'learn-to-code,nonprofits,programming,nodejs,react,d3,careers,education,teachers,javascript'
    #   after   ['learn', 'nodejs', 'react', 'd3', 'education', 'teacher', 'javascript']
    # Corpus-wide on the 1,000-repo mined set: ~644 repos survive with ~1,289 distinct topics.
    preprocessed_df = preprocessing.preprocess(df)

    # preprocessed_df = preprocessing.topics_string_to_list(preprocessed_df)
    #   topics_set  {'javascript': 312, 'python': 287, ..., 'teacher': 3}
    topics_set = preprocessing.get_topic_frequency_map_topics_as_list(
        preprocessed_df)
    print(f"#Repos {len(preprocessed_df.index)}")
    print(f"#Topics {len(topics_set)}")

    # CNBN exporter: one directory per topic, the README copied into each.
    #   dataset_j/CNBN/learn/freeCodeCamp,freeCodeCamp.txt
    #   dataset_j/CNBN/react/freeCodeCamp,freeCodeCamp.txt        ... x7, one per topic
    # NOTE: this tree is currently orphaned - cnbn/ trains from custom_dataset/CNBN/folds_sync/,
    # built separately by generate_datatset.py. See docs/ARCHITECTURE.md section 4, item 12.
    print("Exporting CNBN")
    export_to_CNBN.export(preprocessed_df, README_FOLDER,
                          CNBN_OUT, featured=True)

    # TopFilter exporter: the CrossRec graph triplet.
    #   projects.txt                            'freeCodeCamp___freeCodeCamp'
    #   dicth_freeCodeCamp___freeCodeCamp       '1\tfreeCodeCamp___freeCodeCamp'
    #                                           '2\t#DEP#learn'  '3\t#DEP#nodejs'  ...
    #   graph_freeCodeCamp___freeCodeCamp       '1#2'  '1#3'  ...  '1#8'
    print("Exporting TopFiter")
    export_to_TopFilter.export(preprocessed_df, directory=TOPFILTER_OUT)

    # clean_df.to_csv('filtered_df.csv', index=False)
