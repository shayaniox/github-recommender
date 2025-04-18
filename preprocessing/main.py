import preprocessing
import export_to_TopFilter
import export_to_CNBN
import pandas as pd



def get_raw_dataset(filename):
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
INPUT_CSV_PATH = "repositories.csv"

if __name__ == '__main__':
    ### data preprocessing
    print("Loading Dataset")
    df = get_raw_dataset(INPUT_CSV_PATH)
    export_to_CNBN.remove_repository_without_readme_file(df, README_FOLDER)

    print("Preprocessing dataset")
    preprocessed_df = preprocessing.preprocess(df)
    #preprocessed_df = preprocessing.topics_string_to_list(preprocessed_df)
    topics_set = preprocessing.get_topic_frequency_map_topics_as_list(preprocessed_df)
    print(f"#Repos {len(preprocessed_df.index)}")
    print(f"#Topics {len(topics_set)}")

    # CNBN exporter
    print("Exporting CNBN")
    export_to_CNBN.export(preprocessed_df, README_FOLDER, "dataset_j/CNBN/", featured=True)


    #TopFilter exporter
    print("Exporting TopFiter")
    export_to_TopFilter.export(preprocessed_df, directory = "outpath/TopFilter")

    # clean_df.to_csv('filtered_df.csv', index=False)

