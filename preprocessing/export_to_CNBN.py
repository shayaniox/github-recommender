import os
import pandas as pd
import shutil
import preprocessing

def search_file(dir, filename):
    """Locate a README on disk.

        dir='repo_readme', filename='freeCodeCamp,freeCodeCamp.txt'
            -> 'repo_readme/freeCodeCamp,freeCodeCamp.txt'   or None if absent

    The commented-out os.walk below is what made this a "search"; what remains is a
    one-line existence check.
    """
    if os.path.isfile(os.path.join(dir,filename)):
        return os.path.join(dir, filename)
    # for dirpath, dirnames, filenames in os.walk(dir):
    #     for filename in [f for f in filenames if f==filename ]:
    #         return os.path.join(dirpath, filename)

def export(df, dir, out, featured = False):
    """Write the directory-per-topic tree a text classifier expects.

    For a row with

        repo    'freeCodeCamp/freeCodeCamp'
        topics  ['learn', 'nodejs', 'react', 'd3', 'education', 'teacher', 'javascript']

    the same README is copied seven times, once into each topic's directory:

        dataset_j/CNBN/learn/freeCodeCamp,freeCodeCamp.txt
        dataset_j/CNBN/nodejs/freeCodeCamp,freeCodeCamp.txt
        dataset_j/CNBN/react/freeCodeCamp,freeCodeCamp.txt
        ...                                                   <- directory name IS the label

    A multi-label repository therefore becomes N repeated single-label training samples.

    Two caveats. `featured=True` computes and sorts a frequency map that is never used -
    `topics_set` ends up holding every topic in the frame, so the filter admits all of them.
    And nothing here ever deletes: the tree accumulates across runs, which is why it holds
    ~2,357 topic directories against a ~1,289-topic vocabulary.
    """
    if featured:
        print("featured")
        topics_set = preprocessing.get_topic_frequency_map_topics_as_list(df)
        topics_set = {k: v for k, v in
            sorted(topics_set.items(), key=lambda item: item[1], reverse=True)}
    for index, row in df.iterrows():
        filename = f"{row['repo'].replace('/',',')}.txt"
        source = search_file(dir,filename)
        if source == None:
            # print(filename)
            continue
        for topic in row['topics']:
            if(not featured or topic in topics_set):
                path = os.path.join(out,topic)
                if not os.path.isdir(path):
                    os.makedirs(path)
                shutil.copy(source, os.path.join(path,filename))
        # if index % 100 == 0 : print(index)

def remove_repository_without_readme_file(df, dir):
    """Drop rows whose README was never mined.

        repo 'freeCodeCamp/freeCodeCamp' -> looks for 'repo_readme/freeCodeCamp,freeCodeCamp.txt'
        present -> row kept        absent -> row dropped in place

    The `/` -> `,` substitution is the filename convention miner.py wrote with.
    """
    for index, row in df.iterrows():
        print(f"\rindex: {index}", end="")
        filename = f"{row['repo'].replace('/',',')}.txt"
        source = search_file(dir, filename)
        if source == None:
            df.drop(index, inplace=True)
        if index > 1000:
            print("WTF")
    print()
    return df
