import pandas as pd
import os.path

def export(dataset, directory):
    """Write the CrossRec graph triplet, one dicth_/graph_ pair per repository.

    For a row with

        repo    'freeCodeCamp/freeCodeCamp'
        topics  ['learn', 'nodejs', 'react', 'd3', 'education', 'teacher', 'javascript']

    three artifacts appear in `directory`:

        projects.txt  (one line appended)
            freeCodeCamp___freeCodeCamp             <- line number is the project ID

        dicth_freeCodeCamp___freeCodeCamp           <- node dictionary, ID <TAB> artifact
            1   freeCodeCamp___freeCodeCamp         <- ID 1 is always the repo itself
            2   #DEP#learn
            3   #DEP#nodejs
            ...
            8   #DEP#javascript

        graph_freeCodeCamp___freeCodeCamp           <- edge list, a star from the repo node
            1#2
            1#3
            ...
            1#8

    The `#DEP#` prefix is inherited from CrossRec, where the items were software
    *dependencies*; here they are topics. Slashes become `___` so the name is a valid
    filename: 'freeCodeCamp/freeCodeCamp' -> 'freeCodeCamp___freeCodeCamp'.
    """
    projects_list = []
    for index, row in dataset.iterrows():
        #print(row['repo'], row['topics'])
        projects_list.append(row["repo"].replace("/","___"))
        dicth_filename = f'dicth_{row["repo"].replace("/","___")}'
        graph_filename = f'graph_{row["repo"].replace("/","___")}'
        with open(os.path.join(directory, dicth_filename),"w") as writer:
            writer.write(f'1\t{row["repo"].replace("/", "___")}\n')
            for i,topic in enumerate(row['topics']):
                writer.write(f'{i+2}\t#DEP#{topic}\n')

        with open(os.path.join(directory, graph_filename),"w") as writer:
            for i in range(2,len(row['topics'])+2):
                writer.write(f"1#{str(i)}\n")
    with open(os.path.join(directory,"projects.txt"), "w") as writer:
        for proj in projects_list:
            writer.write(f"{proj}\n")