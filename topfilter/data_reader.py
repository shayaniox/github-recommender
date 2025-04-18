import os
import math
import logging
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import heapq
import csv

class DataReader:
    def __init__(self, src_dir: str):
        self.src_dir = src_dir
        self.eASEOutput = defaultdict(list)
        self.logger = logging.getLogger(__name__)

    def get_number_of_projects(self, filename: str) -> int:
        count = 0
        try:
            with open(filename, 'r') as f:
                for _ in f:
                    count += 1
        except IOError as e:
            self.logger.error(f"Error reading file {filename}: {e}")
        return count

    def read_repository_list(self, filename: str) -> Dict[int, str]:
        ret = {}
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    line = line.strip()
                    vals = line.split("\t")
                    id = int(vals[0].strip())
                    uri = vals[1].strip()
                    ret[id] = uri
        except IOError as e:
            self.logger.error(f"Error reading file {filename}: {e}")
        return ret

    def read_project_list(self, filename: str, start_pos: int, end_pos: int) -> Dict[int, str]:
        ret = {}
        count = 1
        id = start_pos
        try:
            with open(filename, 'r') as reader:
                while count < start_pos:
                    reader.readline()
                    count += 1
                
                while True:
                    line = reader.readline()
                    if not line or count > end_pos:
                        break
                    line = line.strip()
                    repo = line.split(",")[0].strip()
                    ret[id] = repo
                    id += 1
                    count += 1
        except IOError as e:
            self.logger.error(f"Error reading file {filename}: {e}")
        return ret

    def read_dictionary(self, filename: str) -> Dict[int, str]:
        vector = {}
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    vals = line.split("\t")
                    ID = int(vals[0].strip())
                    artifact = vals[1].strip()
                    if ID == 1 or "#DEP#" in artifact:
                        vector[ID] = artifact
        except IOError as e:
            self.logger.error(f"Error reading file {filename}: {e}")
        return vector

    def extract_half_dictionary(self, filename: str, ground_truth_path: str, get_also_users: bool) -> Dict[int, str]:
        dictionary = {}
        ret = {}
        lib_count = 0
        
        fname = os.path.basename(filename).replace("dicth_", "")
        ground_truth_file = os.path.join(ground_truth_path, fname)
        
        try:
            with open(filename, 'r') as reader, open(ground_truth_file, 'w') as writer:
                for line in reader:
                    vals = line.split("\t")
                    ID = int(vals[0].strip())
                    artifact = vals[1].strip()
                    dictionary[ID] = artifact
                    if "#DEP#" in artifact:
                        lib_count += 1
                
                half = round(lib_count / 2)
                enough_lib = False
                lib_count = 0
                
                for key, artifact in dictionary.items():
                    if lib_count == half:
                        enough_lib = True
                    
                    if "#DEP#" in artifact:
                        if not enough_lib:
                            ret[key] = artifact
                        else:
                            content = f"{key}\t{artifact}"
                            writer.write(content + "\n")
                            writer.flush()
                        lib_count += 1
                    else:
                        if get_also_users or "#DEP#" not in artifact:
                            ret[key] = artifact
        except IOError as e:
            self.logger.error(f"Error processing dictionary {filename}: {e}")
        
        return ret

    def load_EASE_output(self):
        filename = os.path.join(self.src_dir, "training_data.csv")
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    values = line.split(";")
                    repo_name = values[0].replace(".txt", "")
                    for z in values[1:]:
                        self.eASEOutput[repo_name].append("#DEP# " + z.strip())
        except IOError as e:
            self.logger.error(f"Error loading EASE output {filename}: {e}")

    def get_EASE_topic(self, project_name: str, n: int) -> List[str]:
        if not self.eASEOutput:
            self.load_EASE_output()
        return self.eASEOutput.get(project_name.replace("___", "/"), [])[:n]

    def get_EASE_output(self):
        if not self.eASEOutput:
            self.load_EASE_output()
        return self.eASEOutput

    def extract_EASE_dictionary(self, filename: str, number_of_topics: int, ground_truth_path: str) -> Dict[int, str]:
        if not self.eASEOutput:
            self.load_EASE_output()
            
        reponame = os.path.basename(filename).replace("dicth_", "").replace("___", "/")
        topics = self.eASEOutput.get(reponame, [])[:number_of_topics]
        reponame = "git://github.com/" + reponame
        ret = {1: reponame}
        
        i = 2
        for topic in topics:
            ret[i] = topic.strip()
            i += 1
            
        fname = os.path.basename(filename).replace("dicth_", "")
        ground_truth_file = os.path.join(ground_truth_path, fname)
        
        try:
            with open(filename, 'r') as reader, open(ground_truth_file, 'w') as writer:
                dictionary = {}
                for line in reader:
                    vals = line.split("\t")
                    ID = int(vals[0].strip())
                    artifact = vals[1].strip()
                    if artifact.startswith("#DEP#"):
                        dictionary[ID] = artifact
                
                for key, artifact in dictionary.items():
                    content = f"{key}\t{artifact}"
                    writer.write(content + "\n")
                    writer.flush()
        except IOError as e:
            self.logger.error(f"Error processing EASE dictionary {filename}: {e}")
        
        return ret

    def get_libraries(self, filename: str) -> Set[str]:
        libraries = set()
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    vals = line.split("\t")
                    library = vals[1].strip()
                    if "#DEP#" in library:
                        libraries.add(library)
        except IOError as e:
            self.logger.error(f"Error reading libraries from {filename}: {e}")
        return libraries

    def get_most_similar_projects(self, filename: str, size: int) -> Dict[int, str]:
        projects = {}
        count = 0
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    if count >= size:
                        break
                    vals = line.split("\t")
                    URI = vals[1].strip()
                    projects[count] = URI
                    count += 1
        except IOError as e:
            self.logger.error(f"Error reading similar projects from {filename}: {e}")
        return projects

    def get_similarity_matrix(self, filename: str, size: int) -> Dict[int, float]:
        sim = {}
        count = 0
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    if count >= size:
                        break
                    vals = line.split("\t")
                    val = float(vals[2].strip())
                    sim[count] = val
                    count += 1
        except IOError as e:
            self.logger.error(f"Error reading similarity matrix from {filename}: {e}")
        return sim

    def read_recommendation_file(self, filename: str) -> Dict[int, str]:
        ret = {}
        id = 1
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    if id > 50:
                        break
                    vals = line.split("\t")
                    library = vals[0].strip()
                    ret[id] = library
                    id += 1
        except IOError as e:
            self.logger.error(f"Error reading recommendation file {filename}: {e}")
        return ret

    def read_all_recommendations(self, filename: str) -> Dict[int, str]:
        ret = {}
        id = 1
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    vals = line.split("\t")
                    val = float(vals[1].strip())
                    if val != 0:
                        library = vals[0].strip()
                        ret[id] = library
                        id += 1
        except IOError as e:
            self.logger.error(f"Error reading all recommendations from {filename}: {e}")
        return ret

    def read_long_tail_items(self, filename: str) -> Set[str]:
        ret = set()
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    vals = line.split("\t")
                    library = vals[0].strip()
                    ret.add(library)
        except IOError as e:
            self.logger.error(f"Error reading long tail items from {filename}: {e}")
        return ret

    def read_recommendation_file_with_size(self, filename: str, size: int) -> Set[str]:
        ret = set()
        count = 0
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    if count >= size:
                        break
                    vals = line.split("\t")
                    library = vals[0].strip()
                    ret.add(library)
                    count += 1
        except IOError as e:
            self.logger.error(f"Error reading recommendation file {filename} with size {size}: {e}")
        return ret

    def read_recommendation_scores(self, filename: str) -> Dict[str, float]:
        ret = {}
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    vals = line.split("\t")
                    item = vals[0].strip()
                    score = float(vals[1].strip())
                    ret[item] = score
        except IOError as e:
            self.logger.error(f"Error reading recommendation scores from {filename}: {e}")
        return ret

    def read_ground_truth_file(self, filename: str) -> Set[str]:
        ret = set()
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    vals = line.split("\t")
                    library = vals[1].strip()
                    ret.add(library)
        except IOError as e:
            self.logger.error(f"Error reading ground truth file {filename}: {e}")
        return ret

    def read_ground_truth_score(self, filename: str) -> Dict[str, float]:
        ret = {}
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    vals = line.split("\t")
                    temp = vals[1].strip().split("%")
                    item = temp[0].strip()
                    rating = float(temp[1].strip())
                    ret[item] = rating
        except IOError as e:
            self.logger.error(f"Error reading ground truth scores from {filename}: {e}")
        return ret

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Runner.main()
