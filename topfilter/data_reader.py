import os
import math
import logging
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import heapq
import csv

_KG_VOCAB = None
_kg_canonical = None
if os.environ.get("KG_NORMALISE") == "1":
    from kg_normalise import canonical as _kg_canonical, load_vocabulary as _kg_load
    _KG_VOCAB = _kg_load()


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
        """Read a 1-based, inclusive window of `projects.txt`.

            projects.txt  line 1  'freeCodeCamp___freeCodeCamp'
                          line 2  'pytorch___pytorch'   ...

            start_pos=131, end_pos=195  ->  {131: 'freeCodeCamp___freeCodeCamp', 132: ...}

        The key is the line number, which is the project ID the fold windows are expressed in.
        """
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
        """Half mode: show the first half of a test repo's real topics, hold back the rest.

            dicth_freeCodeCamp___freeCodeCamp
                1  freeCodeCamp___freeCodeCamp
                2  #DEP#learn        3  #DEP#nodejs    4  #DEP#react     5  #DEP#d3
                6  #DEP#education    7  #DEP#teacher   8  #DEP#javascript

            lib_count=7, half=round(7/2)=4

            returns   {2: #DEP#learn, 3: #DEP#nodejs, 4: #DEP#react, 5: #DEP#d3}   visible
            writes    GroundTruth/  '6\t#DEP#education'  '7\t#DEP#teacher'
                                    '8\t#DEP#javascript'                          held out

        Note the repo node (ID 1) is returned only when `get_also_users` is set; callers pass
        False, so the visible dictionary is topics only. Contrast extract_EASE_dictionary,
        which returns no real topics at all.
        """
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
        """Parse CNBN's seed file into `self.eASEOutput`, cached after the first call.

            line   'freeCodeCamp/freeCodeCamp;javascript;education;react;nodejs;learn-to-code;...'
            ->     eASEOutput['freeCodeCamp/freeCodeCamp'] = ['#DEP#javascript', '#DEP#education',
                                                              '#DEP#react', '#DEP#nodejs', ...]

        "EASE output" is a CrossRec-era name for whatever seeds the collaborative stage; here
        it is always CNBN. The '.txt' strip handles rows CNBN wrote keyed by filename.
        """
        filename = os.path.join(self.src_dir, "training_data.csv")
        try:
            with open(filename, 'r') as reader:
                for line in reader:
                    values = line.split(";")
                    repo_name = values[0].replace(".txt", "")
                    for z in values[1:]:
                        # No space after the prefix: export_to_TopFilter writes "#DEP#<topic>",
                        # and ground truth is compared by exact string, so a spaced variant
                        # can never match and silently zeroes nDCG/EPC.
                        topic = z.strip()
                        if _KG_VOCAB is not None:
                            topic = _kg_canonical(topic, _KG_VOCAB) or topic
                        self.eASEOutput[repo_name].append("#DEP#" + topic)
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
        """Bayesian mode: the test repo's visible profile is CNBN's top 5 and nothing else.

            filename    .../dicth_freeCodeCamp___freeCodeCamp
            reponame    'freeCodeCamp/freeCodeCamp'          ('___' -> '/')
            topics      ['#DEP#javascript', '#DEP#education', '#DEP#react', '#DEP#nodejs',
                         '#DEP#learn-to-code']                first 5 of CNBN's 20

            returns   {1: 'git://github.com/freeCodeCamp/freeCodeCamp',
                       2: '#DEP#javascript', 3: '#DEP#education', 4: '#DEP#react',
                       5: '#DEP#nodejs',     6: '#DEP#learn-to-code'}

            writes    GroundTruth/freeCodeCamp___freeCodeCamp
                        2  #DEP#learn      3  #DEP#nodejs   ...   8  #DEP#javascript
                      - EVERY real topic, read from dicth_ but never returned

        So the repo's own topics never reach the similarity vector or the user-item matrix;
        they exist in this fold only as the answer key. Note also that ground truth here is
        the FULL topic set, so a seed CNBN got right counts as a hit when scored.

        The vocabularies differ, too: seeds are CNBN's raw mined labels ('learn-to-code')
        while ground truth is Stage C's normalized ones ('learn') - section 4, item 12.
        """
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
        """Every topic a repo carries, straight from its dicth_ file. No holding back.

            dicth_bravo___llm-gateway  ->  {'#DEP#python', '#DEP#api', '#DEP#docker',
                                            '#DEP#llm', '#DEP#chat', '#DEP#bot'}

        This is the TRAINING-side reader: training repos always expose their owner-defined
        topics in full. Only test repos go through extract_EASE_dictionary / _half_dictionary.
        """
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
        """The top `size` neighbours, keyed by RANK (0 = most similar), not by project ID.

            Similarities/freeCodeCamp___freeCodeCamp   (already sorted descending)
                freeCodeCamp___freeCodeCamp <TAB> echo___fastapi-starter  <TAB> 0.7515
                freeCodeCamp___freeCodeCamp <TAB> bravo___llm-gateway     <TAB> 0.4090
                ...

            size=20  ->  {0: 'echo___fastapi-starter', 1: 'bravo___llm-gateway', ...}

        The rank keying matters: row k of the user-item matrix and sim[k] must refer to the
        same neighbour, and both are built from this ordering.
        """
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
        """The same file, same order, reading column 3 instead of column 2.

            ->  {0: 0.7515, 1: 0.4090, 2: 0.2900, ...}    rank -> similarity score

        Scores can exceed 1: the cosine denominator is sqrt(norm1*norm2) rather than
        norm1*norm2 (section 4, item 6).
        """
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
        """The answer key written earlier this fold by extract_EASE_dictionary.

            GroundTruth/freeCodeCamp___freeCodeCamp
                2  #DEP#learn
                3  #DEP#nodejs   ...

            ->  {'#DEP#learn', '#DEP#nodejs', '#DEP#react', '#DEP#d3',
                 '#DEP#education', '#DEP#teacher', '#DEP#javascript'}

        Compared to recommendations by exact string, prefix included - which is why the
        '#DEP# ' (with a space) variant once made nDCG and EPC structurally 0.
        """
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
