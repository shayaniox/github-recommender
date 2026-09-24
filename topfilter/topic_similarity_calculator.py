"""Stage F, step 1 - IDF-weighted similarity between each test repo and every training repo.

Worked on a six-repo toy fold (a real one has ~579). Test repo 'acme/chatbot-server' with
CNBN seeds [python, api, docker, rest-api, server]; training repos:

    T1 bravo/llm-gateway       python api docker llm chat bot
    T2 delta/slack-assistant   python bot chat rest-api nlp
    T3 echo/fastapi-starter    python api rest-api docker async
    ...

IDF = log(7/freq) over the 7 repos in the combined graph:

    python 0.154   docker 0.560   api 0.847   rest-api 0.847   chat/bot 1.253   server 1.946

so a topic almost everyone has barely moves the needle and a rare one dominates. Cosine over
those weights ranks T3 first. Full arithmetic in docs/CF-WALKTHROUGH.md.
"""

import os
import math
import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import heapq
import csv
from data_reader import DataReader
from graph import Graph


class TopicSimilarityCalculator:
    def __init__(self, source_dir: str, sub_folder: str, tr_start_pos1: int, tr_end_pos1: int,
                 tr_start_pos2: int, tr_end_pos2: int, te_start_pos: int, te_end_pos: int, bayesian: bool):
        self.src_dir = source_dir
        self.sub_folder = sub_folder
        self.ground_truth = os.path.join(
            self.src_dir, self.sub_folder, "GroundTruth")
        self.sim_dir = os.path.join(
            self.src_dir, self.sub_folder, "Similarities")
        # A fold's output directories were never created by the code - they survived from
        # earlier runs, so a fold that had never been run (2..10) or a cleaned tree produced
        # nothing but "No such file or directory" errors.
        for directory in (self.ground_truth, self.sim_dir):
            os.makedirs(directory, exist_ok=True)
        self.training_start_pos1 = tr_start_pos1
        self.training_end_pos1 = tr_end_pos1
        self.training_start_pos2 = tr_start_pos2
        self.training_end_pos2 = tr_end_pos2
        self.testing_start_pos = te_start_pos
        self.testing_end_pos = te_end_pos
        self.bayesian = bayesian
        self.num_of_EASE_input = 5
        self.logger = logging.getLogger(__name__)

    def compute_weight_cosine_similarity(self):
        reader = DataReader(self.src_dir)
        training_projects = {}

        if self.training_start_pos1 < self.training_end_pos1:
            projects_file = os.path.join(self.src_dir, "projects.txt")
            training_projects = reader.read_project_list(
                projects_file, self.training_start_pos1, self.training_end_pos1)

        if self.training_start_pos2 < self.training_end_pos2:
            projects_file = os.path.join(self.src_dir, "projects.txt")
            temp_projects = reader.read_project_list(
                projects_file, self.training_start_pos2, self.training_end_pos2)
            training_projects.update(temp_projects)

        projects_file = os.path.join(self.src_dir, "projects.txt")
        testing_projects = reader.read_project_list(
            projects_file, self.testing_start_pos, self.testing_end_pos)

        graph = None

        all_training_libs = set()
        training_dictionaries = {}

        for key_training, training_pro in training_projects.items():
            training_filename = training_pro.replace(
                "git://github.com/", "").replace("/", "__")
            training_graph_file = os.path.join(
                self.src_dir, f"graph_{training_filename}")
            training_dict_file = os.path.join(
                self.src_dir, f"dicth_{training_filename}")

            training_libs = reader.get_libraries(training_dict_file)
            all_training_libs.update(training_libs)

            training_dict = reader.read_dictionary(training_dict_file)
            training_dictionaries[key_training] = training_dict
            training_graph = Graph(training_graph_file, training_dict)

            if graph is None:
                graph = Graph()
                graph.OutLinks = training_graph.OutLinks.copy()
                graph.dictionary = training_graph.dictionary.copy()
                graph.nodeCount = training_graph.nodeCount
            else:
                graph.combine(training_graph, training_dict)

        get_also_users = False

        for key_testing, testing_pro in testing_projects.items():
            try:
                all_libs = set(all_training_libs)
                combined_graph = Graph()
                combined_graph.OutLinks = graph.OutLinks.copy()
                combined_graph.dictionary = graph.dictionary.copy()
                combined_graph.nodeCount = graph.nodeCount

                sim = {}
                filename = testing_pro.replace(
                    "git://github.com/", "").replace("/", "__")
                testing_graph_file = os.path.join(
                    self.src_dir, f"graph_{filename}")
                testing_dict_file = os.path.join(
                    self.src_dir, f"dicth_{filename}")

                testing_dict = (reader.extract_EASE_dictionary(testing_dict_file, self.num_of_EASE_input, self.ground_truth)
                                if self.bayesian
                                else reader.extract_half_dictionary(testing_dict_file, self.ground_truth, get_also_users))

                #   testing_libs  {'#DEP#python', '#DEP#api', '#DEP#docker', '#DEP#rest-api',
                #                    '#DEP#server'}   <- the ID 1 repo entry is filtered out here
                # In bayesian mode these are CNBN's 5 seeds; the repo's real topics are not
                # here at all, having just been written out as ground truth.
                testing_libs = {
                    v for v in testing_dict.values() if v.startswith("#DEP#")}
                all_libs.update(testing_libs)

                testing_graph = Graph(testing_graph_file, testing_dict)
                combined_graph.combine(testing_graph, testing_dict)

                lib_weight = {}
                graph_edges = combined_graph.get_out_links()

                # Count how many repos link to each topic. Only repo nodes have outgoing
                # edges, so len(graph_edges) is the repository count.
                #   lib_weight  {python: 6, docker: 4, api: 3, rest-api: 3, chat: 2, server: 1}
                for start_node, outlinks in graph_edges.items():
                    for end_node in outlinks:
                        lib_weight[end_node] = lib_weight.get(end_node, 0) + 1

                # Frequencies become IDF in place: log(7/6)=0.154 for python, log(7/1)=1.946
                # for server. Rare topics carry the similarity; ubiquitous ones barely count.
                number_of_projects = len(graph_edges)
                for lib_id, freq in lib_weight.items():
                    weight = number_of_projects / freq
                    idf = math.log(weight)
                    lib_weight[lib_id] = idf

                for key_training, training_pro in training_projects.items():
                    training_filename = training_pro.replace(
                        "git://github.com/", "").replace("/", "__")
                    training_dict_file = os.path.join(
                        self.src_dir, f"dicth_{training_filename}")
                    training_libs = reader.get_libraries(training_dict_file)

                    # Both vectors span the union of the two topic sets, so a topic only one
                    # side has contributes 0 to the dot product but still to that side's norm.
                    #   test  {python, api, docker, rest-api, server}
                    #   T3    {python, api, rest-api, docker, async}
                    #   union {python, api, docker, rest-api, server, async}  -> 6 columns
                    union = testing_libs.union(training_libs)
                    lib_set = list(union)

                    if len(union) != len(lib_set):
                        self.logger.info("Something went wrong!")

                    vector1 = [0.0] * len(lib_set)
                    vector2 = [0.0] * len(lib_set)

                    for i, lib in enumerate(lib_set):
                        if lib in testing_libs:
                            try:
                                lib_id = combined_graph.dictionary[lib]
                                vector1[i] = lib_weight.get(lib_id, 0.0)
                            except KeyError:
                                self.logger.error(lib)

                        if lib in training_libs:
                            # Same guard as the testing branch above. A topic that a repo's
                            # dicth_ lists but its graph_ has no edge for never enters the
                            # combined graph's dictionary, and the unguarded lookup raised
                            # KeyError into the catch-all below - abandoning EVERY project in
                            # the fold, which is why fold 1 produced no similarities at all.
                            # A topic with no edge carries no weight, so 0.0 is correct.
                            try:
                                lib_id = combined_graph.dictionary[lib]
                                vector2[i] = lib_weight.get(lib_id, 0.0)
                            except KeyError:
                                self.logger.debug(
                                    f"topic absent from combined graph: {lib}")

                    val = self.cosine_similarity(vector1, vector2)
                    sim[str(key_training)] = val

                sorted_sim = sorted(
                    sim.items(), key=lambda x: x[1], reverse=True)
                output_file = os.path.join(self.sim_dir, filename)

                with open(output_file, 'w') as writer:
                    for key, score in sorted_sim:
                        content = f"{testing_pro}\t{training_projects[int(key)]}\t{score}"
                        writer.write(content + "\n")
            except IOError as e:
                self.logger.error(
                    f"Error processing testing project {testing_pro}: {e}")

            except Exception as e:
                self.logger.error(f"Error WTF {testing_pro}: {e}")

    def cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """NOT the cosine, despite the name - see docs/ARCHITECTURE.md section 4, item 6.

            vector1 (test)  [0.154, 0.847, 0.560, 0.847, 1.946, 0.000]   norm1 = 2.358
            vector2 (T3)    [0.154, 0.847, 0.560, 0.000, 0.000, 1.946]   norm2 = 2.358
            dot             1.772

            returned   1.772 / sqrt(2.358 * 2.358) = 0.752     <- what this code computes
            cosine     1.772 / (2.358 * 2.358)     = 0.319     <- what the name promises

        norm1 and norm2 are already magnitudes, so the extra sqrt rescales every score by
        sqrt(norm1*norm2) - a factor that varies per pair, so it is not a harmless monotone
        transform and scores can exceed 1. similarity_calculator.py has the correct form and
        is unused.
        """
        sclar = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
        norm1 = math.sqrt(sum(v * v for v in vector1))
        norm2 = math.sqrt(sum(v * v for v in vector2))
        return sclar / math.sqrt(norm1 * norm2) if (norm1 * norm2) > 0 else 0.0
