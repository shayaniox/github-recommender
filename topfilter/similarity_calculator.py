import os
import math
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict, OrderedDict
import heapq

class SimilarityCalculator:
    """
    Abstract base class for computing similarity between projects using weighted cosine similarity.
    """
    
    def __init__(self, source_dir: str, sub_folder: str, tr_start_pos1: int, tr_end_pos1: int,
                 tr_start_pos2: int, tr_end_pos2: int, te_start_pos: int, te_end_pos: int, bayesian: bool):
        self.src_dir = source_dir
        self.sub_folder = sub_folder
        self.ground_truth = str(Path(self.src_dir) / self.sub_folder / "GroundTruth")
        self.sim_dir = str(Path(self.src_dir) / self.sub_folder / "Similarities")
        
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
        """
        Compute similarity between every testing project and all training projects
        using Cosine Similarity with Weight.
        """
        from data_reader import DataReader
        from graph import Graph
        
        reader = DataReader(self.src_dir)
        training_projects = {}
        
        # Read training projects
        if self.training_start_pos1 < self.training_end_pos1:
            projects_file = str(Path(self.src_dir) / "projects.txt")
            training_projects = reader.read_project_list(
                projects_file, self.training_start_pos1, self.training_end_pos1)
        
        if self.training_start_pos2 < self.training_end_pos2:
            projects_file = str(Path(self.src_dir) / "projects.txt")
            temp_projects = reader.read_project_list(
                projects_file, self.training_start_pos2, self.training_end_pos2)
            training_projects.update(temp_projects)
        
        # Read testing projects
        projects_file = str(Path(self.src_dir) / "projects.txt")
        testing_projects = reader.read_project_list(
            projects_file, self.testing_start_pos, self.testing_end_pos)
        
        # Initialize graph and dictionaries
        graph = None
        all_training_libs = set()
        training_dictionaries = {}
        
        # Process training projects
        for key_training, training_pro in training_projects.items():
            training_filename = training_pro.replace("git://github.com/", "").replace("/", "__")
            training_graph_file = str(Path(self.src_dir) / f"graph_{training_filename}")
            training_dict_file = str(Path(self.src_dir) / f"dicth_{training_filename}")
            
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
        
        # Process testing projects
        for key_testing, testing_pro in testing_projects.items():
            try:
                all_libs = set(all_training_libs)
                combined_graph = Graph()
                combined_graph.OutLinks = graph.OutLinks.copy()
                combined_graph.dictionary = graph.dictionary.copy()
                combined_graph.nodeCount = graph.nodeCount
                
                sim = {}
                filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
                testing_graph_file = str(Path(self.src_dir) / f"graph_{filename}")
                testing_dict_file = str(Path(self.src_dir) / f"dicth_{filename}")
                
                # Get testing dictionary based on bayesian flag
                testing_dict = (reader.extract_EASE_dictionary(testing_dict_file, self.num_of_EASE_input, self.ground_truth) 
                              if self.bayesian 
                              else reader.extract_half_dictionary(testing_dict_file, self.ground_truth, get_also_users))
                
                testing_libs = {v for v in testing_dict.values() if v.startswith("#DEP#")}
                all_libs.update(testing_libs)
                
                testing_graph = Graph(testing_graph_file, testing_dict)
                combined_graph.combine(testing_graph, testing_dict)
                
                # Calculate library weights using IDF
                lib_weight = {}
                graph_edges = combined_graph.get_out_links()
                
                for start_node, outlinks in graph_edges.items():
                    for end_node in outlinks:
                        lib_weight[end_node] = lib_weight.get(end_node, 0) + 1
                
                number_of_projects = len(graph_edges)
                for lib_id, freq in lib_weight.items():
                    weight = number_of_projects / freq
                    idf = math.log(weight)
                    lib_weight[lib_id] = idf
                
                # Calculate similarity with each training project
                for key_training, training_pro in training_projects.items():
                    training_filename = training_pro.replace("git://github.com/", "").replace("/", "__")
                    training_dict_file = str(Path(self.src_dir) / f"dicth_{training_filename}")
                    training_libs = reader.get_libraries(training_dict_file)
                    
                    union = testing_libs.union(training_libs)
                    lib_set = list(union)
                    
                    if len(union) != len(lib_set):
                        self.logger.info("Something went wrong!")
                    
                    # Create vectors for cosine similarity
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
                            lib_id = combined_graph.dictionary[lib]
                            vector2[i] = lib_weight.get(lib_id, 0.0)
                    
                    val = self.cosine_similarity(vector1, vector2)
                    sim[str(key_training)] = val
                
                # Sort similarities and write to file
                sorted_sim = sorted(sim.items(), key=lambda x: x[1], reverse=True)
                output_file = str(Path(self.sim_dir) / filename)
                
                with open(output_file, 'w') as writer:
                    for key, score in sorted_sim:
                        content = f"{testing_pro}\t{training_projects[int(key)]}\t{score}"
                        writer.write(content + "\n")
            except IOError as e:
                self.logger.error(f"Error processing testing project {testing_pro}: {e}")

    def cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            v1: First vector
            v2: Second vector
            
        Returns:
            Cosine similarity between v1 and v2
        """
        dot_product = sum(x*y for x, y in zip(v1, v2))
        norm_v1 = math.sqrt(sum(x*x for x in v1))
        norm_v2 = math.sqrt(sum(x*x for x in v2))
        
        if norm_v1 * norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)