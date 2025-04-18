import os
import math
import logging
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import heapq
import csv
from data_reader import DataReader

class RecommendationEngine:
    def __init__(self, source_dir: str, sub_folder: str, num_of_neighbours: int, 
                 testing_start_pos: int, testing_end_pos: int, bayesian: bool):
        self.src_dir = source_dir
        self.sub_folder = sub_folder
        self.num_of_neighbours = num_of_neighbours
        self.rec_dir = os.path.join(self.src_dir, sub_folder, "Recommendations")
        self.sim_dir = os.path.join(self.src_dir, sub_folder, "Similarities")
        self.ground_truth = os.path.join(self.src_dir, sub_folder, "GroundTruth")
        self.reader = DataReader(source_dir)
        self.testing_start_pos = testing_start_pos
        self.testing_end_pos = testing_end_pos
        self.bayesian = bayesian
        self.num_of_EASE_input = 5
        self.logger = logging.getLogger(__name__)

    def build_user_item_matrix(self, testing_pro: str, lib_set: List[str]) -> List[List[float]]:
        filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
        testing_filename = filename
        testing_dict_filename = os.path.join(self.src_dir, f"dicth_{testing_filename}")
        
        testing_dictionary = (self.reader.extract_EASE_dictionary(testing_dict_filename, self.num_of_EASE_input, self.ground_truth) 
                            if self.bayesian 
                            else self.reader.extract_half_dictionary(testing_dict_filename, self.ground_truth, False))
        
        testing_libs = {v for v in testing_dictionary.values() if v.startswith("#DEP#")}
        
        tmp = os.path.join(self.sim_dir, filename)
        sim_projects = self.reader.get_most_similar_projects(tmp, self.num_of_neighbours)
        
        all_neighbour_libs = {}
        libraries = set()
        
        for key, project in sim_projects.items():
            filename = project.replace("git://github.com/", "").replace("/", "__")
            tmp = os.path.join(self.src_dir, f"dicth_{filename}")
            libs = self.reader.get_libraries(tmp)
            all_neighbour_libs[key] = libs
            libraries.update(libs)
        
        all_neighbour_libs[self.num_of_neighbours] = testing_libs
        libraries.update(testing_libs)
        
        for lib in libraries:
            lib_set.append(lib)
        
        num_rows = self.num_of_neighbours + 1
        num_cols = len(libraries)
        
        user_item_matrix = [[0.0 for _ in range(num_cols)] for _ in range(num_rows)]
        
        for i in range(self.num_of_neighbours):
            tmp_libs = all_neighbour_libs.get(i, set())
            for j in range(num_cols):
                user_item_matrix[i][j] = 1.0 if lib_set[j] in tmp_libs else 0.0
        
        tmp_libs = all_neighbour_libs.get(self.num_of_neighbours, set())
        for j in range(num_cols):
            user_item_matrix[self.num_of_neighbours][j] = 1.0 if lib_set[j] in tmp_libs else -1.0
        
        return user_item_matrix

    def user_based_recommendation(self):
        projects_file = os.path.join(self.src_dir, "projects.txt")
        testing_projects = self.reader.read_project_list(projects_file, self.testing_start_pos, self.testing_end_pos)
        
        for key_testing, testing_pro in testing_projects.items():
            recommendations = {}
            similarities = {}
            lib_set = []
            
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            tmp = os.path.join(self.sim_dir, filename)
            similarities = self.reader.get_similarity_matrix(tmp, self.num_of_neighbours)
            
            user_item_matrix = self.build_user_item_matrix(testing_pro, lib_set)
            avg_rating = 1.0
            val1 = sum(similarities.values())
            
            N = len(lib_set)
            
            for j in range(N):
                if user_item_matrix[self.num_of_neighbours][j] == -1:
                    val2 = 0.0
                    for k in range(self.num_of_neighbours):
                        tmp_rating = sum(user_item_matrix[k]) / N
                        val2 += (user_item_matrix[k][j] - tmp_rating) * similarities.get(k, 0.0)
                    
                    recommendations[str(j)] = avg_rating + val2 / val1 if val1 != 0 else 0.0
            
            sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            tmp = os.path.join(self.rec_dir, filename)
            
            try:
                with open(tmp, 'w') as writer:
                    if self.bayesian:
                        ease_dict_file = os.path.join(self.src_dir, f"dicth_{filename}")
                        ease_topic = self.reader.extract_EASE_dictionary(
                            ease_dict_file, self.num_of_EASE_input, self.ground_truth)
                        ease_topic.pop(1, None)
                        for v in ease_topic.values():
                            content = f"{v}\t2"
                            writer.write(content + "\n")
                    
                    for key, score in sorted_recommendations:
                        content = f"{lib_set[int(key)]}\t{score}"
                        writer.write(content + "\n")
            except IOError as e:
                self.logger.error(f"Error writing recommendations to {tmp}: {e}")

    def new_item_based_recommendation(self):
        projects_file = os.path.join(self.src_dir, "projects.txt")
        testing_projects = self.reader.read_project_list(projects_file, self.testing_start_pos, self.testing_end_pos)
        
        for key_testing, testing_pro in testing_projects.items():
            recommendations = {}
            lib_set = []
            user_item_matrix = self.build_user_item_matrix(testing_pro, lib_set)
            
            N = len(lib_set)
            
            for j in range(N):
                if user_item_matrix[self.num_of_neighbours][j] == -1:
                    avg_item_rating = 0.0
                    count = 0
                    
                    for l in range(self.num_of_neighbours):
                        if user_item_matrix[l][j] != 0:
                            avg_item_rating += user_item_matrix[l][j]
                            count += 1
                    
                    avg_item_rating = avg_item_rating / count if count > 0 else 0.0
                    
                    tmp1 = 0.0
                    tmp2 = 0.0
                    
                    for k in range(N):
                        if k != j and user_item_matrix[self.num_of_neighbours][k] != -1:
                            v1 = 0
                            v2 = 0
                            v3 = 0
                            
                            for l in range(self.num_of_neighbours):
                                if user_item_matrix[l][k] == 1 and user_item_matrix[l][j] == 1:
                                    v1 += 1
                                if user_item_matrix[l][k] == 1:
                                    v2 += 1
                                if user_item_matrix[l][j] == 1:
                                    v3 += 1
                            
                            sim = math.sqrt(v1) / (math.sqrt(v2) + math.sqrt(v3)) if (math.sqrt(v2) + math.sqrt(v3)) != 0 else 0.0
                            
                            tmp1 += sim * (user_item_matrix[self.num_of_neighbours][k] - 1.0)
                            tmp2 += sim
                    
                    val = (tmp1 / tmp2) + avg_item_rating if tmp2 != 0 else avg_item_rating
                    recommendations[str(j)] = val
            
            sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            tmp = os.path.join(self.rec_dir, filename)
            
            try:
                with open(tmp, 'w') as writer:
                    for key, score in sorted_recommendations:
                        content = f"{lib_set[int(key)]}\t{score}"
                        writer.write(content + "\n")
            except IOError as e:
                self.logger.error(f"Error writing recommendations to {tmp}: {e}")

    def item_based_recommendation(self):
        projects_file = os.path.join(self.src_dir, "projects.txt")
        testing_projects = self.reader.read_project_list(projects_file, self.testing_start_pos, self.testing_end_pos)
        
        for key_testing, testing_pro in testing_projects.items():
            recommendations = {}
            lib_set = []
            user_item_matrix = self.build_user_item_matrix(testing_pro, lib_set)
            
            N = len(lib_set)
            
            for j in range(N):
                if user_item_matrix[self.num_of_neighbours][j] == -1:
                    avg_item_rating = sum(row[j] for row in user_item_matrix[:self.num_of_neighbours]) / self.num_of_neighbours
                    
                    tmp1 = 0.0
                    tmp2 = 0.0
                    
                    for k in range(N):
                        if k != j and user_item_matrix[self.num_of_neighbours][k] != -1:
                            vector1 = [row[k] for row in user_item_matrix[:self.num_of_neighbours]]
                            vector2 = [row[j] for row in user_item_matrix[:self.num_of_neighbours]]
                            sim = self.cosine_similarity(vector1, vector2)
                            tmp1 += sim * (user_item_matrix[self.num_of_neighbours][k] - 1.0)
                            tmp2 += sim
                    
                    val = (tmp1 / tmp2) + avg_item_rating if tmp2 != 0 else avg_item_rating
                    recommendations[str(j)] = val
            
            sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            tmp = os.path.join(self.rec_dir, filename)
            
            try:
                with open(tmp, 'w') as writer:
                    for key, score in sorted_recommendations:
                        content = f"{lib_set[int(key)]}\t{score}"
                        writer.write(content + "\n")
            except IOError as e:
                self.logger.error(f"Error writing recommendations to {tmp}: {e}")

    def cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        sclar = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
        norm1 = math.sqrt(sum(v * v for v in vector1))
        norm2 = math.sqrt(sum(v * v for v in vector2))
        
        if norm1 * norm2 > 0 and sclar > 0:
            return sclar / math.sqrt(norm1 * norm2)
        return 0.0
