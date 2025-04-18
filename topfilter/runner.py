import os
import math
import time
import logging
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import heapq
import csv
from data_reader import DataReader
from topic_similarity_calculator import TopicSimilarityCalculator
from recommendation_engine import RecommendationEngine
from validator import Validator

class Runner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._prop_file = "evaluation.properties"
        self.num_of_neighbours = 20

    def load_configurations(self) -> str:
        try:
            with open(self._prop_file, 'r') as f:
                for line in f:
                    if line.startswith("sourceDirectory"):
                        return line.split("=")[1].strip()
        except IOError as e:
            self.logger.error(f"Error loading configurations from {self._prop_file}: {e}")
        return ""

    def run(self, bayesian: bool):
        self.logger.info("TopFilter: Recommender System!")
        
        self.src_dir = "/home/shayan/projects/github-recommender/dataset/topfilter/D1/"

        dr = DataReader(self.src_dir)
        projects_file = os.path.join(self.src_dir, "projects.txt")
        num_of_projects = dr.get_number_of_projects(projects_file)
        
        self.ten_fold_cross_validation(bayesian, num_of_projects)
        self.logger.info(f"Current time: {int(time.time() * 1000)}")

        validator = Validator(self.src_dir, bayesian)
        validator.run()
        self.logger.info(f"Neighbor: {self.num_of_neighbours}")
        self.logger.info(f"Dataset: {self.src_dir}")

    def ten_fold_cross_validation(self, bayesian: bool, num_of_projects: int):
        step = math.ceil(num_of_projects / 10)
        
        for i in range(10):
            training_start_pos1 = 1
            training_end_pos1 = i * step
            training_start_pos2 = (i + 1) * step + 1
            training_end_pos2 = num_of_projects
            testing_start_pos = 1 + i * step
            testing_end_pos = (i + 1) * step
            
            k = i + 1
            sub_folder = f"Round{k}"
            
            self.logger.info(f"Computing similarities fold {i}")
            
            calculator = TopicSimilarityCalculator(
                self.src_dir, sub_folder,
                training_start_pos1, training_end_pos1,
                training_start_pos2, training_end_pos2,
                testing_start_pos, testing_end_pos,
                bayesian
            )
            
            calculator.compute_weight_cosine_similarity()
            self.logger.info(f"\tComputed similarities fold {i}")
            
            self.logger.info(f"Computing recommendations fold {i}")
            engine = RecommendationEngine(
                self.src_dir, sub_folder, self.num_of_neighbours,
                testing_start_pos, testing_end_pos, bayesian
            )
            engine.user_based_recommendation()
            self.logger.info(f"\tComputed recommendations fold {i}")
            break

    @staticmethod
    def main():
        runner = Runner()
        try:
            runner.run(True)
        except Exception as e:
            runner.logger.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Runner.main()
