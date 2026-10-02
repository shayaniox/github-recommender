"""Stage F+G driver: ten folds of collaborative filtering, then evaluation.

RUNNING EXAMPLE, fold 3 of ~644 projects (step = ceil(644/10) = 65):

    projects.txt lines 131..195   -> the test window        (testing_start_pos..testing_end_pos)
    lines 1..130 and 196..644     -> the training windows   (two, one either side)

    for 'freeCodeCamp/freeCodeCamp', a test project in that window, the fold produces
        outpath/TopFilter/Round3/GroundTruth/freeCodeCamp___freeCodeCamp
        outpath/TopFilter/Round3/Similarities/freeCodeCamp___freeCodeCamp
        outpath/TopFilter/Round3/Recommendations/freeCodeCamp___freeCodeCamp

The windows are contiguous index ranges, which only yields a valid split because
scripts/make_folds.py rewrote projects.txt into the shared shuffled order first.
"""

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

    # def load_configurations(self) -> str:
    #     try:
    #         with open(self._prop_file, 'r') as f:
    #             for line in f:
    #                 if line.startswith("sourceDirectory"):
    #                     return line.split("=")[1].strip()
    #     except IOError as e:
    #         self.logger.error(f"Error loading configurations from {self._prop_file}: {e}")
    #     return ""

    def run(self, bayesian: bool):
        self.logger.info("TopFilter: Recommender System!")

        # TOPFILTER_SRC_DIR selects the dataset directory (dicth_/graph_/projects.txt/
        # training_data.csv) so separate runs can use separate copies instead of overwriting
        # one shared outpath/TopFilter. Default: <repo>/outpath/TopFilter/.
        default_src = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outpath", "TopFilter")
        self.src_dir = os.path.join(os.path.abspath(
            os.environ.get("TOPFILTER_SRC_DIR", default_src)), "")

        dr = DataReader(self.src_dir)
        projects_file = os.path.join(self.src_dir, "projects.txt")
        num_of_projects = dr.get_number_of_projects(projects_file)

        # bayesian=True is the hybrid path: CNBN's top 5 stand in as each test repo's visible
        # profile. bayesian=False is "half mode": the first half of its real topics is shown
        # instead. main() below runs half mode only when TOPFILTER_HALF=1.
        self.ten_fold_cross_validation(bayesian, num_of_projects)
        self.logger.info(f"Current time: {int(time.time() * 1000)}")

        validator = Validator(self.src_dir, bayesian)
        validator.run()
        self.logger.info(f"Neighbor: {self.num_of_neighbours}")
        self.logger.info(f"Dataset: {self.src_dir}")

    def ten_fold_cross_validation(self, bayesian: bool, num_of_projects: int):
        # ceil, matching validator.compute_evaluation_metrics - a floor here would score a
        # different window than was recommended for whenever N is not a multiple of 10.
        #   num_of_projects=644 -> step=65
        #   i=0  test 1..65      train 66..644
        #   i=2  test 131..195   train 1..130 and 196..644      <- the running example's fold
        #   i=9  test 586..650   train 1..585    (the last window is short: 644-585 = 59)
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

    @staticmethod
    def main():
        runner = Runner()
        try:
            # TOPFILTER_HALF=1 runs CF alone (half mode): no CNBN seeds, no training_data.csv.
            runner.run(os.environ.get("TOPFILTER_HALF") != "1")
        except Exception as e:
            runner.logger.error(f"Error in main execution: {e}")
            import traceback
            runner.logger.error(traceback.format_exc())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Runner.main()
