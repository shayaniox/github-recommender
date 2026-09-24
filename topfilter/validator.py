import logging
import math
from pathlib import Path
from typing import Dict, Set
from data_reader import DataReader
from metrics import Metrics

class Validator:
    """Stage G - score every fold's recommendations against its held-out ground truth.

    Reads, per fold and per test repository:

        Round3/Recommendations/acme___chatbot-server
            #DEP#python 2.0 | #DEP#api 2.0 | ... | #DEP#async 1.2 | #DEP#bot 1.166 | ...
        Round3/GroundTruth/acme___chatbot-server
            {#DEP#python, #DEP#api, #DEP#docker, #DEP#rest-api, #DEP#machine-learning,
             #DEP#chat, #DEP#bot, #DEP#fast}

    and writes cumulative curves over cut-offs 1..20 into Round3/Results/. At k=10 the
    example above scores 6 hits out of 8: P@10 = 0.60, R@10 = 0.75.

    The `_b` metric variants score the bayesian/hybrid path; the unsuffixed ones score the
    CF output on its own.
    """
    
    def __init__(self, src_dir: str, bayesian: bool):
        self.src_dir = src_dir
        self.bayesian = bayesian
        self.num_of_libraries = 20
        self.num_of_EASE_input = 5
        self.logger = logging.getLogger(__name__)
        self.input_file = "projects.txt"

    def run(self):
        """
        Run the 10-fold cross validation and compute all evaluation metrics.
        """
        self.logger.info("Ten-fold cross validation")
        reader = DataReader(self.src_dir)
        projects_file = Path(self.src_dir) / self.input_file
        num_of_projects = reader.get_number_of_projects(str(projects_file))
        self.compute_evaluation_metrics(num_of_projects)

    def compute_evaluation_metrics(self, num_of_projects: int):
        """
        Compute various evaluation metrics for the recommendation system.
        """
        # ceil, not //: runner.ten_fold_cross_validation slices ceil(N/10) windows, and a
        # validator using floor would score a different test window than was recommended
        # for whenever N is not a multiple of 10.
        step = math.ceil(num_of_projects / 10)
        #   step=65 for 644 projects; fold i scores projects [1+i*65 .. (i+1)*65]
        #   cut_off_value=20 - curves are swept over the top 1..20 recommendations
        cut_off_value = self.num_of_libraries
        recall_rate = 0.0
        vals = {}
        name = "EPC"

        for i in range(10):
            training_start_pos1 = 1
            training_end_pos1 = i * step
            training_start_pos2 = (i + 1) * step + 1
            training_end_pos2 = num_of_projects
            testing_start_pos = 1 + i * step
            testing_end_pos = (i + 1) * step
            k = i + 1
            sub_folder = f"Round{k}"

            # A fold with no recommendations (an interrupted run, or a re-score before the
            # later folds have been computed) is skipped rather than scored as zeros.
            rec_dir = Path(self.src_dir) / sub_folder / "Recommendations"
            if not rec_dir.exists() or not any(
                    p.name != ".gitkeep" for p in rec_dir.iterdir()):
                self.logger.warning(f"{sub_folder}: no recommendations, skipping")
                continue

            metrics = Metrics(
                k, self.num_of_libraries, self.src_dir, sub_folder,
                training_start_pos1, training_end_pos1,
                training_start_pos2, training_end_pos2,
                testing_start_pos, testing_end_pos
            )

            # self.logger.info("==============Long tail==============")
            # metrics.long_tail()

            # The _b family evaluates the hybrid list, whose first 5 entries are CNBN seeds.
            # num_of_EASE_input tells them how many that is, so a seed already scored does not
            # consume a second rank slot further down the file (section 4, item 14).
            if self.bayesian:
                metrics.catalog_coverage_b()
                metrics.success_rate_b(self.num_of_EASE_input)
                metrics.compute_average_success_rate_b()
                metrics.precision_recall_b(self.num_of_EASE_input)
                metrics.compute_average_precision_recall_b()

            # Compute standard metrics
            metrics.success_rate()
            metrics.success_rate_n()
            metrics.precision_recall()
            recall_rate += metrics.recall_rate()
            
            metrics.compute_average_precision_recall()
            metrics.compute_average_success_rate_n()
            metrics.compute_average_success_rate()
            metrics.catalog_coverage()
            metrics.entropy_analysis()
            metrics.epc_analysis()
            metrics.ndcg_analysis()

            # Collect results
            vals.update(metrics.get_some_scores(cut_off_value, "EPC"))
            vals.update(metrics.get_some_scores(cut_off_value, "Entropy"))
            vals.update(metrics.get_some_scores(cut_off_value, "Entropy"))  # Duplicate in original?

        # Write results to file
        res_dir = Path(self.src_dir) / "Results"
        res_dir.mkdir(exist_ok=True)
        output_file = res_dir / f"{name}@{cut_off_value}"

        with open(output_file, 'w') as writer:
            for score in vals.values():
                writer.write(f"{score}\n")