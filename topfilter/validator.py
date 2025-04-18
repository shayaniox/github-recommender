import logging
from pathlib import Path
from typing import Dict, Set
from data_reader import DataReader
from metrics import Metrics

class Validator:
    """
    Main validator class that runs evaluation metrics for the recommendation system.
    Supports both regular and Bayesian validation modes.
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
        step = num_of_projects // 10
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

            metrics = Metrics(
                k, self.num_of_libraries, self.src_dir, sub_folder,
                training_start_pos1, training_end_pos1,
                training_start_pos2, training_end_pos2,
                testing_start_pos, testing_end_pos
            )

            # self.logger.info("==============Long tail==============")
            # metrics.long_tail()

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
            break

        # Write results to file
        res_dir = Path(self.src_dir) / "Results"
        res_dir.mkdir(exist_ok=True)
        output_file = res_dir / f"{name}@{cut_off_value}"

        with open(output_file, 'w') as writer:
            for score in vals.values():
                writer.write(f"{score}\n")