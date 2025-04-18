import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from data_reader import DataReader

class BayesianValidator:
    """
    Validator for Bayesian recommendation system that computes precision, recall, success rate, and coverage metrics.
    """

    def __init__(self, src_dir: str):
        self.src_dir = src_dir
        self.reader = DataReader(src_dir)
        self.logger = logging.getLogger(__name__)

    def get_real_repo_topic(self) -> Dict[str, Set[str]]:
        """
        Get the real topics for repositories from dictionary files.

        Returns:
            Dictionary mapping repository names to their topics
        """
        multimap = self.reader.get_EASE_output()
        real_topics = defaultdict(set)

        for repo in multimap.keys():
            parsed_repo = repo.replace("/", "__")
            file_path = Path(self.src_dir) / f"dicth_{parsed_repo}"
            topics = self.reader.get_libraries(str(file_path))
            if topics:
                real_topics[repo].update(topics)

        return real_topics

    def precision_recall_success_rate(self):
        """
        Compute precision, recall, and success rate metrics for top 20 recommendations.
        """
        precision = {i: 0.0 for i in range(1, 21)}
        recall = {i: 0.0 for i in range(1, 21)}
        success_rate = {i: 0.0 for i in range(1, 21)}

        real = self.get_real_repo_topic()
        ease_result = self.reader.get_EASE_output()

        for reponame, real_topics in real.items():
            repo_result = list(ease_result.get(reponame, []))

            for i in range(1, 21):
                i_list = set(repo_result[:i])
                intersection = i_list.intersection(real_topics)

                precision[i] += len(intersection) / i
                recall[i] += len(intersection) / len(real_topics) if real_topics else 0
                success_rate[i] += 1 if intersection else 0

        # Calculate averages
        num_repos = len(real)
        if num_repos > 0:
            for i in range(1, 21):
                precision[i] /= num_repos
                recall[i] /= num_repos
                success_rate[i] /= num_repos

        # Log results
        for i in range(1, 21):
            self.logger.info(f"PR: {precision[i]}")
        for i in range(1, 21):
            self.logger.info(f"REC: {recall[i]}")
        for i in range(1, 21):
            self.logger.info(f"SR: {success_rate[i]}")

    def coverage(self):
        """
        Compute coverage metrics for top 20 recommendations.
        """
        all_topics = set()
        coverage = defaultdict(set)

        real_topic = self.get_real_repo_topic()
        ease_result = self.reader.get_EASE_output()

        # Collect all real topics
        for topics in real_topic.values():
            all_topics.update(topics)

        # Calculate coverage for each cutoff
        for reponame in real_topic.keys():
            repo_result = list(ease_result.get(reponame, []))

            for i in range(1, 21):
                i_list = set(repo_result[:i])
                coverage[i].update(i_list)

        # Log coverage percentages
        total_topics = len(all_topics)
        if total_topics > 0:
            for i in range(1, 21):
                coverage_pct = (len(coverage[i]) / total_topics) * 100
                self.logger.info(f"COVERAGE {i}: {coverage_pct}")
