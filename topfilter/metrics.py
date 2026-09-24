import os
from pathlib import Path
import logging
from typing import Dict, List, Set, Tuple
import io
import os
from pathlib import Path
import logging
from typing import Dict, List, Set, Tuple, Optional
import math
from collections import defaultdict
from data_reader import DataReader


class Metrics:
    """Twelve metric families over one fold's recommendations.

    Inputs, per test repository (continuing the six-repo toy fold):

        Recommendations/acme___chatbot-server   ranked, '<topic>\t<score>' per line
            #DEP#python 2.0   #DEP#api 2.0   #DEP#docker 2.0   #DEP#rest-api 2.0
            #DEP#server 2.0   #DEP#async 1.2  #DEP#bot 1.166   #DEP#chat 1.166 ...

        GroundTruth/acme___chatbot-server       unranked set, exact-string compared
            #DEP#python  #DEP#api  #DEP#docker  #DEP#rest-api
            #DEP#machine-learning  #DEP#chat  #DEP#bot  #DEP#fast

    Outputs are cumulative curves swept over cut-offs 1..20, one file per metric:

        rank   1..4  hit    5 miss (server)   6 miss (async)   7,8 hit (bot, chat)
        P@10 = 6/10 = 0.60          R@10 = 6/8 = 0.75          SR@1 = 1 (rank 1 correct)

    Comparison is by exact string INCLUDING the '#DEP#' prefix, which is why a seed written
    as '#DEP# python' once made nDCG and EPC structurally zero.
    """

    # How many leading entries of a bayesian recommendation file are CNBN seeds.
    _NUM_OF_MNBN_TOPIC = 5

    def __init__(self, k: int, num_libs: int, src_dir: str, sub_folder: str, 
                 tr_start_pos1: int, tr_end_pos1: int, tr_start_pos2: int, 
                 tr_end_pos2: int, te_start_pos: int, te_end_pos: int):
        self.logger = logging.getLogger(__name__)

        self.fold = k
        self.num_libs = num_libs
        self.src_dir = src_dir
        self.ground_truth = str(Path(self.src_dir) / sub_folder / "GroundTruth")
        self.rec_dir = str(Path(self.src_dir) / sub_folder / "Recommendations")
        self.pr_dir = str(Path(self.src_dir) / sub_folder / "PrecisionRecall")
        self.pr_dir_b = str(Path(self.src_dir) / sub_folder / "PrecisionRecallB")
        self.success_rate_dir = str(Path(self.src_dir) / sub_folder / "SuccesRate")
        self.success_rate_dir_b = str(Path(self.src_dir) / sub_folder / "SuccesRateB")
        self.success_rate_dir_n = str(Path(self.src_dir) / sub_folder / "SuccesRateN")
        self.fs_dir = str(Path(self.src_dir) / sub_folder / "FScore")
        self.res_dir = str(Path(self.src_dir) / "Results")

        # Only some of these were created at the point of use, so a fold that had never been
        # run logged one "No such file or directory" per project and silently produced an
        # EMPTY metric file: PRC_Round* came out 0 bytes because PrecisionRecall/ did not
        # exist. The bayesian paths happened to mkdir theirs, which is why the hybrid numbers
        # survived and the non-bayesian ones did not.
        for directory in (self.pr_dir, self.pr_dir_b, self.success_rate_dir,
                          self.success_rate_dir_b, self.success_rate_dir_n,
                          self.fs_dir, self.res_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)

        self.reader = DataReader(self.src_dir)
        self.training_start_pos1 = tr_start_pos1
        self.training_end_pos1 = tr_end_pos1
        self.training_start_pos2 = tr_start_pos2
        self.training_end_pos2 = tr_end_pos2
        self.testing_start_pos = te_start_pos
        self.testing_end_pos = te_end_pos

        projects_file = str(Path(self.src_dir) / "projects.txt")
        self.testing_projects = self.reader.read_project_list(
            projects_file, self.testing_start_pos, self.testing_end_pos)

    def mean_absolute_error(self) -> None:
        key_testing_projects = self.testing_projects.keys()
        results = {}

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")

            rec_file = str(Path(self.rec_dir) / filename)
            recommendations = self.reader.read_recommendation_scores(rec_file)

            gt_file = str(Path(self.ground_truth) / filename)
            ground_truth = self.reader.read_ground_truth_score(gt_file)

            key_set = ground_truth.keys()
            score = 0.0

            for key in key_set:
                g_score = ground_truth[key]
                r_score = recommendations.get(key, 0.0)
                score += abs(g_score - r_score)

            if ground_truth:
                score /= len(ground_truth)
                results[str(key_testing)] = score
                self.logger.info(f"{testing_pro} \t{score}")

        output_file = str(Path(self.res_dir) / f"MAE_Round{self.fold}")
        try:
            with open(output_file, 'w') as writer:
                for d in results.values():
                    writer.write(f"{d}\n")
        except IOError as e:
            self.logger.error(e)

    def recall_rate(self) -> float:
        key_testing_projects = self.testing_projects.keys()
        count = 0

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")

            rec_file = str(Path(self.rec_dir) / filename)
            recommendation_file = self.reader.read_recommendation_file_with_size(rec_file, self.num_libs)

            gt_file = str(Path(self.ground_truth) / filename)
            ground_truth_file = self.reader.read_ground_truth_file(gt_file)

            common = set(recommendation_file) & set(ground_truth_file)
            if not common:
                count += 1

        output_file = str(Path(self.res_dir) / f"Recall_Round{self.fold}")
        total = len(key_testing_projects)
        recall_rate = (total - count) / total

        try:
            with open(output_file, 'w') as writer:
                writer.write(f"{recall_rate}\n")
        except IOError as e:
            self.logger.error(e)

        return recall_rate

    def success_rate(self) -> None:
        key_testing_projects = self.testing_projects.keys()

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            gt_data = str(Path(self.rec_dir) / filename)

            recommendation_file = self.reader.read_recommendation_file(gt_data)
            gt_data = str(Path(self.ground_truth) / filename)
            ground_truth_file = self.reader.read_ground_truth_file(gt_data)

            key_set = recommendation_file.keys()
            temp = set()

            success_rate_folder = Path(self.success_rate_dir)
            success_rate_folder.mkdir(exist_ok=True)
            success_rate_path = str(success_rate_folder / filename)

            try:
                with open(success_rate_path, 'w') as writer:
                    count = 1
                    for key in key_set:
                        temp.add(recommendation_file[key])
                        common = temp & set(ground_truth_file)
                        size = len(common)
                        content = f"{key}\t{'1' if size else '0'}"
                        writer.write(f"{content}\n")

                        count += 1
                        if count > self.num_libs:
                            break
            except IOError as e:
                self.logger.error(e)

    def success_rate_b(self, number_of_topics_from_ease: int) -> None:
        key_testing_projects = self.testing_projects.keys()

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            rec_file = str(Path(self.rec_dir) / filename)

            ease_topics = self.reader.get_EASE_topic(testing_pro, number_of_topics_from_ease)
            recommendation_data = self.reader.read_recommendation_file(rec_file)

            gt_file = str(Path(self.ground_truth) / filename)
            ground_truth_data = self.reader.read_ground_truth_file(gt_file)

            # NOTE: this used to re-read the ground truth via get_libraries(dicth_<repo>).
            # In bayesian mode the GroundTruth file already holds exactly that set - the
            # repo's full topic set (extract_EASE_dictionary writes every #DEP# topic) - and
            # the two were verified identical for 644/644 projects, so removing the line has
            # no numeric effect. There is no held-out topic split in bayesian mode.

            key_set = recommendation_data.keys()
            temp = set(ease_topics)

            success_rate_folder = Path(self.success_rate_dir_b)
            success_rate_folder.mkdir(exist_ok=True)
            success_rate_path = str(success_rate_folder / filename)

            try:
                with open(success_rate_path, 'w') as writer:
                    i = 1
                    temp_set = set()
                    for element in ease_topics:
                        temp_set.add(element)
                        common = temp_set & set(ground_truth_data)
                        size = len(common)
                        content = f"{i}\t{'1' if size else '0'}"
                        writer.write(f"{content}\n")
                        i += 1

                    # The recommendation file already begins with the same seeds that were
                    # just scored at ranks 1..N_ease: the Topics Merger prepends them. Placing
                    # entry k at rank k + N_ease therefore re-scored the seeds at ranks 6..10
                    # and pushed the first genuinely new topic to rank 11, outside the reported
                    # window - which is why every _b curve was flat from N=5 to N=10. A topic
                    # already scored must not consume a second rank slot.
                    rank = number_of_topics_from_ease
                    for key in key_set:
                        topic = recommendation_data[key]
                        if topic in temp:
                            continue
                        temp.add(topic)
                        rank += 1

                        common = temp & set(ground_truth_data)
                        size = len(common)
                        content = f"{rank}\t{'1' if size else '0'}"
                        writer.write(f"{content}\n")

                        if rank >= self.num_libs:
                            break

                    # A recommendation list shorter than num_libs leaves the metric at its
                    # final level for the remaining ranks. compute_average_success_rate_b
                    # sums by LINE NUMBER and divides by the project count, so a short file
                    # silently drags the average down; pad it out. This was invisible while
                    # the duplicated seeds guaranteed every file 20 lines.
                    final = '1' if (temp & set(ground_truth_data)) else '0'
                    while rank < self.num_libs:
                        rank += 1
                        writer.write(f"{rank}\t{final}\n")
            except IOError as e:
                self.logger.error(e)

    def success_rate_n(self) -> None:
        key_testing_projects = self.testing_projects.keys()

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            gt_data = str(Path(self.rec_dir) / filename)

            recommendation_file = self.reader.read_recommendation_file(gt_data)
            gt_data = str(Path(self.ground_truth) / filename)
            ground_truth_file = self.reader.read_ground_truth_file(gt_data)

            key_set = recommendation_file.keys()
            temp = set()

            success_rate_folder = Path(self.success_rate_dir_n)
            success_rate_folder.mkdir(exist_ok=True)
            success_rate_path = str(success_rate_folder / filename)

            try:
                with open(success_rate_path, 'w') as writer:
                    count = 1
                    for key in key_set:
                        temp.add(recommendation_file[key])
                        common = temp & set(ground_truth_file)
                        size = len(common)
                        content = f"{key}\t{size}"
                        writer.write(f"{content}\n")

                        count += 1
                        if count > self.num_libs:
                            break
            except IOError as e:
                self.logger.error(e)

    def precision_recall(self) -> None:
        key_testing_projects = self.testing_projects.keys()

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")

            rec_file = str(Path(self.rec_dir) / filename)
            recommendation_file = self.reader.read_recommendation_file(rec_file)

            gt_file = str(Path(self.ground_truth) / filename)
            ground_truth_file = self.reader.read_ground_truth_file(gt_file)

            total_of_relevant = len(ground_truth_file)
            key_set = recommendation_file.keys()
            temp = set()

            output_file = str(Path(self.pr_dir) / filename)
            try:
                with open(output_file, 'w') as writer:
                    count = 1
                    for key in key_set:
                        temp.add(recommendation_file[key])
                        common = temp & set(ground_truth_file)
                        size = len(common)

                        precision = size / key if key != 0 else 0
                        recall = size / total_of_relevant if total_of_relevant != 0 else 0

                        val1 = 2 * recall * precision
                        val2 = recall + precision
                        f_score = val1 / val2 if val1 and val2 else 0

                        content = f"{key}\t{recall}\t{precision}"
                        writer.write(f"{content}\n")

                        count += 1
                        if count > self.num_libs:
                            break
            except IOError as e:
                self.logger.error(e)

    def precision_recall_b(self, number_of_topics_from_ease: int) -> None:
        key_testing_projects = self.testing_projects.keys()

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")

            rec_file = str(Path(self.rec_dir) / filename)
            recommendation_file = self.reader.read_recommendation_file(rec_file)

            gt_file = str(Path(self.ground_truth) / filename)
            ground_truth_data = self.reader.read_ground_truth_file(gt_file)

            # NOTE: this used to re-read the ground truth via get_libraries(dicth_<repo>).
            # In bayesian mode the GroundTruth file already holds exactly that set - the
            # repo's full topic set (extract_EASE_dictionary writes every #DEP# topic) - and
            # the two were verified identical for 644/644 projects, so removing the line has
            # no numeric effect. There is no held-out topic split in bayesian mode.

            total_of_relevant = len(ground_truth_data)
            ease_topics = self.reader.get_EASE_topic(testing_pro, number_of_topics_from_ease)
            key_set = recommendation_file.keys()
            temp = set(ease_topics)

            output_file = str(Path(self.pr_dir_b) / filename)
            Path(self.pr_dir_b).mkdir(exist_ok=True)

            try:
                with open(output_file, 'w') as writer:
                    i = 1
                    temp_set = set()
                    for element in ease_topics:
                        temp_set.add(element)
                        common = temp_set & set(ground_truth_data)
                        size = len(common)

                        precision = size / i if i != 0 else 0
                        recall = size / total_of_relevant if total_of_relevant != 0 else 0

                        content = f"{i}\t{recall}\t{precision}"
                        writer.write(f"{content}\n")
                        i += 1

                    # See success_rate_b: the recommendation file repeats the seeds that were
                    # already scored at ranks 1..N_ease, so a duplicate must not consume a rank
                    # slot. Scoring entry k at rank k + N_ease made precision@10 exactly
                    # precision@5 / 2 by construction, for every configuration.
                    rank = number_of_topics_from_ease
                    for key in key_set:
                        topic = recommendation_file[key]
                        if topic in temp:
                            continue
                        temp.add(topic)
                        rank += 1

                        common = temp & set(ground_truth_data)
                        size = len(common)

                        precision = size / rank
                        recall = size / total_of_relevant if total_of_relevant != 0 else 0

                        content = f"{rank}\t{recall}\t{precision}"
                        writer.write(f"{content}\n")

                        if rank >= self.num_libs:
                            break

                    # See success_rate_b: pad a short list to num_libs. The hit count stays
                    # fixed, so recall holds flat while precision keeps falling as hits/rank -
                    # the standard convention, which charges a system for returning fewer
                    # than N items rather than rewarding it.
                    size = len(temp & set(ground_truth_data))
                    while rank < self.num_libs:
                        rank += 1
                        writer.write(f"{rank}\t"
                                     f"{size / total_of_relevant if total_of_relevant else 0}\t"
                                     f"{size / rank}\n")
            except IOError as e:
                self.logger.error(e)
    def compute_average_precision_recall(self) -> None:
        key_testing_projects = self.testing_projects.keys()
        precision_map = {}
        recall_map = {}

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            pr_file = str(Path(self.pr_dir) / filename)

            try:
                with open(pr_file, 'r') as f:
                    id = 1
                    for line in f:
                        vals = line.strip().split("\t")
                        recall = float(vals[1])
                        precision = float(vals[2])

                        if id in recall_map:
                            recall_map[id] += recall
                            precision_map[id] += precision
                        else:
                            recall_map[id] = recall
                            precision_map[id] = precision

                        id += 1
                        if id > self.num_libs:
                            break
            except IOError as e:
                self.logger.error(e)

        size = len(self.testing_projects)
        output_file = str(Path(self.res_dir) / f"PRC_Round{self.fold}")

        try:
            with open(output_file, 'w') as writer:
                for key in sorted(precision_map.keys()):
                    recall = recall_map[key] / size if size != 0 else 0
                    precision = precision_map[key] / size if size != 0 else 0
                    content = f"{key}\t{recall}\t{precision}"
                    writer.write(f"{content}\n")
        except IOError as e:
            self.logger.error(e)

    def compute_average_precision_recall_b(self) -> None:
        key_testing_projects = self.testing_projects.keys()
        precision_map = {}
        recall_map = {}

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            pr_file = str(Path(self.pr_dir_b) / filename)

            try:
                with open(pr_file, 'r') as f:
                    id = 1
                    for line in f:
                        vals = line.strip().split("\t")
                        recall = float(vals[1])
                        precision = float(vals[2])

                        if id in recall_map:
                            recall_map[id] += recall
                            precision_map[id] += precision
                        else:
                            recall_map[id] = recall
                            precision_map[id] = precision

                        id += 1
                        if id > self.num_libs:
                            break
            except IOError as e:
                self.logger.error(e)

        size = len(self.testing_projects)
        output_file = str(Path(self.res_dir) / f"PRCB_Round{self.fold}")

        try:
            with open(output_file, 'w') as writer:
                for key in sorted(precision_map.keys()):
                    recall = recall_map[key] / size if size != 0 else 0
                    precision = precision_map[key] / size if size != 0 else 0
                    content = f"{key}\t{recall}\t{precision}"
                    writer.write(f"{content}\n")
        except IOError as e:
            self.logger.error(e)

    def init_sr_map(self) -> Dict[int, float]:
        return {i: 0.0 for i in range(1, 21)}

    def compute_average_success_rate_n(self) -> None:
        key_testing_projects = self.testing_projects.keys()
        sr_map1 = self.init_sr_map()
        sr_map2 = self.init_sr_map()
        sr_map3 = self.init_sr_map()
        sr_map4 = self.init_sr_map()

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            sr_file = str(Path(self.success_rate_dir_n) / filename)

            try:
                # Carried forward by the fill loop below when a project's file is short; an
                # empty file (a project with no recommendations at all) must contribute 0,
                # not crash the whole evaluation on an unbound name.
                success_rate = 0.0
                with open(sr_file, 'r') as f:
                    id = 1
                    for line in f:
                        vals = line.strip().split("\t")
                        success_rate = float(vals[1])

                        sr1 = sr_map1[id]
                        sr2 = sr_map2[id]
                        sr3 = sr_map3[id]
                        sr4 = sr_map4[id]

                        if success_rate == 1:
                            sr1 += 1
                        elif 1 < success_rate < 3:
                            sr1 += 1
                            sr2 += 1
                        elif 2 < success_rate < 4:
                            sr1 += 1
                            sr2 += 1
                            sr3 += 1
                        elif success_rate > 3:
                            sr1 += 1
                            sr2 += 1
                            sr3 += 1
                            sr4 += 1

                        sr_map1[id] = sr1
                        sr_map2[id] = sr2
                        sr_map3[id] = sr3
                        sr_map4[id] = sr4

                        id += 1
                        if id > self.num_libs:
                            break
            except IOError as e:
                self.logger.error(e)

        size = len(self.testing_projects)
        output_file = str(Path(self.res_dir) / f"SR_STAR_Round{self.fold}")
        
        try:
            with open(output_file, 'w') as writer:
                for key in sorted(sr_map1.keys()):
                    sr1 = sr_map1[key] / size if size != 0 else 0
                    sr2 = sr_map2[key] / size if size != 0 else 0
                    sr3 = sr_map3[key] / size if size != 0 else 0
                    sr4 = sr_map4[key] / size if size != 0 else 0
                    content = f"{key}\t{sr1:.03f}\t{sr2:.03f}\t{sr3:.03f}\t{sr4:.03f}"
                    writer.write(f"{content}\n")
        except IOError as e:
            self.logger.error(e)

    def compute_average_success_rate_b(self) -> None:
        key_testing_projects = self.testing_projects.keys()
        sr_map = {}

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            sr_file = str(Path(self.success_rate_dir_b) / filename)

            try:
                # Carried forward by the fill loop below when a project's file is short; an
                # empty file (a project with no recommendations at all) must contribute 0,
                # not crash the whole evaluation on an unbound name.
                success_rate = 0.0
                with open(sr_file, 'r') as f:
                    id = 1
                    for line in f:
                        vals = line.strip().split("\t")
                        success_rate = float(vals[1])
                        
                        if id in sr_map:
                            sr_map[id] += success_rate
                        else:
                            sr_map[id] = success_rate
                        
                        id += 1
                        if id > self.num_libs:
                            break
            except IOError as e:
                self.logger.error(e)

        size = len(self.testing_projects)
        output_file = str(Path(self.res_dir) / f"SRB_Round{self.fold}")
        
        try:
            with open(output_file, 'w') as writer:
                for key in sorted(sr_map.keys()):
                    sr = sr_map[key] / size if size != 0 else 0
                    content = f"{key}\t{sr}"
                    writer.write(f"{content}\n")
        except IOError as e:
            self.logger.error(e)

    def compute_average_success_rate(self) -> None:
        key_testing_projects = self.testing_projects.keys()
        sr_map = {}

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            sr_file = str(Path(self.success_rate_dir) / filename)

            try:
                # Carried forward by the fill loop below when a project's file is short; an
                # empty file (a project with no recommendations at all) must contribute 0,
                # not crash the whole evaluation on an unbound name.
                success_rate = 0.0
                with open(sr_file, 'r') as f:
                    id = 1
                    for line in f:
                        vals = line.strip().split("\t")
                        success_rate = float(vals[1])
                        
                        if id in sr_map:
                            sr_map[id] += success_rate
                        else:
                            sr_map[id] = success_rate
                        
                        id += 1
                        if id > self.num_libs:
                            break
                    
                    # Fill remaining positions if needed
                    while id <= self.num_libs:
                        if id in sr_map:
                            sr_map[id] += success_rate
                        else:
                            sr_map[id] = success_rate
                        id += 1
            except IOError as e:
                self.logger.error(e)

        size = len(self.testing_projects)
        output_file = str(Path(self.res_dir) / f"SR_Round{self.fold}")
        
        try:
            with open(output_file, 'w') as writer:
                for key in sorted(sr_map.keys()):
                    sr = sr_map[key] / size if size != 0 else 0
                    content = f"{key}\t{sr}"
                    writer.write(f"{content}\n")
        except IOError as e:
            self.logger.error(e)

    def get_f_scores(self, cut_off_value: int) -> Dict[str, float]:
        key_testing_projects = self.testing_projects.keys()
        f_scores = {}

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            fs_file = str(Path(self.fs_dir) / filename)

            try:
                with open(fs_file, 'r') as f:
                    id = 1
                    for line in f:
                        if id == cut_off_value:
                            vals = line.strip().split("\t")
                            f_score = float(vals[1])
                            f_scores[filename] = f_score
                            break
                        id += 1
            except IOError as e:
                self.logger.error(e)

        return f_scores

    def get_precision_recall_scores(self, cut_off_value: int, 
                                  recall_map: Dict[str, float], 
                                  precision_map: Dict[str, float]) -> None:
        key_testing_projects = self.testing_projects.keys()

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            pr_file = str(Path(self.pr_dir) / filename)

            try:
                with open(pr_file, 'r') as f:
                    id = 1
                    for line in f:
                        if id == cut_off_value:
                            vals = line.strip().split("\t")
                            recall = float(vals[1])
                            precision = float(vals[2])
                            recall_map[filename] = recall
                            precision_map[filename] = precision
                            break
                        id += 1
            except IOError as e:
                self.logger.error(e)

    def get_some_scores(self, cut_off_value: int, name: str) -> Dict[str, float]:
        vals = {}
        score_file = str(Path(self.res_dir) / f"{name}_Round{self.fold}")

        try:
            with open(score_file, 'r') as f:
                id = 1
                for line in f:
                    if id == cut_off_value:
                        val = float(line.strip())
                        vals[score_file] = val
                        break
                    id += 1
        except IOError as e:
            self.logger.error(e)

        return vals

    def get_all_items(self) -> Set[str]:
        all_items = set()
        training_projects = self.reader.read_project_list(
            str(Path(self.src_dir) / "projects.txt"), 
            self.training_start_pos1, self.training_end_pos1)

        if self.training_start_pos2 != 0 and self.training_end_pos2 != 0:
            temp_projects = self.reader.read_project_list(
                str(Path(self.src_dir) / "projects.txt"), 
                self.training_start_pos2, self.training_end_pos2)
            training_projects.update(temp_projects)

        for key, project in training_projects.items():
            filename = project.replace("git://github.com/", "").replace("/", "__")
            training_dict_file = str(Path(self.src_dir) / f"dicth_{filename}")
            all_items.update(self.reader.get_libraries(training_dict_file))

        return all_items

    def catalog_coverage_b(self) -> None:
        key_testing_projects = self.testing_projects.keys()
        all_items = len(self.get_all_items())
        size_map = defaultdict(set)

        for i in range(1, 11):
            for key_testing in key_testing_projects:
                testing_pro = self.testing_projects[key_testing]
                testing_recs_path = str(Path(self.rec_dir) / testing_pro)
                ease_topics = self.reader.get_EASE_topic(testing_pro, self._NUM_OF_MNBN_TOPIC)
                recs = list(self.reader.read_recommendation_file_with_size(testing_recs_path, i))
                ease_topics.extend(recs)
                
                try:
                    if len(ease_topics) >= i:
                        size_map[i].update(ease_topics[:i])
                    else:
                        size_map[i].update(ease_topics)
                except Exception as e:
                    self.logger.error(f"{i} {testing_pro.replace('___', '/')} {str(e)}")
                    raise

        output_file = str(Path(self.res_dir) / f"Catalog{self.fold}")
        try:
            with open(output_file, 'w') as writer:
                for size, items in sorted(size_map.items()):
                    coverage = len(items) / all_items
                    writer.write(f"{size}\t{coverage}\n")
        except IOError as e:
            self.logger.error(str(e))

    def catalog_coverage(self) -> None:
        key_testing_projects = self.testing_projects.keys()
        all_items = len(self.get_all_items())
        size_map = defaultdict(set)

        for i in range(1, 11):
            for key_testing in key_testing_projects:
                testing_pro = self.testing_projects[key_testing]
                testing_recs_path = str(Path(self.rec_dir) / testing_pro)
                recs = list(self.reader.read_recommendation_file_with_size(testing_recs_path, i))
                
                try:
                    if len(recs) >= i:
                        size_map[i].update(recs[:i])
                    else:
                        size_map[i].update(recs)
                except Exception as e:
                    self.logger.error(f"{i} {testing_pro.replace('___', '/')} {str(e)}")
                    raise

        output_file = str(Path(self.res_dir) / f"Catalog{self.fold}")
        try:
            with open(output_file, 'w') as writer:
                for size, items in sorted(size_map.items()):
                    coverage = len(items) / all_items
                    writer.write(f"{size}\t{coverage}\n")
        except IOError as e:
            self.logger.error(str(e))

    def long_tail(self, n: int, long_tail_items: Set[str], rec: Dict[str, Dict[int, str]]) -> float:
        all_recs = defaultdict(int)
        total = 0

        for project, recommendations in rec.items():
            top_n = {k: v for k, v in recommendations.items() if k <= n}
            for lib in top_n.values():
                all_recs[lib] += 1
                total += 1

        count = sum(all_recs[lib] for lib in long_tail_items if lib in all_recs)
        return count / total if total != 0 else 0

    def long_tail_analysis(self) -> None:
        key_testing_projects = self.testing_projects.keys()
        long_tail_items = self.reader.read_long_tail_items("/home/utente/Documents/Journals/EMSE/Longtail.txt")
        rec = {}

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            rec_file = str(Path(self.rec_dir) / filename)
            recommendations = self.reader.read_all_recommendations(rec_file)
            rec[filename] = recommendations

        output_file = str(Path(self.res_dir) / f"LongTail_Round{self.fold}")
        try:
            with open(output_file, 'w') as writer:
                for i in range(1, self.num_libs + 1):
                    longtail = self.long_tail(i, long_tail_items, rec)
                    writer.write(f"{longtail}\n")
        except IOError as e:
            self.logger.error(str(e))

    def ndcg(self, n: int, rec: Dict[str, Dict[int, str]], gt: Dict[str, Set[str]]) -> float:
        total_ndcg = 0

        for project, recommendations in rec.items():
            ground_truth = gt[project]
            top_n = {k: v for k, v in recommendations.items() if k <= n}
            
            dcg = 0.0
            for pos, lib in top_n.items():
                rel = 1 if lib in ground_truth else 0
                dcg += rel / math.log2(pos + 1)
            
            # Ideal DCG
            ideal_relevances = [1] * min(len(ground_truth), n) + [0] * max(0, n - len(ground_truth))
            idcg = sum(rel / math.log2(i + 1) for i, rel in enumerate(ideal_relevances, 1))
            
            total_ndcg += dcg / idcg if idcg != 0 else 0

        return total_ndcg / len(rec) if rec else 0

    def ndcg_analysis(self) -> None:
        key_testing_projects = self.testing_projects.keys()
        rec = {}
        gt = {}

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            
            rec_file = str(Path(self.rec_dir) / filename)
            recommendations = self.reader.read_all_recommendations(rec_file)
            rec[filename] = recommendations
            
            gt_file = str(Path(self.ground_truth) / filename)
            ground_truth = self.reader.read_ground_truth_file(gt_file)
            gt[filename] = ground_truth

        output_file = str(Path(self.res_dir) / f"nDCG_Round{self.fold}")
        try:
            with open(output_file, 'w') as writer:
                for i in range(1, self.num_libs + 1):
                    ndcg = self.ndcg(i, rec, gt)
                    writer.write(f"{ndcg}\n")
        except IOError as e:
            self.logger.error(str(e))

    def entropy(self, all_items: Set[str], rec: Dict[int, Dict[int, str]], n: int) -> float:
        item_freq = defaultdict(int)
        total = 0

        for project_recs in rec.values():
            top_n = {k: v for k, v in project_recs.items() if k <= n}
            for lib in top_n.values():
                item_freq[lib] += 1
                total += 1

        entropy_val = 0.0
        for item in all_items:
            freq = item_freq[item]
            if freq > 0:
                prob = freq / total
                entropy_val -= prob * math.log(prob)

        return entropy_val

    def entropy_analysis(self) -> None:
        key_testing_projects = self.testing_projects.keys()
        all_items = self.get_all_items()
        rec = {}

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            rec_file = str(Path(self.rec_dir) / filename)
            recommendations = self.reader.read_all_recommendations(rec_file)
            rec[key_testing] = recommendations

        output_file = str(Path(self.res_dir) / f"Entropy_Round{self.fold}")
        try:
            with open(output_file, 'w') as writer:
                for i in range(1, self.num_libs + 1):
                    entropy_val = self.entropy(all_items, rec, i)
                    writer.write(f"{entropy_val}\n")
        except IOError as e:
            self.logger.error(str(e))

    def epc_analysis(self) -> None:
        key_testing_projects = self.testing_projects.keys()
        rec = {}

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            rec_file = str(Path(self.rec_dir) / filename)
            recommendations = self.reader.read_all_recommendations(rec_file)
            rec[filename] = recommendations

        output_file = str(Path(self.res_dir) / f"EPC_Round{self.fold}")
        try:
            with open(output_file, 'w') as writer:
                for i in range(1, self.num_libs + 1):
                    epc = self.epc(i, rec)
                    writer.write(f"{epc}\n")
        except IOError as e:
            self.logger.error(str(e))

    def epc(self, n: int, rec: Dict[str, Dict[int, str]]) -> float:
        pop = self.popularity()
        numerator = 0.0
        denominator = 0.0

        for project, recommendations in rec.items():
            gt_file = str(Path(self.ground_truth) / project)
            ground_truth = self.reader.read_ground_truth_file(gt_file)
            top_n = {k: v for k, v in recommendations.items() if k <= n}
            
            for pos, lib in top_n.items():
                if lib in ground_truth:
                    rel = 1
                    numerator += rel * (1 - pop.get(lib, 0)) / math.log2(pos + 1)
                    denominator += rel / math.log2(pos + 1)

        return numerator / denominator if denominator != 0 else 0

    def popularity(self) -> Dict[str, float]:
        pop = defaultdict(int)
        key_testing_projects = self.testing_projects.keys()

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            rec_file = str(Path(self.rec_dir) / filename)
            recommendations = self.reader.read_recommendation_file(rec_file)
            
            for lib in recommendations.values():
                pop[lib] += 1

        max_pop = max(pop.values()) if pop else 1
        return {lib: count / max_pop for lib, count in pop.items()}

    def frequency(self) -> Dict[str, float]:
        freq = defaultdict(int)
        key_testing_projects = self.testing_projects.keys()

        for key_testing in key_testing_projects:
            testing_pro = self.testing_projects[key_testing]
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            rec_file = str(Path(self.rec_dir) / filename)
            recommendations = self.reader.read_recommendation_file(rec_file)
            
            for lib in recommendations.values():
                freq[lib] += 1

        return dict(freq)

    def ebn(self, start_pos: int, end_pos: int) -> float:
        testing_projects = self.reader.read_project_list(
            str(Path(self.src_dir) / "projects.txt"), start_pos, end_pos)
        all_recs = {}
        total_ebn = 0.0

        # First pass: collect all recommendations
        for key, project in testing_projects.items():
            filename = project.replace("git://github.com/", "").replace("/", "__")
            rec_file = str(Path(self.rec_dir) / filename)
            recommendations = self.reader.read_recommendation_file(rec_file)
            all_recs[key] = set(recommendations.values())

        # Second pass: compute EBN for each project
        for key, project in testing_projects.items():
            filename = project.replace("git://github.com/", "").replace("/", "__")
            rec_file = str(Path(self.rec_dir) / filename)
            recommendations = self.reader.read_recommendation_file(rec_file)
            
            project_ebn = 0.0
            for pos, lib in sorted(recommendations.items()):
                count = 0
                for other_key, other_recs in all_recs.items():
                    if other_key != key and lib in other_recs:
                        count += 1
                
                prob = count / len(testing_projects)
                if prob > 0:
                    project_ebn += -prob * math.log2(prob)
            
            total_ebn += project_ebn

        avg_ebn = total_ebn / len(testing_projects) if testing_projects else 0
        self.logger.info(f"EBN is: {avg_ebn}")
        return avg_ebn
