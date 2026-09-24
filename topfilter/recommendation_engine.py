import os
import math
import logging
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import heapq
import csv
from data_reader import DataReader
from kg import build_adjacency, expand as kg_expand

KG_PATH = "sedkgraph.csv"
DEP_PREFIX = "#DEP#"

class RecommendationEngine:
    def __init__(self, source_dir: str, sub_folder: str, num_of_neighbours: int, 
                 testing_start_pos: int, testing_end_pos: int, bayesian: bool):
        self.src_dir = source_dir
        self.sub_folder = sub_folder
        self.num_of_neighbours = num_of_neighbours
        self.rec_dir = os.path.join(self.src_dir, sub_folder, "Recommendations")
        self.sim_dir = os.path.join(self.src_dir, sub_folder, "Similarities")
        self.ground_truth = os.path.join(self.src_dir, sub_folder, "GroundTruth")
        # See TopicSimilarityCalculator: these were never created, only inherited.
        for directory in (self.rec_dir, self.sim_dir, self.ground_truth):
            os.makedirs(directory, exist_ok=True)
        self.reader = DataReader(source_dir)
        self.testing_start_pos = testing_start_pos
        self.testing_end_pos = testing_end_pos
        self.bayesian = bayesian
        self.num_of_EASE_input = 5
        self.logger = logging.getLogger(__name__)

        # Ablation: keep bayesian mode's setup (CNBN's 5 seeds are the visible profile, the
        # repo's full topic set is the ground truth) but do NOT prepend the seeds to the
        # output. The result is CF's own ranking scored on exactly the hybrid's task, which
        # is the only way to read off what the Topics Merger's prepend is worth. Scoring a
        # NO_SEED_PREPEND=1 run must use the unsuffixed metrics: the _b family re-inserts the
        # seeds from training_data.csv itself and would undo the ablation.
        self.no_seed_prepend = os.environ.get("NO_SEED_PREPEND") == "1"

        # Knowledge-graph expansion of the merged topic list. KG_ENABLED=1 turns it on;
        # KG_ORIGINALS caps how many original (CNBN + CF) topics are kept ahead of the
        # neighbours, so the neighbours land inside the evaluated cut-off instead of past
        # it. Unset KG_ORIGINALS keeps every original and appends neighbours behind them.
        self.kg_enabled = os.environ.get("KG_ENABLED") == "1"
        self.kg_originals = int(os.environ["KG_ORIGINALS"]) if os.environ.get("KG_ORIGINALS") else None
        # "append" (default) adds neighbours after the originals; "rerank" instead boosts
        # existing candidates by KG agreement and introduces nothing new.
        self.kg_mode = os.environ.get("KG_MODE", "append")
        self.kg_lambda = float(os.environ.get("KG_LAMBDA", "0.5"))
        self.kg_sources = int(os.environ.get("KG_SOURCES", "10"))
        self.kg_adjacency = build_adjacency(KG_PATH) if self.kg_enabled else {}
        if self.kg_enabled:
            self.logger.info(
                f"KG merge enabled: mode={self.kg_mode}, {len(self.kg_adjacency)} topics, "
                f"lambda={self.kg_lambda}, sources={self.kg_sources}, "
                f"originals kept = {self.kg_originals if self.kg_originals else 'all'}"
            )

    def build_user_item_matrix(self, testing_pro: str, lib_set: List[str]) -> List[List[float]]:
        """Assemble the 21 x |topics| ratings matrix: 20 neighbour rows plus the active repo.

        Continuing the six-repo toy fold from topic_similarity_calculator (real folds take 20
        neighbours out of ~579; here only 6 exist, so rows 6..19 stay all-zero and contribute
        nothing to either sum):

            lib_set  ['api','async','bot','chat','cli','django','docker','html','jupyter',
                      'llm','machine-learning','nlp','pandas','python','rest-api','rust',
                      'server','tool','web']                    19 columns, sorted

            row 0  T3  echo/fastapi-starter     1.0 where that neighbour has the topic
            row 1  T1  bravo/llm-gateway        ...ordered by SIMILARITY RANK, not project ID
            row 2  T2  delta/slack-assistant
            ...
            row 20 the active repo:   1.0 for api/docker/python/rest-api/server  (visible)
                                     -1.0 for the other 14                       (score these)

        -1 is the marker the scoring loop looks for, so a topic the repo already shows as
        visible is never re-recommended.
        """
        filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
        testing_filename = filename
        testing_dict_filename = os.path.join(self.src_dir, f"dicth_{testing_filename}")
        
        testing_dictionary = (self.reader.extract_EASE_dictionary(testing_dict_filename, self.num_of_EASE_input, self.ground_truth) 
                            if self.bayesian 
                            else self.reader.extract_half_dictionary(testing_dict_filename, self.ground_truth, False))
        
        # The active repo's visible profile - CNBN's 5 seeds in bayesian mode.
        #   {'#DEP#python', '#DEP#api', '#DEP#docker', '#DEP#rest-api', '#DEP#server'}
        testing_libs = {v for v in testing_dictionary.values() if v.startswith("#DEP#")}
        
        tmp = os.path.join(self.sim_dir, filename)
        sim_projects = self.reader.get_most_similar_projects(tmp, self.num_of_neighbours)
        
        all_neighbour_libs = {}
        libraries = set()
        
        # Neighbours expose their OWNER-DEFINED topics in full - get_libraries reads dicth_
        # straight through. Nothing is held back on the training side.
        #   all_neighbour_libs[0]  {'#DEP#python','#DEP#api','#DEP#rest-api','#DEP#docker',
        #                           '#DEP#async'}      <- T3, the top-ranked neighbour
        for key, project in sim_projects.items():
            filename = project.replace("git://github.com/", "").replace("/", "__")
            tmp = os.path.join(self.src_dir, f"dicth_{filename}")
            libs = self.reader.get_libraries(tmp)
            all_neighbour_libs[key] = libs
            libraries.update(libs)
        
        all_neighbour_libs[self.num_of_neighbours] = testing_libs
        libraries.update(testing_libs)
        
        # Sorted, not raw set order: `libraries` is a set, so its iteration order varies
        # with PYTHONHASHSEED between processes. That reorders the columns of the
        # user-item matrix and breaks score ties differently on every run, which made two
        # identical runs disagree on the top-10 of ~15% of projects.
        for lib in sorted(libraries):
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
            # Mean-centered user-based CF over the matrix above:
            #     score(j) = 1.0 + SUM_k (r[k][j] - mean(row k)) * sim[k] / SUM_k sim[k]
            #
            # Worked for 'chat', which T1 (row 1) and T2 (row 2) carry, with SUM sim = 1.578:
            #     k=0 T3  (0 - 0.2632) * 0.752 = -0.1978
            #     k=1 T1  (1 - 0.3158) * 0.409 = +0.2798
            #     k=2 T2  (1 - 0.2632) * 0.290 = +0.2137
            #     k=3 T4  (0 - 0.2632) * 0.119 = -0.0313
            #     k=4 T6  (0 - 0.2105) * 0.008 = -0.0018
            #                              sum = +0.2626  ->  1 + 0.2626/1.578 = 1.166
            #
            # Across all 14 candidates:
            #     async 1.200   bot 1.166   chat 1.166   llm 0.983   nlp 0.907
            #     django/web/html 0.799     machine-learning 0.729   cli/rust/tool 0.723
            #
            # Note 'async' wins on ONE very similar neighbour, outranking chat/bot which two
            # mid-similarity neighbours share - that is what mean-centering rewards.
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
            
            # Stable sort, and `recommendations` was filled by ascending j, so ties keep
            # lib_set (alphabetical) order: 'bot' 1.166 lands ahead of 'chat' 1.166.
            sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            
            filename = testing_pro.replace("git://github.com/", "").replace("/", "__")
            tmp = os.path.join(self.rec_dir, filename)
            
            # Topics Merger: CNBN's topics (bayesian seeds) followed by CF's ranked topics,
            # then — when the KG is enabled — the one-hop neighbours of both branches.
            merged = []
            if self.bayesian and not self.no_seed_prepend:
                ease_dict_file = os.path.join(self.src_dir, f"dicth_{filename}")
                ease_topic = self.reader.extract_EASE_dictionary(
                    ease_dict_file, self.num_of_EASE_input, self.ground_truth)
                # pop(1) drops the repo's own node, leaving just the 5 seeds. They go in at a
                # FLAT 2.0, above any CF score, so CNBN's posterior ordering alone decides
                # ranks 1-5 of the final file. This prepend is the hybrid mechanism.
                #   merged  [('#DEP#python', 2.0), ('#DEP#api', 2.0), ('#DEP#docker', 2.0),
                #            ('#DEP#rest-api', 2.0), ('#DEP#server', 2.0)]
                ease_topic.pop(1, None)
                merged.extend((v, 2.0) for v in ease_topic.values())

            # Then the CF output, translating column index back to topic name:
            #   [..., ('#DEP#async', 1.200), ('#DEP#bot', 1.166), ('#DEP#chat', 1.166), ...]
            merged.extend((lib_set[int(key)], score) for key, score in sorted_recommendations)

            if self.kg_enabled:
                merged = (self._kg_rerank(merged) if self.kg_mode == "rerank"
                          else self._expand_with_kg(merged))

            try:
                with open(tmp, 'w') as writer:
                    for label, score in merged:
                        writer.write(f"{label}\t{score}\n")
            except IOError as e:
                self.logger.error(f"Error writing recommendations to {tmp}: {e}")

    def _kg_rerank(self, merged):
        """Re-rank existing candidates by knowledge-graph agreement.

        The append strategy can only reach the evaluated window by pushing correct CF
        predictions out of it, so it loses precision wherever it becomes visible. Here the
        KG never introduces a topic: it boosts candidates the recommender already produced,
        in proportion to how many source topics they neighbour. A topic supported by
        several confident sources rises; everything else keeps its relative order.

            score'(t) = score(t) + lambda * (encounters(t) / max_encounters)

        Sources are the high-confidence head of the merged list (the CNBN seeds plus the
        top CF topics), not the whole list, so weak tail candidates do not vote.
        """
        seen = set()
        candidates = []
        for label, score in merged:
            if label not in seen:
                seen.add(label)
                candidates.append((label, score))
        if not candidates:
            return candidates

        head = candidates[:self.kg_sources]
        sources = [self._bare(label) for label, _ in head]
        encounters = dict(kg_expand(sources, self.kg_adjacency))
        if not encounters:
            return candidates
        top = max(encounters.values())

        boosted = []
        for label, score in candidates:
            bonus = self.kg_lambda * (encounters.get(self._bare(label), 0) / top)
            boosted.append((label, score + bonus))
        # Stable sort keeps the original order among equally-boosted candidates.
        boosted.sort(key=lambda item: item[1], reverse=True)
        return boosted

    @staticmethod
    def _bare(label):
        return label[len(DEP_PREFIX):] if label.startswith(DEP_PREFIX) else label

    def _expand_with_kg(self, merged):
        """Append one-hop SED-KGraph neighbours of the merged topic list.

        Sources are every original topic from both branches; a neighbour's rank is the
        number of distinct sources it is adjacent to (kg.expand). Originals keep their
        order ahead of the neighbours, so the KG can only add, never displace a correct
        prediction. Neighbour scores sit strictly below every original score so that a
        consumer sorting by score sees the same order as one reading positionally.
        """
        seen = set()
        originals = []
        for label, score in merged:
            if label not in seen:
                seen.add(label)
                originals.append((label, score))

        if self.kg_originals is not None:
            originals = originals[:self.kg_originals]

        # SED-KGraph is keyed by bare topic names; the pipeline carries "#DEP#<topic>".
        sources = [label[len(DEP_PREFIX):] if label.startswith(DEP_PREFIX) else label
                   for label, _ in originals]
        neighbours = kg_expand(sources, self.kg_adjacency)
        if not neighbours:
            return originals

        floor = min((score for _, score in originals), default=0.0)
        top_encounters = neighbours[0][1]
        expanded = list(originals)
        for topic, encounters in neighbours:
            label = f"{DEP_PREFIX}{topic}"
            if label in seen:
                continue
            seen.add(label)
            expanded.append((label, floor - 1.0 + encounters / (top_encounters + 1.0)))
        return expanded

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
        scalar = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
        norm1 = math.sqrt(sum(v * v for v in vector1))
        norm2 = math.sqrt(sum(v * v for v in vector2))
        
        if norm1 * norm2 > 0 and scalar > 0:
            return scalar / math.sqrt(norm1 * norm2)
        return 0.0
