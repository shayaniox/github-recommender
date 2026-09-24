"""Stage E, part 2 - the content-based recommender (CNBN = Complement NB + kNowledge graph).

RUNNING EXAMPLE, `freeCodeCamp/freeCodeCamp` as a fold-3 test repository.

    in    custom_dataset/CNBN/folds_sync/test3/<topic>/freeCodeCamp,freeCodeCamp.txt
          - the README text, reachable under each of its 10 topic directories

    out   training_data.csv line:
          'freeCodeCamp/freeCodeCamp;javascript;education;react;nodejs;learn-to-code;...'
          - 20 topics, highest posterior first; TopFilter later reads only the first 5

Because Stage D copies a README once per topic, this is a SINGLE-label classifier over the
topic vocabulary: `clf.classes_` is every topic, and `predict_proba` is one distribution over
it summing to 1.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import operator
import os
import time
import logging
from metrics import calculate_metrics, get_topics, precision, recall, success_rate, top_rank
from crawler import map_repos2topics
from kg import build_adjacency, expand as kg_expand

logger = logging.getLogger(__name__)

class ModelConfig:
    def __init__(
        self,
        model_type: str = 'CNB',
        num_topics: int = 20,
        # Overridable so two variants can run concurrently without clobbering each other's
        # outputs; unset, the defaults are exactly the pipeline's normal paths.
        results_path: Path = Path(os.environ.get('CNBN_RESULTS_DIR', 'results')),
        ease_output_path: Path = Path(
            os.environ.get('CNBN_EASE_OUT', 'outpath/TopFilter/training_data.csv')),
        kg_enabled: bool = False,
        timestamp_path: Path = Path('timestamp.csv'),
        lang_file: Path = Path('lang_file.txt'),
        topics_file: Path = Path('topics_134.txt')
    ):
        self.model_type = model_type
        self.num_topics = num_topics
        self.results_path = results_path
        self.ease_output_path = ease_output_path
        self.kg_enabled = kg_enabled
        self.timestamp_path = timestamp_path
        self.lang_file = lang_file
        self.topics_file = topics_file

def load_data(path_topic: Path) -> Tuple[List[str], List[str]]:
    """Load dataset from directory structure. Directory name = class label.

        path_topic  custom_dataset/CNBN/folds_sync/train3/
                      learn-to-code/freeCodeCamp,freeCodeCamp.txt
                      nodejs/freeCodeCamp,freeCodeCamp.txt
                      nodejs/expressjs,express.txt
                      ...

        X  ['# freeCodeCamp\n\nLearn to code...', '# freeCodeCamp\n\nLearn to code...', ...]
        y  ['learn-to-code',                       'nodejs',                             ...]

    Note X[0] and X[1] are the SAME text under two labels - that repetition is the design,
    not a bug. A fold of ~579 repos yields ~5,800 samples over ~1,300 distinct labels.
    """
    X: List[str] = []
    y: List[str] = []
    
    try:
        for topic in path_topic.iterdir():
            if not topic.is_dir():
                continue
                
            for file_path in topic.glob('*.txt'):
                try:
                    with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                        X.append(f.read())
                        y.append(topic.name)
                except IOError as e:
                    logger.warning(f"Error reading {file_path}: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error loading data from {path_topic}: {str(e)}")
        raise
        
    logger.info(f"Loaded {len(X)} samples with {len(set(y))} topics")
    return X, y

def predict_topics(
    dirs: List[str],
    test_dir: Path,
    train_data: List[str],
    labels: List[str],
    num_topics: int,
    list_test: Path,
    model: str,
    accumulate: bool = False
) -> Dict[str, List[str]]:
    """Predict topics for test repositories.

    `accumulate=False` truncates the output files; `True` appends to them. A ten-fold run
    calls this once per fold, and every fold's test set is disjoint, so appending builds one
    `training_data.csv` covering the whole corpus exactly once - which is what TopFilter
    needs, since it reads a single seed file for all ten of its own folds.
    """
    # KG_ENABLED=1 turns on knowledge-graph expansion; unset reproduces the no-KG baseline.
    config = ModelConfig(
        model_type=model,
        num_topics=num_topics,
        kg_enabled=os.environ.get('KG_ENABLED') == '1',
    )
    logger.info(f'KG expansion on CNBN branch: {config.kg_enabled}')
    config.results_path.mkdir(parents=True, exist_ok=True)

    semantic_dict = build_adjacency("sedkgraph.csv")
    
    try:
        # Feature extraction. TF-IDF over raw README text:
        #   train_data   ['# freeCodeCamp\n\nLearn to code for free...', ...]
        #   train_tfidf  sparse (5800 x ~90000) matrix, row = one README copy
        logger.info("Extracting features from dataset")
        count_vect = TfidfVectorizer(
            input='content',
            stop_words='english',
            lowercase=True,
            analyzer='word'
        )
        train_vectors = count_vect.fit_transform(train_data)
        # TfidfVectorizer ALREADY emits TF-IDF. Passing its output through a TfidfTransformer
        # applies IDF a second time - squaring it and re-normalising - which is not what the
        # method describes. The historical runs in docs/METRICS.md were all produced with the
        # double weighting, so it stays reproducible behind CNBN_DOUBLE_TFIDF=1; correct
        # single weighting is the default.
        double_tfidf = os.environ.get('CNBN_DOUBLE_TFIDF') == '1'
        if double_tfidf:
            tfidf_transformer = TfidfTransformer()
            train_tfidf = tfidf_transformer.fit_transform(train_vectors)
        else:
            tfidf_transformer = None
            train_tfidf = train_vectors
        logger.info(f'double TF-IDF weighting: {double_tfidf}')

        # Model training
        logger.info(f"Training {model} model")
        start_training = time.time()
        
        if model == 'MNB':
            clf = MultinomialNB()
        else:
            clf = ComplementNB()
            
        #   labels        ['learn-to-code', 'nodejs', 'nodejs', 'react', ...]
        #   clf.classes_  ['3d', 'ai', 'android', ..., 'zsh']   sorted, ~1,300 entries
        clf.fit(train_tfidf, labels)
        end_training = time.time()
        training_time = end_training - start_training
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Initialize prediction structures
        predicted_topics: Dict[str, List[str]] = {}
        test_repos: List[str] = []
        
        # Read test repository list
        with list_test.open('r', encoding='utf-8', errors='ignore') as f:
            test_repos = [line.strip() for line in f if line.strip()]
        
        # Process each test directory
        for dir_name in dirs:
            dir_path = test_dir / dir_name
            if not dir_path.is_dir():
                logger.warning(f"Directory not found: {dir_path}")
                continue
                
            for file_path in dir_path.glob('*.txt'):
                # A repository's README is copied into one directory per topic it carries,
                # so the same file is reached ~10 times per fold and used to be classified
                # ~10 times, each run overwriting the previous identical result. The file
                # name identifies the repository, so one prediction per repository suffices.
                if file_path.name in predicted_topics:
                    continue
                try:
                    with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                        repo_content = [f.read()]
                        
                    # Transform test data
                    X_new_counts = count_vect.transform(repo_content)
                    X_new_tfidf = (tfidf_transformer.transform(X_new_counts)
                                   if tfidf_transformer is not None else X_new_counts)
                    
                    # Predict probabilities over every topic class.
                    #   probas[0]  [0.0001, 0.0003, ..., 0.0002]   one per clf.classes_ entry,
                    #                                              summing to 1
                    # `predicted` (the argmax label) is computed but never used.
                    predicted = clf.predict(X_new_tfidf)
                    probas = clf.predict_proba(X_new_tfidf)
                    
                    # Get top N predictions with probabilities
                    topics_with_probs = []
                    for i, class_ in enumerate(clf.classes_):
                        topics_with_probs.append((class_, probas[0][i]))
                    
                    # Sort by probability (descending)
                    #   before  [('3d', 0.0001), ('ai', 0.0003), ..., ('zsh', 0.0002)]
                    #   after   [('javascript', 0.28), ('education', 0.19), ('react', 0.11), ...]
                    # `list.sort` is stable and reverse=True does NOT reverse equal elements, so
                    # ties keep clf.classes_ order - i.e. equal posteriors break alphabetically.
                    topics_with_probs.sort(key=operator.itemgetter(1), reverse=True)
                    
                    # Extract top topics. num_topics=20 here; TopFilter reads only the first 5
                    # (data_reader.extract_EASE_dictionary), so the other 15 are written unread.
                    #   top_topics  ['javascript', 'education', 'react', 'nodejs', 'learn-to-code',
                    #                'programming', 'd3', ... ]                       20 entries
                    top_topics = [topic for topic, prob in topics_with_probs[:num_topics]]

                    # KG expansion belongs to the Topics Merger, which sees CNBN's and CF's
                    # outputs together and can rank neighbours by how many of them a topic is
                    # adjacent to. Expanding here as well would bury CNBN's own predictions
                    # under neighbours before CF ever sees them, so this branch only expands
                    # for the CNBN-alone ablation.
                    if config.kg_enabled:
                        #   top_topics  [..., 'javascript', 'react', ...]
                        #   neighbours  ['typescript', 'frontend', 'web', ...]  one-hop in sedkgraph
                        #   merged      originals first, neighbours appended, duplicates dropped
                        #               (dict.fromkeys keeps first occurrence, so order survives)
                        neighbours = [topic for topic, _ in kg_expand(top_topics, semantic_dict)]
                        top_topics = list(dict.fromkeys(top_topics + neighbours))

                    predicted_topics[file_path.name] = top_topics
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue
        
        # Calculate metrics
        logger.info("Calculating metrics")
        #   test_repos     ['freeCodeCamp/freeCodeCamp', 'pytorch/pytorch', ...]
        #   actual_topics  {'freeCodeCamp/freeCodeCamp': ['learn-to-code', 'nonprofits', ...]}
        actual_topics = map_repos2topics(test_repos)
        
        # Prepare output files
        results_files = [
            config.results_path / 'results_1_topics.csv',
            config.results_path / 'results_7_topics.csv',
            config.results_path / 'results_9_topics.csv'
        ]
        
        # Truncate and write the header on the FIRST fold only. These files used to be
        # opened in append mode with the header written only when absent, so each run
        # silently piled onto the previous one (176,650 rows across 88,358 repos from a
        # 100-repo test set). Folds 2-10 append to the file fold 1 created.
        if not accumulate:
            for res_file in results_files:
                with res_file.open('w', encoding='utf-8') as f:
                    f.write("repo,success@1,success@2,success@3,success@4,success@5,"
                            "success@6,success@7,success@8,success@9,success@10,"
                            "precision,recall,top_rank\n")
        
        # Calculate and write metrics
        for repo in test_repos:
            if '/' not in repo:
                continue
                
            repo_key = repo.replace('/', ',') + '.txt'
            act_topics, pred_1, pred_7, pred_9 = get_topics(
                actual_topics, repo,
                predicted_topics, repo_key
            )
            
            # Write metrics for different prediction lengths
            for pred_list, res_file in zip([pred_1, pred_7, pred_9], results_files):
                with res_file.open('a', encoding='utf-8') as f:
                    f.write(f"{repo},")
                    for n in range(1, 11):
                        f.write(f"{success_rate(pred_list, act_topics, n)},")
                    f.write(f"{precision(pred_list, act_topics)},")
                    f.write(f"{recall(pred_list, act_topics)},")
                    f.write(f"{top_rank(pred_list, act_topics)}\n")
        
        # Write training data to file.
        # TopFilter's DataReader.load_EASE_output splits these lines on ';', so the
        # separator must be ';' for the CNBN -> TopFilter handoff to carry any topics.
        # It also reads the file from its own source directory, so write a copy there.
        #   results/training_data.csv             reporting copy
        #   outpath/TopFilter/training_data.csv   the live bridge TopFilter reads
        # Same bytes to both; TopFilter's DataReader only ever looks in its own src_dir.
        # Line format, ';'-separated with '/' restored in the repo name:
        #   'freeCodeCamp/freeCodeCamp;javascript;education;react;nodejs;learn-to-code;...'
        training_data_file = config.results_path / 'training_data.csv'
        for destination in (training_data_file, config.ease_output_path):
            if destination is None:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('a' if accumulate else 'w', encoding='utf-8') as f:
                for repo, topics in predicted_topics.items():
                    f.write(f"{repo.replace(',', '/')};{';'.join(topics)}\n")
        
        # Write timestamps
        with config.timestamp_path.open('w', encoding='utf-8') as f:
            f.write("training,testing\n")
            f.write(f"{training_time},0\n")  # Testing time not implemented
        
        return predicted_topics
        
    except Exception as e:
        logger.error(f"Error during topic prediction: {str(e)}")
        raise