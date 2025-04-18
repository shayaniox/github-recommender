from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import operator
import time
import logging
import csv
from metrics import calculate_metrics, get_topics, precision, recall, success_rate, top_rank
from crawler import map_repos2topics

logger = logging.getLogger(__name__)

class ModelConfig:
    def __init__(
        self,
        model_type: str = 'CNB',
        num_topics: int = 20,
        results_path: Path = Path('results'),
        timestamp_path: Path = Path('timestamp.csv'),
        lang_file: Path = Path('lang_file.txt'),
        topics_file: Path = Path('topics_134.txt')
    ):
        self.model_type = model_type
        self.num_topics = num_topics
        self.results_path = results_path
        self.timestamp_path = timestamp_path
        self.lang_file = lang_file
        self.topics_file = topics_file

def load_data(path_topic: Path) -> Tuple[List[str], List[str]]:
    """Load dataset from directory structure."""
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
    model: str
) -> Dict[str, List[str]]:
    """Predict topics for test repositories."""
    config = ModelConfig(model_type=model, num_topics=num_topics)
    config.results_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Feature extraction
        logger.info("Extracting features from dataset")
        count_vect = TfidfVectorizer(
            input='content',
            stop_words='english',
            lowercase=True,
            analyzer='word'
        )
        train_vectors = count_vect.fit_transform(train_data)
        tfidf_transformer = TfidfTransformer()
        train_tfidf = tfidf_transformer.fit_transform(train_vectors)

        # Model training
        logger.info(f"Training {model} model")
        start_training = time.time()
        
        if model == 'MNB':
            clf = MultinomialNB()
        else:
            clf = ComplementNB()
            
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
                try:
                    with file_path.open('r', encoding='utf-8', errors='ignore') as f:
                        repo_content = [f.read()]
                        
                    # Transform test data
                    X_new_counts = count_vect.transform(repo_content)
                    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
                    
                    # Predict probabilities
                    predicted = clf.predict(X_new_tfidf)
                    probas = clf.predict_proba(X_new_tfidf)
                    
                    # Get top N predictions with probabilities
                    topics_with_probs = []
                    for i, class_ in enumerate(clf.classes_):
                        topics_with_probs.append((class_, probas[0][i]))
                    
                    # Sort by probability (descending)
                    topics_with_probs.sort(key=operator.itemgetter(1), reverse=True)
                    
                    # Extract top topics
                    top_topics = [topic for topic, prob in topics_with_probs[:num_topics]]
                    predicted_topics[file_path.name] = top_topics
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue
        
        # Calculate metrics
        logger.info("Calculating metrics")
        actual_topics = map_repos2topics(test_repos)
        
        # Prepare output files
        results_files = [
            config.results_path / 'results_1_topics.csv',
            config.results_path / 'results_7_topics.csv',
            config.results_path / 'results_9_topics.csv'
        ]
        
        # Write headers if files don't exist
        for res_file in results_files:
            if not res_file.exists():
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
        
        # Write training data to file
        training_data_file = config.results_path / 'training_data.csv'
        with training_data_file.open('w', encoding='utf-8') as f:
            for repo, topics in predicted_topics.items():
                f.write(f"{repo.replace(',', '/')},{','.join(topics)}\n")
        
        # Write timestamps
        with config.timestamp_path.open('w', encoding='utf-8') as f:
            f.write("training,testing\n")
            f.write(f"{training_time},0\n")  # Testing time not implemented
        
        return predicted_topics
        
    except Exception as e:
        logger.error(f"Error during topic prediction: {str(e)}")
        raise