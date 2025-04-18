from pathlib import Path
from typing import Dict, List, Tuple, Optional
from nltk.stem import PorterStemmer
import logging
import shutil
from distutils.dir_util import copy_tree

logger = logging.getLogger(__name__)

def get_topics(
    dict_actual: Dict[str, List[str]],
    key_actual: str,
    dict_predicted: Dict[str, List[str]],
    key_predicted: str
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Get actual and predicted topics for comparison.
    
    Args:
        dict_actual: Dictionary containing actual topics
        key_actual: Key for actual topics
        dict_predicted: Dictionary containing predicted topics
        key_predicted: Key for predicted topics
        
    Returns:
        Tuple of (actual_topics, top1_predicted, top7_predicted, top9_predicted)
    """
    list_actual = dict_actual.get(key_actual, [])
    list_predicted = dict_predicted.get(key_predicted, [])
    
    return (
        list_actual,
        list_predicted[:1],
        list_predicted[:7],
        list_predicted[:9]
    )

def stemming_topics(
    predicted: List[str],
    actual: List[str]
) -> Tuple[List[str], List[str]]:
    """Apply stemming to both predicted and actual topics."""
    stemmer = PorterStemmer()
    stemmed_pred = [stemmer.stem(p) for p in predicted]
    stemmed_act = [stemmer.stem(a) for a in actual]
    return stemmed_pred, stemmed_act

def success_rate(
    predicted: List[str],
    actual: List[str],
    n: int
) -> int:
    """Calculate success rate for top-n predictions."""
    if not actual:
        return 0
    match = [value for value in predicted if value in actual]
    return 1 if len(match) >= n else 0

def precision(predicted: List[str], actual: List[str]) -> float:
    """Calculate precision metric."""
    if not actual:
        return 0.0
    true_p = len([value for value in predicted if value in actual])
    false_p = len([value for value in predicted if value not in actual])
    return (true_p / (true_p + false_p)) * 100 if (true_p + false_p) > 0 else 0.0

def recall(predicted: List[str], actual: List[str]) -> float:
    """Calculate recall metric."""
    if not actual:
        return 0.0
    true_p = len([value for value in predicted if value in actual])
    false_n = len([value for value in actual if value not in predicted])
    return (true_p / (true_p + false_n)) * 100 if (true_p + false_n) > 0 else 0.0

def top_rank(predicted: List[str], actual: List[str]) -> int:
    """Check if top prediction is correct."""
    if not predicted:
        return 0
    top = predicted[0]
    return 1 if top in actual else 0

def remove_dashes(actual: List[str]) -> List[str]:
    """Remove dashes from topics."""
    return [topic.replace("-", "") if "-" in topic else topic for topic in actual]

def calculate_metrics(
    act_topics: List[str],
    pred_topics: List[str],
    out_results_path: Path,
    repo: str
) -> None:
    """Compute and write all metrics for a repository."""
    if not act_topics:
        return
        
    try:
        with out_results_path.open('a+', encoding='utf-8', errors='ignore') as out_results:
            # Write metrics implementation here
            pass
            
    except IOError as e:
        logger.error(f"Failed to write metrics for {repo}: {str(e)}")