"""Stage E driver: ten folds of CNBN, appending into one shared seed file.

Per round, with `freeCodeCamp/freeCodeCamp` in fold 3:

    train_dir  custom_dataset/CNBN/folds_sync/train3/   ~579 repos, ~5,800 README copies
    test_dir   custom_dataset/CNBN/folds_sync/test3/    ~65 repos
    dirs       ['learn-to-code', 'nodejs', 'react', ...]  test3's topic subdirectories
    test_file  custom_dataset/CNBN/folds_sync/test_3.txt

Fold 1 truncates `training_data.csv`, folds 2-10 append. Test sets are disjoint, so after
round 10 the file holds every repository exactly once - which is what TopFilter needs, since
it reads one seed file for all ten of its own folds.
"""

import os
from pathlib import Path
from typing import List
from model import load_data, predict_topics
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s- %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hybridrec.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_experiment(round_num: int, model_type: str = 'CNB', accumulate: bool = False) -> None:
    """Run a single experiment round."""
    try:
        # Configure paths using pathlib
        base_dir = Path(os.environ.get("CNBN_FOLDS_DIR",
                                       "./custom_dataset/CNBN/folds_sync"))
        train_dir = base_dir / f"train{round_num}"
        test_dir = base_dir / f"test{round_num}"
        
        if not train_dir.exists() or not test_dir.exists():
            raise FileNotFoundError(f"Dataset directories not found for round {round_num}")

        logger.info(f"Starting round {round_num} with {model_type} model")
        logger.info(f"Training directory: {train_dir}")
        logger.info(f"Testing directory: {test_dir}")

        # Get test subdirectories (topics). These are the topic directories present in the
        # TEST tree, i.e. the labels this fold's held-out repos carry - not the class space.
        #   dirs  ['learn-to-code', 'nodejs', 'react', 'd3', 'careers', ...]
        dirs = [d.name for d in test_dir.iterdir() if d.is_dir()]
        if not dirs:
            logger.warning(f"No test directories found in {test_dir}")
            return

        # Load training data
        #   train_data    ['# freeCodeCamp\n\nLearn to code...', ...]   ~5,800 README copies
        #   train_labels  ['learn-to-code', 'nodejs', ...]                one label per copy
        train_data, train_labels = load_data(train_dir)
        
        test_file = base_dir / f"test_{round_num}.txt"
        # Run prediction
        # num_topics=20 is what gets WRITTEN; TopFilter consumes only the first 5 of each row.
        predict_topics(
            dirs=dirs,
            test_dir=test_dir,
            train_data=train_data,
            labels=train_labels,
            num_topics=20,
            list_test=Path(test_file),
            model=model_type,
            accumulate=accumulate
        )

        logger.info(f"Completed round {round_num} successfully")

    except Exception as e:
        logger.error(f"Error in round {round_num}: {str(e)}")
        raise

def main() -> None:
    """Main execution function."""
    try:
        model_type = 'CNB'  # Can be changed to 'MNB' for Multinomial Naive Bayes
        
        for i in range(1, 11):
            logger.info('*' * 40)
            logger.info(f'Starting round {i}')
            
            # Fold 1 truncates the shared output files; folds 2-10 append, so
            # training_data.csv ends up covering every repository exactly once.
            run_experiment(round_num=i, model_type=model_type, accumulate=(i > 1))

            logger.info(f'Completed round {i}')
            logger.info('*' * 40)

    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()