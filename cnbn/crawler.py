from pathlib import Path
from typing import Dict, List, Optional, Set
import csv
import logging
from dataclasses import dataclass, field
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s: %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CrawlerConfig:
    csv_path: Path = Path('repo_topics.csv')
    default_topics: List[str] = field(default_factory=lambda: ["topic1", "topic2", "topic3"])
    encoding: str = 'utf-8'
    required_columns: int = 4

def map_repos2topics(repo_list: List[str], config: Optional[CrawlerConfig] = None) -> Dict[str, List[str]]:
    """Maps repository identifiers to their corresponding topics from a CSV file."""
    config = config or CrawlerConfig()
    repo_to_topics: Dict[str, List[str]] = {}
    
    try:
        if not config.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {config.csv_path}")
            
        with config.csv_path.open(mode='r', encoding=config.encoding, errors='ignore') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) < config.required_columns:
                    continue
                
                repo_name = row[0].strip()
                if repo_name in repo_list:
                    topics = [topic.strip() for topic in row[3:] if topic.strip()]
                    repo_to_topics[repo_name] = topics
                    
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        # Fallback to default topics
        for repo in repo_list:
            repo_to_topics[repo] = config.default_topics.copy()
    
    return repo_to_topics