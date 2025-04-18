import csv
from pathlib import Path
from sklearn.model_selection import KFold
import shutil

def prepare_datasets(
    csv_path: Path, 
    readme_dir: Path = Path("./repo_readme"),
    output_dir: Path = Path("custom_dataset/CNBN/ten_folder_100")
):
    # Read repository data
    with csv_path.open('r', encoding='utf-8') as f:
        reader = csv.reader(f)
        repos = [row for row in reader if len(row) > 4]  # Skip malformed rows

    # Create 10-fold split
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Prepare output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(repos), 1):
        # Create train/test directories
        train_dir = output_dir / f"train{fold}"
        test_dir = output_dir / f"test{fold}"
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # Prepare test file
        test_file = output_dir / f"test_{fold}.txt"
        with test_file.open('w', encoding='utf-8') as f:
            for idx in test_idx:
                repo_name = repos[idx][0]
                f.write(f"{repo_name}\n")
        
        # Process each repository
        for idx, repo in enumerate(repos):
            repo_name = repo[0]
            topics = repo[4:]
            
            if not topics:  # Skip repos with no topics
                continue
                
            # Determine if this is train or test repo
            target_dir = train_dir if idx in train_idx else test_dir
            
            # Use existing README file
            readme_file = readme_dir / f"{repo_name.replace('/', ',')}.txt"
            if not readme_file.exists():
                continue
                
            for topic in topics:
                topic_dir = target_dir / topic
                topic_dir.mkdir(exist_ok=True)
                
                # Create symlink to original README (or copy if symlinks not supported)
                repo_file = topic_dir / f"{repo_name.replace('/', ',')}.txt"
                # try:
                #     if not repo_file.exists():
                #         repo_file.symlink_to(readme_file)  # Use symlinks to save space
                #         # Alternative for Windows: shutil.copy2(readme_file, repo_file)
                # except OSError:
                shutil.copy2(readme_file, repo_file)

if __name__ == "__main__":
    prepare_datasets(
        csv_path=Path("repositories.csv"),
        readme_dir=Path("repo_readme")
    )
