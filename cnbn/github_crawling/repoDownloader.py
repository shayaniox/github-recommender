import os
import git
from github import Github
import re

def download_repos(repo_url, main_dir):
    """
    Downloads a GitHub repository to a local directory.
    
    Args:
        repo_url (str): The GitHub repository URL or identifier (e.g., 'username/repo')
        main_dir (str): The main directory where repositories will be stored
    
    Returns:
        str: Path to the downloaded repository or None if download failed
    """
    try:
        # Clean up the repo URL to extract username and repo name
        repo_url = repo_url.strip()
        
        # Handle different URL formats
        if "github.com" in repo_url:
            # Extract username/repo from URL
            match = re.search(r'github\.com[:/]([^/]+)/([^/\n]+)', repo_url)
            if match:
                username, repo_name = match.groups()
                repo_name = repo_name.replace('.git', '')
            else:
                print(f"Could not parse GitHub URL: {repo_url}")
                return None
        else:
            # Assume format is already username/repo
            parts = repo_url.split('/')
            if len(parts) >= 2:
                username, repo_name = parts[0], parts[1]
            else:
                print(f"Invalid repository identifier: {repo_url}")
                return None
        
        # Create target directory
        repo_path = os.path.join(main_dir, f"{username}_{repo_name}")
        os.makedirs(repo_path, exist_ok=True)
        
        # Clone the repository
        git_url = f"https://github.com/{username}/{repo_name}.git"
        print(f"Cloning {git_url} to {repo_path}")
        git.Repo.clone_from(git_url, repo_path)
        
        return repo_path
    
    except Exception as e:
        print(f"Error downloading repository {repo_url}: {str(e)}")
        return None