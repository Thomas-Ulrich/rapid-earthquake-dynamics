import subprocess
import requests
import re
from pathlib import Path


def find_git_root(start_path):
    """Walk up from start_path to find the .git directory"""
    current = Path(start_path).resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return None


def get_git_remote_url(git_root):
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=git_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return url
    except subprocess.CalledProcessError:
        return None


def parse_github_owner_repo(url):
    # Supports both HTTPS and SSH URLs
    https_pattern = r"https://github\.com/(.*?)/(.*?)(\.git)?$"
    ssh_pattern = r"git@github\.com:(.*?)/(.*?)(\.git)?$"

    for pattern in [https_pattern, ssh_pattern]:
        match = re.match(pattern, url)
        if match:
            owner, repo = match.group(1), match.group(2)
            return owner, repo
    return None, None


def get_version_info(owner, repo):
    base_url = f"https://api.github.com/repos/{owner}/{repo}"

    tags_resp = requests.get(f"{base_url}/tags")
    tags = tags_resp.json()
    latest_tag = tags[0]["name"] if tags else None

    repo_info = requests.get(base_url).json()
    default_branch = repo_info.get("default_branch", "main")

    commit_resp = requests.get(f"{base_url}/commits/{default_branch}")
    latest_commit = commit_resp.json().get("sha", None)

    return {
        "owner": owner,
        "repo": repo,
        "latest_tag": latest_tag,
        "default_branch": default_branch,
        "latest_commit": latest_commit,
    }


def get_repo_info():
    script_path = Path(__file__).resolve()
    git_root = find_git_root(script_path)

    if git_root:
        remote_url = get_git_remote_url(git_root)
        owner, repo = parse_github_owner_repo(remote_url)

        if owner and repo:
            info = get_version_info(owner, repo)
            return info
        else:
            print("Could not parse owner/repo from remote URL.")
    else:
        print("Could not find Git root from script location.")

    return {}


if __name__ == "__main__":
    info = get_repo_info()
    print(info)
