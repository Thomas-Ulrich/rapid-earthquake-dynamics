import subprocess
import requests
import re
from pathlib import Path
import semver


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


def get_git_info():
    try:
        current_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        local_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        remote_default = subprocess.check_output(
            ["git", "remote", "show", "origin"], text=True
        )
        default_branch = None
        for line in remote_default.splitlines():
            if "HEAD branch:" in line:
                default_branch = line.strip().split(":")[1].strip()
                break

        return current_branch, local_commit, default_branch
    except subprocess.CalledProcessError:
        return None, None, None


def get_commit_distance(from_ref, to_ref):
    try:
        result = subprocess.check_output(
            ["git", "rev-list", "--left-right", "--count", f"{from_ref}...{to_ref}"],
            text=True,
        )
        behind, ahead = map(int, result.strip().split())
        return {"ahead_by": ahead, "behind_by": behind}
    except subprocess.CalledProcessError:
        return None


def compute_commit_distance_from_local_head_to_ref(current_branch, target_ref):
    """
    Calculate the commit distance from the current HEAD to a target reference
    via the remote branch.

    Args:
        current_branch (str): Name of the current branch
        target_ref (str): Target reference (branch name, tag name, or commit SHA)

    Returns:
        dict: {"ahead_by": int, "behind_by": int} or None if calculation failed
    """
    if not current_branch:
        return None

    # Define the remote branch
    remote_branch = f"origin/{current_branch}"

    # Step 1: Calculate distance from HEAD to remote branch
    distance_to_remote = get_commit_distance(remote_branch, "HEAD")

    # Step 2: Calculate distance from target reference to remote branch
    distance_remote_to_ref = get_commit_distance(target_ref, remote_branch)

    # If both distances were calculated successfully, combine them
    if distance_to_remote and distance_remote_to_ref:
        return {
            "ahead_by": distance_to_remote["ahead_by"]
            + distance_remote_to_ref["ahead_by"],
            "behind_by": distance_to_remote["behind_by"]
            + distance_remote_to_ref["behind_by"],
        }

    return None


def get_highest_semver(tags):
    """
    Find the highest semantic version tag from a list of GitHub tag objects.

    Args:
        tags (list): List of GitHub tag objects with at least 'name' and 'commit' keys
                    where 'commit' is a dict with 'sha' key

    Returns:
        tuple: (highest_version_tag, highest_version_sha) or (None, None)
        if no valid semver tags found
    """
    highest_version = None
    highest_version_tag = None
    highest_version_sha = None

    for tag in tags:
        tag_name = tag["name"]
        tag_sha = tag["commit"]["sha"]

        # Strip 'v' prefix if present
        version_str = tag_name
        if tag_name.startswith("v"):
            version_str = tag_name[1:]

        try:
            parsed_version = semver.VersionInfo.parse(version_str)
            if highest_version is None or parsed_version > highest_version:
                highest_version = parsed_version
                highest_version_tag = tag_name
                highest_version_sha = tag_sha
        except ValueError:
            # Skip tags that aren't valid semantic versions
            continue

    return highest_version_tag, highest_version_sha


def get_version_info(owner, repo):
    current_branch, local_commit, default_branch = get_git_info()

    # GitHub tag info
    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    tags_resp = requests.get(f"{base_url}/tags")
    tags = tags_resp.json()

    highest_version_tag, highest_version_sha = get_highest_semver(tags)

    # GitHub latest commit on default branch
    latest_commit = None
    try:
        latest_commit = requests.get(f"{base_url}/commits/{default_branch}").json()[
            "sha"
        ]
    except Exception:
        pass

    main_branch = f"origin/{default_branch}"
    commit_distance_to_main = compute_commit_distance_from_local_head_to_ref(
        current_branch, main_branch
    )
    if highest_version_tag:
        commit_distance_to_highest_version = (
            compute_commit_distance_from_local_head_to_ref(
                current_branch, highest_version_tag
            )
        )
    else:
        commit_distance_to_highest_version = None

    return {
        "owner": owner,
        "repo": repo,
        "highest_version_tag": highest_version_tag,
        "current_branch": current_branch,
        "local_commit": local_commit,
        "latest_commit_on_main": latest_commit,
        "commit_distance_to_main": commit_distance_to_main,
        "commit_distance_to_highest_version": commit_distance_to_highest_version,
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
