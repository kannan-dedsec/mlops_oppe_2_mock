import os
import subprocess
import sys

# -----------------------------
# USER CONFIGURATION
# -----------------------------

# Extensions to track
TRACKED_EXTENSIONS = (
    ".py",
    ".dvc",
    ".yaml",
    ".yml",
    ".json",
    ".md",
)

GIT_REMOTE_NAME = "origin"
GIT_BRANCH = "main"

# -----------------------------
# Utilities
# -----------------------------
def run(cmd, capture=False):
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=capture
    )

def get_changed_files():
    """
    Returns a list of changed file paths including:
    - untracked (??)
    - modified (M)
    - added (A)
    - renamed (R -> new path)
    Excludes deleted files.
    """
    result = run(
        ["git", "status", "--porcelain=v1", "-z"],
        capture=True
    )

    entries = result.stdout.split("\0")
    files = []

    for entry in entries:
        if not entry:
            continue

        status = entry[:2]
        path = entry[3:]

        # Skip deleted files
        if "D" in status:
            continue

        # Handle renames: "old -> new"
        if "->" in path:
            path = path.split("->", 1)[1].strip()

        files.append(path)

    return files

def filter_by_extension(files):
    return [
        f for f in files
        if f.endswith(TRACKED_EXTENSIONS)
    ]

# -----------------------------
# Main logic
# -----------------------------
def main():
    # Ensure we're in a git repo
    if not os.path.isdir(".git"):
        print("ERROR: This directory is not a git repository.")
        sys.exit(1)

    changed_files = get_changed_files()
    relevant_files = filter_by_extension(changed_files)

    if not relevant_files:
        print("No relevant changes detected. Nothing to commit.")
        return

    print("\nThe following files will be committed:")
    for f in relevant_files:
        print(f"  - {f}")

    # Stage files
    run(["git", "add", "--"] + relevant_files)

    # Prompt commit message
    commit_msg = input("\nEnter commit message: ").strip()
    if not commit_msg:
        print("ERROR: Commit message cannot be empty.")
        sys.exit(1)

    run(["git", "commit", "-m", commit_msg])
    run(["git", "push", GIT_REMOTE_NAME, GIT_BRANCH])

    print("\nGit add, commit, and push completed successfully.")

# -----------------------------
if __name__ == "__main__":
    main()
