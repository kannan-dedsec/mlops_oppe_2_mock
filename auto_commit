import os
import subprocess
import sys

# -----------------------------
# USER CONFIGURATION
# -----------------------------

# Extensions to track
TRACKED_EXTENSIONS = [
    ".py",
    ".dvc",
    ".yaml",
    ".yml",
    ".json",
    ".md",
]

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
    result = run(["git", "status", "--porcelain"], capture=True)
    files = []

    for line in result.stdout.splitlines():
        # Format: XY filename
        files.append(line[3:])

    return files

def filter_by_extension(files):
    return [
        f for f in files
        if any(f.endswith(ext) for ext in TRACKED_EXTENSIONS)
    ]

# -----------------------------
# Main logic
# -----------------------------
def main():
    # Ensure we're in a git repo
    if not os.path.exists(".git"):
        print("ERROR: This directory is not a git repository.")
        sys.exit(1)

    changed_files = get_changed_files()
    relevant_files = filter_by_extension(changed_files)

    if not relevant_files:
        print("No relevant changes detected. Nothing to commit.")
        return

    print("The following files will be committed:")
    for f in relevant_files:
        print(f"  - {f}")

    # Stage files
    for f in relevant_files:
        run(["git", "add", f])

    # Prompt commit message
    commit_msg = input("\nEnter commit message: ").strip()
    if not commit_msg:
        print("ERROR: Commit message cannot be empty.")
        sys.exit(1)

    run(["git", "commit", "-m", commit_msg])

    # Push using existing auth
    run(["git", "push", GIT_REMOTE_NAME, GIT_BRANCH])

    print("\nGit add, commit, and push completed successfully.")

if __name__ == "__main__":
    main()
