import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(command, shell=False):
    """Helper function to execute terminal commands."""
    logger.info(
        f"Running: {' '.join(command) if isinstance(command, list) else command}"
    )
    try:
        result = subprocess.run(
            command, shell=shell, check=True, capture_output=True, text=True
        )
        if result.stdout.strip():
            logger.info(result.stdout.strip())
        if result.stderr.strip():
            logger.warning(f"Warnings/Errors: {result.stderr.strip()}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(e.stderr)
        sys.exit(e.returncode)


def commit_data(
    lakefs_repo, branch_name, local_folder, commit_message, merge_to_main=False
):
    # Commit the data to LakeFS
    run_command(["lakectl", "local", "commit", local_folder, "-m", commit_message])

    # Optional: Merge into main
    if merge_to_main:
        run_command(
            [
                "lakectl",
                "merge",
                f"lakefs://{lakefs_repo}/{branch_name}",
                f"lakefs://{lakefs_repo}/main",
            ]
        )
        run_command(
            [
                "lakectl",
                "local",
                "checkout",
                local_folder,
                "--ref",
                f"lakefs://{lakefs_repo}/main",
                "--yes",
            ]
        )

    # Commit changes to Git
    run_command(["git", "add", "."])
    run_command(["git", "commit", "-m", commit_message])
    run_command(["git", "push"])


if __name__ == "__main__":
    # Adjust parameters here
    LAKEFS_REPO = "anomaly-detection-data"
    BRANCH_NAME = "augmented"
    LOCAL_FOLDER = "data"
    COMMIT_MESSAGE = "Adding some augmented pictures"
    MERGE_TO_MAIN = False  # True if it should be automatically merged into main
    commit_data(LAKEFS_REPO, BRANCH_NAME, LOCAL_FOLDER, COMMIT_MESSAGE, MERGE_TO_MAIN)
