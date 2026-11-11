import subprocess
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_TRAINING_ROOT = Path(__file__).resolve().parent
load_dotenv(MODEL_TRAINING_ROOT / ".env", override=False)


def run_command(command, shell=False):
    """Helper function to execute terminal commands."""
    logger.info(
        f"Running: {' '.join(command) if isinstance(command, list) else command}"
    )
    try:
        result = subprocess.run(
            command, shell=shell, check=True, capture_output=True, text=True, cwd=MODEL_TRAINING_ROOT
        )
        if result.stdout.strip():
            logger.info(result.stdout.strip())
        if result.stderr.strip():
            logger.warning(f"Warnings/Errors: {result.stderr.strip()}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(e.stderr)
        sys.exit(e.returncode)


def prepare_branch(lakefs_repo, new_branch, local_folder):
    # 1. Create new branch
    run_command(
        [
            "lakectl",
            "branch",
            "create",
            f"lakefs://{lakefs_repo}/{new_branch}",
            "--source",
            f"lakefs://{lakefs_repo}/main",
        ]
    )

    # 2. Switch locally to the new branch
    run_command(
        [
            "lakectl",
            "local",
            "checkout",
            local_folder,
            "--ref",
            f"lakefs://{lakefs_repo}/{new_branch}",
            "--yes",
        ]
    )


if __name__ == "__main__":
    # Adjust parameters here
    LAKEFS_REPO = "anomaly-detection-data"
    NEW_BRANCH = "augmented"
    LOCAL_FOLDER = "data"
    prepare_branch(LAKEFS_REPO, NEW_BRANCH, LOCAL_FOLDER)
