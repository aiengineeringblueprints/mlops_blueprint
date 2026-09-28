import os
import subprocess
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL_TRAINING_ROOT = Path(__file__).resolve().parent
load_dotenv(MODEL_TRAINING_ROOT / ".env", override=False)

PROJECT_ROOT = MODEL_TRAINING_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_training.s3_bucket import ensure_bucket, s3_client  # noqa: E402


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


def setup_project(bucket: str, lakefs_repo: str, local_data_folder: str) -> None:
    """Create the bucket, the LakeFS repository and commit the local data as the first version."""
    ensure_bucket(s3_client(), bucket)
    run_command(["lakectl", "repo", "create", f"lakefs://{lakefs_repo}", f"s3://{bucket}/{lakefs_repo}"])
    # Link the local folder to the (empty) branch path, then commit the local files: lakectl
    # uploads them itself, so no S3 client is needed on the host.
    run_command(["lakectl", "local", "init", f"lakefs://{lakefs_repo}/main/data", local_data_folder])
    run_command(["lakectl", "local", "commit", local_data_folder, "-m", "Commit initial data version"])


if __name__ == "__main__":
    # Change parameters here
    BUCKET = os.getenv("S3_BUCKET", "anomaly-detection")
    LAKEFS_REPO = os.getenv("LAKEFS_REPO", "anomaly-detection-data")
    LOCAL_DATA_FOLDER = "data"
    setup_project(BUCKET, LAKEFS_REPO, LOCAL_DATA_FOLDER)
