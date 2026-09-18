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


def setup_project(minio_bucket, lakefs_repo, local_data_folder):
    # Create MinIO Bucket
    run_command(["mc", "mb", f"localminio/{minio_bucket}"])

    # Create LakeFS Repository
    run_command(
        ["lakectl", "repo", "create", f"lakefs://{lakefs_repo}", f"s3://{minio_bucket}"]
    )

    # Copy data
    run_command(
        [
            "mc",
            "cp",
            f"./{local_data_folder}/",
            f"lakefs/{lakefs_repo}/main/data",
            "--recursive",
        ]
    )

    # Initialize local LakeFS repository
    run_command(
        [
            "lakectl",
            "local",
            "init",
            f"lakefs://{lakefs_repo}/main/data",
            local_data_folder,
        ]
    )

    # Create initial commit
    run_command(
        [
            "lakectl",
            "commit",
            f"lakefs://{lakefs_repo}/main",
            "-m",
            "Commit initial data version",
        ]
    )


if __name__ == "__main__":
    # Change parameters here
    MINIO_BUCKET = "anomaly-detection"
    LAKEFS_REPO = "anomaly-detection-data"
    LOCAL_DATA_FOLDER = "data"
    setup_project(MINIO_BUCKET, LAKEFS_REPO, LOCAL_DATA_FOLDER)
