"""S3 helpers for the training workflow: work with any S3-compatible endpoint."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

EXISTS_CODES = ("BucketAlreadyOwnedByYou", "BucketAlreadyExists")


def s3_client() -> Any:
    """boto3 client for the configured endpoint (S3_HOST_ENDPOINT, empty means AWS)."""
    import boto3

    endpoint = os.getenv("S3_HOST_ENDPOINT") or None
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.getenv("S3_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
    )


def ensure_bucket(client: Any, name: str) -> bool:
    """Create the bucket unless it exists. Returns True if it was created."""
    from botocore.exceptions import ClientError

    kwargs: dict[str, Any] = {"Bucket": name}
    region = getattr(getattr(client, "meta", None), "region_name", None)
    if region and region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}

    try:
        client.create_bucket(**kwargs)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in EXISTS_CODES:
            logger.info("bucket %s already exists", name)
            return False
        raise
    logger.info("bucket %s created", name)
    return True
