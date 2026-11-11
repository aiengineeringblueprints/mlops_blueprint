"""Unit tests for idempotent bucket creation (no network)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_training.s3_bucket import ensure_bucket  # noqa: E402


class StubClient:
    def __init__(self, error_code: str | None = None) -> None:
        self.error_code = error_code
        self.created: list[str] = []

    def create_bucket(self, Bucket: str) -> dict[str, str]:  # noqa: N803 - boto3 signature
        if self.error_code:
            raise ClientError({"Error": {"Code": self.error_code, "Message": "x"}}, "CreateBucket")
        self.created.append(Bucket)
        return {}


class RegionStubClient:
    """Stub with a boto3-like `.meta.region_name`, recording the create_bucket call."""

    class _Meta:
        def __init__(self, region_name: str) -> None:
            self.region_name = region_name

    def __init__(self, region_name: str) -> None:
        self.meta = self._Meta(region_name)
        self.call_kwargs: dict[str, Any] | None = None

    def create_bucket(self, **kwargs: Any) -> dict[str, str]:
        self.call_kwargs = kwargs
        return {}


def test_creates_a_missing_bucket() -> None:
    client = StubClient()
    assert ensure_bucket(client, "data") is True
    assert client.created == ["data"]


@pytest.mark.parametrize("code", ["BucketAlreadyOwnedByYou", "BucketAlreadyExists"])
def test_existing_bucket_is_not_an_error(code: str) -> None:
    assert ensure_bucket(StubClient(code), "data") is False


def test_other_errors_propagate() -> None:
    with pytest.raises(ClientError):
        ensure_bucket(StubClient("AccessDenied"), "data")


def test_region_constraint_added_outside_us_east_1() -> None:
    client = RegionStubClient("eu-central-1")
    assert ensure_bucket(client, "data") is True
    assert client.call_kwargs == {
        "Bucket": "data",
        "CreateBucketConfiguration": {"LocationConstraint": "eu-central-1"},
    }
