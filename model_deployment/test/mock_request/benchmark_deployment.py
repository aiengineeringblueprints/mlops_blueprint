"""Benchmark inference latency and throughput across deployment backends."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import boto3
import httpx
import numpy as np
from PIL import Image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
def find_repo_root(start: Path) -> Path:
    """Find the repository root containing data/ and deploy.py."""
    for candidate in (start, *start.parents):
        if (candidate / "deploy.py").is_file():
            return candidate

    return start

HERE = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(HERE)
DEFAULT_IMAGE = (
    REPO_ROOT
    / "data"
    / "val"
    / "Standard-Schraube"
    / "IMG_8059.JPG"
)
DEFAULT_IMAGE_SIZE = (224, 224)


def load_batch(image_path: str) -> np.ndarray:
    """Load and normalize one image into a single inference batch."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(DEFAULT_IMAGE_SIZE)

    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def percentile(values: list[float], percentile_value: float) -> float:
    """Calculate a percentile from latency measurements."""
    if not values:
        return 0.0

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def benchmark_http_request(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
) -> tuple[bool, float]:
    """Send one HTTP inference request and return success and latency."""
    started = time.perf_counter()

    try:
        response = client.post(url, json=payload)
        response.raise_for_status()
        response.json()

        latency = time.perf_counter() - started
        return True, latency

    except Exception as exc:
        latency = time.perf_counter() - started
        logger.debug("HTTP request failed: %s", exc)
        return False, latency


def benchmark_sagemaker_request(
    client: Any,
    endpoint_name: str,
    batch: np.ndarray,
) -> tuple[bool, float]:
    """Invoke one SageMaker endpoint request and return success and latency."""
    started = time.perf_counter()

    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(batch.tolist()),
        )

        response["Body"].read()

        latency = time.perf_counter() - started
        return True, latency

    except Exception as exc:
        latency = time.perf_counter() - started
        logger.debug("SageMaker request failed: %s", exc)
        return False, latency


def run_warmup_http(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    requests: int,
) -> None:
    """Send HTTP warm-up requests before collecting benchmark measurements."""
    logger.info("Running %d warm-up requests", requests)

    for _ in range(requests):
        success, _ = benchmark_http_request(client, url, payload)

        if not success:
            raise RuntimeError("HTTP warm-up request failed.")


def run_warmup_sagemaker(
    client: Any,
    endpoint_name: str,
    batch: np.ndarray,
    requests: int,
) -> None:
    """Send SageMaker warm-up requests before benchmark measurements."""
    logger.info("Running %d warm-up requests", requests)

    for _ in range(requests):
        success, _ = benchmark_sagemaker_request(
            client,
            endpoint_name,
            batch,
        )

        if not success:
            raise RuntimeError("SageMaker warm-up request failed.")


def execute_http_benchmark(
    url: str,
    batch: np.ndarray,
    requests: int,
    concurrency: int,
    warmup: int,
    timeout: float,
) -> tuple[list[float], int, int, float]:
    """Benchmark an HTTP-based Docker Compose or KServe endpoint."""
    payload = {"inputs": batch.tolist()}

    limits = httpx.Limits(
        max_connections=max(concurrency, 10),
        max_keepalive_connections=max(concurrency, 10),
    )

    with httpx.Client(timeout=timeout, limits=limits) as client:
        run_warmup_http(client, url, payload, warmup)

        logger.info(
            "Starting HTTP benchmark: requests=%d concurrency=%d",
            requests,
            concurrency,
        )

        latencies: list[float] = []
        successful = 0
        failed = 0

        benchmark_started = time.perf_counter()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    benchmark_http_request,
                    client,
                    url,
                    payload,
                )
                for _ in range(requests)
            ]

            for future in as_completed(futures):
                success, latency = future.result()

                if success:
                    successful += 1
                    latencies.append(latency)
                else:
                    failed += 1

        duration = time.perf_counter() - benchmark_started

    return latencies, successful, failed, duration


def execute_sagemaker_benchmark(
    endpoint_name: str,
    region: str,
    batch: np.ndarray,
    requests: int,
    concurrency: int,
    warmup: int,
) -> tuple[list[float], int, int, float]:
    """Benchmark an AWS SageMaker real-time inference endpoint."""
    client = boto3.client("sagemaker-runtime", region_name=region)

    run_warmup_sagemaker(
        client,
        endpoint_name,
        batch,
        warmup,
    )

    logger.info(
        "Starting SageMaker benchmark: requests=%d concurrency=%d",
        requests,
        concurrency,
    )

    latencies: list[float] = []
    successful = 0
    failed = 0

    benchmark_started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                benchmark_sagemaker_request,
                client,
                endpoint_name,
                batch,
            )
            for _ in range(requests)
        ]

        for future in as_completed(futures):
            success, latency = future.result()

            if success:
                successful += 1
                latencies.append(latency)
            else:
                failed += 1

    duration = time.perf_counter() - benchmark_started

    return latencies, successful, failed, duration


def report_results(
    requests: int,
    successful: int,
    failed: int,
    duration: float,
    latencies: list[float],
) -> None:
    """Log the final benchmark statistics."""
    success_rate = (successful / requests * 100.0) if requests else 0.0
    throughput = successful / duration if duration > 0 else 0.0

    logger.info("Benchmark completed")
    logger.info("Total requests: %d", requests)
    logger.info("Successful requests: %d", successful)
    logger.info("Failed requests: %d", failed)
    logger.info("Success rate: %.2f%%", success_rate)
    logger.info("Total duration: %.3f s", duration)
    logger.info("Throughput: %.3f requests/s", throughput)

    if not latencies:
        logger.warning("No successful requests available for latency statistics.")
        return

    latency_ms = [value * 1000.0 for value in latencies]

    logger.info("Mean latency: %.3f ms", sum(latency_ms) / len(latency_ms))
    logger.info("Median latency: %.3f ms", percentile(latency_ms, 50))
    logger.info("P95 latency: %.3f ms", percentile(latency_ms, 95))
    logger.info("Minimum latency: %.3f ms", min(latency_ms))
    logger.info("Maximum latency: %.3f ms", max(latency_ms))


def parse_args() -> argparse.Namespace:
    """Parse benchmark command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark model inference latency and throughput."
    )

    parser.add_argument(
        "--backend",
        required=True,
        choices=["http", "sagemaker"],
        help="Deployment backend type.",
    )

    parser.add_argument(
        "--url",
        help="HTTP prediction endpoint for Docker Compose or KServe.",
    )

    parser.add_argument(
        "--endpoint-name",
        help="AWS SageMaker endpoint name.",
    )

    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region for SageMaker.",
    )

    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Input image used for inference.",
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of measured requests.",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent requests.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warm-up requests.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP request timeout in seconds.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the benchmark for the selected deployment backend."""
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    if args.requests <= 0:
        raise ValueError("--requests must be greater than zero.")

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be greater than zero.")

    if args.warmup < 0:
        raise ValueError("--warmup must not be negative.")

    batch = load_batch(str(image_path))

    if args.backend == "http":
        if not args.url:
            raise ValueError("--url is required for the http backend.")

        logger.info("Benchmarking HTTP endpoint: %s", args.url)

        results = execute_http_benchmark(
            url=args.url,
            batch=batch,
            requests=args.requests,
            concurrency=args.concurrency,
            warmup=args.warmup,
            timeout=args.timeout,
        )

    else:
        if not args.endpoint_name:
            raise ValueError(
                "--endpoint-name is required for the sagemaker backend."
            )

        logger.info(
            "Benchmarking SageMaker endpoint: %s",
            args.endpoint_name,
        )

        results = execute_sagemaker_benchmark(
            endpoint_name=args.endpoint_name,
            region=args.region,
            batch=batch,
            requests=args.requests,
            concurrency=args.concurrency,
            warmup=args.warmup,
        )

    latencies, successful, failed, duration = results

    report_results(
        requests=args.requests,
        successful=successful,
        failed=failed,
        duration=duration,
        latencies=latencies,
    )


if __name__ == "__main__":
    main()