"""Generate inference traffic for monitoring/dashboard visualization.

Examples:

Docker Compose:
    python3 mock_request.py \
        --backend http \
        --url http://127.0.0.1:8000/predict \
        --requests 120 \
        --interval 0.5

Kubernetes/KServe after port-forward:
    python3 mock_request.py \
        --backend http \
        --url http://127.0.0.1:18000/predict \
        --requests 120 \
        --interval 0.5

AWS SageMaker:
    python3 mock_request.py \
        --backend sagemaker \
        --endpoint-name <sagemaker-endpoint-name> \
        --region eu-central-1 \
        --requests 120 \
        --interval 0.5

This script is intended to create ordinary inference traffic so monitoring
systems have enough samples to display request-rate and latency graphs.
It is not a performance/load benchmark.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Schraube-Zylinder_ganz",
    "Schraube-Zylinder_oben",
    "Schraube-Zylinder_unten",
    "Standard-Schraube",
    "Zylinder",
]


def find_project_root(start: Path) -> Path:
    """Locate the repository root from the split training/deployment structure."""
    for candidate in (start, *start.parents):
        if (candidate / "model_deployment").is_dir() and (candidate / "model_training").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root containing model_training/ and model_deployment/")


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(HERE)
TRAINING_ROOT = PROJECT_ROOT / "model_training"

DEFAULT_IMAGE = (
    TRAINING_ROOT
    / "data"
    / "val"
    / "Standard-Schraube"
    / "IMG_8059.JPG"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate repeated inference requests for Docker Compose, "
            "Kubernetes/KServe, or AWS SageMaker."
        )
    )

    parser.add_argument(
        "--backend",
        choices=("http", "sagemaker"),
        required=True,
        help=(
            "http for Docker Compose/Kubernetes/KServe; "
            "sagemaker for an AWS SageMaker real-time endpoint."
        ),
    )

    parser.add_argument(
        "--url",
        help=(
            "HTTP prediction URL for Docker Compose/KServe, "
            "for example http://127.0.0.1:8000/predict."
        ),
    )

    parser.add_argument(
        "--endpoint-name",
        help="AWS SageMaker endpoint name.",
    )

    parser.add_argument(
        "--region",
        default="eu-central-1",
        help="AWS region for SageMaker. Default: eu-central-1.",
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Input image. Default: {DEFAULT_IMAGE}",
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=120,
        help="Number of inference requests to send. Default: 120.",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help=(
            "Seconds to wait between requests. Default: 0.5. "
            "Use a non-zero interval to create visible time-series graphs."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds. Default: 30.",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log progress every N requests. Default: 10.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate backend-specific arguments."""
    if args.requests < 1:
        raise RuntimeError("--requests must be at least 1")

    if args.interval < 0:
        raise RuntimeError("--interval cannot be negative")

    if args.log_every < 1:
        raise RuntimeError("--log-every must be at least 1")

    if not args.image.is_file():
        raise RuntimeError(f"Image not found: {args.image}")

    if args.backend == "http" and not args.url:
        raise RuntimeError("--url is required when --backend=http")

    if args.backend == "sagemaker" and not args.endpoint_name:
        raise RuntimeError(
            "--endpoint-name is required when --backend=sagemaker"
        )


def load_batch(image_path: Path) -> np.ndarray:
    """Load and normalize one image exactly like the smoke tests."""
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def extract_prediction(payload: dict) -> tuple[str, float]:
    """Extract the predicted category and confidence from a response."""
    predictions = payload.get("predictions")

    if not isinstance(predictions, list) or not predictions:
        raise RuntimeError(f"Invalid prediction response: {payload}")

    probabilities = np.asarray(predictions[0], dtype=np.float32)

    if probabilities.size == 0:
        raise RuntimeError("Prediction response contains no values")

    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])

    if index >= len(CLASS_NAMES):
        label = f"index {index}"
    else:
        label = CLASS_NAMES[index]

    return label, confidence


def invoke_http(
    url: str,
    body: bytes,
    timeout: float,
) -> tuple[str, float]:
    """Invoke Docker Compose or KServe over HTTP."""
    request = Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            status = int(getattr(response, "status", 200))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} from {url}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc

    if not 200 <= status < 300:
        raise RuntimeError(f"HTTP {status} from {url}: {payload}")

    result = json.loads(payload)

    if not isinstance(result, dict):
        raise RuntimeError("Prediction response was not a JSON object")

    return extract_prediction(result)


def create_sagemaker_client(region: str):
    """Create the SageMaker Runtime client only when AWS is selected."""
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for --backend=sagemaker"
        ) from exc

    return boto3.client("sagemaker-runtime", region_name=region)


def invoke_sagemaker(
    client,
    endpoint_name: str,
    body: bytes,
) -> tuple[str, float]:
    """Invoke one AWS SageMaker real-time endpoint."""
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=body,
    )

    payload = response["Body"].read().decode("utf-8")
    result = json.loads(payload)

    if not isinstance(result, dict):
        raise RuntimeError("SageMaker response was not a JSON object")

    return extract_prediction(result)


def main() -> int:
    """Generate repeated inference traffic."""
    try:
        args = parse_args()
        validate_args(args)

        batch = load_batch(args.image)
        body = json.dumps({"inputs": batch.tolist()}).encode("utf-8")

        sagemaker_client = None

        if args.backend == "sagemaker":
            sagemaker_client = create_sagemaker_client(args.region)
            target = (
                f"SageMaker endpoint={args.endpoint_name} "
                f"region={args.region}"
            )
        else:
            target = args.url

        logger.info("Traffic target: %s", target)
        logger.info("Image: %s", args.image)
        logger.info(
            "Sending %d requests with %.3fs interval",
            args.requests,
            args.interval,
        )

        successes = 0
        failures = 0
        latencies_ms: list[float] = []
        last_label = None
        last_confidence = None

        started = time.monotonic()

        for request_number in range(1, args.requests + 1):
            request_started = time.monotonic()

            try:
                if args.backend == "http":
                    label, confidence = invoke_http(
                        args.url,
                        body,
                        args.timeout,
                    )
                else:
                    label, confidence = invoke_sagemaker(
                        sagemaker_client,
                        args.endpoint_name,
                        body,
                    )

                latency_ms = (
                    time.monotonic() - request_started
                ) * 1000.0

                latencies_ms.append(latency_ms)
                successes += 1
                last_label = label
                last_confidence = confidence

            except Exception as exc:
                failures += 1
                logger.error(
                    "Request %d/%d failed: %s",
                    request_number,
                    args.requests,
                    exc,
                )

            if (
                request_number == 1
                or request_number % args.log_every == 0
                or request_number == args.requests
            ):
                if last_label is not None:
                    logger.info(
                        "Progress %d/%d: successes=%d failures=%d "
                        "latest=%s (%.1f%%)",
                        request_number,
                        args.requests,
                        successes,
                        failures,
                        last_label,
                        last_confidence * 100,
                    )
                else:
                    logger.info(
                        "Progress %d/%d: successes=%d failures=%d",
                        request_number,
                        args.requests,
                        successes,
                        failures,
                    )

            if (
                args.interval > 0
                and request_number < args.requests
            ):
                time.sleep(args.interval)

        elapsed = time.monotonic() - started

        if latencies_ms:
            average_ms = float(np.mean(latencies_ms))
            p95_ms = float(np.percentile(latencies_ms, 95))
        else:
            average_ms = 0.0
            p95_ms = 0.0

        logger.info(
            "Traffic generation completed: "
            "requests=%d successes=%d failures=%d "
            "elapsed=%.1fs avg_client_latency=%.1fms "
            "p95_client_latency=%.1fms",
            args.requests,
            successes,
            failures,
            elapsed,
            average_ms,
            p95_ms,
        )

        if successes == 0:
            return 1

        return 0

    except Exception:
        logger.exception("Inference traffic generation FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
