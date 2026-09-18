"""Manual Kubernetes/KServe monitoring smoke test."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)

IMAGE_SIZE = (224, 224)
DEFAULT_IMAGE = REPO_ROOT / "data" / "val" / "Standard-Schraube" / "IMG_8059.JPG"
IMAGE_PATH = Path(os.getenv("IMAGE_PATH", str(DEFAULT_IMAGE)))

# Keep the same port-forward used by the KServe inference test.
BASE_URL = os.getenv("KSERVE_BASE_URL", "http://127.0.0.1:18000").rstrip("/")

CLASS_NAMES = [
    "Schraube-Zylinder_ganz",
    "Schraube-Zylinder_oben",
    "Schraube-Zylinder_unten",
    "Standard-Schraube",
    "Zylinder",
]


def get_text(url: str, timeout: float = 10.0) -> str:
    """Fetch one text response."""
    with urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
        status = int(getattr(response, "status", 200))

    if not 200 <= status < 300:
        raise RuntimeError(f"GET {url} returned HTTP {status}")

    return payload


def get_json(url: str, timeout: float = 10.0) -> dict:
    """Fetch and decode one JSON response."""
    payload = get_text(url, timeout=timeout)
    result = json.loads(payload)

    if not isinstance(result, dict):
        raise RuntimeError(f"GET {url} did not return a JSON object")

    return result


def get_metric_sum(
    metrics_text: str,
    metric_name: str,
    default: float | None = None,
) -> float:
    """Return the sum of all samples for one Prometheus metric."""
    total = 0.0
    found = False

    for line in metrics_text.splitlines():
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        sample_token = stripped.split(None, 1)[0]
        sample_name = sample_token.split("{", 1)[0]

        if sample_name != metric_name:
            continue

        try:
            value = float(stripped.rsplit(None, 1)[1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(
                f"Could not parse metric sample: {stripped}"
            ) from exc

        total += value
        found = True

    if found:
        return total

    if default is not None:
        return default

    raise RuntimeError(
        f"Metric {metric_name!r} was not found in /metrics output"
    )


def load_batch(image_path: Path) -> np.ndarray:
    """Load and normalize one image for inference."""
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict(batch: np.ndarray) -> list:
    """Send one real prediction request."""
    request = Request(
        f"{BASE_URL}/predict",
        data=json.dumps({"inputs": batch.tolist()}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
        status = int(getattr(response, "status", 200))

    if not 200 <= status < 300:
        raise RuntimeError(
            f"Prediction request returned HTTP {status}: {payload}"
        )

    result = json.loads(payload)
    predictions = result.get("predictions")

    if not isinstance(predictions, list) or not predictions:
        raise RuntimeError(f"Invalid prediction response: {result}")

    return predictions[0]


def verify_health() -> None:
    """Verify that the deployed model reports healthy."""
    payload = get_json(f"{BASE_URL}/health")

    if payload.get("status") != "ok":
        raise RuntimeError(
            f"Health endpoint did not report status=ok: {payload}"
        )

    logger.info(
        "Health check PASSED: model=%s version=%s",
        payload.get("model"),
        payload.get("version"),
    )


def verify_metrics_definitions(metrics_text: str) -> None:
    """Verify the metrics expected by the blueprint."""
    required = (
        "# HELP inference_requests_total",
        "# HELP inference_errors_total",
        "# HELP inference_duration_seconds",
        "# HELP model_up",
    )

    missing = [item for item in required if item not in metrics_text]

    if missing:
        raise RuntimeError(
            "/metrics is missing expected metric definitions: "
            + ", ".join(missing)
        )

    logger.info("Expected Prometheus metric definitions found")


def main() -> int:
    """Run the Kubernetes/KServe monitoring smoke test."""
    if not IMAGE_PATH.is_file():
        logger.error("Image not found: %s", IMAGE_PATH)
        return 1

    try:
        logger.info("Testing KServe monitoring endpoint: %s", BASE_URL)

        verify_health()

        metrics_before = get_text(f"{BASE_URL}/metrics")
        verify_metrics_definitions(metrics_before)

        model_up = get_metric_sum(metrics_before, "model_up")
        requests_before = get_metric_sum(
            metrics_before,
            "inference_requests_total",
            default=0.0,
        )
        errors_before = get_metric_sum(
            metrics_before,
            "inference_errors_total",
            default=0.0,
        )

        if model_up < 1:
            raise RuntimeError(
                f"model_up does not report a loaded model: {model_up}"
            )

        logger.info(
            "Metrics before inference: model_up=%.0f requests=%.0f errors=%.0f",
            model_up,
            requests_before,
            errors_before,
        )

        probabilities = predict(load_batch(IMAGE_PATH))
        array = np.asarray(probabilities, dtype=np.float32)

        if array.size == 0:
            raise RuntimeError("Prediction response contains no values")

        index = int(np.argmax(array))
        confidence = float(array[index])
        category = (
            CLASS_NAMES[index]
            if index < len(CLASS_NAMES)
            else f"index {index}"
        )

        logger.info(
            "Prediction: %s (%.1f%% confidence)",
            category,
            confidence * 100,
        )

        metrics_after = get_text(f"{BASE_URL}/metrics")

        requests_after = get_metric_sum(
            metrics_after,
            "inference_requests_total",
            default=0.0,
        )

        latency_count = get_metric_sum(
            metrics_after,
            "inference_duration_seconds_count",
            default=0.0,
        )

        if requests_after < requests_before + 1:
            raise RuntimeError(
                "Inference request counter did not increase: "
                f"before={requests_before}, after={requests_after}"
            )

        if latency_count < 1:
            raise RuntimeError(
                "Inference latency histogram did not record the request"
            )

        logger.info(
            "Metrics after inference: requests %.0f -> %.0f, "
            "latency_count=%.0f",
            requests_before,
            requests_after,
            latency_count,
        )

        logger.info("KServe monitoring smoke test PASSED")
        return 0

    except Exception:
        logger.exception("KServe monitoring smoke test FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
