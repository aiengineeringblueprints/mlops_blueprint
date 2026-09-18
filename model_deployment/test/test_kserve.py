"""Manual Kubernetes/KServe inference smoke test."""

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

# Use this after port-forwarding the KServe predictor to local port 18000.
ENDPOINT_URL = os.getenv("KSERVE_ENDPOINT_URL", "http://127.0.0.1:18000/predict")

CLASS_NAMES = [
    "Schraube-Zylinder_ganz",
    "Schraube-Zylinder_oben",
    "Schraube-Zylinder_unten",
    "Standard-Schraube",
    "Zylinder",
]


def load_batch(image_path: Path) -> np.ndarray:
    """Load and normalize one image exactly like the existing smoke test."""
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict(batch: np.ndarray) -> list:
    """Send one prediction request to the port-forwarded KServe predictor."""
    body = {"inputs": batch.tolist()}

    request = Request(
        ENDPOINT_URL,
        data=json.dumps(body).encode("utf-8"),
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


def main() -> int:
    """Run one real image prediction and show the predicted category."""
    if not IMAGE_PATH.is_file():
        logger.error("Image not found: %s", IMAGE_PATH)
        return 1

    try:
        logger.info("Testing image: %s", IMAGE_PATH)
        logger.info("KServe endpoint: %s", ENDPOINT_URL)

        probabilities = predict(load_batch(IMAGE_PATH))
        array = np.asarray(probabilities, dtype=np.float32)

        if array.size == 0:
            raise RuntimeError("Prediction response contains no values")

        index = int(np.argmax(array))
        confidence = float(array[index])

        if index >= len(CLASS_NAMES):
            raise RuntimeError(
                f"Predicted class index {index} is outside the configured "
                f"{len(CLASS_NAMES)} categories"
            )

        category = CLASS_NAMES[index]

        logger.info(
            "Prediction: %s (%.1f%% confidence)",
            category,
            confidence * 100,
        )

        return 0

    except Exception:
        logger.exception("KServe inference test FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
