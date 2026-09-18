"""Manual SageMaker endpoint smoke test.

This is an integration helper, not a unit test. It uses the normal AWS credential chain.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import boto3
import numpy as np
from dotenv import load_dotenv
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_deployment import stable_deployment_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_SIZE = (224, 224)
DEFAULT_IMAGE = REPO_ROOT / "data" / "val" / "Standard-Schraube" / "IMG_8059.JPG"
CLASS_NAMES = [
    "Schraube-Zylinder_ganz",
    "Schraube-Zylinder_oben",
    "Schraube-Zylinder_unten",
    "Standard-Schraube",
    "Zylinder",
]


def build_client():
    """Use boto3's standard credential provider chain; never require embedded keys."""
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError("AWS_REGION or AWS_DEFAULT_REGION must be configured")
    return boto3.client("sagemaker-runtime", region_name=region)


def load_batch(image_path):
    """Load and normalize one image for inference."""
    img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict(client, endpoint_name, batch):
    """Invoke the SageMaker endpoint and return one prediction."""
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps({"inputs": batch.tolist()}),
    )
    payload = json.loads(response["Body"].read())
    return payload["predictions"][0]


def main():
    """Run one SageMaker inference smoke test."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    model_name = os.getenv("MODEL_NAME", "champion_model")
    endpoint_name = os.getenv("ENDPOINT_NAME", stable_deployment_name(model_name, "aws_sagemaker"))
    image_path = Path(os.getenv("IMAGE_PATH", DEFAULT_IMAGE))
    if not image_path.is_file():
        logger.error("Image not found: %s", image_path)
        return 1
    try:
        client = build_client()
        logger.info("Scoring %s against endpoint %s", image_path.name, endpoint_name)
        probabilities = predict(client, endpoint_name, load_batch(image_path))
    except Exception:
        logger.exception("SageMaker integration smoke test failed")
        return 1
    index = int(np.argmax(probabilities))
    label = CLASS_NAMES[index] if index < len(CLASS_NAMES) else f"index {index}"
    logger.info("Prediction: %s (%.1f%% confidence)", label, probabilities[index] * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
