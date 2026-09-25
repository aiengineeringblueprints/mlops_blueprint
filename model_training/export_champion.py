"""Export the registered champion model as a Keras file plus class list and model card."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

MODEL_TRAINING_ROOT = Path(__file__).resolve().parent
load_dotenv(MODEL_TRAINING_ROOT / ".env", override=False)

logger = logging.getLogger(__name__)

MODEL_CARD_TEMPLATE = """# Model card: {model_name} (alias {alias}, version {version})

- Format: Keras 3 model file (`model.keras`). Load with `keras.saving.load_model("model.keras", compile=False)`.
- Input: a batch of RGB images as float32 arrays of shape (n, 224, 224, 3), pixel values scaled to [0, 1].
- Output: one softmax vector per instance; index i corresponds to `classes.json[i]`.
- Source: MLflow run {run_id}, selected as champion by {champion_metric} = {champion_value:.4f}.
- Data version: {dataset_source}
"""


def class_names(train_dir: Path) -> list[str]:
    """Class names in the order the training script's OneHotEncoder assigns output indices (sorted)."""
    return sorted(p.name for p in train_dir.iterdir() if p.is_dir())


def export(output_dir: Path, train_dir: Path, model_name: str, alias: str, tracking_uri: str) -> int:
    """Download the alias model, save model.keras, classes.json and MODEL_CARD.md."""
    import mlflow
    import mlflow.tensorflow

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(model_name, alias)
    run = client.get_run(version.run_id)
    model = mlflow.tensorflow.load_model(f"models:/{model_name}@{alias}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.keras")
    (output_dir / "classes.json").write_text(json.dumps(class_names(train_dir)) + "\n")
    metric = run.data.tags.get("champion_metric", "validation_accuracy")
    inputs = run.inputs.dataset_inputs if run.inputs else []
    dataset_source = inputs[0].dataset.source if inputs else "not recorded"
    (output_dir / "MODEL_CARD.md").write_text(
        MODEL_CARD_TEMPLATE.format(
            model_name=model_name, alias=alias, version=version.version, run_id=version.run_id,
            champion_metric=metric, champion_value=float(run.data.metrics.get(metric, float("nan"))),
            dataset_source=dataset_source,
        )
    )
    logger.info("exported %s@%s (version %s) to %s", model_name, alias, version.version, output_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the champion model to a directory.")
    parser.add_argument("--output-dir", type=Path, default=MODEL_TRAINING_ROOT / "output")
    parser.add_argument("--train-dir", type=Path, default=MODEL_TRAINING_ROOT / "data" / "train")
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "screw_classifier"))
    parser.add_argument("--alias", default=os.getenv("MODEL_ALIAS", "champion"))
    parser.add_argument("--tracking-uri", default=os.getenv("TRACKING_URI", "sqlite:///mlflow.db"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    try:
        return export(args.output_dir, args.train_dir, args.model_name, args.alias, args.tracking_uri)
    except Exception as exc:  # the CLI reports any failure as exit code 1
        logger.error("export failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
