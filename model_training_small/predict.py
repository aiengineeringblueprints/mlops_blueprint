"""Predict failure probabilities for raw rows with the trained model and its scaler.

    python predict.py --input raw.csv --output predictions.csv [--model models/trained_nn_model.keras]

The input has the raw columns of the dataset (date, device, metric1 ... metric9; a failure
column is ignored); the output repeats date and device and adds ``failure_probability``,
one row per input row in input order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import keras

SCALER_PATH = Path("models/scaler.joblib")
MODEL_PATH = Path("models/trained_nn_model.keras")


def predict(input_csv: Path, output_csv: Path, model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH) -> pd.DataFrame:
    frame = pd.read_csv(input_csv)
    bundle = joblib.load(scaler_path)
    features = frame[bundle["features"]].to_numpy(dtype="float32")
    scaled = bundle["scaler"].transform(features)
    model = keras.saving.load_model(str(model_path), compile=False)
    probabilities = np.asarray(model.predict(scaled, verbose=0)).reshape(-1)
    result = frame[[c for c in ("date", "device") if c in frame.columns]].copy()
    result["failure_probability"] = probabilities
    result.to_csv(output_csv, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--scaler", type=Path, default=SCALER_PATH)
    args = parser.parse_args()
    result = predict(args.input, args.output, args.model, args.scaler)
    print(f"{len(result)} predictions written to {args.output}")


if __name__ == "__main__":
    main()
