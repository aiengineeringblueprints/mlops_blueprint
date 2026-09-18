"""Shared MLflow model loading, input normalization, and prediction execution."""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd


def load_serving_model(model_path: str):
    """Load the MLflow PyFunc model used by every serving protocol."""
    return mlflow.pyfunc.load_model(model_path)


def signature_prefers_float32(metadata: Any) -> bool:
    """Return whether a simple MLflow input signature explicitly requests float32."""
    try:
        schema = metadata.get_input_schema()
        inputs = getattr(schema, "inputs", None) or []
        if len(inputs) != 1:
            return False
        dtype = getattr(inputs[0], "dtype", None)
        if dtype is None:
            dtype = getattr(inputs[0], "type", None)
        return "float32" in str(dtype).lower()
    except Exception:
        return False


def normalize_input(payload: Any, model_metadata: Any = None) -> Any:
    """Normalize JSON-compatible model input into a supported prediction format."""
    if payload is None:
        raise ValueError(
            "Model input must not be null. "
            "Provide a JSON array, object, string, number, or boolean."
        )

    if isinstance(payload, list):
        if not payload:
            raise ValueError(
                "Model input must not be an empty list. "
                "Provide one or more input values or record objects."
            )

        if all(isinstance(item, dict) for item in payload):
            try:
                return pd.DataFrame(payload)
            except Exception as exc:
                raise ValueError(f"Could not normalize tabular records: {exc}") from exc

        try:
            array = np.asarray(payload)
        except Exception as exc:
            raise ValueError(
                f"Could not convert the input list to an array: {exc}"
            ) from exc

        if array.dtype == object:
            raise ValueError(
                "Input contains an inconsistent or ragged structure. "
                "Provide a regular numeric array with consistent dimensions "
                "or a list of record dictionaries."
            )

        if np.issubdtype(array.dtype, np.number) and signature_prefers_float32(
            model_metadata
        ):
            array = array.astype(np.float32, copy=False)

        return array

    if isinstance(payload, dict):
        if not payload:
            raise ValueError(
                "Model input object must not be empty. "
                "Provide one or more input fields."
            )
        return payload

    if isinstance(payload, (str, int, float, bool)):
        return payload

    raise ValueError(
        f"Unsupported model input type: {type(payload).__name__}. "
        "Expected a JSON array, JSON object, string, number, or boolean."
    )


def extract_payload(body: Any, *, kserve_v1: bool = False) -> Any:
    """Extract the model payload from generic or KServe V1 request bodies."""
    if isinstance(body, dict):
        has_instances = "instances" in body
        has_inputs = "inputs" in body
        if has_instances and has_inputs:
            raise ValueError(
                "Request must contain only one of 'instances' or 'inputs'."
            )
        if has_instances:
            return body["instances"]
        if has_inputs:
            return body["inputs"]
        if kserve_v1:
            raise ValueError("KServe V1 request must contain 'instances' or 'inputs'.")
    return body


def to_json_compatible(value: Any) -> Any:
    """Convert prediction output to JSON-compatible Python values."""
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_prediction(model, payload: Any) -> dict:
    """Execute the single shared prediction path used by every HTTP protocol."""
    inputs = normalize_input(payload, getattr(model, "metadata", None))
    predictions = model.predict(inputs)
    return {"predictions": to_json_compatible(predictions)}
