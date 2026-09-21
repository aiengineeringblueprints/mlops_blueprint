"""Shared model loading, input normalization, and prediction execution."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ONNXModelAdapter:
    """Run ONNX models through ONNX Runtime while preserving a common predict API."""

    def __init__(self, path: str):
        import onnxruntime as ort

        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inputs = list(self.session.get_inputs())
        self.outputs = list(self.session.get_outputs())

    @staticmethod
    def _dtype(type_name: str):
        mapping = {
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int16)": np.int16,
            "tensor(int8)": np.int8,
            "tensor(uint64)": np.uint64,
            "tensor(uint32)": np.uint32,
            "tensor(uint16)": np.uint16,
            "tensor(uint8)": np.uint8,
            "tensor(bool)": np.bool_,
        }
        return mapping.get(str(type_name).lower())

    def _array_for_input(self, value: Any, input_meta) -> np.ndarray:
        array = np.asarray(value)
        dtype = self._dtype(getattr(input_meta, "type", ""))
        if dtype is not None and array.dtype != dtype:
            array = array.astype(dtype, copy=False)
        return array

    def predict(self, inputs: Any):
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.to_numpy()

        if isinstance(inputs, dict):
            available = {meta.name: meta for meta in self.inputs}
            unknown = sorted(set(inputs) - set(available))
            missing = sorted(set(available) - set(inputs))
            if unknown or missing:
                raise ValueError(
                    "ONNX multi-input requests must use the model input names exactly. "
                    f"missing={missing or 'none'}, unknown={unknown or 'none'}"
                )
            feed = {
                name: self._array_for_input(value, available[name])
                for name, value in inputs.items()
            }
        else:
            if len(self.inputs) != 1:
                names = ", ".join(meta.name for meta in self.inputs)
                raise ValueError(
                    "This ONNX model has multiple inputs. Send a JSON object keyed by "
                    f"the ONNX input names: {names}"
                )
            meta = self.inputs[0]
            feed = {meta.name: self._array_for_input(inputs, meta)}

        results = self.session.run(None, feed)
        if len(results) == 1:
            return results[0]
        return {meta.name: value for meta, value in zip(self.outputs, results)}


class KerasModelAdapter:
    """Run native Keras/H5 models through TensorFlow/Keras."""

    def __init__(self, path: str):
        import tensorflow as tf

        self.model = tf.keras.models.load_model(path, compile=False)

    def predict(self, inputs: Any):
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.to_numpy()
        return self.model.predict(inputs, verbose=0)


class SklearnModelAdapter:
    """Serve joblib/pickle estimators exposing the conventional predict method."""

    def __init__(self, model: Any):
        if not callable(getattr(model, "predict", None)):
            raise TypeError(
                "The loaded joblib/pickle object does not provide a callable predict() method."
            )
        self.model = model

    def predict(self, inputs: Any):
        return self.model.predict(inputs)


class PyTorchModelAdapter:
    """Serve TorchScript/full-module PyTorch artifacts on CPU.

    State-dict-only .pt/.pth checkpoints are intentionally unsupported because they
    cannot be reconstructed without the original Python model architecture.
    """

    def __init__(self, path: str):
        import torch

        self.torch = torch
        try:
            model = torch.jit.load(path, map_location="cpu")
        except Exception as jit_exc:
            try:
                model = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as load_exc:
                raise RuntimeError(
                    "Could not load the PyTorch artifact as TorchScript or a serialized full module."
                ) from load_exc
            if isinstance(model, dict):
                raise RuntimeError(
                    "This .pt/.pth file appears to contain a state_dict/checkpoint mapping. "
                    "State-dict-only checkpoints are unsupported because the model class/architecture "
                    "is not contained in the standalone artifact."
                ) from jit_exc
            if not callable(model):
                raise RuntimeError(
                    "The loaded PyTorch object is not callable and cannot be served directly."
                ) from jit_exc

        if callable(getattr(model, "eval", None)):
            model.eval()
        self.model = model

    def _tensor(self, value: Any):
        if isinstance(value, pd.DataFrame):
            value = value.to_numpy()
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32, copy=False)
        return self.torch.as_tensor(array)

    def predict(self, inputs: Any):
        with self.torch.no_grad():
            if isinstance(inputs, dict):
                output = self.model(**{key: self._tensor(value) for key, value in inputs.items()})
            else:
                output = self.model(self._tensor(inputs))
        if isinstance(output, dict):
            return {key: self._detach(value) for key, value in output.items()}
        if isinstance(output, (list, tuple)):
            return [self._detach(value) for value in output]
        return self._detach(output)

    def _detach(self, value: Any):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return value


class TFLiteModelAdapter:
    """Serve TensorFlow Lite artifacts through the TensorFlow Lite interpreter."""

    def __init__(self, path: str):
        import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()

    @staticmethod
    def _prepare_array(detail: dict, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=detail["dtype"])

    @staticmethod
    def _needs_resize(detail: dict, array: np.ndarray) -> tuple[bool, bool, tuple[int, ...]]:
        expected = tuple(int(x) for x in detail.get("shape", ()))
        signature = tuple(int(x) for x in detail.get("shape_signature", expected))
        dynamic = any(dim < 0 for dim in signature)
        mismatch = bool(expected) and tuple(array.shape) != expected
        return mismatch, dynamic, expected

    def predict(self, inputs: Any):
        input_details = list(self.interpreter.get_input_details())
        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.to_numpy()

        if isinstance(inputs, dict):
            by_name = {detail["name"]: detail for detail in input_details}
            unknown = sorted(set(inputs) - set(by_name))
            missing = sorted(set(by_name) - set(inputs))
            if unknown or missing:
                raise ValueError(
                    "TensorFlow Lite multi-input requests must use the input tensor names exactly. "
                    f"missing={missing or 'none'}, unknown={unknown or 'none'}"
                )
            prepared = {
                name: self._prepare_array(by_name[name], value)
                for name, value in inputs.items()
            }
        else:
            if len(input_details) != 1:
                names = ", ".join(detail["name"] for detail in input_details)
                raise ValueError(
                    "This TensorFlow Lite model has multiple inputs. Send a JSON object keyed by "
                    f"the tensor names: {names}"
                )
            detail = input_details[0]
            prepared = {detail["name"]: self._prepare_array(detail, inputs)}

        details_by_name = {detail["name"]: detail for detail in input_details}
        resized = False
        for name, array in prepared.items():
            detail = details_by_name[name]
            mismatch, dynamic, expected = self._needs_resize(detail, array)
            if mismatch and not dynamic:
                raise ValueError(
                    f"TensorFlow Lite input '{name}' expects shape {expected}, "
                    f"received {tuple(array.shape)}."
                )
            if mismatch and dynamic:
                self.interpreter.resize_tensor_input(detail["index"], array.shape, strict=False)
                resized = True

        if resized:
            self.interpreter.allocate_tensors()
            details_by_name = {
                detail["name"]: detail for detail in self.interpreter.get_input_details()
            }

        for name, array in prepared.items():
            detail = details_by_name[name]
            self.interpreter.set_tensor(detail["index"], array)

        self.interpreter.invoke()
        output_details = self.interpreter.get_output_details()
        outputs = [self.interpreter.get_tensor(detail["index"]) for detail in output_details]
        if len(outputs) == 1:
            return outputs[0]
        return {detail["name"]: value for detail, value in zip(output_details, outputs)}


class XGBoostModelAdapter:
    """Serve XGBoost native model files through xgboost.Booster."""

    def __init__(self, path: str):
        import xgboost as xgb

        self.xgb = xgb
        self.model = xgb.Booster()
        self.model.load_model(path)

    def predict(self, inputs: Any):
        if isinstance(inputs, pd.DataFrame):
            matrix = inputs
        else:
            matrix = np.asarray(inputs)
            if matrix.ndim == 1:
                matrix = np.expand_dims(matrix, axis=0)
        return self.model.predict(self.xgb.DMatrix(matrix))


def _load_pickle_model(path: str, *, prefer_joblib: bool) -> Any:
    """Load a trusted local serialized estimator with joblib/pickle compatibility."""
    if prefer_joblib:
        import joblib

        return joblib.load(path)

    try:
        import joblib

        return joblib.load(path)
    except Exception:
        with open(path, "rb") as stream:
            return pickle.load(stream)


def load_serving_model(
    model_path: str,
    *,
    model_source: str = "mlflow",
    model_format: str = "mlflow",
    model_file: str | None = None,
):
    """Load an MLflow model or one explicitly supported standalone local model."""
    source = (model_source or "mlflow").strip().lower()
    fmt = (model_format or "mlflow").strip().lower()

    if source == "mlflow":
        import mlflow.pyfunc

        return mlflow.pyfunc.load_model(model_path)

    if source != "local":
        raise ValueError(f"Unsupported MODEL_SOURCE: {model_source!r}")
    if not model_file:
        raise ValueError("MODEL_FILE is required for standalone local-model serving.")

    path = Path(model_path) / model_file
    if not path.is_file():
        raise FileNotFoundError(f"Standalone model file not found in serving image: {path}")

    if fmt == "keras":
        return KerasModelAdapter(str(path))
    if fmt == "onnx":
        return ONNXModelAdapter(str(path))
    if fmt == "joblib":
        return SklearnModelAdapter(_load_pickle_model(str(path), prefer_joblib=True))
    if fmt == "pickle":
        return SklearnModelAdapter(_load_pickle_model(str(path), prefer_joblib=False))
    if fmt == "pytorch":
        return PyTorchModelAdapter(str(path))
    if fmt == "tflite":
        return TFLiteModelAdapter(str(path))
    if fmt == "xgboost":
        return XGBoostModelAdapter(str(path))

    raise ValueError(f"Unsupported local MODEL_FORMAT: {model_format!r}")


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
