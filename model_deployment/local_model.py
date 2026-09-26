"""Resolve standalone local model files without requiring MLflow."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Optional

from model_deployment import DeploymentError, ResolvedModel


SUPPORTED_LOCAL_MODEL_FORMATS = (
    "keras",
    "onnx",
    "joblib",
    "pickle",
    "pytorch",
    "tflite",
    "xgboost",
)

_EXTENSION_FORMATS = {
    ".keras": "keras",
    ".h5": "keras",
    ".hdf5": "keras",
    ".onnx": "onnx",
    ".joblib": "joblib",
    ".pkl": "pickle",
    ".pickle": "pickle",
    ".pt": "pytorch",
    ".pth": "pytorch",
    ".tflite": "tflite",
    ".ubj": "xgboost",
    ".xgb": "xgboost",
    ".xgboost": "xgboost",
    ".json": "xgboost",
    ".model": "xgboost",
}

_FORMAT_ALIASES = {
    "auto": "auto",
    "keras": "keras",
    "tensorflow": "keras",
    "tensorflow_keras": "keras",
    "h5": "keras",
    "onnx": "onnx",
    "onnxruntime": "onnx",
    "joblib": "joblib",
    "sklearn": "joblib",
    "scikit-learn": "joblib",
    "pickle": "pickle",
    "pkl": "pickle",
    "pytorch": "pytorch",
    "torch": "pytorch",
    "torchscript": "pytorch",
    "tflite": "tflite",
    "tensorflow_lite": "tflite",
    "xgboost": "xgboost",
    "xgb": "xgboost",
}


def normalize_local_model_format(value: Optional[str]) -> str:
    """Normalize the configured local-model format or raise an actionable error."""
    raw = (value or "auto").strip().lower()
    normalized = _FORMAT_ALIASES.get(raw)
    if normalized is None:
        raise DeploymentError(
            "LOCAL_MODEL_FORMAT_UNSUPPORTED",
            "Unsupported LOCAL_MODEL_FORMAT "
            f"{value!r}. Supported values: auto, "
            + ", ".join(SUPPORTED_LOCAL_MODEL_FORMATS),
        )
    return normalized


def detect_local_model_format(path: Path, configured_format: Optional[str] = None) -> str:
    """Resolve a supported model format from configuration or the file extension."""
    requested = normalize_local_model_format(configured_format)
    detected = _EXTENSION_FORMATS.get(path.suffix.lower())

    if requested == "auto":
        if detected is None:
            raise DeploymentError(
                "LOCAL_MODEL_FORMAT_UNSUPPORTED",
                f"Could not detect a supported local model format from '{path.name}'. "
                "Set LOCAL_MODEL_FORMAT explicitly when the file extension is non-standard. "
                "Supported formats: "
                + ", ".join(SUPPORTED_LOCAL_MODEL_FORMATS),
            )
        return detected

    if detected is not None and detected != requested:
        raise DeploymentError(
            "LOCAL_MODEL_FORMAT_MISMATCH",
            f"LOCAL_MODEL_FORMAT={requested!r} conflicts with the recognized extension "
            f"'{path.suffix}' ({detected}).",
        )
    return requested


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a local model file without loading it into memory."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def configured_deployment_model_name(default: str | None = None) -> str | None:
    """Return the deployment model identity implied by the current environment."""
    import os

    local_path = (os.getenv("LOCAL_MODEL_PATH") or "").strip()
    if local_path:
        configured = (os.getenv("LOCAL_MODEL_NAME") or "").strip()
        return configured or Path(local_path).expanduser().stem
    configured = (os.getenv("MODEL_NAME") or "").strip()
    return configured or default

def resolve_local_model(
    model_path: str,
    *,
    model_name: Optional[str] = None,
    model_format: Optional[str] = None,
) -> ResolvedModel:
    """Resolve a standalone model file into the common deployment model identity."""
    path = Path(model_path).expanduser().resolve()
    if not path.exists():
        raise DeploymentError(
            "LOCAL_MODEL_NOT_FOUND",
            f"LOCAL_MODEL_PATH does not exist: {path}",
        )
    if not path.is_file():
        raise DeploymentError(
            "LOCAL_MODEL_INVALID",
            f"LOCAL_MODEL_PATH must reference one supported model file, not a directory: {path}",
        )

    resolved_format = detect_local_model_format(path, model_format)
    resolved_name = (model_name or path.stem).strip()
    if not resolved_name:
        raise DeploymentError(
            "LOCAL_MODEL_INVALID",
            f"Could not derive a model name from LOCAL_MODEL_PATH: {path}",
        )

    # A content fingerprint gives local files a deterministic deployment version without
    # introducing a registry dependency. A changed file receives a new version identity.
    version = file_sha256(path)[:12]
    return ResolvedModel(
        name=resolved_name,
        version=version,
        source=str(path),
        run_id=None,
        alias=None,
        source_type="local",
        model_format=resolved_format,
    )
