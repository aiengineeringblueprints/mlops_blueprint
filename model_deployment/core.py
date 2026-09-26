"""Shared deployment data structures and naming helpers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
import subprocess
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DeploymentRequest:
    """User request resolved from CLI arguments and environment configuration."""

    target: str
    model_name: Optional[str]
    tracking_uri: Optional[str]
    version: Optional[str] = None
    alias: str = "champion"
    dry_run: bool = False
    local_model_path: Optional[str] = None
    local_model_format: str = "auto"


@dataclass(frozen=True)
class ResolvedModel:
    """Model artifact selected for deployment from MLflow or a local file."""

    name: str
    version: str
    source: str
    run_id: Optional[str]
    alias: Optional[str]
    source_type: str = "mlflow"
    model_format: str = "mlflow"


@dataclass(frozen=True)
class DeploymentPlan:
    """Traceable deployment plan produced before infrastructure changes."""

    deployment_target: str
    model_name: str
    registered_version: str
    alias: Optional[str]
    model_uri: str
    image: str
    docker_platform: Optional[str]
    deployment_name: str
    serving_backend: str
    monitoring_backend: str
    service_name: Optional[str] = None
    model_endpoint: Optional[str] = None
    metrics_endpoint: Optional[str] = None
    prometheus_endpoint: Optional[str] = None
    external_prometheus_url: Optional[str] = None
    grafana_endpoint: Optional[str] = None
    monitoring_configuration: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert the deployment plan to a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class DeploymentResult:
    """Structured result returned after backend readiness and monitoring checks."""

    model_name: str
    registered_version: str
    alias: Optional[str]
    deployment_target: str
    image: str
    docker_platform: Optional[str]
    deployment_name: str
    status: str
    monitoring: str
    service_name: Optional[str] = None
    model_endpoint: Optional[str] = None
    metrics_endpoint: Optional[str] = None
    prometheus_endpoint: Optional[str] = None
    external_prometheus_url: Optional[str] = None
    grafana_endpoint: Optional[str] = None
    monitoring_configuration: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert the deployment result to a plain dictionary."""
        return asdict(self)


def exception_message(exc: BaseException) -> str:
    """Return a concise underlying exception message suitable for CLI errors."""
    if isinstance(exc, subprocess.CalledProcessError):
        candidates = (getattr(exc, "stderr", None), getattr(exc, "stdout", None))
    else:
        candidates = (str(exc),)

    keywords = (
        "error",
        "failed",
        "failure",
        "denied",
        "not found",
        "unhealthy",
        "already allocated",
        "cannot",
        "could not",
        "reason",
        "invalid",
        "timeout",
    )

    for candidate in candidates:
        if candidate is None:
            continue
        lines = [line.strip() for line in str(candidate).splitlines() if line.strip()]
        if not lines:
            continue
        for line in reversed(lines):
            if any(keyword in line.lower() for keyword in keywords):
                return line[:800]
        return lines[-1][:800]

    fallback = str(exc).strip()
    return (fallback or exc.__class__.__name__)[:800]


class DeploymentError(RuntimeError):
    """Error with a stable stage code suitable for CLI reporting."""

    def __init__(self, code: str, message: str):
        """Create a deployment error with a stable error code."""
        self.code = code
        super().__init__(f"{code}: {message}")


def short_fingerprint(*parts: object, length: int = 8) -> str:
    """Return a stable short digest for immutable backend configuration identities."""
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def safe_resource_name(name: str, *, max_length: int = 63) -> str:
    """Return a deterministic DNS/AWS-friendly name while preserving uniqueness."""
    raw = str(name).strip().lower().replace("_", "-")
    cleaned = re.sub(r"[^a-z0-9-]+", "-", raw)
    cleaned = re.sub(r"-+", "-", cleaned).strip("-") or "model"
    if cleaned[0].isdigit():
        cleaned = f"m-{cleaned}"
    if len(cleaned) <= max_length:
        return cleaned
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:8]
    head = cleaned[: max_length - len(digest) - 1].rstrip("-")
    return f"{head}-{digest}"



def validate_kubernetes_deployment_name(name: str) -> str:
    """Validate the explicit Kubernetes deployment base name without rewriting it."""
    value = str(name).strip()
    if not value:
        raise DeploymentError(
            "CONFIGURATION_MISSING",
            "KUBE_DEPLOYMENT_NAME is required for Kubernetes/KServe deployment.",
        )
    if len(value) > 45:
        raise DeploymentError(
            "INVALID_KUBERNETES_DEPLOYMENT_NAME",
            "KUBE_DEPLOYMENT_NAME must be at most 45 characters so derived "
            "InferenceService and PodMonitor names remain within Kubernetes limits; "
            f"got {value!r}.",
        )
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", value):
        raise DeploymentError(
            "INVALID_KUBERNETES_DEPLOYMENT_NAME",
            "KUBE_DEPLOYMENT_NAME must contain only lowercase letters, numbers, and "
            "hyphens, and must start and end with a letter or number; "
            f"got {value!r}. Example: champion-model.",
        )
    return value

def stable_deployment_name(model_name: str, target: str) -> str:
    """Return a stable client-facing deployment identity independent of model version."""
    base = safe_resource_name(model_name, max_length=48)
    suffix_by_target = {
        "aws_sagemaker": "endpoint",
        "kubernetes": "inference",
        "docker_compose": "compose",
    }
    suffix = suffix_by_target.get(target, "deployment")
    return safe_resource_name(f"{base}-{suffix}", max_length=63)


def versioned_artifact_name(model_name: str, version: str, suffix: str = "") -> str:
    """Return a deterministic versioned backend resource name."""
    base = safe_resource_name(model_name, max_length=36)
    tail = f"-v{version}{('-' + suffix) if suffix else ''}"
    return safe_resource_name(f"{base}{tail}", max_length=63)
