"""Champion provenance validation and registered-model resolution."""
from __future__ import annotations

import re
from typing import Optional

from model_deployment import DeploymentError, ResolvedModel


class ChampionProvenanceError(RuntimeError):
    """Raised when a champion run lacks trustworthy validation-selection provenance."""


def is_test_metric(metric_name: str) -> bool:
    """Return True when a metric name clearly represents final/test-set evaluation."""
    normalized = metric_name.strip().lower()
    if normalized.startswith("test"):
        return True
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    return "test" in tokens


def validate_champion_provenance(run) -> str:
    """Require champion=true plus a logged, non-test champion metric."""
    data = getattr(run, "data", None)
    tags = getattr(data, "tags", {}) or {}
    metrics = getattr(data, "metrics", {}) or {}

    champion_tag = str(tags.get("champion", "")).strip().lower()
    if champion_tag != "true":
        raise ChampionProvenanceError(
            "Selected run is not explicitly tagged champion=true."
        )

    metric_name = str(tags.get("champion_metric", "")).strip()
    if not metric_name:
        raise ChampionProvenanceError(
            "Champion run does not declare the validation metric used for selection."
        )
    if is_test_metric(metric_name):
        raise ChampionProvenanceError(
            f"Champion test metric '{metric_name}' represents test-set evaluation; "
            "champion selection must use an independent validation metric."
        )
    if metric_name not in metrics:
        raise ChampionProvenanceError(
            f"Champion metric '{metric_name}' is not logged on the selected run."
        )
    return metric_name


def resolve_model_version(
    model_name: str,
    tracking_uri: str,
    version: Optional[str] = None,
    *,
    alias: str = "champion",
    client=None,
) -> ResolvedModel:
    """Resolve an explicit registered version, otherwise the controlled champion alias."""
    if client is None:
        try:
            import mlflow
        except ImportError as exc:
            raise DeploymentError("MODEL_RESOLUTION_FAILED", "MLflow is not installed") from exc
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.MlflowClient()

    try:
        if version not in (None, ""):
            model_version = client.get_model_version(model_name, str(version))
            alias_used = None
        else:
            try:
                model_version = client.get_model_version_by_alias(model_name, alias)
            except Exception as exc:
                raise DeploymentError(
                    "MODEL_RESOLUTION_FAILED",
                    f"Registered model '{model_name}' has no usable alias '{alias}'. "
                    "Register/assign a champion first, or pass --model-version explicitly.",
                ) from exc
            alias_used = alias
            run_id = getattr(model_version, "run_id", None)
            get_run = getattr(client, "get_run", None)
            if not run_id or not callable(get_run):
                raise DeploymentError(
                    "INVALID_CHAMPION_PROVENANCE",
                    f"Alias '{alias}' points to version {model_version.version}, but its source run cannot "
                    "be inspected for champion provenance. Pass an explicit --model-version only "
                    "when intentionally deploying that exact historical version.",
                )
            try:
                validate_champion_provenance(get_run(run_id))
            except ChampionProvenanceError as exc:
                raise DeploymentError(
                    "INVALID_CHAMPION_PROVENANCE",
                    f"Alias '{alias}' points to version {model_version.version}, but its champion provenance "
                    f"is invalid: {exc} Pass an explicit --model-version only when intentionally "
                    "deploying that exact historical version.",
                ) from exc
    except DeploymentError:
        raise
    except Exception as exc:
        requested = f"version {version}" if version else f"alias '{alias}'"
        raise DeploymentError(
            "MODEL_RESOLUTION_FAILED",
            f"Could not resolve {requested} for registered model '{model_name}': {exc}",
        ) from exc

    return ResolvedModel(
        name=model_name,
        version=str(model_version.version),
        source=str(model_version.source),
        run_id=getattr(model_version, "run_id", None),
        alias=alias_used,
    )
