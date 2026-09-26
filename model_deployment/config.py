"""Centralized configuration for the three concrete deployment backends."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Dict, Optional

from dotenv import load_dotenv

from model_deployment import (
    DeploymentError,
    DeploymentRequest,
    exception_message,
    validate_kubernetes_deployment_name,
)


VALID_TARGETS = ("aws_sagemaker", "kubernetes", "docker_compose")


@dataclass(frozen=True)
class DeploymentSettings:
    """Validated settings shared by the common controller and backend adapters."""

    target: str
    serving_backend: str
    monitoring_backend: str
    docker_registry: str
    docker_platform: Optional[str]
    readiness_timeout_seconds: int
    poll_interval_seconds: int
    prometheus_discovery_timeout_seconds: int
    prometheus_discovery_poll_interval_seconds: int

    # AWS SageMaker
    aws_region: Optional[str] = None
    sagemaker_role_arn: Optional[str] = None
    sagemaker_instance_type: Optional[str] = None
    sagemaker_instance_count: int = 1
    sagemaker_variant_name: str = "AllTraffic"
    sagemaker_latency_threshold_us: int = 1_000_000
    sagemaker_alarm_period: int = 60
    sagemaker_evaluation_periods: int = 3

    # Kubernetes / KServe
    kube_context: Optional[str] = None
    kube_deployment_name: Optional[str] = None
    namespace: str = "default"
    cpu_request: Optional[str] = None
    mem_request: Optional[str] = None
    cpu_limit: Optional[str] = None
    mem_limit: Optional[str] = None
    min_replicas: int = 1
    max_replicas: int = 1
    scrape_interval: str = "15s"
    kserve_deployment_mode: str = "Standard"
    image_pull_policy: str = "IfNotPresent"
    image_pull_secret: Optional[str] = None
    prometheus_release_label: str = "prometheus"
    prometheus_url: Optional[str] = None
    node_selector: Dict[str, str] = field(default_factory=dict)
    extra_resource_limits: Dict[str, str] = field(default_factory=dict)
    runtime_class_name: Optional[str] = None

    # Docker Compose
    compose_project_name: Optional[str] = None
    compose_model_port: int = 8000
    compose_prometheus_port: int = 9090
    compose_grafana_port: int = 3000
    compose_enable_prometheus: bool = False
    compose_enable_grafana: bool = False
    compose_external_prometheus_url: Optional[str] = None


def get_value(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return a trimmed environment value or a default."""
    value = os.environ.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else default


def get_int(
    name: str,
    default: int,
    *,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    """Read and range-check an integer environment setting."""
    raw = get_value(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise DeploymentError(
            "CONFIGURATION_INVALID",
            f"{name} must be an integer: {exception_message(exc)}",
        ) from exc
    if value < minimum or (maximum is not None and value > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise DeploymentError(
            "CONFIGURATION_INVALID", f"{name} must be >= {minimum}{upper}; got {value}"
        )
    return value


def get_bool(name: str, default: bool) -> bool:
    """Read a conventional boolean environment setting."""
    raw = get_value(name)
    if raw is None:
        return default
    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise DeploymentError(
        "CONFIGURATION_INVALID",
        f"{name} must be one of true/false, yes/no, on/off, or 1/0",
    )


def get_json_string_map(name: str) -> Dict[str, str]:
    """Parse an optional JSON object and normalize keys/values to strings."""
    raw = get_value(name)
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DeploymentError(
            "CONFIGURATION_INVALID",
            f"{name} must be a valid JSON object: {exception_message(exc)}",
        ) from exc
    if not isinstance(value, dict):
        raise DeploymentError("CONFIGURATION_INVALID", f"{name} must be a JSON object")
    result: Dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise DeploymentError(
                "CONFIGURATION_INVALID", f"{name} keys must be non-empty strings"
            )
        if isinstance(item, (dict, list)) or item is None:
            raise DeploymentError(
                "CONFIGURATION_INVALID", f"{name} values must be scalar strings/numbers"
            )
        result[key.strip()] = str(item)
    return result


def validate_registry(
    registry: str,
    *,
    target: str,
    aws_region: Optional[str] = None,
) -> None:
    """Validate registry syntax without accepting embedded credentials."""
    if not registry:
        return
    if any(ch.isspace() for ch in registry) or "://" in registry or "@" in registry:
        raise DeploymentError(
            "CONFIGURATION_INVALID",
            f"{target} Docker registry must be a registry/repository path without URL schemes, "
            "whitespace, or embedded credentials.",
        )
    if registry.startswith("/") or registry.endswith("/"):
        raise DeploymentError(
            "CONFIGURATION_INVALID",
            f"{target} Docker registry must not start or end with '/'.",
        )
    if target == "aws_sagemaker":
        match = re.match(
            r"^(?P<account>[0-9]{12})\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com(?:\.cn)?(?:/[a-z0-9][a-z0-9._/-]*)?$",
            registry,
        )
        if not match:
            raise DeploymentError(
                "CONFIGURATION_INVALID",
                "AWS SageMaker registry must be a private Amazon ECR registry/repository URI "
                "such as <account>.dkr.ecr.<region>.amazonaws.com/<repository>.",
            )
        if aws_region and match.group("region") != aws_region:
            raise DeploymentError(
                "CONFIGURATION_INVALID",
                f"ECR registry region '{match.group('region')}' does not match AWS_REGION "
                f"'{aws_region}'.",
            )


def load_environment(env_file: Optional[str] = None) -> None:
    """Load an optional local .env file without overriding explicit process values."""
    load_dotenv(dotenv_path=env_file, override=False)


def request_from_environment(
    *,
    target: Optional[str] = None,
    model_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    version: Optional[str] = None,
    alias: Optional[str] = None,
    local_model_path: Optional[str] = None,
    local_model_format: Optional[str] = None,
    local_model_name: Optional[str] = None,
    dry_run: bool = False,
) -> DeploymentRequest:
    """Build a deployment request from CLI values and common environment configuration."""
    target = target or get_value("DEPLOYMENT_BACKEND")
    local_model_path = local_model_path or get_value("LOCAL_MODEL_PATH")
    local_model_format = local_model_format or get_value("LOCAL_MODEL_FORMAT", "auto")
    local_model_name = local_model_name or get_value("LOCAL_MODEL_NAME")
    model_name = model_name or get_value("MODEL_NAME")
    tracking_uri = tracking_uri or get_value("TRACKING_URI")
    version = version if version not in (None, "") else get_value("MODEL_VERSION")
    alias = alias or get_value("MODEL_ALIAS", "champion")

    # LOCAL_MODEL_PATH deliberately takes precedence over MLflow. In local mode the
    # blueprint does not require a tracking URI, registry model, alias, or version.
    if local_model_path:
        effective_model_name = local_model_name or Path(local_model_path).expanduser().stem
        required_values = (("DEPLOYMENT_BACKEND/--target", target),)
    else:
        effective_model_name = model_name
        required_values = (
            ("DEPLOYMENT_BACKEND/--target", target),
            ("MODEL_NAME/--model-name", model_name),
            ("TRACKING_URI/--tracking-uri", tracking_uri),
        )

    missing = [name for name, value in required_values if not value]
    if missing:
        raise DeploymentError(
            "CONFIGURATION_MISSING", "Missing required configuration: " + ", ".join(missing)
        )
    if target not in VALID_TARGETS:
        raise DeploymentError(
            "CONFIGURATION_INVALID",
            f"target must be one of {', '.join(VALID_TARGETS)}; got {target!r}",
        )

    return DeploymentRequest(
        target=str(target),
        model_name=str(effective_model_name) if effective_model_name else None,
        tracking_uri=str(tracking_uri) if tracking_uri else None,
        version=str(version) if version is not None and not local_model_path else None,
        alias=str(alias or "champion"),
        dry_run=dry_run,
        local_model_path=str(local_model_path) if local_model_path else None,
        local_model_format=str(local_model_format or "auto"),
    )


def build_common_settings(
    target: str,
    *,
    serving_backend: str,
    monitoring_backend: str,
    registry: str = "",
    platform: Optional[str] = None,
    **kwargs,
) -> DeploymentSettings:
    """Build settings shared by all deployment backends."""
    return DeploymentSettings(
        target=target,
        serving_backend=serving_backend,
        monitoring_backend=monitoring_backend,
        docker_registry=registry,
        docker_platform=platform,
        readiness_timeout_seconds=get_int("DEPLOYMENT_READINESS_TIMEOUT_SECONDS", 900, minimum=1),
        poll_interval_seconds=get_int("DEPLOYMENT_POLL_INTERVAL_SECONDS", 10, minimum=1),
        prometheus_discovery_timeout_seconds=get_int(
            "PROMETHEUS_DISCOVERY_TIMEOUT_SECONDS", 120, minimum=1
        ),
        prometheus_discovery_poll_interval_seconds=get_int(
            "PROMETHEUS_DISCOVERY_POLL_INTERVAL_SECONDS", 5, minimum=1
        ),
        **kwargs,
    )


def validate_compose_ports(
    model_port: int,
    prometheus_port: int,
    grafana_port: int,
    *,
    enable_prometheus: bool,
    enable_grafana: bool,
) -> None:
    """Ensure only ports actually bound by enabled Compose services are unique."""
    bound = [("COMPOSE_MODEL_PORT", model_port)]
    if enable_prometheus:
        bound.append(("COMPOSE_PROMETHEUS_PORT", prometheus_port))
    if enable_grafana:
        bound.append(("COMPOSE_GRAFANA_PORT", grafana_port))
    values = [value for _, value in bound]
    if len(set(values)) != len(values):
        names = ", ".join(name for name, _ in bound)
        raise DeploymentError(
            "CONFIGURATION_INVALID",
            f"Enabled Docker Compose services must use distinct host ports: {names}.",
        )


def load_settings(target: str) -> DeploymentSettings:
    """Load and validate settings for one concrete deployment backend."""
    if target not in VALID_TARGETS:
        raise DeploymentError("CONFIGURATION_INVALID", f"Unsupported target {target!r}")

    if target == "aws_sagemaker":
        region = get_value("AWS_REGION")
        role = get_value("SAGEMAKER_ROLE_ARN")
        instance_type = get_value("SAGEMAKER_INSTANCE_TYPE")
        registry = get_value("AWS_DOCKER_REGISTRY", "") or ""
        platform = get_value("AWS_DOCKER_PLATFORM")
        missing = [
            key
            for key, value in (
                ("AWS_REGION", region),
                ("SAGEMAKER_ROLE_ARN", role),
                ("SAGEMAKER_INSTANCE_TYPE", instance_type),
                ("AWS_DOCKER_REGISTRY", registry),
                ("AWS_DOCKER_PLATFORM", platform),
            )
            if not value
        ]
        if missing:
            raise DeploymentError(
                "CONFIGURATION_MISSING",
                "AWS SageMaker deployment requires: " + ", ".join(missing),
            )
        validate_registry(registry, target=target, aws_region=region)
        return build_common_settings(
            target,
            serving_backend="sagemaker",
            monitoring_backend="cloudwatch",
            registry=registry,
            platform=platform,
            aws_region=region,
            sagemaker_role_arn=role,
            sagemaker_instance_type=instance_type,
            sagemaker_instance_count=get_int("SAGEMAKER_INSTANCE_COUNT", 1, minimum=1),
            sagemaker_variant_name=get_value("SAGEMAKER_VARIANT_NAME", "AllTraffic") or "AllTraffic",
            sagemaker_latency_threshold_us=get_int(
                "SAGEMAKER_LATENCY_THRESHOLD_US", 1_000_000, minimum=1
            ),
            sagemaker_alarm_period=get_int("SAGEMAKER_ALARM_PERIOD", 60, minimum=1),
            sagemaker_evaluation_periods=get_int(
                "SAGEMAKER_EVALUATION_PERIODS", 3, minimum=1
            ),
        )

    if target == "kubernetes":
        registry = get_value("KUBE_DOCKER_REGISTRY", "") or ""
        kube_context = get_value("KUBE_CONTEXT")
        kube_deployment_name = get_value("KUBE_DEPLOYMENT_NAME")
        cpu_request = get_value("KUBE_CPU_REQUEST")
        platform = get_value("KUBE_DOCKER_PLATFORM")
        mem_request = get_value("KUBE_MEM_REQUEST")
        missing = [
            key
            for key, value in (
                ("KUBE_CONTEXT", kube_context),
                ("KUBE_DEPLOYMENT_NAME", kube_deployment_name),
                ("KUBE_DOCKER_REGISTRY", registry),
                ("KUBE_CPU_REQUEST", cpu_request),
                ("KUBE_MEM_REQUEST", mem_request),
                ("KUBE_DOCKER_PLATFORM", platform),
            )
            if not value
        ]
        if missing:
            raise DeploymentError(
                "CONFIGURATION_MISSING",
                "Kubernetes/KServe deployment requires: " + ", ".join(missing),
            )
        validate_registry(registry, target=target)
        kube_deployment_name = validate_kubernetes_deployment_name(kube_deployment_name)
        min_replicas = get_int("KUBE_MIN_REPLICAS", 1, minimum=1)
        max_replicas = get_int("KUBE_MAX_REPLICAS", 1, minimum=1)
        if max_replicas < min_replicas:
            raise DeploymentError(
                "CONFIGURATION_INVALID",
                "KUBE_MAX_REPLICAS must be >= KUBE_MIN_REPLICAS",
            )
        mode = get_value("KUBE_KSERVE_DEPLOYMENT_MODE", "Standard") or "Standard"
        if mode not in ("Standard", "Knative", "ModelMesh"):
            raise DeploymentError(
                "CONFIGURATION_INVALID",
                "KUBE_KSERVE_DEPLOYMENT_MODE must be Standard, Knative, or ModelMesh",
            )
        return build_common_settings(
            target,
            serving_backend="kserve",
            monitoring_backend="prometheus",
            registry=registry,
            platform=platform,
            kube_context=kube_context,
            kube_deployment_name=kube_deployment_name,
            namespace=get_value("KUBE_NAMESPACE", "default") or "default",
            cpu_request=cpu_request,
            mem_request=mem_request,
            cpu_limit=get_value("KUBE_CPU_LIMIT", cpu_request),
            mem_limit=get_value("KUBE_MEM_LIMIT", mem_request),
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            scrape_interval=get_value("KUBE_SCRAPE_INTERVAL", "15s") or "15s",
            kserve_deployment_mode=mode,
            image_pull_policy=get_value("KUBE_IMAGE_PULL_POLICY", "IfNotPresent") or "IfNotPresent",
            image_pull_secret=get_value("KUBE_IMAGE_PULL_SECRET"),
            prometheus_release_label=get_value("KUBE_PROMETHEUS_RELEASE_LABEL", "prometheus") or "prometheus",
            prometheus_url=get_value("KUBE_PROMETHEUS_URL"),
            node_selector=get_json_string_map("KUBE_NODE_SELECTOR_JSON"),
            extra_resource_limits=get_json_string_map("KUBE_EXTRA_RESOURCE_LIMITS_JSON"),
            runtime_class_name=get_value("KUBE_RUNTIME_CLASS"),
        )

    model_port = get_int("COMPOSE_MODEL_PORT", 8000, minimum=1, maximum=65535)
    prometheus_port = get_int("COMPOSE_PROMETHEUS_PORT", 9090, minimum=1, maximum=65535)
    grafana_port = get_int("COMPOSE_GRAFANA_PORT", 3000, minimum=1, maximum=65535)
    enable_prometheus = get_bool("COMPOSE_ENABLE_PROMETHEUS", False)
    enable_grafana = get_bool("COMPOSE_ENABLE_GRAFANA", False)
    external_prometheus_url = get_value("COMPOSE_EXTERNAL_PROMETHEUS_URL")
    platform = get_value("COMPOSE_DOCKER_PLATFORM")
    if not platform:
        raise DeploymentError(
            "CONFIGURATION_MISSING",
            "Docker Compose deployment requires: COMPOSE_DOCKER_PLATFORM",
        )

    validate_compose_ports(
        model_port,
        prometheus_port,
        grafana_port,
        enable_prometheus=enable_prometheus,
        enable_grafana=enable_grafana,
    )
    if enable_grafana and not enable_prometheus and not external_prometheus_url:
        raise DeploymentError(
            "CONFIGURATION_MISSING",
            "COMPOSE_EXTERNAL_PROMETHEUS_URL is required when bundled Grafana is enabled "
            "without bundled Prometheus.",
        )

    return build_common_settings(
        target,
        serving_backend="docker_compose",
        monitoring_backend="prometheus_grafana",
        registry="",
        platform=platform,
        compose_project_name=get_value("COMPOSE_PROJECT_NAME"),
        compose_model_port=model_port,
        compose_prometheus_port=prometheus_port,
        compose_grafana_port=grafana_port,
        compose_enable_prometheus=enable_prometheus,
        compose_enable_grafana=enable_grafana,
        compose_external_prometheus_url=external_prometheus_url,
    )
