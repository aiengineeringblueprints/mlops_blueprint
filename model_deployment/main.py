"""Common deployment control for the three concrete deployment backends."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional, Sequence

from model_deployment import DeploymentError, DeploymentPlan, DeploymentResult, stable_deployment_name
from model_deployment.champion import resolve_model_version
from model_deployment.local_model import resolve_local_model
from model_deployment.config import VALID_TARGETS, load_environment, load_settings, request_from_environment
from model_deployment.container import build_image, image_tag
from model_deployment.deployments import TARGET_ADAPTERS
from model_deployment.deployments.docker_compose import (
    compose_monitoring_description,
    compose_project_name,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(HERE, ".env")
load_environment(ENV_FILE)

logger = logging.getLogger(__name__)


def resolve_args(argv: Optional[Sequence[str]] = None):
    """Parse CLI options and merge them with deployment environment configuration."""
    load_environment(ENV_FILE)
    parser = argparse.ArgumentParser(description="Deploy an MLflow-registered model or supported standalone local model.")
    parser.add_argument("--backend", "--target", dest="target", choices=VALID_TARGETS)
    parser.add_argument("--model-name")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--model-version", help="Explicit registered version override")
    parser.add_argument("--model-alias", help="Alias used when no explicit version is supplied")
    parser.add_argument("--local-model-path", help="Standalone local model file; bypasses MLflow when set")
    parser.add_argument("--local-model-format", help="Local format override; default is auto detection")
    parser.add_argument("--local-model-name", help="Optional deployment name for a standalone local model")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and log the deployment plan only")
    args = parser.parse_args(argv)
    return request_from_environment(
        target=args.target,
        model_name=args.model_name,
        tracking_uri=args.tracking_uri,
        version=args.model_version,
        alias=args.model_alias,
        local_model_path=args.local_model_path,
        local_model_format=args.local_model_format,
        local_model_name=args.local_model_name,
        dry_run=args.dry_run,
    )


def build_plan_identity(request, settings, resolved) -> dict:
    """Return backend-specific stable identity, endpoints, and monitoring topology."""
    if request.target == "docker_compose":
        project = compose_project_name(settings, resolved.name)
        model_endpoint = f"http://localhost:{settings.compose_model_port}"
        return {
            "deployment_name": project,
            "service_name": "model-service",
            "model_endpoint": model_endpoint,
            "metrics_endpoint": f"{model_endpoint}/metrics",
            "prometheus_endpoint": (
                f"http://localhost:{settings.compose_prometheus_port}"
                if settings.compose_enable_prometheus
                else None
            ),
            "external_prometheus_url": (
                settings.compose_external_prometheus_url
                if not settings.compose_enable_prometheus
                else None
            ),
            "grafana_endpoint": (
                f"http://localhost:{settings.compose_grafana_port}"
                if settings.compose_enable_grafana
                else None
            ),
            "monitoring_configuration": compose_monitoring_description(settings),
        }

    if request.target == "kubernetes":
        return {
            "deployment_name": stable_deployment_name(settings.kube_deployment_name, request.target),
            "service_name": None,
            "model_endpoint": None,
            "metrics_endpoint": None,
            "prometheus_endpoint": settings.prometheus_url,
            "external_prometheus_url": None,
            "grafana_endpoint": None,
            "monitoring_configuration": "Prometheus/Grafana via Kubernetes PodMonitor",
        }

    return {
        "deployment_name": stable_deployment_name(resolved.name, request.target),
        "service_name": None,
        "model_endpoint": None,
        "metrics_endpoint": None,
        "prometheus_endpoint": None,
        "external_prometheus_url": None,
        "grafana_endpoint": None,
        "monitoring_configuration": "Amazon CloudWatch",
    }


def build_plan(request, settings, resolved) -> DeploymentPlan:
    """Build a traceable deployment plan before infrastructure changes are made."""
    image_identity = (
        settings.kube_deployment_name if request.target == "kubernetes" else resolved.name
    )
    image = image_tag(image_identity, resolved.version, settings.docker_registry)
    identity = build_plan_identity(request, settings, resolved)
    return DeploymentPlan(
        deployment_target=request.target,
        model_name=resolved.name,
        registered_version=resolved.version,
        alias=resolved.alias,
        model_uri=resolved.source,
        image=image,
        docker_platform=settings.docker_platform,
        deployment_name=identity["deployment_name"],
        serving_backend=settings.serving_backend,
        monitoring_backend=settings.monitoring_backend,
        service_name=identity["service_name"],
        model_endpoint=identity["model_endpoint"],
        metrics_endpoint=identity["metrics_endpoint"],
        prometheus_endpoint=identity["prometheus_endpoint"],
        external_prometheus_url=identity["external_prometheus_url"],
        grafana_endpoint=identity["grafana_endpoint"],
        monitoring_configuration=identity["monitoring_configuration"],
    )


def run(request) -> DeploymentResult | DeploymentPlan:
    """Execute the common deployment flow and dispatch only backend-specific operations."""
    settings = load_settings(request.target)
    adapter = TARGET_ADAPTERS[request.target]
    if not request.dry_run:
        logger.info("stage=preflight target=%s backend=%s", request.target, adapter.name)
        adapter.preflight(settings)

    if request.local_model_path:
        logger.info(
            "stage=model_resolution source=local model=%s target=%s path=%s format=%s",
            request.model_name,
            request.target,
            request.local_model_path,
            request.local_model_format,
        )
        resolved = resolve_local_model(
            request.local_model_path,
            model_name=request.model_name,
            model_format=request.local_model_format,
        )
    else:
        logger.info(
            "stage=model_resolution source=mlflow model=%s target=%s requested=%s",
            request.model_name,
            request.target,
            f"version:{request.version}" if request.version else f"alias:{request.alias}",
        )
        if not request.model_name or not request.tracking_uri:
            raise DeploymentError(
                "CONFIGURATION_MISSING",
                "MLflow deployment requires MODEL_NAME and TRACKING_URI when LOCAL_MODEL_PATH is not set.",
            )
        resolved = resolve_model_version(
            request.model_name,
            request.tracking_uri,
            request.version,
            alias=request.alias,
        )
    plan = build_plan(request, settings, resolved)
    if request.dry_run:
        logger.info(
            "Deployment plan:\n%s",
            json.dumps(plan.as_dict(), indent=2, sort_keys=True),
        )
        return plan

    logger.info(
        "stage=image_prepare model=%s version=%s target=%s image=%s",
        resolved.name,
        resolved.version,
        request.target,
        plan.image,
    )
    push_image = request.target in {"aws_sagemaker", "kubernetes"}
    image = build_image(
        resolved.name,
        resolved.version,
        resolved.source,
        request.target,
        settings.docker_registry,
        platform=settings.docker_platform,
        push=push_image,
        image_name=(settings.kube_deployment_name if request.target == "kubernetes" else None),
        source_type=resolved.source_type,
        model_format=resolved.model_format,
    )

    logger.info(
        "stage=deployment target=%s backend=%s model=%s version=%s deployment=%s",
        request.target,
        adapter.name,
        resolved.name,
        resolved.version,
        plan.deployment_name,
    )
    deployment_name = adapter.deploy(settings, resolved, image)

    logger.info(
        "stage=monitoring target=%s backend=%s deployment=%s",
        request.target,
        settings.monitoring_backend,
        deployment_name,
    )
    monitoring_status = adapter.configure_monitoring(
        settings,
        resolved.name,
        deployment_name,
    )

    identity = build_plan_identity(request, settings, resolved)
    result = DeploymentResult(
        model_name=resolved.name,
        registered_version=resolved.version,
        alias=resolved.alias,
        deployment_target=request.target,
        image=image,
        docker_platform=settings.docker_platform,
        deployment_name=deployment_name,
        status="Ready",
        monitoring=monitoring_status,
        service_name=identity["service_name"],
        model_endpoint=identity["model_endpoint"],
        metrics_endpoint=identity["metrics_endpoint"],
        prometheus_endpoint=identity["prometheus_endpoint"],
        external_prometheus_url=identity["external_prometheus_url"],
        grafana_endpoint=identity["grafana_endpoint"],
        monitoring_configuration=identity["monitoring_configuration"],
    )
    logger.info(
        "Deployment result:\n%s",
        json.dumps(result.as_dict(), indent=2, sort_keys=True),
    )
    logger.info(
        "Deployment successful: model=%s version=%s target=%s deployment=%s",
        resolved.name,
        resolved.version,
        request.target,
        deployment_name,
    )
    return result


def cli(argv: Optional[Sequence[str]] = None) -> int:
    """Run the deployment CLI and translate deployment errors into exit codes."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        request = resolve_args(argv)
        run(request)
        return 0
    except DeploymentError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Deployment interrupted")
        return 130
    except Exception as exc:
        logger.exception("UNEXPECTED_DEPLOYMENT_FAILURE: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(cli())
