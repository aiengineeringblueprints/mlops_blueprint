"""Kubernetes/KServe deployment and monitoring helpers."""
from __future__ import annotations

import logging
from pathlib import Path
import time

import yaml
from jinja2 import Template

from model_deployment import DeploymentError, ResolvedModel, safe_resource_name, stable_deployment_name
from model_deployment.config import DeploymentSettings
from model_deployment.monitoring import wait_prometheus_target

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
TEMPLATES_DIR = HERE / "templates"


def get_kserve_ready_condition(resource: dict):
    """Return the KServe Ready condition when it is present."""
    conditions = resource.get("status", {}).get("conditions", []) or []
    for condition in conditions:
        if condition.get("type") == "Ready":
            return condition
    return None


def wait_kserve_ready(
    custom_api,
    *,
    namespace: str,
    name: str,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> None:
    """Wait until the KServe InferenceService reports Ready=True."""
    deadline = time.monotonic() + timeout_seconds
    last_reason = "Ready condition not reported"
    while time.monotonic() < deadline:
        try:
            resource = custom_api.get_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=name,
            )
        except Exception as exc:
            raise DeploymentError(
                "DEPLOYMENT_FAILED",
                f"Could not inspect KServe InferenceService '{namespace}/{name}': {exc}",
            ) from exc
        condition = get_kserve_ready_condition(resource)
        if condition:
            status = str(condition.get("status", "")).lower()
            if status == "true":
                return
            last_reason = ": ".join(
                item for item in (condition.get("reason"), condition.get("message")) if item
            ) or f"Ready={condition.get('status')}"
            if status == "false" and condition.get("reason") in {
                "PredictorConfigurationReady",
                "RevisionFailed",
                "IngressNotConfigured",
            }:
                raise DeploymentError(
                    "DEPLOYMENT_FAILED",
                    f"KServe InferenceService '{namespace}/{name}' is not ready: {last_reason}",
                )
        time.sleep(poll_interval_seconds)
    raise DeploymentError(
        "DEPLOYMENT_TIMEOUT",
        f"KServe InferenceService '{namespace}/{name}' did not become Ready within "
        f"{timeout_seconds}s (last condition: {last_reason}).",
    )


def render_inference_service(
    settings: DeploymentSettings,
    resolved: ResolvedModel,
    image: str,
) -> dict:
    """Render the KServe InferenceService manifest for the resolved model."""
    name = stable_deployment_name(settings.kube_deployment_name, settings.target)
    template = Template(
        (TEMPLATES_DIR / "inferenceservice.yaml.j2").read_text(encoding="utf-8")
    )
    manifest = yaml.safe_load(
        template.render(
            service_name=name,
            model_name=resolved.name,
            deployment_model_name=settings.kube_deployment_name,
            model_version=resolved.version,
            image=image,
            namespace=settings.namespace,
            min_replicas=settings.min_replicas,
            max_replicas=settings.max_replicas,
            cpu_request=settings.cpu_request,
            mem_request=settings.mem_request,
            cpu_limit=settings.cpu_limit,
            mem_limit=settings.mem_limit,
            deployment_mode=settings.kserve_deployment_mode,
            image_pull_policy=settings.image_pull_policy,
        )
    )
    predictor = manifest["spec"]["predictor"]
    if settings.image_pull_secret:
        predictor["imagePullSecrets"] = [{"name": settings.image_pull_secret}]
    if settings.node_selector:
        predictor["nodeSelector"] = dict(settings.node_selector)
    if settings.runtime_class_name:
        predictor["runtimeClassName"] = settings.runtime_class_name
    if settings.extra_resource_limits:
        limits = predictor["containers"][0].setdefault("resources", {}).setdefault("limits", {})
        limits.update(settings.extra_resource_limits)
    return manifest


def require_crd(api_extensions, name: str, description: str) -> None:
    """Require one Kubernetes CRD and translate lookup errors to deployment errors."""
    try:
        api_extensions.read_custom_resource_definition(name)
    except Exception as exc:
        status = getattr(exc, "status", None)
        if status == 404:
            raise DeploymentError(
                "PREREQUISITE_MISSING",
                f"{description} CRD '{name}' is not installed.",
            ) from exc
        raise DeploymentError(
            "PREREQUISITE_UNREACHABLE",
            f"Could not inspect required CRD '{name}': {exc}",
        ) from exc


def preflight_kserve(
    settings: DeploymentSettings,
    *,
    version_api=None,
    api_extensions=None,
    core_api=None,
) -> None:
    """Fail early when required Kubernetes/KServe prerequisites are absent."""
    if not settings.docker_registry:
        raise DeploymentError(
            "PREREQUISITE_MISSING",
            "No image registry is configured for the Kubernetes/KServe backend.",
        )

    if version_api is None or api_extensions is None or core_api is None:
        try:
            from kubernetes import client as kclient, config as kconfig

            kconfig.load_kube_config(context=settings.kube_context)
            version_api = version_api or kclient.VersionApi()
            api_extensions = api_extensions or kclient.ApiextensionsV1Api()
            core_api = core_api or kclient.CoreV1Api()
        except Exception as exc:
            raise DeploymentError(
                "PREREQUISITE_UNREACHABLE",
                f"Unable to load/connect to Kubernetes context '{settings.kube_context}': {exc}",
            ) from exc

    try:
        version_api.get_code()
    except Exception as exc:
        raise DeploymentError(
            "PREREQUISITE_UNREACHABLE",
            f"Unable to connect to Kubernetes cluster for context '{settings.kube_context}': {exc}",
        ) from exc

    require_crd(api_extensions, "inferenceservices.serving.kserve.io", "KServe InferenceService")
    if settings.monitoring_backend == "prometheus":
        try:
            require_crd(
                api_extensions,
                "podmonitors.monitoring.coreos.com",
                "Prometheus Operator PodMonitor",
            )
        except DeploymentError as exc:
            if exc.code == "PREREQUISITE_MISSING":
                raise DeploymentError(
                    "PREREQUISITE_MISSING",
                    "PodMonitor CRD is unavailable. Install/configure Prometheus Operator "
                    "before using the blueprint's Prometheus monitoring integration.",
                ) from exc
            raise

    try:
        core_api.read_namespace(settings.namespace)
    except Exception as exc:
        if getattr(exc, "status", None) == 404:
            raise DeploymentError(
                "PREREQUISITE_MISSING",
                f"Target namespace '{settings.namespace}' does not exist. Create it before deployment.",
            ) from exc
        raise DeploymentError(
            "PREREQUISITE_UNREACHABLE",
            f"Could not inspect target namespace '{settings.namespace}': {exc}",
        ) from exc

    if settings.image_pull_secret:
        try:
            core_api.read_namespaced_secret(settings.image_pull_secret, settings.namespace)
        except Exception as exc:
            if getattr(exc, "status", None) == 404:
                raise DeploymentError(
                    "PREREQUISITE_MISSING",
                    f"Configured imagePullSecret '{settings.image_pull_secret}' does not exist "
                    f"in namespace '{settings.namespace}'.",
                ) from exc
            raise DeploymentError(
                "PREREQUISITE_UNREACHABLE",
                f"Could not inspect imagePullSecret '{settings.image_pull_secret}' in "
                f"namespace '{settings.namespace}': {exc}",
            ) from exc


def deploy_kserve(
    settings: DeploymentSettings,
    resolved: ResolvedModel,
    image: str,
    *,
    custom_api=None,
) -> str:
    """Reconcile a stable KServe InferenceService and wait for Ready=True."""
    if custom_api is None:
        from kubernetes import client as kclient, config as kconfig

        kconfig.load_kube_config(context=settings.kube_context)
        custom_api = kclient.CustomObjectsApi()

    manifest = render_inference_service(settings, resolved, image)
    name = manifest["metadata"]["name"]

    try:
        custom_api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=settings.namespace,
            plural="inferenceservices",
            body=manifest,
        )
        logger.info("Created KServe InferenceService '%s/%s'", settings.namespace, name)
    except Exception as exc:
        status = getattr(exc, "status", None)
        if status != 409:
            raise DeploymentError(
                "DEPLOYMENT_SUBMISSION_FAILED",
                f"Could not create KServe InferenceService '{settings.namespace}/{name}': {exc}",
            ) from exc
        try:
            custom_api.patch_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=settings.namespace,
                plural="inferenceservices",
                name=name,
                body=manifest,
            )
            logger.info("Updated KServe InferenceService '%s/%s'", settings.namespace, name)
        except Exception as patch_exc:
            raise DeploymentError(
                "DEPLOYMENT_SUBMISSION_FAILED",
                f"Could not update KServe InferenceService '{settings.namespace}/{name}': {patch_exc}",
            ) from patch_exc

    wait_kserve_ready(
        custom_api,
        namespace=settings.namespace,
        name=name,
        timeout_seconds=settings.readiness_timeout_seconds,
        poll_interval_seconds=settings.poll_interval_seconds,
    )
    return name


def render_podmonitor(settings: DeploymentSettings, endpoint: str) -> dict:
    """Render the PodMonitor used to scrape the KServe workload."""
    template = Template((TEMPLATES_DIR / "podmonitor.yaml.j2").read_text(encoding="utf-8"))
    monitor_name = safe_resource_name(f"{endpoint}-monitor", max_length=63)
    return yaml.safe_load(
        template.render(
            service_name=endpoint,
            monitor_name=monitor_name,
            namespace=settings.namespace,
            interval=settings.scrape_interval,
            prometheus_release_label=settings.prometheus_release_label,
        )
    )


def configure_kserve_monitoring(
    settings: DeploymentSettings,
    model_name: str,
    endpoint: str,
    *,
    custom_api=None,
) -> str:
    """Create/update the KServe PodMonitor and optionally verify Prometheus discovery."""
    del model_name
    if custom_api is None:
        from kubernetes import client as kclient, config as kconfig

        kconfig.load_kube_config(context=settings.kube_context)
        custom_api = kclient.CustomObjectsApi()

    manifest = render_podmonitor(settings, endpoint)
    name = manifest["metadata"]["name"]
    try:
        custom_api.create_namespaced_custom_object(
            group="monitoring.coreos.com",
            version="v1",
            namespace=settings.namespace,
            plural="podmonitors",
            body=manifest,
        )
        logger.info("Created PodMonitor '%s/%s'", settings.namespace, name)
    except Exception as exc:
        if getattr(exc, "status", None) != 409:
            raise DeploymentError(
                "MONITORING_CONFIGURATION_FAILED",
                f"Could not create PodMonitor '{settings.namespace}/{name}': {exc}",
            ) from exc
        try:
            custom_api.patch_namespaced_custom_object(
                group="monitoring.coreos.com",
                version="v1",
                namespace=settings.namespace,
                plural="podmonitors",
                name=name,
                body=manifest,
            )
            logger.info("Updated PodMonitor '%s/%s'", settings.namespace, name)
        except Exception as patch_exc:
            raise DeploymentError(
                "MONITORING_CONFIGURATION_FAILED",
                f"Could not update PodMonitor '{settings.namespace}/{name}': {patch_exc}",
            ) from patch_exc

    if settings.prometheus_url:
        wait_prometheus_target(settings, endpoint)
        return f"Prometheus PodMonitor {settings.namespace}/{name}; target discovered and up"

    return (
        f"Prometheus PodMonitor {settings.namespace}/{name}; configured but target discovery "
        "not externally verified (KUBE_PROMETHEUS_URL not configured)"
    )


def teardown_kserve(
    deployment_name: str,
    namespace: str,
    kube_context: str,
    *,
    execute: bool = False,
    custom_api=None,
) -> str:
    """Remove the exact KServe InferenceService and matching PodMonitor."""
    name = stable_deployment_name(deployment_name, "kubernetes")
    monitor = safe_resource_name(f"{name}-monitor", max_length=63)
    if not execute:
        return (
            f"DRY-RUN: delete exact InferenceService {namespace}/{name} "
            f"and PodMonitor {namespace}/{monitor}"
        )
    if custom_api is None:
        from kubernetes import client as kclient, config as kconfig

        kconfig.load_kube_config(context=kube_context)
        custom_api = kclient.CustomObjectsApi()
    for group, plural, resource_name in (
        ("monitoring.coreos.com", "podmonitors", monitor),
        ("serving.kserve.io", "inferenceservices", name),
    ):
        try:
            custom_api.delete_namespaced_custom_object(
                group=group,
                version="v1" if group == "monitoring.coreos.com" else "v1beta1",
                namespace=namespace,
                plural=plural,
                name=resource_name,
                body={},
            )
        except Exception as exc:
            if getattr(exc, "status", None) != 404:
                raise DeploymentError(
                    "TEARDOWN_FAILED",
                    f"Could not delete exact resource '{namespace}/{resource_name}': {exc}",
                ) from exc
    return f"Deleted exact KServe deployment resources for {namespace}/{name}."
