"""Prometheus target discovery and readiness verification shared by deployment backends."""
from __future__ import annotations

import json
import time
from typing import Callable, Iterable, Optional
from urllib.request import urlopen

from model_deployment import DeploymentError, exception_message
from model_deployment.config import DeploymentSettings


def prometheus_target_matches_kserve(target: dict, endpoint: str, namespace: str) -> bool:
    """Match a KServe workload using structured Kubernetes discovery labels."""
    labels = {}
    labels.update(target.get("discoveredLabels", {}) or {})
    labels.update(target.get("labels", {}) or {})
    namespace_labels = {
        str(labels.get("namespace", "")),
        str(labels.get("kubernetes_namespace", "")),
        str(labels.get("__meta_kubernetes_namespace", "")),
    }
    identity_values = {
        str(labels.get("service", "")),
        str(labels.get("pod", "")),
        str(labels.get("inferenceservice", "")),
        str(labels.get("serving_kserve_io_inferenceservice", "")),
        str(labels.get("__meta_kubernetes_pod_label_serving_kserve_io_inferenceservice", "")),
    }
    namespace_ok = namespace in namespace_labels
    identity_ok = endpoint in identity_values or any(endpoint in value for value in identity_values)
    return namespace_ok and identity_ok and target.get("health") in (None, "", "up")


def fetch_prometheus_targets(prometheus_url: str, *, urlopen_fn=urlopen) -> list[dict]:
    """Fetch Prometheus active-target data from its HTTP API."""
    url = prometheus_url.rstrip("/") + "/api/v1/targets?state=active"
    with urlopen_fn(url, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "success":
        raise RuntimeError("Prometheus targets API did not return success")
    targets = payload.get("data", {}).get("activeTargets", [])
    if not isinstance(targets, list):
        raise RuntimeError("Prometheus activeTargets response is not a list")
    return targets


def wait_prometheus_target(
    settings: DeploymentSettings,
    endpoint: str,
    *,
    fetch_targets: Optional[Callable[[], Iterable[dict]]] = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Poll Prometheus until the deployed KServe workload is an operational target."""
    if not settings.prometheus_url and fetch_targets is None:
        raise DeploymentError(
            "MONITORING_CONFIGURATION_FAILED",
            "Prometheus target verification requires a configured Prometheus URL.",
        )
    if fetch_targets is None:
        fetch_targets = lambda: fetch_prometheus_targets(str(settings.prometheus_url))

    deadline = monotonic_fn() + settings.prometheus_discovery_timeout_seconds
    last_detail = "target not yet reported"
    while monotonic_fn() < deadline:
        try:
            targets = list(fetch_targets())
            if any(
                prometheus_target_matches_kserve(target, endpoint, settings.namespace)
                for target in targets
            ):
                return
            last_detail = f"{len(targets)} active target(s) checked; matching workload not ready"
        except Exception as exc:
            last_detail = f"Prometheus query failed: {exception_message(exc)}"
        sleep_fn(settings.prometheus_discovery_poll_interval_seconds)

    raise DeploymentError(
        "MONITORING_DISCOVERY_TIMEOUT",
        f"Prometheus did not report an operational target for KServe workload "
        f"'{settings.namespace}/{endpoint}' within "
        f"{settings.prometheus_discovery_timeout_seconds}s ({last_detail}).",
    )


def prometheus_target_matches_compose(target: dict) -> bool:
    """Match the Docker Compose model-service scrape target and require it to be healthy."""
    labels = {}
    labels.update(target.get("discoveredLabels", {}) or {})
    labels.update(target.get("labels", {}) or {})
    if str(labels.get("job", "")) != "model-service":
        return False
    scrape_url = str(target.get("scrapeUrl", ""))
    if scrape_url and "model-service:8080/metrics" not in scrape_url:
        return False
    return target.get("health") in (None, "", "up")


def wait_compose_prometheus_target(
    settings: DeploymentSettings,
    project_name: str,
    *,
    fetch_targets: Optional[Callable[[], Iterable[dict]]] = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Wait until bundled Compose Prometheus reports the model-service target as healthy."""
    if not settings.compose_enable_prometheus:
        return
    if fetch_targets is None:
        prometheus_url = f"http://127.0.0.1:{settings.compose_prometheus_port}"
        fetch_targets = lambda: fetch_prometheus_targets(prometheus_url)

    deadline = monotonic_fn() + settings.prometheus_discovery_timeout_seconds
    last_detail = "model-service target not yet ready"
    while monotonic_fn() < deadline:
        try:
            targets = list(fetch_targets())
            if any(prometheus_target_matches_compose(target) for target in targets):
                return
            last_detail = f"{len(targets)} Prometheus target(s) checked; model-service not up"
        except Exception as exc:
            last_detail = f"Prometheus query failed: {exception_message(exc)}"
        sleep_fn(settings.prometheus_discovery_poll_interval_seconds)

    raise DeploymentError(
        "MONITORING_DISCOVERY_TIMEOUT",
        f"Bundled Prometheus for Compose project '{project_name}' did not report the "
        f"model-service target as healthy within {settings.prometheus_discovery_timeout_seconds}s "
        f"({last_detail}).",
    )
