"""Manual Prometheus smoke test for the Kubernetes/KServe deployment."""

from __future__ import annotations

import json
import logging
import os
import sys
from urllib.parse import urlencode
from urllib.request import urlopen
from pathlib import Path

from dotenv import load_dotenv



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEPLOYMENT_ROOT = PROJECT_ROOT / "model_deployment"
TRAINING_ROOT = PROJECT_ROOT / "model_training"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_deployment.local_model import configured_deployment_model_name

load_dotenv(DEPLOYMENT_ROOT / ".env", override=False)

# Port-forward Prometheus to 19090, or override TEST_PROMETHEUS_URL.
PROMETHEUS_BASE_URL = os.getenv("TEST_PROMETHEUS_URL", "http://127.0.0.1:19090").rstrip("/")

EXPECTED_NAMESPACE = os.getenv("KUBE_NAMESPACE", "mlops")
EXPECTED_MODEL = configured_deployment_model_name("champion_model") or "champion_model"
KUBE_DEPLOYMENT_NAME = os.getenv("KUBE_DEPLOYMENT_NAME", "champion-model")
EXPECTED_JOB_FRAGMENT = os.getenv(
    "TEST_PODMONITOR_NAME", f"{KUBE_DEPLOYMENT_NAME}-inference-monitor"
)


def get_text(url: str, timeout: float = 10.0) -> str:
    """Fetch one text response."""
    with urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
        status = int(getattr(response, "status", 200))

    if not 200 <= status < 300:
        raise RuntimeError(f"GET {url} returned HTTP {status}")

    return payload


def get_json(url: str, timeout: float = 10.0) -> dict:
    """Fetch and decode one JSON response."""
    payload = get_text(url, timeout=timeout)
    result = json.loads(payload)

    if not isinstance(result, dict):
        raise RuntimeError(f"GET {url} did not return a JSON object")

    return result


def prometheus_query(query: str) -> dict:
    """Run one Prometheus instant query."""
    url = f"{PROMETHEUS_BASE_URL}/api/v1/query?" + urlencode({"query": query})
    payload = get_json(url)

    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query {query!r} failed: {payload}")

    return payload


def get_query_results(query: str) -> list[dict]:
    """Return vector results for one Prometheus query."""
    payload = prometheus_query(query)
    result = payload.get("data", {}).get("result", [])

    if not isinstance(result, list):
        raise RuntimeError(
            f"Prometheus query {query!r} returned invalid result: {payload}"
        )

    return result


def verify_readiness() -> None:
    """Verify that Prometheus is ready."""
    payload = get_text(f"{PROMETHEUS_BASE_URL}/-/ready")

    if "ready" not in payload.lower():
        raise RuntimeError(
            f"Unexpected Prometheus readiness response: {payload!r}"
        )

    logger.info("Prometheus readiness PASSED")


def verify_target() -> None:
    """Verify that Prometheus discovered the KServe PodMonitor target."""
    payload = get_json(
        f"{PROMETHEUS_BASE_URL}/api/v1/targets?state=active"
    )

    if payload.get("status") != "success":
        raise RuntimeError(
            f"Could not fetch active Prometheus targets: {payload}"
        )

    targets = payload.get("data", {}).get("activeTargets", [])

    for target in targets:
        labels = {}
        labels.update(target.get("discoveredLabels", {}) or {})
        labels.update(target.get("labels", {}) or {})

        job = str(labels.get("job", ""))
        namespace = str(labels.get("namespace", ""))

        if EXPECTED_JOB_FRAGMENT not in job:
            continue

        if namespace and namespace != EXPECTED_NAMESPACE:
            continue

        health = str(target.get("health", "")).lower()
        scrape_url = str(target.get("scrapeUrl", ""))

        if health != "up":
            raise RuntimeError(
                "KServe Prometheus target found but is not UP: "
                f"health={health!r}, scrape_url={scrape_url!r}"
            )

        logger.info(
            "KServe Prometheus target PASSED: job=%s health=%s scrape=%s",
            job,
            health,
            scrape_url,
        )
        return

    raise RuntimeError(
        "Could not find an active Prometheus target for "
        f"{EXPECTED_JOB_FRAGMENT!r}"
    )


def matching_metric_samples(query: str) -> list[dict]:
    """Return samples belonging to the deployed KServe model."""
    results = get_query_results(query)

    return [
        item
        for item in results
        if item.get("metric", {}).get("namespace") == EXPECTED_NAMESPACE
        and item.get("metric", {}).get("model") == EXPECTED_MODEL
    ]


def verify_model_up() -> None:
    """Verify that Prometheus reports the deployed model as up."""
    matching = matching_metric_samples("model_up")

    if not matching:
        raise RuntimeError(
            f"model_up returned no sample for model {EXPECTED_MODEL!r}"
        )

    for item in matching:
        metric = item.get("metric", {})
        value = item.get("value", [])

        if len(value) < 2:
            continue

        numeric_value = float(value[1])

        if numeric_value >= 1:
            logger.info(
                "model_up PASSED: model=%s version=%s pod=%s value=%.0f",
                metric.get("model"),
                metric.get("version"),
                metric.get("pod"),
                numeric_value,
            )
            return

    raise RuntimeError(
        f"model_up did not report value=1 for model {EXPECTED_MODEL!r}"
    )


def verify_required_metric(query: str) -> None:
    """Verify that a metric expected after successful inference is present."""
    matching = matching_metric_samples(query)

    if not matching:
        raise RuntimeError(
            f"Prometheus query {query!r} returned no sample for "
            f"namespace={EXPECTED_NAMESPACE!r} model={EXPECTED_MODEL!r}"
        )

    values = [
        float(item["value"][1])
        for item in matching
        if len(item.get("value", [])) >= 2
    ]

    if not values:
        raise RuntimeError(f"Metric {query!r} has no readable values")

    logger.info(
        "%s PASSED: samples=%d values=%s",
        query,
        len(values),
        ", ".join(f"{value:g}" for value in values),
    )


def verify_error_metric() -> None:
    """Report the inference error counter, treating no sample as zero errors.

    Prometheus labeled counters may not emit a sample until the first matching
    event occurs. Therefore an absent inference_errors_total sample is valid
    when no failed inference has occurred yet.
    """
    matching = matching_metric_samples("inference_errors_total")

    if not matching:
        logger.info(
            "inference_errors_total PASSED: value=0 "
            "(no failed inference sample has been created yet)"
        )
        return

    values = [
        float(item["value"][1])
        for item in matching
        if len(item.get("value", [])) >= 2
    ]

    if not values:
        raise RuntimeError(
            "inference_errors_total returned samples without readable values"
        )

    logger.info(
        "inference_errors_total PASSED: samples=%d values=%s",
        len(values),
        ", ".join(f"{value:g}" for value in values),
    )


def main() -> int:
    """Run the Prometheus integration smoke test."""
    try:
        logger.info(
            "Testing Prometheus endpoint: %s",
            PROMETHEUS_BASE_URL,
        )

        verify_readiness()
        verify_target()
        verify_model_up()

        verify_required_metric("inference_requests_total")
        verify_error_metric()
        verify_required_metric("inference_duration_seconds_count")

        logger.info("KServe Prometheus smoke test PASSED")
        return 0

    except Exception:
        logger.exception("KServe Prometheus smoke test FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
