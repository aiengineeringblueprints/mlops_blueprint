"""Manual Docker Compose deployment smoke test.

This is an integration helper, not a unit test. It validates the running
model-service and, when enabled, the bundled Prometheus and Grafana services.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from dotenv import load_dotenv
from PIL import Image


def find_repo_root(start: Path) -> Path:
    """Find the repository root so this file can live anywhere under model_deployment/test."""
    for candidate in (start, *start.parents):
        if (candidate / "deploy.py").is_file() and (
            candidate / "model_deployment"
        ).is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate repository root containing deploy.py and model_deployment/"
    )


HERE = Path(__file__).resolve().parent
REPO_ROOT = find_repo_root(HERE)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_deployment import safe_resource_name

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

IMAGE_SIZE = (224, 224)
DEFAULT_IMAGE = REPO_ROOT / "data" / "val" / "Standard-Schraube" / "IMG_8059.JPG"

CLASS_NAMES = [
    "Schraube-Zylinder_ganz",
    "Schraube-Zylinder_oben",
    "Schraube-Zylinder_unten",
    "Standard-Schraube",
    "Zylinder",
]

MODEL_SERVICE_NAME = "model-service"
PROMETHEUS_SERVICE_NAME = "prometheus"
GRAFANA_SERVICE_NAME = "grafana"


def get_bool(name: str, default: bool = False) -> bool:
    """Read one boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise RuntimeError(f"{name} must be true or false, got {value!r}")


def get_json(url: str, timeout: float = 10.0) -> dict:
    """Fetch and decode one JSON document."""
    with urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
        status = int(getattr(response, "status", 200))

    if not 200 <= status < 300:
        raise RuntimeError(f"GET {url} returned HTTP {status}")

    result = json.loads(payload)
    if not isinstance(result, dict):
        raise RuntimeError(f"GET {url} did not return a JSON object")

    return result


def get_text(url: str, timeout: float = 10.0) -> str:
    """Fetch one text response."""
    with urlopen(url, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
        status = int(getattr(response, "status", 200))

    if not 200 <= status < 300:
        raise RuntimeError(f"GET {url} returned HTTP {status}")

    return payload


def post_json(url: str, body: dict, timeout: float = 30.0) -> dict:
    """POST JSON and decode a successful JSON response."""
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
        status = int(getattr(response, "status", 200))

    if not 200 <= status < 300:
        raise RuntimeError(f"POST {url} returned HTTP {status}")

    result = json.loads(payload)
    if not isinstance(result, dict):
        raise RuntimeError(f"POST {url} did not return a JSON object")

    return result


def post_json_expect_status(
    url: str,
    body: dict,
    expected_status: int,
    timeout: float = 10.0,
) -> None:
    """POST JSON and require one expected non-success status."""
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            status = int(getattr(response, "status", 200))
    except HTTPError as exc:
        status = exc.code

    if status != expected_status:
        raise RuntimeError(
            f"POST {url} returned HTTP {status}, expected HTTP {expected_status}"
        )


def load_batch(image_path: Path) -> np.ndarray:
    """Load and normalize one image exactly like the SageMaker smoke test."""
    image = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def get_project_name(model_name: str) -> str:
    """Return the configured or default Docker Compose project name."""
    configured = (os.getenv("COMPOSE_PROJECT_NAME") or "").strip()
    return safe_resource_name(configured or f"mlops-{model_name}", max_length=48)


def get_compose_file(project_name: str) -> Path:
    """Return the generated Compose manifest for the running project."""
    return (
        REPO_ROOT
        / "model_deployment"
        / "deployments"
        / "docker_compose"
        / ".compose"
        / project_name
        / "compose.yaml"
    )


def get_running_services(project_name: str, compose_file: Path) -> set[str]:
    """Return running services for only the selected Compose project."""
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "-p",
            project_name,
            "-f",
            str(compose_file),
            "ps",
            "--services",
            "--status",
            "running",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def verify_running_services(
    project_name: str,
    compose_file: Path,
    enable_prometheus: bool,
    enable_grafana: bool,
) -> None:
    """Verify that only the expected Compose services are running."""
    if not compose_file.is_file():
        raise RuntimeError(f"Generated Compose file not found: {compose_file}")

    expected = {MODEL_SERVICE_NAME}
    if enable_prometheus:
        expected.add(PROMETHEUS_SERVICE_NAME)
    if enable_grafana:
        expected.add(GRAFANA_SERVICE_NAME)

    running = get_running_services(project_name, compose_file)
    if running != expected:
        raise RuntimeError(
            f"Unexpected running services for project {project_name}: "
            f"expected={sorted(expected)}, actual={sorted(running)}"
        )

    logger.info("Running Compose services verified: %s", ", ".join(sorted(running)))


def verify_health(model_base_url: str, model_name: str) -> dict:
    """Verify the model-service health endpoint."""
    payload = get_json(f"{model_base_url}/health")

    if payload.get("status") != "ok":
        raise RuntimeError(f"Model health endpoint did not report ok: {payload}")

    if payload.get("model") and payload.get("model") != model_name:
        raise RuntimeError(
            f"Model health response reports {payload.get('model')!r}, "
            f"expected {model_name!r}"
        )

    logger.info(
        "Model service healthy: model=%s version=%s",
        payload.get("model"),
        payload.get("version"),
    )
    return payload


def get_metric_sum(
    metrics_text: str, metric_name: str, default: float | None = None
) -> float:
    """Return the sum of all samples for one exact Prometheus metric name."""
    total = 0.0
    found = False

    for line in metrics_text.splitlines():
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        sample_token = stripped.split(None, 1)[0]
        sample_name = sample_token.split("{", 1)[0]

        if sample_name != metric_name:
            continue

        try:
            value = float(stripped.rsplit(None, 1)[1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"Could not parse metric sample: {stripped}") from exc

        total += value
        found = True

    if found:
        return total

    if default is not None:
        return default

    raise RuntimeError(f"Metric {metric_name!r} was not found in /metrics output")


def verify_metrics_endpoint(model_base_url: str) -> tuple[str, float, float]:
    """Verify metric definitions and model_up before generating inference samples.

    Prometheus metrics that have labels do not emit samples until a label-set is
    instantiated. Therefore request/error/latency samples are checked after the
    corresponding requests, not before them.
    """
    metrics = get_text(f"{model_base_url}/metrics")

    required_definitions = (
        "# HELP inference_requests_total",
        "# HELP inference_errors_total",
        "# HELP inference_duration_seconds",
        "# HELP model_up",
    )
    missing = [name for name in required_definitions if name not in metrics]
    if missing:
        raise RuntimeError(
            "/metrics is missing expected metric definitions: " + ", ".join(missing)
        )

    model_up = get_metric_sum(metrics, "model_up")
    if model_up < 1:
        raise RuntimeError(f"model_up did not report a loaded model: {model_up}")

    request_count = get_metric_sum(
        metrics,
        "inference_requests_total",
        default=0.0,
    )
    error_count = get_metric_sum(
        metrics,
        "inference_errors_total",
        default=0.0,
    )

    logger.info(
        "Model /metrics available: model_up=%.0f requests=%.0f errors=%.0f",
        model_up,
        request_count,
        error_count,
    )
    return metrics, request_count, error_count


def predict(model_base_url: str, batch: np.ndarray) -> list:
    """Send one real prediction request to the generic Compose endpoint."""
    payload = post_json(
        f"{model_base_url}/predict",
        {"inputs": batch.tolist()},
    )

    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise RuntimeError(f"Prediction response is invalid: {payload}")

    return predictions[0]


def verify_successful_prediction(
    model_base_url: str,
    image_path: Path,
    request_count_before: float,
) -> None:
    """Run one valid inference and verify request and latency metrics."""
    logger.info("Scoring %s against %s/predict", image_path.name, model_base_url)

    probabilities = predict(model_base_url, load_batch(image_path))
    array = np.asarray(probabilities)

    if array.size == 0:
        raise RuntimeError("Prediction response contains no values")

    index = int(np.argmax(array))
    confidence = float(array[index])
    label = CLASS_NAMES[index] if index < len(CLASS_NAMES) else f"index {index}"

    logger.info("Prediction: %s (%.1f%% confidence)", label, confidence * 100)

    metrics_after = get_text(f"{model_base_url}/metrics")

    request_count_after = get_metric_sum(
        metrics_after,
        "inference_requests_total",
        default=0.0,
    )
    if request_count_after < request_count_before + 1:
        raise RuntimeError(
            "Inference request counter did not increase after prediction: "
            f"before={request_count_before}, after={request_count_after}"
        )

    latency_count = get_metric_sum(
        metrics_after,
        "inference_duration_seconds_count",
        default=0.0,
    )
    if latency_count < 1:
        raise RuntimeError(
            "Inference latency histogram did not record the successful prediction"
        )

    logger.info(
        "Successful inference metrics verified: requests %.0f -> %.0f, latency_count=%.0f",
        request_count_before,
        request_count_after,
        latency_count,
    )


def verify_failed_prediction_metric(
    model_base_url: str,
    error_count_before: float,
) -> None:
    """Send one intentionally invalid request and verify the error counter."""
    post_json_expect_status(
        f"{model_base_url}/predict",
        {"inputs": []},
        expected_status=400,
    )

    metrics_after = get_text(f"{model_base_url}/metrics")
    error_count_after = get_metric_sum(
        metrics_after,
        "inference_errors_total",
        default=0.0,
    )

    if error_count_after < error_count_before + 1:
        raise RuntimeError(
            "Inference error counter did not increase after invalid request: "
            f"before={error_count_before}, after={error_count_after}"
        )

    logger.info(
        "Inference error metric verified: errors %.0f -> %.0f",
        error_count_before,
        error_count_after,
    )


def prometheus_target_is_up(payload: dict) -> bool:
    """Return whether bundled Prometheus reports model-service as an UP target."""
    if payload.get("status") != "success":
        return False

    targets = payload.get("data", {}).get("activeTargets", [])
    if not isinstance(targets, list):
        return False

    for target in targets:
        labels = {}
        labels.update(target.get("discoveredLabels", {}) or {})
        labels.update(target.get("labels", {}) or {})

        if str(labels.get("job", "")) != MODEL_SERVICE_NAME:
            continue

        scrape_url = str(target.get("scrapeUrl", ""))
        if scrape_url and "model-service:8080/metrics" not in scrape_url:
            continue

        if target.get("health") == "up":
            return True

    return False


def wait_for_prometheus(
    prometheus_base_url: str, timeout_seconds: float = 45.0
) -> None:
    """Verify bundled Prometheus readiness, scrape target, and model metric query."""
    deadline = time.monotonic() + timeout_seconds
    last_detail = "Prometheus not ready"

    while time.monotonic() < deadline:
        try:
            ready = get_text(f"{prometheus_base_url}/-/ready", timeout=5)
            if "ready" not in ready.lower():
                last_detail = f"unexpected readiness response: {ready!r}"
                time.sleep(2)
                continue

            targets = get_json(
                f"{prometheus_base_url}/api/v1/targets?state=active",
                timeout=5,
            )
            if not prometheus_target_is_up(targets):
                last_detail = "model-service target is not UP yet"
                time.sleep(2)
                continue

            query_url = f"{prometheus_base_url}/api/v1/query?" + urlencode(
                {"query": "model_up"}
            )
            query = get_json(query_url, timeout=5)
            result = query.get("data", {}).get("result", [])

            if query.get("status") != "success" or not result:
                last_detail = "model_up query has no result yet"
                time.sleep(2)
                continue

            logger.info(
                "Bundled Prometheus verified: ready, model-service target UP, "
                "model_up query available"
            )
            return

        except Exception as exc:
            last_detail = str(exc)
            time.sleep(2)

    raise RuntimeError(
        "Bundled Prometheus did not become operational within "
        f"{timeout_seconds:.0f}s: {last_detail}"
    )



def wait_for_external_prometheus(
    prometheus_base_url: str,
    model_name: str,
    timeout_seconds: float = 45.0,
) -> None:
    """Verify that an existing external Prometheus scrapes this model service."""
    deadline = time.monotonic() + timeout_seconds
    last_detail = "external Prometheus has not reported the model yet"
    query = f'model_up{{model="{model_name}"}}'

    while time.monotonic() < deadline:
        try:
            ready = get_text(f"{prometheus_base_url}/-/ready", timeout=5)
            if "ready" not in ready.lower():
                last_detail = f"unexpected readiness response: {ready!r}"
                time.sleep(2)
                continue

            query_url = f"{prometheus_base_url}/api/v1/query?" + urlencode(
                {"query": query}
            )
            payload = get_json(query_url, timeout=5)
            result = payload.get("data", {}).get("result", [])

            if payload.get("status") != "success" or not result:
                last_detail = f"{query} returned no samples"
                time.sleep(2)
                continue

            values = []
            for item in result:
                value = item.get("value", [])
                if len(value) >= 2:
                    values.append(float(value[1]))

            if any(value >= 1 for value in values):
                logger.info(
                    "External Prometheus verified: model=%s model_up=1",
                    model_name,
                )
                return

            last_detail = f"{query} returned values={values}"
            time.sleep(2)

        except Exception as exc:
            last_detail = str(exc)
            time.sleep(2)

    raise RuntimeError(
        "External Prometheus did not expose the deployed model metric within "
        f"{timeout_seconds:.0f}s: {last_detail}"
    )


def verify_external_grafana(grafana_base_url: str) -> None:
    """Verify that an existing external Grafana instance is healthy."""
    payload = get_json(f"{grafana_base_url}/api/health", timeout=10)

    if str(payload.get("database", "")).lower() != "ok":
        raise RuntimeError(
            f"External Grafana health endpoint did not report database=ok: {payload}"
        )

    logger.info(
        "External Grafana verified: version=%s database=%s",
        payload.get("version", "unknown"),
        payload.get("database"),
    )


def wait_for_grafana(grafana_base_url: str, timeout_seconds: float = 45.0) -> None:
    """Verify that bundled Grafana is running and its internal database is healthy."""
    deadline = time.monotonic() + timeout_seconds
    last_detail = "Grafana not ready"

    while time.monotonic() < deadline:
        try:
            payload = get_json(f"{grafana_base_url}/api/health", timeout=5)

            if str(payload.get("database", "")).lower() == "ok":
                logger.info(
                    "Bundled Grafana verified: version=%s database=%s",
                    payload.get("version", "unknown"),
                    payload.get("database"),
                )
                return

            last_detail = f"Grafana health response: {payload}"

        except Exception as exc:
            last_detail = str(exc)

        time.sleep(2)

    raise RuntimeError(
        "Bundled Grafana did not become operational within "
        f"{timeout_seconds:.0f}s: {last_detail}"
    )


def verify_grafana_dashboard(project_name: str) -> None:
    """Verify that the framework generated the provisioned Grafana dashboard file."""
    dashboard = (
        REPO_ROOT
        / "model_deployment"
        / "deployments"
        / "docker_compose"
        / ".compose"
        / project_name
        / "grafana"
        / "dashboards"
        / "operational-dashboard.json"
    )

    if not dashboard.is_file():
        raise RuntimeError(f"Provisioned Grafana dashboard not found: {dashboard}")

    payload = json.loads(dashboard.read_text(encoding="utf-8"))
    if payload.get("title") != "MLOps Blueprint - Operational Inference":
        raise RuntimeError(f"Unexpected Grafana dashboard definition: {dashboard}")

    logger.info("Grafana dashboard provisioning verified: %s", dashboard)


def main() -> int:
    """Run the Docker Compose integration smoke test."""
    load_dotenv(REPO_ROOT / ".env", override=False)

    model_name = os.getenv("MODEL_NAME", "champion_model")
    model_port = int(os.getenv("COMPOSE_MODEL_PORT", "8000"))
    prometheus_port = int(os.getenv("COMPOSE_PROMETHEUS_PORT", "9090"))
    grafana_port = int(os.getenv("COMPOSE_GRAFANA_PORT", "3000"))

    enable_prometheus = get_bool("COMPOSE_ENABLE_PROMETHEUS", False)
    enable_grafana = get_bool("COMPOSE_ENABLE_GRAFANA", False)

    # Test-only URLs for pre-existing monitoring infrastructure.
    external_prometheus_url = (
        os.getenv("TEST_EXTERNAL_PROMETHEUS_URL") or ""
    ).strip().rstrip("/")
    external_grafana_url = (
        os.getenv("TEST_EXTERNAL_GRAFANA_URL") or ""
    ).strip().rstrip("/")

    project_name = get_project_name(model_name)
    compose_file = get_compose_file(project_name)
    image_path = Path(os.getenv("IMAGE_PATH", str(DEFAULT_IMAGE)))

    model_base_url = f"http://127.0.0.1:{model_port}"
    prometheus_base_url = f"http://127.0.0.1:{prometheus_port}"
    grafana_base_url = f"http://127.0.0.1:{grafana_port}"

    if not image_path.is_file():
        logger.error("Image not found: %s", image_path)
        return 1

    try:
        logger.info(
            "Testing Docker Compose project=%s model=%s endpoint=%s",
            project_name,
            model_name,
            model_base_url,
        )

        verify_running_services(
            project_name,
            compose_file,
            enable_prometheus,
            enable_grafana,
        )

        verify_health(model_base_url, model_name)

        _, request_count_before, error_count_before = verify_metrics_endpoint(
            model_base_url
        )

        verify_successful_prediction(
            model_base_url,
            image_path,
            request_count_before,
        )

        verify_failed_prediction_metric(
            model_base_url,
            error_count_before,
        )

        if enable_prometheus:
            wait_for_prometheus(prometheus_base_url)
        elif external_prometheus_url:
            wait_for_external_prometheus(
                external_prometheus_url,
                model_name,
            )
        else:
            logger.info(
                "Bundled Prometheus disabled and TEST_EXTERNAL_PROMETHEUS_URL "
                "not set; external Prometheus verification skipped"
            )

        if enable_grafana:
            wait_for_grafana(grafana_base_url)
            verify_grafana_dashboard(project_name)
        elif external_grafana_url:
            verify_external_grafana(external_grafana_url)
        else:
            logger.info(
                "Bundled Grafana disabled and TEST_EXTERNAL_GRAFANA_URL "
                "not set; external Grafana verification skipped"
            )

        logger.info("Docker Compose integration smoke test PASSED")
        return 0

    except Exception:
        logger.exception("Docker Compose integration smoke test FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
