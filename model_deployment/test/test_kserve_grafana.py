"""Manual Grafana configuration smoke test for Kubernetes/KServe monitoring."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)

# Port-forward Grafana to 13000, or override TEST_GRAFANA_URL.
GRAFANA_BASE_URL = os.getenv("TEST_GRAFANA_URL", "http://127.0.0.1:13000").rstrip("/")

GRAFANA_USER = os.getenv("GRAFANA_USER", "admin")
GRAFANA_PASSWORD = os.getenv("GRAFANA_PASSWORD", "")

EXPECTED_DATASOURCE_TYPE = "prometheus"
EXPECTED_PROMETHEUS_HOST = os.getenv(
    "TEST_GRAFANA_PROMETHEUS_HOST",
    "prometheus-kube-prometheus-prometheus.monitoring:9090",
)


def auth_headers() -> dict[str, str]:
    """Build Grafana Basic Authentication headers."""
    if not GRAFANA_PASSWORD:
        raise RuntimeError(
            "GRAFANA_PASSWORD is not set. Export the Grafana admin password first."
        )

    token = base64.b64encode(
        f"{GRAFANA_USER}:{GRAFANA_PASSWORD}".encode("utf-8")
    ).decode("ascii")

    return {
        "Authorization": f"Basic {token}",
        "Accept": "application/json",
    }


def get_json(
    url: str,
    *,
    authenticated: bool = False,
    timeout: float = 10.0,
) -> dict | list:
    """Fetch and decode one JSON response."""
    headers = auth_headers() if authenticated else {"Accept": "application/json"}

    request = Request(url, headers=headers, method="GET")

    with urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
        status = int(getattr(response, "status", 200))

    if not 200 <= status < 300:
        raise RuntimeError(f"GET {url} returned HTTP {status}")

    result = json.loads(payload)

    if not isinstance(result, (dict, list)):
        raise RuntimeError(f"GET {url} returned unexpected JSON")

    return result


def verify_grafana_health() -> None:
    """Verify Grafana runtime health."""
    payload = get_json(f"{GRAFANA_BASE_URL}/api/health")

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected Grafana health response: {payload}")

    if str(payload.get("database", "")).lower() != "ok":
        raise RuntimeError(f"Grafana database is not healthy: {payload}")

    logger.info(
        "Grafana health PASSED: version=%s database=%s",
        payload.get("version", "unknown"),
        payload.get("database"),
    )


def get_prometheus_datasource() -> dict:
    """Find and validate Grafana's Prometheus datasource configuration."""
    payload = get_json(
        f"{GRAFANA_BASE_URL}/api/datasources",
        authenticated=True,
    )

    if not isinstance(payload, list):
        raise RuntimeError(
            f"Grafana /api/datasources returned unexpected payload: {payload}"
        )

    for datasource in payload:
        if str(datasource.get("type", "")).lower() != EXPECTED_DATASOURCE_TYPE:
            continue

        uid = str(datasource.get("uid", ""))
        url = str(datasource.get("url", ""))
        access = str(datasource.get("access", ""))

        if not uid:
            raise RuntimeError(
                f"Prometheus datasource has no UID: {datasource}"
            )

        if EXPECTED_PROMETHEUS_HOST not in url:
            raise RuntimeError(
                "Grafana Prometheus datasource points to an unexpected URL: "
                f"{url!r}"
            )

        logger.info(
            "Prometheus datasource PASSED: name=%s uid=%s url=%s access=%s",
            datasource.get("name"),
            uid,
            url,
            access or "<not reported>",
        )

        return datasource

    raise RuntimeError("Grafana has no Prometheus datasource configured")


def check_optional_datasource_health(datasource: dict) -> None:
    """Check datasource health when the Prometheus plugin exposes that endpoint.

    Grafana does not require every datasource plugin to implement the health
    endpoint. HTTP 404 therefore means unsupported, not failed.
    """
    uid = datasource.get("uid")

    if not uid:
        raise RuntimeError("Prometheus datasource UID is missing")

    url = f"{GRAFANA_BASE_URL}/api/datasources/uid/{uid}/health"
    request = Request(url, headers=auth_headers(), method="GET")

    try:
        with urlopen(request, timeout=10) as response:
            payload = response.read().decode("utf-8")
            status = int(getattr(response, "status", 200))

    except HTTPError as exc:
        if exc.code == 404:
            logger.info(
                "Prometheus datasource health endpoint SKIPPED: "
                "plugin returned HTTP 404 (health check not implemented/exposed)"
            )
            return

        raise

    if not 200 <= status < 300:
        raise RuntimeError(
            f"Datasource health endpoint returned HTTP {status}"
        )

    result = json.loads(payload)

    if not isinstance(result, dict):
        raise RuntimeError(
            f"Unexpected datasource health response: {result}"
        )

    health_status = str(result.get("status", "")).upper()

    if health_status not in {"OK", "SUCCESS"}:
        raise RuntimeError(
            f"Prometheus datasource health check failed: {result}"
        )

    logger.info(
        "Prometheus datasource health PASSED: status=%s message=%s",
        health_status,
        result.get("message", ""),
    )


def main() -> int:
    """Run Grafana runtime and datasource-configuration validation."""
    try:
        logger.info("Testing Grafana endpoint: %s", GRAFANA_BASE_URL)

        verify_grafana_health()
        datasource = get_prometheus_datasource()
        check_optional_datasource_health(datasource)

        logger.info("KServe Grafana configuration smoke test PASSED")
        return 0

    except Exception:
        logger.exception("KServe Grafana configuration smoke test FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
