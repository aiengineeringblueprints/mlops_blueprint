"""Docker Compose deployment helpers for a standalone Docker host."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Callable, Optional
from urllib.request import urlopen

from jinja2 import Template
import yaml

from model_deployment import (
    DeploymentError,
    ResolvedModel,
    exception_message,
    safe_resource_name,
)
from model_deployment.config import DeploymentSettings
from model_deployment.monitoring import wait_compose_prometheus_target

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[1]
TEMPLATES_DIR = HERE / "templates"
RUNTIME_ROOT = HERE / ".compose"
DASHBOARD_SOURCE = PACKAGE_ROOT / "monitoring" / "grafana" / "operational-dashboard.json"
PROMETHEUS_IMAGE = "prom/prometheus:v3.13.1"
GRAFANA_IMAGE = "grafana/grafana:13.2.1"


def compose_project_name(settings: DeploymentSettings, model_name: str) -> str:
    """Return the configured or deterministic Docker Compose project name."""
    configured = (settings.compose_project_name or "").strip()
    return safe_resource_name(configured or f"mlops-{model_name}", max_length=48)


def compose_runtime_dir(project_name: str) -> Path:
    """Return the isolated generated-files directory for one Compose project."""
    return RUNTIME_ROOT / safe_resource_name(project_name, max_length=48)


def build_compose_command(project_name: str, compose_file: Path) -> list[str]:
    """Return the scoped Docker Compose v2 command prefix."""
    return ["docker", "compose", "-p", project_name, "-f", str(compose_file)]


def load_template(name: str) -> Template:
    """Load one backend-owned Docker Compose template."""
    path = TEMPLATES_DIR / name
    if not path.is_file():
        raise DeploymentError(
            "CONFIGURATION_INVALID",
            f"Docker Compose template is missing: {path}",
        )
    return Template(path.read_text(encoding="utf-8"), trim_blocks=True, lstrip_blocks=True)


def render_compose_manifest(
    settings: DeploymentSettings,
    resolved: ResolvedModel,
    image: str,
) -> dict:
    """Render the selected model/Prometheus/Grafana services from the Compose template."""
    rendered = load_template("compose.yaml.j2").render(
        image=image,
        model_name=resolved.name,
        model_version=resolved.version,
        model_port=settings.compose_model_port,
        model_platform=settings.docker_platform,
        enable_prometheus=settings.compose_enable_prometheus,
        enable_grafana=settings.compose_enable_grafana,
        prometheus_image=PROMETHEUS_IMAGE,
        prometheus_port=settings.compose_prometheus_port,
        grafana_image=GRAFANA_IMAGE,
        grafana_port=settings.compose_grafana_port,
    )
    manifest = yaml.safe_load(rendered)
    if not isinstance(manifest, dict) or "services" not in manifest:
        raise DeploymentError(
            "CONFIGURATION_INVALID",
            "Rendered Docker Compose template does not contain a services mapping.",
        )
    return manifest


def render_compose_prometheus_config() -> dict:
    """Render the bundled Prometheus scrape configuration for model-service:/metrics."""
    rendered = load_template("prometheus.yml.j2").render()
    config = yaml.safe_load(rendered)
    if not isinstance(config, dict) or "scrape_configs" not in config:
        raise DeploymentError(
            "MONITORING_CONFIGURATION_FAILED",
            "Rendered Prometheus configuration is invalid.",
        )
    return config


def write_grafana_provisioning(root: Path, settings: DeploymentSettings) -> None:
    """Provision bundled Grafana with either bundled or external Prometheus as datasource."""
    datasource_url = (
        "http://prometheus:9090"
        if settings.compose_enable_prometheus
        else settings.compose_external_prometheus_url
    )
    if not datasource_url:
        raise DeploymentError(
            "MONITORING_CONFIGURATION_FAILED",
            "Grafana requires bundled Prometheus or COMPOSE_EXTERNAL_PROMETHEUS_URL.",
        )

    datasource_dir = root / "grafana" / "provisioning" / "datasources"
    dashboard_provider_dir = root / "grafana" / "provisioning" / "dashboards"
    dashboards_dir = root / "grafana" / "dashboards"
    datasource_dir.mkdir(parents=True, exist_ok=True)
    dashboard_provider_dir.mkdir(parents=True, exist_ok=True)
    dashboards_dir.mkdir(parents=True, exist_ok=True)

    datasource = {
        "apiVersion": 1,
        "datasources": [
            {
                "name": "Prometheus",
                "type": "prometheus",
                "access": "proxy",
                "url": datasource_url,
                "isDefault": True,
                "editable": False,
            }
        ],
    }
    (datasource_dir / "prometheus.yml").write_text(
        yaml.safe_dump(datasource, sort_keys=False),
        encoding="utf-8",
    )
    dashboard_provider = {
        "apiVersion": 1,
        "providers": [
            {
                "name": "MLOps Blueprint",
                "orgId": 1,
                "folder": "MLOps Blueprint",
                "type": "file",
                "disableDeletion": True,
                "editable": False,
                "options": {"path": "/var/lib/grafana/dashboards"},
            }
        ],
    }
    (dashboard_provider_dir / "dashboard.yml").write_text(
        yaml.safe_dump(dashboard_provider, sort_keys=False),
        encoding="utf-8",
    )

    if not DASHBOARD_SOURCE.is_file():
        raise DeploymentError(
            "MONITORING_CONFIGURATION_FAILED",
            f"Grafana dashboard asset is missing: {DASHBOARD_SOURCE}",
        )
    json.loads(DASHBOARD_SOURCE.read_text(encoding="utf-8"))
    shutil.copy2(DASHBOARD_SOURCE, dashboards_dir / "operational-dashboard.json")


def write_compose_files(
    settings: DeploymentSettings,
    resolved: ResolvedModel,
    image: str,
    *,
    runtime_root: Optional[Path] = None,
) -> Path:
    """Generate only the Compose and monitoring files required by the selected services."""
    project = compose_project_name(settings, resolved.name)
    root = (runtime_root or RUNTIME_ROOT) / project
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    compose_file = root / "compose.yaml"
    compose_file.write_text(
        yaml.safe_dump(render_compose_manifest(settings, resolved, image), sort_keys=False),
        encoding="utf-8",
    )

    if settings.compose_enable_prometheus:
        (root / "prometheus.yml").write_text(
            yaml.safe_dump(render_compose_prometheus_config(), sort_keys=False),
            encoding="utf-8",
        )

    if settings.compose_enable_grafana:
        write_grafana_provisioning(root, settings)

    return compose_file


def preflight_compose(
    settings: DeploymentSettings,
    *,
    run_cmd: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    runtime_root: Optional[Path] = None,
) -> None:
    """Verify Docker, Compose v2, templates, dashboard assets, and runtime writability."""
    try:
        run_cmd(["docker", "info"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise DeploymentError(
            "PREREQUISITE_UNREACHABLE",
            "Docker daemon is not available to the deployment host: "
            f"{exception_message(exc)}",
        ) from exc
    try:
        run_cmd(["docker", "compose", "version"], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise DeploymentError(
            "PREREQUISITE_MISSING",
            "Docker Compose v2 plugin is not available. Install/enable 'docker compose': "
            f"{exception_message(exc)}",
        ) from exc

    load_template("compose.yaml.j2")
    if settings.compose_enable_prometheus:
        load_template("prometheus.yml.j2")
    if settings.compose_enable_grafana and not DASHBOARD_SOURCE.is_file():
        raise DeploymentError(
            "PREREQUISITE_MISSING",
            f"Grafana dashboard asset is missing: {DASHBOARD_SOURCE}",
        )

    root = runtime_root or RUNTIME_ROOT
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise DeploymentError(
            "PREREQUISITE_MISSING",
            f"Compose runtime directory is not writable: {root}: {exception_message(exc)}",
        ) from exc


def get_running_services(
    project_name: str,
    compose_file: Path,
    *,
    run_cmd: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> set[str]:
    """Return running service names for one scoped Compose project."""
    completed = run_cmd(
        build_compose_command(project_name, compose_file)
        + ["ps", "--services", "--status", "running"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def wait_compose_service_running(
    settings: DeploymentSettings,
    project_name: str,
    compose_file: Path,
    service_name: str,
    *,
    run_cmd: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Wait until one selected Compose service is reported as running."""
    deadline = monotonic_fn() + settings.readiness_timeout_seconds
    last_detail = f"{service_name} is not running"
    while monotonic_fn() < deadline:
        try:
            running = get_running_services(project_name, compose_file, run_cmd=run_cmd)
            if service_name in running:
                return
            last_detail = f"running services: {', '.join(sorted(running)) or 'none'}"
        except Exception as exc:
            last_detail = exception_message(exc)
        sleep_fn(settings.poll_interval_seconds)
    raise DeploymentError(
        "DEPLOYMENT_TIMEOUT",
        f"Docker Compose service '{service_name}' did not become ready within "
        f"{settings.readiness_timeout_seconds}s ({last_detail}).",
    )


def wait_compose_model_ready(
    settings: DeploymentSettings,
    project_name: str,
    compose_file: Path,
    *,
    run_cmd: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    urlopen_fn=urlopen,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Wait until model-service is running and its /health endpoint succeeds."""
    deadline = monotonic_fn() + settings.readiness_timeout_seconds
    endpoint = f"http://127.0.0.1:{settings.compose_model_port}/health"
    last_detail = "model-service not yet ready"
    while monotonic_fn() < deadline:
        try:
            running = get_running_services(project_name, compose_file, run_cmd=run_cmd)
            if "model-service" not in running:
                last_detail = "model-service container is not running"
            else:
                with urlopen_fn(endpoint, timeout=3) as response:
                    if 200 <= int(getattr(response, "status", 200)) < 300:
                        return
                    last_detail = f"/health returned HTTP {getattr(response, 'status', '?')}"
        except Exception as exc:
            last_detail = exception_message(exc)
        sleep_fn(settings.poll_interval_seconds)
    raise DeploymentError(
        "DEPLOYMENT_TIMEOUT",
        f"Docker Compose model service did not become ready within "
        f"{settings.readiness_timeout_seconds}s ({last_detail}).",
    )


def deploy_compose(
    settings: DeploymentSettings,
    resolved: ResolvedModel,
    image: str,
    *,
    run_cmd: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    runtime_root: Optional[Path] = None,
    wait_ready: bool = True,
) -> str:
    """Generate, start, and verify the selected Docker Compose services."""
    project = compose_project_name(settings, resolved.name)
    compose_file = write_compose_files(settings, resolved, image, runtime_root=runtime_root)
    try:
        run_cmd(
            build_compose_command(project, compose_file) + ["up", "-d", "--remove-orphans"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise DeploymentError(
            "DEPLOYMENT_SUBMISSION_FAILED",
            f"Could not start Docker Compose project '{project}': {exception_message(exc)}",
        ) from exc
    if wait_ready:
        wait_compose_model_ready(settings, project, compose_file, run_cmd=run_cmd)
    return project


def compose_monitoring_description(settings: DeploymentSettings) -> str:
    """Describe the selected bundled/external Docker Compose monitoring topology."""
    if settings.compose_enable_prometheus and settings.compose_enable_grafana:
        return "bundled Prometheus + bundled Grafana"
    if settings.compose_enable_prometheus:
        return "bundled Prometheus + external/not-bundled Grafana"
    if settings.compose_enable_grafana:
        return "external Prometheus + bundled Grafana"
    return "external/no bundled Prometheus + external/no bundled Grafana"


def configure_compose_monitoring(
    settings: DeploymentSettings,
    model_name: str,
    project_name: str,
    *,
    run_cmd: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str:
    """Verify only bundled monitoring services while leaving external systems untouched."""
    del model_name
    compose_file = compose_runtime_dir(project_name) / "compose.yaml"

    if settings.compose_enable_prometheus:
        wait_compose_service_running(
            settings,
            project_name,
            compose_file,
            "prometheus",
            run_cmd=run_cmd,
        )
        wait_compose_prometheus_target(settings, project_name)

    if settings.compose_enable_grafana:
        wait_compose_service_running(
            settings,
            project_name,
            compose_file,
            "grafana",
            run_cmd=run_cmd,
        )

    description = compose_monitoring_description(settings)
    if not settings.compose_enable_prometheus and settings.compose_external_prometheus_url:
        return (
            f"{description}; external Prometheus is configured at "
            f"{settings.compose_external_prometheus_url} and is not administered by the framework"
        )
    if not settings.compose_enable_prometheus and not settings.compose_enable_grafana:
        return (
            f"{description}; model /metrics remains exposed for organization-managed monitoring"
        )
    return f"Docker Compose monitoring ready: {description}"


def teardown_compose_project(
    project_name: str,
    *,
    execute: bool = False,
    run_cmd: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    runtime_root: Optional[Path] = None,
) -> str:
    """Stop only the named Compose project without deleting unrelated images or volumes."""
    root = (runtime_root or RUNTIME_ROOT) / safe_resource_name(project_name, max_length=48)
    compose_file = root / "compose.yaml"
    if not execute:
        return f"DRY-RUN: docker compose -p {project_name} down using {compose_file}"
    if not compose_file.is_file():
        raise DeploymentError(
            "TEARDOWN_FAILED",
            f"Compose definition for project '{project_name}' was not found at {compose_file}.",
        )
    try:
        run_cmd(
            build_compose_command(project_name, compose_file) + ["down"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise DeploymentError(
            "TEARDOWN_FAILED",
            f"Could not stop Docker Compose project '{project_name}': {exception_message(exc)}",
        ) from exc
    return (
        f"Stopped Docker Compose project {project_name}. "
        "Images and named volumes were preserved."
    )
