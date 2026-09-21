"""Prepare deterministic serving images from MLflow or standalone local model artifacts."""
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
import shutil
import subprocess

from model_deployment import DeploymentError, safe_resource_name

logger = logging.getLogger(__name__)
CONTAINER_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CONTAINER_DIR.parent
DOCKERFILE = CONTAINER_DIR / "Dockerfile"
PROMETHEUS_METRICS_TARGETS = frozenset({"kubernetes", "docker_compose"})
SUPPORTED_IMAGE_TARGETS = frozenset({"aws_sagemaker", "kubernetes", "docker_compose"})


def image_tag(model_name: str, version: str, docker_registry: str = "") -> str:
    """Return the deterministic versioned image tag."""
    prefix = f"{docker_registry.rstrip('/')}/" if docker_registry else ""
    return f"{prefix}{safe_resource_name(model_name)}:{version}"


def prepare_ecr(tag: str, docker_registry: str, build_env: dict) -> None:
    """Authenticate Docker and ensure the exact ECR repository exists via boto3."""
    try:
        import boto3
    except ImportError as exc:
        raise DeploymentError("IMAGE_PUSH_FAILED", "boto3 is required for ECR") from exc

    registry_host = docker_registry.split("/", 1)[0]
    host_parts = registry_host.split(".")
    if len(host_parts) < 6 or host_parts[1:3] != ["dkr", "ecr"]:
        raise DeploymentError(
            "IMAGE_PUSH_FAILED",
            f"AWS_DOCKER_REGISTRY '{docker_registry}' is not a recognized private ECR registry URI.",
        )
    region = host_parts[3]
    repository_name = tag.split("/", 1)[1].rsplit(":", 1)[0]
    ecr = boto3.client("ecr", region_name=region)

    try:
        ecr.describe_repositories(repositoryNames=[repository_name])
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code not in {"RepositoryNotFoundException", "ResourceNotFoundException"}:
            raise DeploymentError(
                "IMAGE_PUSH_FAILED",
                f"Could not inspect ECR repository '{repository_name}': {exc}",
            ) from exc
        try:
            ecr.create_repository(repositoryName=repository_name)
        except Exception as create_exc:
            raise DeploymentError(
                "IMAGE_PUSH_FAILED",
                f"Could not create ECR repository '{repository_name}': {create_exc}",
            ) from create_exc

    try:
        auth_data = ecr.get_authorization_token()["authorizationData"][0]
        decoded = base64.b64decode(auth_data["authorizationToken"]).decode("utf-8")
        username, password = decoded.split(":", 1)
        subprocess.run(
            ["docker", "login", "--username", username, "--password-stdin", registry_host],
            input=password,
            check=True,
            text=True,
            env=build_env,
        )
    except DeploymentError:
        raise
    except Exception as exc:
        raise DeploymentError(
            "IMAGE_PUSH_FAILED",
            f"Could not authenticate Docker to ECR host '{registry_host}': {exc}",
        ) from exc


def prometheus_metrics_enabled_for_target(target: str) -> bool:
    """Return whether the canonical deployment backend requires Prometheus instrumentation."""
    if target not in SUPPORTED_IMAGE_TARGETS:
        raise DeploymentError(
            "IMAGE_BUILD_FAILED",
            f"Unsupported deployment target for image build: {target}",
        )
    return target in PROMETHEUS_METRICS_TARGETS


def build_docker_command(
    tag: str,
    model_name: str,
    version: str,
    target: str,
    *,
    platform: str | None = None,
    source_type: str = "mlflow",
    model_format: str = "mlflow",
    model_file: str | None = None,
) -> list[str]:
    """Build the Docker CLI command used for the selected deployment backend."""
    enable_prometheus_metrics = prometheus_metrics_enabled_for_target(target)
    command = ["docker", "build", "-f", str(DOCKERFILE)]
    if platform:
        command.extend(["--platform", platform])
    command.extend(
        [
            "-t",
            tag,
            "--build-arg",
            f"MODEL_NAME={model_name}",
            "--build-arg",
            f"MODEL_VERSION={version}",
            "--build-arg",
            f"ENABLE_PROMETHEUS_METRICS={'true' if enable_prometheus_metrics else 'false'}",
        ]
    )

    if source_type != "mlflow" or model_format != "mlflow" or model_file:
        command.extend(
            [
                "--build-arg",
                f"MODEL_SOURCE={source_type}",
                "--build-arg",
                f"MODEL_FORMAT={model_format}",
                "--build-arg",
                f"MODEL_FILE={model_file or ''}",
            ]
        )
    command.append(str(PACKAGE_ROOT))
    return command


def _stage_mlflow_artifact(model_uri: str, artifact_dir: Path, download_dir: Path) -> None:
    """Download and stage the existing MLflow artifact exactly as before."""
    try:
        import mlflow
    except ImportError as exc:
        raise DeploymentError("IMAGE_BUILD_FAILED", "MLflow is not installed") from exc

    logger.info("Downloading resolved MLflow artifact '%s'", model_uri)
    downloaded = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri, dst_path=str(download_dir)
    )
    downloaded_path = Path(downloaded).resolve()
    if not downloaded_path.exists():
        raise DeploymentError(
            "IMAGE_BUILD_FAILED",
            f"MLflow returned a missing downloaded artifact path: {downloaded_path}",
        )
    if downloaded_path.is_dir():
        shutil.copytree(downloaded_path, artifact_dir, dirs_exist_ok=True)
    else:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded_path, artifact_dir / downloaded_path.name)
    if not (artifact_dir / "MLmodel").is_file():
        raise DeploymentError(
            "IMAGE_BUILD_FAILED",
            f"Resolved MLflow artifact '{model_uri}' does not contain MLmodel at its root.",
        )


def _stage_local_artifact(model_path: str, artifact_dir: Path) -> str:
    """Copy one validated standalone model file into the shared Docker build context."""
    source = Path(model_path).expanduser().resolve()
    if not source.is_file():
        raise DeploymentError(
            "IMAGE_BUILD_FAILED",
            f"Standalone local model file is missing or unreadable: {source}",
        )

    generated_roots = ((CONTAINER_DIR / "model_artifact").resolve(), (CONTAINER_DIR / ".model_artifact_download").resolve())
    if any(source == root or root in source.parents for root in generated_roots):
        raise DeploymentError(
            "IMAGE_BUILD_FAILED",
            "LOCAL_MODEL_PATH must not point inside the deployment builder's generated artifact directories.",
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    destination = artifact_dir / source.name
    shutil.copy2(source, destination)
    logger.info("Staged standalone local model '%s'", source)
    return destination.name


def build_image(
    model_name: str,
    version: str,
    model_uri: str,
    target: str,
    docker_registry: str = "",
    *,
    platform: str | None = None,
    push: bool = True,
    image_name: str | None = None,
    source_type: str = "mlflow",
    model_format: str = "mlflow",
) -> str:
    """Stage the resolved model, build its serving image, and optionally push it."""
    enable_prometheus_metrics = prometheus_metrics_enabled_for_target(target)
    tag = image_tag(image_name or model_name, version, docker_registry)
    artifact_dir = CONTAINER_DIR / "model_artifact"
    download_dir = CONTAINER_DIR / ".model_artifact_download"
    shutil.rmtree(artifact_dir, ignore_errors=True)
    shutil.rmtree(download_dir, ignore_errors=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    stage = "IMAGE_BUILD_FAILED"
    model_file: str | None = None

    try:
        if source_type == "mlflow":
            _stage_mlflow_artifact(model_uri, artifact_dir, download_dir)
        elif source_type == "local":
            model_file = _stage_local_artifact(model_uri, artifact_dir)
        else:
            raise DeploymentError(
                "IMAGE_BUILD_FAILED",
                f"Unsupported model source type for image preparation: {source_type!r}",
            )

        build_env = os.environ.copy()
        build_env["DOCKER_BUILDKIT"] = "1"
        command = build_docker_command(
            tag,
            model_name,
            version,
            target,
            platform=platform,
            source_type=source_type,
            model_format=model_format,
            model_file=model_file,
        )
        logger.info(
            "Building Docker image '%s'%s (model_source=%s model_format=%s prometheus_metrics=%s)",
            tag,
            f" for platform {platform}" if platform else " using Docker host/default platform",
            source_type,
            model_format,
            "enabled" if enable_prometheus_metrics else "disabled",
        )
        subprocess.run(command, check=True, env=build_env)

        if push and docker_registry:
            stage = "IMAGE_PUSH_FAILED"
            if ".dkr.ecr." in docker_registry:
                prepare_ecr(tag, docker_registry, build_env)
            logger.info("Pushing Docker image '%s'", tag)
            subprocess.run(["docker", "push", tag], check=True, env=build_env)
    except DeploymentError:
        raise
    except subprocess.CalledProcessError as exc:
        if stage == "IMAGE_BUILD_FAILED" and platform:
            message = (
                f"Docker build failed with exit code {exc.returncode} for requested platform "
                f"'{platform}'. Ensure Docker/BuildKit supports that platform (and emulation, "
                "when cross-building)."
            )
        else:
            message = f"Docker/registry command failed with exit code {exc.returncode}"
        raise DeploymentError(stage, message) from exc
    except Exception as exc:
        raise DeploymentError(stage, f"Could not prepare image '{tag}': {exc}") from exc
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)
        shutil.rmtree(download_dir, ignore_errors=True)
    return tag
