"""Docker Compose deployment adapter."""
from __future__ import annotations

from model_deployment import ResolvedModel
from model_deployment.config import DeploymentSettings
from model_deployment.deployments.docker_compose.helpers import (
    configure_compose_monitoring,
    deploy_compose,
    preflight_compose,
)


class DockerComposeAdapter:
    """Deploy the shared model server on a standalone Docker Compose host."""

    name = "docker_compose"

    def preflight(self, settings: DeploymentSettings) -> None:
        """Check Docker daemon, Compose v2, templates, and runtime prerequisites."""
        preflight_compose(settings)

    def deploy(
        self,
        settings: DeploymentSettings,
        resolved: ResolvedModel,
        image: str,
    ) -> str:
        """Start the isolated Compose project and wait for model readiness."""
        return deploy_compose(settings, resolved, image)

    def configure_monitoring(
        self,
        settings: DeploymentSettings,
        model_name: str,
        deployment_name: str,
    ) -> str:
        """Verify selected bundled monitoring while leaving external systems untouched."""
        return configure_compose_monitoring(settings, model_name, deployment_name)
