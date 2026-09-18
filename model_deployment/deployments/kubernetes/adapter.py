"""Unified Kubernetes/KServe deployment adapter."""
from __future__ import annotations

from model_deployment import ResolvedModel
from model_deployment.config import DeploymentSettings
from model_deployment.deployments.kubernetes.helpers import (
    configure_kserve_monitoring,
    deploy_kserve,
    preflight_kserve,
)


class KubernetesKServeAdapter:
    """Deploy to standard Kubernetes, K3s, or another compatible cluster via one API path."""

    name = "kserve"

    def preflight(self, settings: DeploymentSettings) -> None:
        """Validate Kubernetes, KServe, namespace, registry, and monitoring prerequisites."""
        preflight_kserve(settings)

    def deploy(
        self,
        settings: DeploymentSettings,
        resolved: ResolvedModel,
        image: str,
    ) -> str:
        """Deploy the resolved model as one stable KServe InferenceService."""
        return deploy_kserve(settings, resolved, image)

    def configure_monitoring(
        self,
        settings: DeploymentSettings,
        model_name: str,
        deployment_name: str,
    ) -> str:
        """Configure PodMonitor and optionally verify the Prometheus target."""
        return configure_kserve_monitoring(settings, model_name, deployment_name)
