"""AWS SageMaker deployment adapter."""
from __future__ import annotations

from model_deployment import ResolvedModel
from model_deployment.config import DeploymentSettings
from model_deployment.deployments.sagemaker.helpers import reconcile_sagemaker_endpoint
from model_deployment.monitoring import configure_sagemaker_monitoring


class SageMakerAdapter:
    """Deploy a registered model to Amazon SageMaker."""

    name = "sagemaker"

    def preflight(self, settings: DeploymentSettings) -> None:
        """Use centralized validation and AWS API operations as backend preflight."""
        del settings

    def deploy(
        self,
        settings: DeploymentSettings,
        resolved: ResolvedModel,
        image: str,
    ) -> str:
        """Reconcile the stable SageMaker endpoint for the resolved model version."""
        return reconcile_sagemaker_endpoint(settings, resolved, image)

    def configure_monitoring(
        self,
        settings: DeploymentSettings,
        model_name: str,
        deployment_name: str,
    ) -> str:
        """Configure CloudWatch operational monitoring for the endpoint."""
        return configure_sagemaker_monitoring(settings, model_name, deployment_name)
