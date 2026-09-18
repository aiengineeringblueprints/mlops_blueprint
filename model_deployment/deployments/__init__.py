"""Public deployment-backend API and canonical adapter registry."""

from model_deployment.deployments.docker_compose import DockerComposeAdapter
from model_deployment.deployments.kubernetes import KubernetesKServeAdapter
from model_deployment.deployments.sagemaker import SageMakerAdapter

TARGET_ADAPTERS = {
    "aws_sagemaker": SageMakerAdapter(),
    "kubernetes": KubernetesKServeAdapter(),
    "docker_compose": DockerComposeAdapter(),
}

__all__ = [
    "DockerComposeAdapter",
    "KubernetesKServeAdapter",
    "SageMakerAdapter",
    "TARGET_ADAPTERS",
]
