"""Public API for the Amazon SageMaker deployment backend."""

from model_deployment.deployments.sagemaker.adapter import SageMakerAdapter
from model_deployment.deployments.sagemaker.helpers import teardown_sagemaker

__all__ = ["SageMakerAdapter", "teardown_sagemaker"]
