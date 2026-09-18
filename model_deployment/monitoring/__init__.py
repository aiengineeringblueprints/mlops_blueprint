"""Public operational-monitoring API shared by deployment backends."""

from model_deployment.monitoring.cloudwatch import configure_sagemaker_monitoring
from model_deployment.monitoring.prometheus import (
    fetch_prometheus_targets,
    wait_compose_prometheus_target,
    wait_prometheus_target,
)

__all__ = [
    "configure_sagemaker_monitoring",
    "fetch_prometheus_targets",
    "wait_compose_prometheus_target",
    "wait_prometheus_target",
]
