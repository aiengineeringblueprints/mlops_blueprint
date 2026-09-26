"""Amazon CloudWatch operational monitoring for SageMaker deployments."""
from __future__ import annotations

from model_deployment import DeploymentError, exception_message
from model_deployment.config import DeploymentSettings


def configure_sagemaker_monitoring(
    settings: DeploymentSettings,
    model_name: str,
    endpoint: str,
    *,
    cw=None,
) -> str:
    """Create/update the SageMaker endpoint latency alarm in CloudWatch."""
    del model_name
    if cw is None:
        import boto3

        cw = boto3.client("cloudwatch", region_name=settings.aws_region)
    try:
        cw.put_metric_alarm(
            AlarmName=f"{endpoint}-high-latency",
            Namespace="AWS/SageMaker",
            MetricName="ModelLatency",
            Dimensions=[
                {"Name": "EndpointName", "Value": endpoint},
                {"Name": "VariantName", "Value": settings.sagemaker_variant_name},
            ],
            Statistic="Average",
            Period=settings.sagemaker_alarm_period,
            EvaluationPeriods=settings.sagemaker_evaluation_periods,
            Threshold=settings.sagemaker_latency_threshold_us,
            ComparisonOperator="GreaterThanThreshold",
            TreatMissingData="notBreaching",
        )
    except Exception as exc:
        raise DeploymentError(
            "MONITORING_CONFIGURATION_FAILED",
            f"Could not configure CloudWatch alarm for '{endpoint}': {exception_message(exc)}",
        ) from exc
    return f"CloudWatch alarm {endpoint}-high-latency"
