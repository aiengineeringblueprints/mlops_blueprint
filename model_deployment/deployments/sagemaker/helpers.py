"""Amazon SageMaker deployment helpers."""
from __future__ import annotations

import logging
import time
from typing import Optional

from model_deployment import (
    DeploymentError,
    ResolvedModel,
    short_fingerprint,
    stable_deployment_name,
    versioned_artifact_name,
)
from model_deployment.config import DeploymentSettings

logger = logging.getLogger(__name__)


def get_client_error_code(exc: Exception) -> Optional[str]:
    """Return the AWS error code from a boto-style exception when available."""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return response.get("Error", {}).get("Code")
    return None


def is_sagemaker_not_found(exc: Exception) -> bool:
    """Return whether SageMaker reports a missing resource."""
    return get_client_error_code(exc) in {"ValidationException", "ResourceNotFound"}


def wait_sagemaker_ready(
    sm,
    endpoint_name: str,
    *,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> None:
    """Wait until a SageMaker endpoint is operational or fail with its backend status."""
    deadline = time.monotonic() + timeout_seconds
    last_status = "Unknown"
    while time.monotonic() < deadline:
        try:
            description = sm.describe_endpoint(EndpointName=endpoint_name)
        except Exception as exc:
            raise DeploymentError(
                "DEPLOYMENT_FAILED",
                f"Could not inspect SageMaker endpoint '{endpoint_name}': {exc}",
            ) from exc
        last_status = description.get("EndpointStatus", "Unknown")
        if last_status == "InService":
            return
        if last_status in {"Failed", "OutOfService"}:
            reason = description.get("FailureReason", "No failure reason returned")
            raise DeploymentError(
                "DEPLOYMENT_FAILED",
                f"SageMaker endpoint '{endpoint_name}' entered {last_status}: {reason}",
            )
        time.sleep(poll_interval_seconds)
    raise DeploymentError(
        "DEPLOYMENT_TIMEOUT",
        f"SageMaker endpoint '{endpoint_name}' did not become InService within "
        f"{timeout_seconds}s (last status: {last_status}).",
    )


def reconcile_sagemaker_endpoint(
    settings: DeploymentSettings,
    resolved: ResolvedModel,
    image: str,
    *,
    sm=None,
) -> str:
    """Create/reuse versioned SageMaker artifacts and reconcile one stable endpoint."""
    if sm is None:
        import boto3

        sm = boto3.client("sagemaker", region_name=settings.aws_region)

    endpoint_name = stable_deployment_name(resolved.name, "aws_sagemaker")
    model_fingerprint = short_fingerprint(image, settings.sagemaker_role_arn)
    aws_model_name = versioned_artifact_name(
        resolved.name, resolved.version, f"m-{model_fingerprint}"
    )
    config_fingerprint = short_fingerprint(
        aws_model_name,
        settings.sagemaker_instance_type,
        settings.sagemaker_instance_count,
        settings.sagemaker_variant_name,
    )
    config_name = versioned_artifact_name(
        resolved.name, resolved.version, f"cfg-{config_fingerprint}"
    )

    try:
        existing_model = sm.describe_model(ModelName=aws_model_name)
        existing_image = existing_model.get("PrimaryContainer", {}).get("Image")
        if existing_image and existing_image != image:
            raise DeploymentError(
                "DEPLOYMENT_CONFLICT",
                f"SageMaker model '{aws_model_name}' already exists with a different image. "
                "Registered versions are treated as immutable.",
            )
        logger.info("Reusing SageMaker Model '%s'", aws_model_name)
    except DeploymentError:
        raise
    except Exception as exc:
        if not is_sagemaker_not_found(exc):
            raise DeploymentError(
                "DEPLOYMENT_SUBMISSION_FAILED",
                f"Could not inspect SageMaker model '{aws_model_name}': {exc}",
            ) from exc
        logger.info("Creating SageMaker Model '%s'", aws_model_name)
        try:
            sm.create_model(
                ModelName=aws_model_name,
                ExecutionRoleArn=settings.sagemaker_role_arn,
                PrimaryContainer={
                    "Image": image,
                    "Environment": {
                        "MODEL_NAME": resolved.name,
                        "MODEL_VERSION": resolved.version,
                    },
                },
                Tags=[
                    {"Key": "mlflow-model", "Value": resolved.name},
                    {"Key": "mlflow-version", "Value": resolved.version},
                ],
            )
        except Exception as create_exc:
            raise DeploymentError(
                "DEPLOYMENT_SUBMISSION_FAILED",
                f"Could not create SageMaker model '{aws_model_name}': {create_exc}",
            ) from create_exc

    try:
        sm.describe_endpoint_config(EndpointConfigName=config_name)
        logger.info("Reusing SageMaker EndpointConfig '%s'", config_name)
    except Exception as exc:
        if not is_sagemaker_not_found(exc):
            raise DeploymentError(
                "DEPLOYMENT_SUBMISSION_FAILED",
                f"Could not inspect SageMaker EndpointConfig '{config_name}': {exc}",
            ) from exc
        logger.info("Creating SageMaker EndpointConfig '%s'", config_name)
        try:
            sm.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        "VariantName": settings.sagemaker_variant_name,
                        "ModelName": aws_model_name,
                        "InitialInstanceCount": settings.sagemaker_instance_count,
                        "InstanceType": settings.sagemaker_instance_type,
                        "InitialVariantWeight": 1.0,
                    }
                ],
                Tags=[
                    {"Key": "mlflow-model", "Value": resolved.name},
                    {"Key": "mlflow-version", "Value": resolved.version},
                ],
            )
        except Exception as create_exc:
            raise DeploymentError(
                "DEPLOYMENT_SUBMISSION_FAILED",
                f"Could not create SageMaker EndpointConfig '{config_name}': {create_exc}",
            ) from create_exc

    endpoint_exists = False
    current_config = None
    try:
        description = sm.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
        current_config = description.get("EndpointConfigName")
    except Exception as exc:
        if not is_sagemaker_not_found(exc):
            raise DeploymentError(
                "DEPLOYMENT_SUBMISSION_FAILED",
                f"Could not inspect SageMaker endpoint '{endpoint_name}': {exc}",
            ) from exc

    try:
        if not endpoint_exists:
            logger.info("Creating stable SageMaker endpoint '%s'", endpoint_name)
            sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        elif current_config != config_name:
            logger.info(
                "Updating stable SageMaker endpoint '%s' to v%s",
                endpoint_name,
                resolved.version,
            )
            sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
        else:
            logger.info(
                "SageMaker endpoint '%s' already targets v%s",
                endpoint_name,
                resolved.version,
            )
    except Exception as exc:
        raise DeploymentError(
            "DEPLOYMENT_SUBMISSION_FAILED",
            f"Could not create/update SageMaker endpoint '{endpoint_name}': {exc}",
        ) from exc

    wait_sagemaker_ready(
        sm,
        endpoint_name,
        timeout_seconds=settings.readiness_timeout_seconds,
        poll_interval_seconds=settings.poll_interval_seconds,
    )
    return endpoint_name


def teardown_sagemaker(
    model_name: str,
    region: str,
    *,
    execute: bool = False,
    sm=None,
    cw=None,
) -> str:
    """Remove the exact stable SageMaker endpoint and its CloudWatch alarm."""
    endpoint = stable_deployment_name(model_name, "aws_sagemaker")
    alarm = f"{endpoint}-high-latency"
    if not execute:
        return f"DRY-RUN: delete exact SageMaker endpoint {endpoint} and CloudWatch alarm {alarm}"
    if sm is None or cw is None:
        import boto3

        sm = sm or boto3.client("sagemaker", region_name=region)
        cw = cw or boto3.client("cloudwatch", region_name=region)
    try:
        sm.delete_endpoint(EndpointName=endpoint)
    except Exception as exc:
        if get_client_error_code(exc) not in {"ValidationException", "ResourceNotFound"}:
            raise DeploymentError(
                "TEARDOWN_FAILED",
                f"Could not delete SageMaker endpoint '{endpoint}': {exc}",
            ) from exc
    try:
        cw.delete_alarms(AlarmNames=[alarm])
    except Exception as exc:
        raise DeploymentError(
            "TEARDOWN_FAILED",
            f"Could not delete CloudWatch alarm '{alarm}': {exc}",
        ) from exc
    return (
        f"Deleted exact SageMaker endpoint {endpoint} and CloudWatch alarm {alarm}. "
        "Versioned models/configs were preserved for traceability."
    )
