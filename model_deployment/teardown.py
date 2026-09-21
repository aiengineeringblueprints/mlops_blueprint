"""Safe backend-aware teardown entry point for exact deployment resources."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional, Sequence

from pathlib import Path
from dotenv import load_dotenv

MODEL_DEPLOYMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DEPLOYMENT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_deployment import (
    DeploymentError,
    safe_resource_name,
    validate_kubernetes_deployment_name,
)
from model_deployment.config import VALID_TARGETS
from model_deployment.local_model import configured_deployment_model_name
from model_deployment.deployments.docker_compose import teardown_compose_project
from model_deployment.deployments.kubernetes import teardown_kserve
from model_deployment.deployments.sagemaker import teardown_sagemaker

logger = logging.getLogger(__name__)



def teardown_deployment(
    target: str,
    model_name: str,
    *,
    execute: bool = False,
    aws_region: str | None = None,
    kube_namespace: str = "default",
    kube_context: str | None = None,
    compose_project_name: str | None = None,
    kube_deployment_name: str | None = None,
) -> str:
    """Dispatch teardown using only the exact deployment identity for the selected backend."""
    if target != "kubernetes" and not model_name:
        raise DeploymentError(
            "CONFIGURATION_MISSING",
            "MODEL_NAME is required for AWS SageMaker and Docker Compose teardown.",
        )

    if target == "aws_sagemaker":
        if not aws_region:
            raise DeploymentError(
                "CONFIGURATION_MISSING",
                "AWS_REGION is required for SageMaker teardown",
            )
        return teardown_sagemaker(model_name, aws_region, execute=execute)

    if target == "kubernetes":
        if not kube_context:
            raise DeploymentError("CONFIGURATION_MISSING", "KUBE_CONTEXT is required")
        deployment_name = validate_kubernetes_deployment_name(kube_deployment_name or "")
        return teardown_kserve(
            deployment_name,
            kube_namespace,
            kube_context,
            execute=execute,
        )

    if target == "docker_compose":
        project = safe_resource_name(
            compose_project_name or f"mlops-{model_name}",
            max_length=48,
        )
        return teardown_compose_project(project, execute=execute)

    raise DeploymentError("CONFIGURATION_INVALID", f"Unsupported target {target!r}")


def resolve_args(argv: Optional[Sequence[str]] = None):
    """Parse teardown arguments using the same canonical backend names as deployment."""
    load_dotenv(MODEL_DEPLOYMENT_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(
        description="Tear down one exact deployment backend resource set."
    )
    parser.add_argument("--backend", "--target", dest="target", required=True, choices=VALID_TARGETS)
    parser.add_argument(
        "--model-name",
        default=configured_deployment_model_name(),
        help="Deployment model identity; defaults to LOCAL_MODEL_NAME/file stem in local mode or MODEL_NAME in MLflow mode",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete the exact deployment resources; default is dry-run",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run safe backend-aware teardown; dry-run remains the default."""
    args = resolve_args(argv)
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        result = teardown_deployment(
            args.target,
            args.model_name,
            execute=args.execute,
            aws_region=os.getenv("AWS_REGION"),
            kube_namespace=os.getenv("KUBE_NAMESPACE", "default"),
            kube_context=os.getenv("KUBE_CONTEXT"),
            compose_project_name=os.getenv("COMPOSE_PROJECT_NAME"),
            kube_deployment_name=os.getenv("KUBE_DEPLOYMENT_NAME"),
        )
        logger.info("Teardown result: %s", result)
        return 0
    except DeploymentError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logger.error("Teardown interrupted")
        return 130
    except Exception as exc:
        logger.exception("UNEXPECTED_TEARDOWN_FAILURE: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
