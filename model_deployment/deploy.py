"""Model-deployment command-line entry point."""

from pathlib import Path
import sys

MODEL_DEPLOYMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DEPLOYMENT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_deployment.main import cli


if __name__ == "__main__":
    raise SystemExit(cli())
