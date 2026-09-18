"""Repository-level entry point for model deployment."""

from model_deployment.main import cli


# Start the shared deployment command-line interface.
if __name__ == "__main__":
    raise SystemExit(cli())
