"""Start the shared model server for all deployment backends."""

from __future__ import annotations

import sys

import uvicorn


def main() -> None:
    """Start Uvicorn and accept SageMaker's optional serve argument."""
    args = sys.argv[1:]

    if args and args != ["serve"]:
        raise RuntimeError(f"Unsupported container arguments: {args}")

    uvicorn.run(
        "serving.app:app",
        host="0.0.0.0",
        port=8080,
    )


if __name__ == "__main__":
    main()