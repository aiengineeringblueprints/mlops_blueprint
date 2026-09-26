"""Champion-selection provenance validation used by the training/registration workflow."""
from __future__ import annotations

import re


class ChampionProvenanceError(RuntimeError):
    """Raised when a champion run lacks trustworthy validation-selection provenance."""


def is_test_metric(metric_name: str) -> bool:
    """Return True when a metric name clearly represents final/test-set evaluation."""
    normalized = metric_name.strip().lower()
    if normalized.startswith("test"):
        return True
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalized) if token]
    return "test" in tokens


def validate_champion_provenance(run) -> str:
    """Require champion=true plus a logged, non-test champion metric."""
    data = getattr(run, "data", None)
    tags = getattr(data, "tags", {}) or {}
    metrics = getattr(data, "metrics", {}) or {}

    champion_tag = str(tags.get("champion", "")).strip().lower()
    if champion_tag != "true":
        raise ChampionProvenanceError(
            "Selected run is not explicitly tagged champion=true."
        )

    metric_name = str(tags.get("champion_metric", "")).strip()
    if not metric_name:
        raise ChampionProvenanceError(
            "Champion run does not declare the validation metric used for selection."
        )
    if is_test_metric(metric_name):
        raise ChampionProvenanceError(
            f"Champion test metric '{metric_name}' represents test-set evaluation; "
            "champion selection must use an independent validation metric."
        )
    if metric_name not in metrics:
        raise ChampionProvenanceError(
            f"Champion metric '{metric_name}' is not logged on the selected run."
        )
    return metric_name
