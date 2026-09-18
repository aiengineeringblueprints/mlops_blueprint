"""Idempotent promotion of the tagged champion run into MLflow Model Registry."""
from __future__ import annotations
import argparse
import logging
import os
from typing import Optional, Tuple
from dotenv import load_dotenv
from model_deployment.champion import ChampionProvenanceError, validate_champion_provenance

logger = logging.getLogger(__name__)


def get_champion_run(experiment_name: str, *, mlflow_module):
    """Return the newest run currently tagged as champion."""
    experiment = mlflow_module.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' was not found.")
    runs = mlflow_module.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.champion = 'true'",
        order_by=["attributes.start_time DESC"],
        max_results=100,
        output_format="list",
    )
    if not runs:
        return None
    if len(runs) > 1:
        logger.warning("%d runs are tagged champion; using the newest run_id=%s", len(runs), runs[0].info.run_id)
    return runs[0]


def get_logged_model_source(
    experiment_id: str,
    run_id: str,
    artifact_path: str,
    *,
    mlflow_module,
) -> Tuple[str, Optional[str]]:
    """Resolve the intended MLflow-3 LoggedModel without ambiguity."""

    logged_models = list(
        mlflow_module.search_logged_models(
            experiment_ids=[experiment_id],
            filter_string=f"source_run_id='{run_id}'",
            output_format="list",
        )
    )
    expected_name = artifact_path.rstrip("/").split("/")[-1] if artifact_path else None

    if expected_name and logged_models:
        named = [
            candidate
            for candidate in logged_models
            if str(getattr(candidate, "name", "")) == expected_name
        ]
        if len(named) == 1:
            model_id = str(named[0].model_id)
            return f"models:/{model_id}", model_id
        if len(named) > 1:
            ids = ", ".join(str(getattr(candidate, "model_id", "?")) for candidate in named)
            raise RuntimeError(
                "AMBIGUOUS_LOGGED_MODEL: Multiple LoggedModels named "
                f"'{expected_name}' were found for champion run {run_id}: {ids}. "
                "Resolve the intended model artifact before registration."
            )

    if len(logged_models) == 1:
        model_id = str(logged_models[0].model_id)
        return f"models:/{model_id}", model_id
    if len(logged_models) > 1:
        names = ", ".join(
            f"{getattr(candidate, 'name', '?')}={getattr(candidate, 'model_id', '?')}"
            for candidate in logged_models
        )
        raise RuntimeError(
            "AMBIGUOUS_LOGGED_MODEL: Multiple LoggedModels were found for champion "
            f"run {run_id}, and expected name '{expected_name}' did not identify exactly one: "
            f"{names}. Resolve the intended model artifact before registration."
        )

    logger.warning(
        "No MLflow 3 LoggedModel found for run_id=%s; using runs:/ artifact path '%s'",
        run_id, artifact_path,
    )
    return f"runs:/{run_id}/{artifact_path}", None


def ensure_registered_model_exists(client, model_name: str) -> None:
    """Create the registered model only when it does not already exist."""
    try:
        client.get_registered_model(model_name)
        return
    except Exception:
        pass
    try:
        client.create_registered_model(model_name)
    except Exception as exc:
        # A concurrent process may have created it after our read.
        try:
            client.get_registered_model(model_name)
            return
        except Exception:
            raise exc


def find_existing_version(
    client,
    model_name: str,
    *,
    run_id: str,
    source_uri: str,
    logged_model_id: Optional[str],
    legacy_source_uri: Optional[str] = None,
):
    """Find the same registered artifact using registry metadata, not model name alone."""
    versions = list(client.search_model_versions(f"name='{model_name}'"))
    for mv in versions:
        if str(getattr(mv, "run_id", "")) == run_id and str(getattr(mv, "source", "")) == source_uri:
            return mv

    if legacy_source_uri:
        legacy_matches = [
            mv
            for mv in versions
            if str(getattr(mv, "run_id", "")) == run_id
            and str(getattr(mv, "source", "")) == legacy_source_uri
        ]
        if legacy_matches:
            return max(legacy_matches, key=lambda mv: int(mv.version))

    for mv in versions:
        try:
            detailed = client.get_model_version(model_name, str(mv.version))
            tags = getattr(detailed, "tags", {}) or {}
        except Exception:
            continue
        if tags.get("source_run_id") != run_id:
            continue
        if logged_model_id and tags.get("logged_model_id") == logged_model_id:
            return detailed
        if tags.get("source_uri") == source_uri:
            return detailed
    return None


def set_trace_tags(client, model_name: str, version: str, *, run_id: str, source_uri: str, logged_model_id: Optional[str]) -> None:
    """Store source information on the registered model version."""
    client.set_model_version_tag(model_name, version, "source_run_id", run_id)
    client.set_model_version_tag(model_name, version, "source_uri", source_uri)
    if logged_model_id:
        client.set_model_version_tag(model_name, version, "logged_model_id", logged_model_id)


def register_champion(
    experiment_name: str,
    model_name: str,
    alias: str = "champion",
    artifact_path: str = "model",
    tracking_uri: Optional[str] = None,
    *,
    mlflow_module=None,
    client=None,
):
    """Register or reuse the champion model version and assign its alias."""
    if mlflow_module is None:
        import mlflow as mlflow_module
    tracking_uri = tracking_uri or os.getenv("TRACKING_URI", "sqlite:///mlflow.db")
    mlflow_module.set_tracking_uri(tracking_uri)
    if client is None:
        client = mlflow_module.MlflowClient()

    champion_run = get_champion_run(experiment_name, mlflow_module=mlflow_module)
    if champion_run is None:
        raise RuntimeError(f"No run tagged champion was found in experiment '{experiment_name}'.")

    run_id = champion_run.info.run_id
    experiment_id = champion_run.info.experiment_id
    try:
        validate_champion_provenance(champion_run)
    except ChampionProvenanceError as exc:
        raise RuntimeError(f"INVALID_CHAMPION_PROVENANCE: {exc}") from exc

    ensure_registered_model_exists(client, model_name)
    source_uri, logged_model_id = get_logged_model_source(
        experiment_id,
        run_id,
        artifact_path,
        mlflow_module=mlflow_module,
    )
    existing = find_existing_version(
        client,
        model_name,
        run_id=run_id,
        source_uri=source_uri,
        logged_model_id=logged_model_id,
        legacy_source_uri=f"runs:/{run_id}/{artifact_path}",
    )
    if existing is None:
        mv = client.create_model_version(name=model_name, source=source_uri, run_id=run_id)
        version = str(mv.version)
        logger.info("Registered champion as new version %s", version)
    else:
        version = str(existing.version)
        logger.info("Reusing existing registered version %s for champion run_id=%s", version, run_id)

    set_trace_tags(
        client,
        model_name,
        version,
        run_id=run_id,
        source_uri=source_uri,
        logged_model_id=logged_model_id,
    )
    client.set_registered_model_alias(model_name, alias, version)
    logger.info("Alias '%s' now points to %s version %s", alias, model_name, version)
    return version


def parse_args(argv=None):
    """Read registry command-line arguments and environment defaults."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Register/reuse the tagged champion in MLflow Model Registry.")
    parser.add_argument("--experiment", default=os.getenv("EXPERIMENT_NAME"))
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME"))
    parser.add_argument("--alias", default=os.getenv("MODEL_ALIAS", "champion"))
    parser.add_argument("--artifact-path", default=os.getenv("ARTIFACT_PATH", "model"))
    parser.add_argument("--tracking-uri", default=os.getenv("TRACKING_URI", "sqlite:///mlflow.db"))
    args = parser.parse_args(argv)
    if not args.experiment:
        parser.error("--experiment or EXPERIMENT_NAME is required")
    if not args.model_name:
        parser.error("--model-name or MODEL_NAME is required")
    return args


def main(argv=None) -> int:
    """Run champion registration and return a CLI exit code."""
    load_dotenv()
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
    args = parse_args(argv)
    try:
        register_champion(
            args.experiment,
            args.model_name,
            args.alias,
            args.artifact_path,
            args.tracking_uri,
        )
        return 0
    except Exception as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
