"""Shared FastAPI serving application for all deployment backends."""

from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, HTTPException, Request, Response

from . import (
    create_metrics,
    extract_payload,
    load_serving_model,
    run_prediction,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model_artifact")
MODEL_NAME = os.environ.get("MODEL_NAME", "model")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "0")
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "mlflow")
MODEL_FORMAT = os.environ.get("MODEL_FORMAT", "mlflow")
MODEL_FILE = os.environ.get("MODEL_FILE") or None
LABELS = {"model": MODEL_NAME, "version": MODEL_VERSION}
METRICS = create_metrics()

app = FastAPI(title=f"Model Server: {MODEL_NAME}")

try:
    logger.info(
        "Loading model '%s' (v%s) source=%s format=%s from %s",
        MODEL_NAME,
        MODEL_VERSION,
        MODEL_SOURCE,
        MODEL_FORMAT,
        MODEL_PATH,
    )
    MODEL = load_serving_model(
        MODEL_PATH,
        model_source=MODEL_SOURCE,
        model_format=MODEL_FORMAT,
        model_file=MODEL_FILE,
    )
    METRICS.model_up.labels(**LABELS).set(1)
except Exception:
    METRICS.model_up.labels(**LABELS).set(0)
    logger.exception("Model loading failed")
    raise


async def handle_prediction_request(
    request: Request, *, kserve_v1: bool = False
) -> dict:
    """Parse one request, execute the shared prediction path, and record operational metrics."""
    METRICS.requests.labels(**LABELS).inc()
    started = time.perf_counter()
    try:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid JSON payload: "
                    f"{str(exc).strip() or exc.__class__.__name__}"
                ),
            ) from exc
        try:
            payload = extract_payload(body, kserve_v1=kserve_v1)
            return run_prediction(MODEL, payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception(
                "Inference failed for model=%s version=%s", MODEL_NAME, MODEL_VERSION
            )
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {str(exc).strip() or exc.__class__.__name__}",
            ) from exc
    except HTTPException:
        METRICS.errors.labels(**LABELS).inc()
        raise
    finally:
        METRICS.latency.labels(**LABELS).observe(time.perf_counter() - started)


@app.get("/health")
@app.get("/ping")
def health():
    """Return model-server liveness information."""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "source": MODEL_SOURCE,
        "format": MODEL_FORMAT,
    }


@app.get("/v1/models")
def kserve_list_models():
    """Return the model list expected by the KServe V1 protocol."""
    return {"models": [MODEL_NAME]}


@app.get("/v1/models/{requested_model}")
def kserve_model_ready(requested_model: str):
    """Report readiness for the configured KServe V1 model name."""
    if requested_model != MODEL_NAME:
        raise HTTPException(
            status_code=404, detail=f"Model '{requested_model}' is not served here."
        )
    return {"name": MODEL_NAME, "ready": True}


@app.post("/predict")
async def predict(request: Request):
    """Serve the generic prediction endpoint."""
    return await handle_prediction_request(request)


@app.post("/invocations")
async def invocations(request: Request):
    """Serve the SageMaker-compatible invocation endpoint."""
    return await handle_prediction_request(request)


@app.post("/v1/models/{requested_model}:predict")
async def kserve_predict(requested_model: str, request: Request):
    """Serve the KServe V1 prediction endpoint."""
    if requested_model != MODEL_NAME:
        raise HTTPException(
            status_code=404, detail=f"Model '{requested_model}' is not served here."
        )
    return await handle_prediction_request(request, kserve_v1=True)


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics only when the image enabled instrumentation."""
    if not METRICS.enabled:
        raise HTTPException(
            status_code=404,
            detail="Prometheus metrics are disabled for this deployment backend.",
        )
    return Response(content=METRICS.render(), media_type=METRICS.content_type)
