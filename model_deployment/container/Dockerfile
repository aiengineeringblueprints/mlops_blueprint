# Use the established shared Python serving base image.
FROM python:3.12-slim

# Model identity and source passed during image build.
ARG MODEL_NAME
ARG MODEL_VERSION
ARG MODEL_SOURCE=mlflow
ARG MODEL_FORMAT=mlflow
ARG MODEL_FILE=""

# Enable Prometheus metrics only when required.
ARG ENABLE_PROMETHEUS_METRICS=false

WORKDIR /app

# Least privilege (NFR-07): the server runs as an unprivileged user. Packages are installed as
# root below; the application files are handed to that user, and port 8080 needs no privileges.
RUN useradd --system --uid 10001 --create-home --home-dir /home/serving --shell /usr/sbin/nologin serving

# Install only the common serving dependencies first. Model-source/runtime packages
# are selected below so a standalone local model does not require the MLflow runtime.
RUN pip install --no-cache-dir \
      "fastapi>=0.111,<1" \
      "uvicorn[standard]>=0.30,<1" \
      "numpy>=1.26,<2.3" \
      "pandas>=2.2,<3" \
    && if [ "$ENABLE_PROMETHEUS_METRICS" = "true" ]; then \
      pip install --no-cache-dir "prometheus-client>=0.20,<1"; \
    fi

# Install exactly the runtime needed for the selected model source/format.
# TensorFlow provides both Keras loading and the TensorFlow Lite Interpreter.
RUN if [ "$MODEL_SOURCE" = "mlflow" ]; then \
      pip install --no-cache-dir "mlflow>=3,<4"; \
    elif [ "$MODEL_FORMAT" = "keras" ] || [ "$MODEL_FORMAT" = "tflite" ]; then \
      pip install --no-cache-dir "tensorflow>=2.20,<2.22"; \
    elif [ "$MODEL_FORMAT" = "onnx" ]; then \
      pip install --no-cache-dir "onnxruntime>=1.20,<2"; \
    elif [ "$MODEL_FORMAT" = "joblib" ] || [ "$MODEL_FORMAT" = "pickle" ]; then \
      pip install --no-cache-dir "joblib>=1.3,<2" "scikit-learn>=1.4,<2"; \
    elif [ "$MODEL_FORMAT" = "pytorch" ]; then \
      pip install --no-cache-dir "torch>=2.2,<3"; \
    elif [ "$MODEL_FORMAT" = "xgboost" ]; then \
      pip install --no-cache-dir "xgboost>=2,<4"; \
    else \
      echo "Unsupported model runtime: source=$MODEL_SOURCE format=$MODEL_FORMAT" >&2; exit 2; \
    fi

# Copy the resolved MLflow artifact or the selected standalone model file.
COPY --chown=serving:serving container/model_artifact/ ./model_artifact/

# Preserve existing MLflow model-specific dependency installation. A standalone
# single-file model intentionally does not infer arbitrary custom dependencies.
RUN if [ "$MODEL_SOURCE" = "mlflow" ] && [ -f "./model_artifact/requirements.txt" ]; then \
      pip install --no-cache-dir -r ./model_artifact/requirements.txt; \
    fi

COPY --chown=serving:serving serving/ ./serving/

ENV MODEL_NAME=${MODEL_NAME} \
    MODEL_VERSION=${MODEL_VERSION} \
    MODEL_PATH=/app/model_artifact \
    MODEL_SOURCE=${MODEL_SOURCE} \
    MODEL_FORMAT=${MODEL_FORMAT} \
    MODEL_FILE=${MODEL_FILE} \
    ENABLE_PROMETHEUS_METRICS=${ENABLE_PROMETHEUS_METRICS}

ENV HOME=/home/serving
USER serving

EXPOSE 8080
ENTRYPOINT ["python", "-m", "serving.entrypoint"]
