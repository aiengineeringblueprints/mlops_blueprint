# Use a lightweight Python base image
FROM python:3.12-slim

# Model identity passed during image build
ARG MODEL_NAME
ARG MODEL_VERSION

# Enable Prometheus metrics only when required
ARG ENABLE_PROMETHEUS_METRICS=false

# Set the application working directory
WORKDIR /app

# Install common serving dependencies
RUN pip install --no-cache-dir \
      "mlflow>=3,<4" \
      "fastapi>=0.111,<1" \
      "uvicorn[standard]>=0.30,<1" \
      "numpy>=1.26,<2.3" \
      "pandas>=2.2,<3" \
    && if [ "$ENABLE_PROMETHEUS_METRICS" = "true" ]; then \
      pip install --no-cache-dir "prometheus-client>=0.20,<1"; \
    fi

# Copy the resolved MLflow model artifact
COPY container/model_artifact/ ./model_artifact/

# Install model-specific dependencies when available
RUN if [ -f "./model_artifact/requirements.txt" ]; then \
      pip install --no-cache-dir -r ./model_artifact/requirements.txt; \
    fi

# Copy the shared model-serving application
COPY serving/ ./serving/

# Configure model and monitoring runtime settings
ENV MODEL_NAME=${MODEL_NAME} \
    MODEL_VERSION=${MODEL_VERSION} \
    MODEL_PATH=/app/model_artifact \
    ENABLE_PROMETHEUS_METRICS=${ENABLE_PROMETHEUS_METRICS}

# Expose the model-serving port
EXPOSE 8080

# Start the shared serving entrypoint
ENTRYPOINT ["python", "-m", "serving.entrypoint"]