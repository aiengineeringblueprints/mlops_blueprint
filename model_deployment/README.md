# Model Deployment

This folder deploys a model through one common workflow to **AWS SageMaker**, **Kubernetes/KServe**, or **Docker Compose**.

The deployment source can be either:

- an existing **MLflow registered model** (the original/default workflow), or
- one supported **standalone local model file** configured with `LOCAL_MODEL_PATH`.

When `LOCAL_MODEL_PATH` is set, MLflow tracking/registry resolution is skipped completely. The local model is fingerprinted, packaged into the same serving image workflow, and then passed to the existing deployment backend.

## 1. File Structure

```text
model_deployment/
├── deploy.py                             # Deployment command
├── .env.example                          # Deployment-only runtime configuration
├── requirements.txt                      # Deployment-host dependencies
├── main.py                               # Common deployment workflow
├── config.py                             # Reads and validates deployment settings
├── core.py                               # Shared data classes, errors, and naming helpers
├── champion.py                           # Resolves the MLflow version or champion alias
├── local_model.py                         # Validates/fingerprints supported standalone model files
├── teardown.py                           # Teardown only the selected deployment resources
├── container/
│   ├── Dockerfile                        # Shared model-serving image
│   └── containerize.py                   # Stages MLflow/local artifacts, builds, and pushes images
├── deployments/
│   ├── __init__.py                       # Deployment backend package
│   ├── sagemaker/                        # SageMaker endpoint deployment
│   │   ├── __init__.py                   # SageMaker backend package
│   │   ├── adapter.py                    # Connects the common workflow to SageMaker
│   │   └── helpers.py                    # Creates, reuses, and checks SageMaker resources
│   ├── kubernetes/                       # Kubernetes/KServe deployment
│   │   ├── __init__.py                   # Kubernetes backend package
│   │   ├── adapter.py                    # Connects the common workflow to KServe
│   │   ├── helpers.py                    # Deploys and checks KServe resources
│   │   └── templates/                    # Kubernetes manifest templates
│   │       ├── inferenceservice.yaml.j2  # KServe InferenceService template
│   │       └── podmonitor.yaml.j2        # Prometheus PodMonitor template
│   └── docker_compose/                   # Single-host Docker Compose deployment
│       ├── __init__.py                   # Docker Compose backend package
│       ├── adapter.py                    # Connects the common workflow to Docker Compose
│       ├── helpers.py                    # Renders, starts, and checks Compose services
│       └── templates/                    # Docker Compose configuration templates
│           ├── compose.yaml.j2           # Model, Prometheus, and Grafana services
│           └── prometheus.yml.j2         # Prometheus scrape configuration
├── monitoring/
│   ├── cloudwatch.py                     # SageMaker CloudWatch alarm
│   ├── prometheus.py                     # Prometheus target checks
│   └── grafana/                          # Reusable Grafana dashboard
├── serving/
│   ├── app.py                            # FastAPI serving endpoints
│   ├── entrypoint.py                     # Start shared model server
│   ├── inference.py                      # Input handling plus MLflow/native model adapters
│   └── metrics.py                        # Prometheus or no-op metrics
└── test/                                 # Testing for the endpoint working
```

## 2. `.env` Setup

Install only the deployment dependencies and create the deployment-specific environment file:

```bash
python -m pip install -r model_deployment/requirements.txt
cp model_deployment/.env.example model_deployment/.env
```

`model_deployment/.env` is loaded explicitly by the deployment commands and tests. Training-only variables are kept separately in `model_training/.env`.

Always configure the deployment backend:

```env
DEPLOYMENT_BACKEND=<aws_sagemaker|kubernetes|docker_compose>
```

Then choose **one model-source mode**.

### Option A — Existing MLflow Registry workflow

Leave `LOCAL_MODEL_PATH` empty and configure the existing MLflow values:

```env
LOCAL_MODEL_PATH=
TRACKING_URI=sqlite:///mlflow.db
MODEL_NAME=champion_model
MODEL_ALIAS=champion
MODEL_VERSION=
```

`MODEL_VERSION` is optional and overrides the alias when an exact registered version is required. The MLflow workflow is unchanged from the original blueprint.
If the training workflow uses a local SQLite MLflow database, configure `TRACKING_URI` with an absolute path so deployment can resolve the same registry regardless of the shell working directory.

### Option B — Standalone local model (no MLflow setup required)

Set `LOCAL_MODEL_PATH` to one supported model file:

```env
LOCAL_MODEL_PATH=/absolute/path/to/model.onnx
LOCAL_MODEL_NAME=                      # optional; defaults to the filename stem
LOCAL_MODEL_FORMAT=auto                # normally keep auto
```

When `LOCAL_MODEL_PATH` is non-empty, it takes precedence over `TRACKING_URI`, `MODEL_NAME`, `MODEL_ALIAS`, and `MODEL_VERSION` for deployment. The file is hashed with SHA-256 and the first 12 hexadecimal characters are used as its deterministic deployment-version identity. Changing the file therefore produces a new local model version without requiring a registry.

Supported standalone formats are:

| Local artifact                                | Runtime/adapter              | Notes                                                                                |
| --------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------ |
| `.keras`                                      | TensorFlow/Keras             | Native Keras model                                                                   |
| `.h5`, `.hdf5`                                | TensorFlow/Keras             | HDF5 Keras model                                                                     |
| `.onnx`                                       | ONNX Runtime                 | CPU execution provider in the shared server                                          |
| `.joblib`                                     | joblib / scikit-learn        | Loaded object must expose `predict()`                                                |
| `.pkl`, `.pickle`                             | joblib/pickle + scikit-learn | Loaded object must expose `predict()`; trusted files only                            |
| `.pt`, `.pth`                                 | PyTorch                      | TorchScript or serialized full module; state-dict-only checkpoints are not supported |
| `.tflite`                                     | TensorFlow Lite              | Uses the TensorFlow Lite interpreter                                                 |
| `.ubj`, `.xgb`, `.xgboost`, `.json`, `.model` | XGBoost                      | Native `Booster.load_model()` artifacts                                              |

For a non-standard XGBoost filename such as `model.bin`, set an explicit override:

```env
LOCAL_MODEL_PATH=/absolute/path/to/model.bin
LOCAL_MODEL_FORMAT=xgboost
```

Accepted explicit `LOCAL_MODEL_FORMAT` values are `keras`, `onnx`, `joblib`, `pickle`, `pytorch`, `tflite`, and `xgboost`. Unsupported formats fail before image deployment.

The local-file mode intentionally supports **model execution**, not automatic reconstruction of arbitrary training code or preprocessing pipelines. The `/predict` request must already match the model's expected input structure/shape. Keras models requiring custom Python layers/objects must have those dependencies available in the serving image; the blueprint does not reconstruct arbitrary custom training code. Serialized pickle/joblib models may require compatible library versions, and `.pt/.pth` state dictionaries cannot be served without the original model architecture. Only load trusted pickle/joblib/PyTorch serialized files. These are explicit portability limitations of standalone-file deployment.

The serving image installs only the runtime required by the selected source/format. A local standalone deployment therefore does not require the MLflow Python runtime or an MLflow server/registry.

Check the resolved plan before changing infrastructure:

```bash
python model_deployment/deploy.py --backend <backend> --dry-run
```

## 3. Kubernetes / KServe Deployment

### Prerequisites

- Docker is running on the deployment host.
- `kubectl` can access the target cluster through `KUBE_CONTEXT`.
- KServe is installed.
- The target namespace already exists.
- The cluster can pull the image from `KUBE_DOCKER_REGISTRY`.
- Prometheus Operator/`PodMonitor` is available when Prometheus monitoring is used.

`KUBE_DOCKER_REGISTRY` only defines the image destination. Authenticate the deployment host separately, for example:

```bash
docker login <registry-host>
```

For a private registry, also configure `KUBE_IMAGE_PULL_SECRET` when the cluster needs pull credentials.

### Configure

Set the Kubernetes section in `model_deployment/.env`:

```env
DEPLOYMENT_BACKEND=kubernetes
KUBE_CONTEXT=<context>
KUBE_NAMESPACE=mlops
KUBE_DEPLOYMENT_NAME=champion-model
KUBE_DOCKER_REGISTRY=<registry>/<repository-prefix>
KUBE_DOCKER_PLATFORM=<linux/amd64-or-linux/arm64>
```

`KUBE_DEPLOYMENT_NAME` must already be a valid lowercase Kubernetes name. It is not silently rewritten.

### Deploy

```bash
python model_deployment/deploy.py --backend kubernetes --dry-run
python model_deployment/deploy.py --backend kubernetes
```

### Test

Check the deployed KServe resources:

```bash
kubectl --context <context> -n <namespace> get inferenceservice,podmonitor
```

Port-forward the predictor service:

```bash
kubectl --context <context> -n <namespace> port-forward svc/<deployment-name>-inference-predictor 18000:80
```

Then check health:

```bash
curl http://127.0.0.1:18000/health
```

Optional smoke tests:

```bash
python model_deployment/test/test_kserve.py
python model_deployment/test/test_kserve_monitoring.py
python model_deployment/test/test_kserve_prometheus.py
python model_deployment/test/test_kserve_grafana.py
```

```bash
python3 mock_request.py \
        --backend http \
        --url http://127.0.0.1:18000/predict \
        --requests 120 \
        --interval 0.5
```
```bash
python3 model_deployment/test/mock_request/benchmark_deployment.py \
  --backend http \
  --url http://127.0.0.1:18000/predict \
  --requests 100 \
  --concurrency 5 \
  --warmup 5
```

## 4. AWS SageMaker Deployment

### Prerequisites

- Docker is running.
- AWS credentials are available through `model_deployment/.env` or the host environment.
- The SageMaker execution role exists.
- The configured ECR repository path and AWS region belong together.

### Configure

Set the AWS section in `model_deployment/.env`, especially:

```env
DEPLOYMENT_BACKEND=aws_sagemaker
AWS_ACCESS_KEY_ID=<ACCESS_KEY>
AWS_SECRET_ACCESS_KEY=<SECRET_ACCESS>
AWS_REGION=us-east-1
AWS_DOCKER_REGISTRY=<account>.dkr.ecr.us-east-1.amazonaws.com/mlops-blueprint
SAGEMAKER_ROLE_ARN=<role-arn>
SAGEMAKER_INSTANCE_TYPE=ml.m5.xlarge
AWS_DOCKER_PLATFORM=linux/amd64
```

Set `AWS_DOCKER_PLATFORM` to the architecture required by the SageMaker instance. This is important when deploying from an ARM64 host.

### Deploy

```bash
python model_deployment/deploy.py --backend aws_sagemaker --dry-run
python model_deployment/deploy.py --backend aws_sagemaker
```

The workflow builds the image, authenticates to ECR, pushes the image, reconciles the SageMaker endpoint, waits for readiness, and configures the CloudWatch latency alarm.

### Test

Check the endpoint state:

```bash
aws sagemaker describe-endpoint \
  --endpoint-name <endpoint-name> \
  --region <region> \
  --query "{Status:EndpointStatus,FailureReason:FailureReason}"
```

The expected status is `InService`.

Optional inference smoke test:

```bash
python model_deployment/test/test_sagemaker.py
```

```bash
python3 mock_request.py \
        --backend sagemaker \
        --endpoint-name <sagemaker-endpoint-name> \
        --region eu-central-1 \
        --requests 120 \
        --interval 0.5
```
```bash
python3 model_deployment/test/mock_request/benchmark_deployment.py \
  --backend sagemaker \
  --endpoint-name <YOUR_SAGEMAKER_ENDPOINT_NAME> \
  --region <YOUR_AWS_REGION> \
  --requests 100 \
  --concurrency 5 \
  --warmup 5
```

## 5. Docker Compose Deployment

### Prerequisites

- Docker is running.
- Docker Compose v2 is available.
- The selected host ports are free.

### Configure

Set the Compose section in `model_deployment/.env`, for example:

```env
DEPLOYMENT_BACKEND=docker_compose
COMPOSE_MODEL_PORT=8000
COMPOSE_DOCKER_PLATFORM=<linux/amd64-or-linux/arm64>
COMPOSE_ENABLE_PROMETHEUS=true
COMPOSE_PROMETHEUS_PORT=9090
COMPOSE_ENABLE_GRAFANA=true
COMPOSE_GRAFANA_PORT=3000
```

Monitoring choices are independent:

```text
Prometheus=true,  Grafana=true   -> bundled Prometheus + Grafana
Prometheus=false, Grafana=false  -> model service only; use existing monitoring
Prometheus=false, Grafana=true   -> bundled Grafana + external Prometheus
Prometheus=true,  Grafana=false  -> bundled Prometheus only
```

For bundled Grafana with external Prometheus, set `COMPOSE_EXTERNAL_PROMETHEUS_URL`.

### Deploy

```bash
python model_deployment/deploy.py --backend docker_compose --dry-run
python model_deployment/deploy.py --backend docker_compose
```

### Test

Check the model service:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

Optional smoke test:

```bash
python model_deployment/test/test_docker_compose.py
```

```bash
python3 mock_request.py \
        --backend http \
        --url http://127.0.0.1:8000/predict \
        --requests 120 \
        --interval 0.5
```

The example tests default to the thesis validation image under `model_training/data/val/...`. If that dataset is not present on a deployment-only host, set `IMAGE_PATH` in `model_deployment/.env` to any compatible test input.

## 6. Teardown and Changing Deployment

Teardown is a dry-run by default:

```bash
python -m model_deployment.teardown --backend <backend>
```

To actually remove the selected deployment:

```bash
python -m model_deployment.teardown --backend <backend> --execute
```

In standalone local-model mode, teardown uses `LOCAL_MODEL_NAME` when configured or otherwise the local model filename stem, matching deployment identity. Kubernetes teardown continues to use `KUBE_DEPLOYMENT_NAME`.

To change backend, model version, ports, resources, or monitoring settings:

1. Update `model_deployment/.env`.
2. Run `python model_deployment/deploy.py --backend <backend> --dry-run`.
3. Run `python model_deployment/deploy.py --backend <backend>`.
4. Teardown the previous deployment only when it is no longer required.

The teardown code is scoped to the selected deployment and does not intentionally remove unrelated infrastructure.
