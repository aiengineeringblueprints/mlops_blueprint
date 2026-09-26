# Model Deployment

This folder implements the deployment part of the MLOps blueprint. It uses one common workflow to deploy a resolved model to **AWS SageMaker**, **Kubernetes/KServe**, or **Docker Compose**.

The model source can be either:

- an existing **MLflow registered model**, or
- one supported **standalone local model file** configured with `LOCAL_MODEL_PATH`.

When `LOCAL_MODEL_PATH` is set, MLflow resolution is skipped. The local file is fingerprinted and passed through the same image-preparation and backend deployment workflow.

> `model_deployment/.env.example` and this README are intended to be version-controlled. Keep real credentials and host-specific values in `model_deployment/.env`; `.env` is ignored by Git.

---

## 1. Prerequisites

Install these on the deployment host before starting:

- **Git**
- **Python 3.10+**
- **Docker**
- **Docker Compose v2** (`docker compose`)

Verify the basic tools:

```bash
git --version
python3 --version
docker --version
docker compose version
```

Docker must also be running:

```bash
docker info
```

Additional prerequisites depend on the selected backend and are listed under Step 5.

---

## 2. Clone the Repository and Create a Python Environment

Clone the repository and enter its root directory:

```bash
git clone <repository-url>
cd <repository-directory>
```

Create and activate a virtual environment.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the deployment-host dependencies:

```bash
python -m pip install --upgrade pip
pip install -r model_deployment/requirements.txt
```

### Optional: use `uv`

`uv` can be used instead of `pip` for faster environment creation and dependency installation. It is optional; the blueprint does not depend on it.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r model_deployment/requirements.txt
```

---

## 3. Create the Deployment Configuration

Copy the example configuration:

Linux/macOS:

```bash
cp model_deployment/.env.example model_deployment/.env
```

Windows PowerShell:

```powershell
Copy-Item model_deployment/.env.example model_deployment/.env
```

The deployment code loads `model_deployment/.env`. Explicit environment variables already defined in the host process take precedence over values from the file.

### Required Docker platform

Each backend now requires its own Docker platform variable when that backend is selected:

- AWS SageMaker: `AWS_DOCKER_PLATFORM`
- Kubernetes/KServe: `KUBE_DOCKER_PLATFORM`
- Docker Compose: `COMPOSE_DOCKER_PLATFORM`

Common Docker platform values include:

```text
linux/amd64
linux/arm64
linux/arm/v7
linux/arm/v6
linux/386
```

Select the platform that matches the target runtime. Useful checks are:

```bash
uname -m
docker info --format '{{.Architecture}}'
```

Typical mappings are:

```text
x86_64 / amd64   -> linux/amd64
aarch64 / arm64  -> linux/arm64
armv7l            -> linux/arm/v7
```

For SageMaker, the selected platform must also be supported by the configured SageMaker instance type.

---

## 4. Choose the Model Source

Choose **one** source mode in `model_deployment/.env`.

### 4.1 MLflow Registry source

Leave `LOCAL_MODEL_PATH` empty and configure the registry source:

```env
LOCAL_MODEL_PATH=
TRACKING_URI=sqlite:////absolute/path/to/model_training/mlflow.db
MODEL_NAME=champion_model
MODEL_ALIAS=champion
MODEL_VERSION=
```

`MODEL_VERSION` is optional. When it is empty, the blueprint resolves `MODEL_ALIAS` and validates champion provenance. When an explicit version is supplied, that exact registered version is used.

If MLflow stores artifacts in an S3-compatible service such as MinIO, also set:

```env
MLFLOW_S3_ENDPOINT_URL=http://<host>:<port>
```

### 4.2 Standalone local model source

Set an absolute path to one supported model file:

```env
LOCAL_MODEL_PATH=/absolute/path/to/model.keras
LOCAL_MODEL_NAME=
LOCAL_MODEL_FORMAT=auto
```

`LOCAL_MODEL_PATH` takes precedence over MLflow configuration. `LOCAL_MODEL_NAME` is optional and defaults to the file stem. `LOCAL_MODEL_FORMAT=auto` detects the format from the extension.

Supported standalone formats are:

| Artifact | Runtime / adapter | Important limitation |
| --- | --- | --- |
| `.keras`, `.h5`, `.hdf5` | TensorFlow/Keras | Custom layers must be available in the serving image |
| `.onnx` | ONNX Runtime | CPU execution provider |
| `.joblib` | joblib/scikit-learn | Loaded object must expose `predict()` |
| `.pkl`, `.pickle` | pickle/joblib | Trusted files only; compatible libraries required |
| `.pt`, `.pth` | PyTorch | TorchScript/full module supported; state-dict-only checkpoints are not |
| `.tflite` | TensorFlow Lite | Uses the TFLite interpreter |
| `.ubj`, `.xgb`, `.xgboost`, `.json`, `.model` | XGBoost | Loaded using `Booster.load_model()` |

For a non-standard filename, set the format explicitly, for example:

```env
LOCAL_MODEL_PATH=/absolute/path/to/model.bin
LOCAL_MODEL_FORMAT=xgboost
```

The local-model mode executes the supplied model artifact; it does not reconstruct arbitrary preprocessing or training code.

---

## 5. Choose One Deployment Backend

Set `DEPLOYMENT_BACKEND` to exactly one of:

```env
DEPLOYMENT_BACKEND=aws_sagemaker | kubernetes | docker_compose
```

Then follow only the matching subsection below.

### 5.1 AWS SageMaker

#### Additional prerequisites

- An AWS account and credentials with the required ECR/SageMaker permissions
- A SageMaker execution role
- An ECR repository path in the same region used by the deployment

The AWS CLI is useful for manual verification but the blueprint itself uses `boto3`.

#### Required and important variables

| Variable | Purpose |
| --- | --- |
| `AWS_REGION` | Region for ECR, SageMaker, and CloudWatch |
| `AWS_DOCKER_REGISTRY` | ECR repository used for the deployment image |
| `AWS_DOCKER_PLATFORM` | **Required** target image platform |
| `SAGEMAKER_ROLE_ARN` | SageMaker execution role |
| `SAGEMAKER_INSTANCE_TYPE` | Inference instance type |
| `SAGEMAKER_INSTANCE_COUNT` | Number of inference instances |
| `SAGEMAKER_VARIANT_NAME` | SageMaker production variant name |
| `SAGEMAKER_LATENCY_THRESHOLD_US` | CloudWatch latency-alarm threshold |
| `SAGEMAKER_ALARM_PERIOD` | CloudWatch alarm period |
| `SAGEMAKER_EVALUATION_PERIODS` | Number of alarm evaluation periods |

Example:

```env
DEPLOYMENT_BACKEND=aws_sagemaker
AWS_REGION=eu-central-1
AWS_DOCKER_REGISTRY=<account>.dkr.ecr.eu-central-1.amazonaws.com/mlops-blueprint
AWS_DOCKER_PLATFORM=linux/amd64
SAGEMAKER_ROLE_ARN=<role-arn>
SAGEMAKER_INSTANCE_TYPE=ml.m5.xlarge
SAGEMAKER_INSTANCE_COUNT=1
```

Credentials can be supplied through the normal AWS provider chain or through the corresponding variables in `model_deployment/.env`.

### 5.2 Kubernetes / KServe

#### Additional prerequisites

- `kubectl`
- Access to the target cluster through the configured kubeconfig context
- KServe installed
- Target namespace already created
- Prometheus Operator/`PodMonitor` available for the blueprint's Prometheus integration
- A container registry reachable by the deployment host and the cluster

Authenticate the deployment host separately before pushing, for example:

```bash
docker login <registry-host>
```

A private image repository may additionally require `KUBE_IMAGE_PULL_SECRET` so the cluster can pull the image.

#### Required and important variables

| Variable | Purpose |
| --- | --- |
| `KUBE_CONTEXT` | kubeconfig context selecting the cluster |
| `KUBE_NAMESPACE` | Namespace for deployment resources |
| `KUBE_DEPLOYMENT_NAME` | Stable KServe/Kubernetes deployment name |
| `KUBE_DOCKER_REGISTRY` | Registry/repository prefix for the image |
| `KUBE_DOCKER_PLATFORM` | **Required** target image platform |
| `KUBE_CPU_REQUEST`, `KUBE_MEM_REQUEST` | Predictor resource requests |
| `KUBE_CPU_LIMIT`, `KUBE_MEM_LIMIT` | Predictor resource limits |
| `KUBE_MIN_REPLICAS`, `KUBE_MAX_REPLICAS` | Replica bounds |
| `KUBE_KSERVE_DEPLOYMENT_MODE` | KServe mode (`Standard`, `Knative`, or `ModelMesh`) |
| `KUBE_IMAGE_PULL_POLICY` | Kubernetes image pull policy |
| `KUBE_SCRAPE_INTERVAL` | PodMonitor scrape interval |
| `KUBE_PROMETHEUS_RELEASE_LABEL` | Prometheus Operator release label |
| `KUBE_PROMETHEUS_URL` | Optional URL for automatic target verification |
| `KUBE_NODE_SELECTOR_JSON` | Optional node selector, e.g. ARM64 nodes |
| `KUBE_EXTRA_RESOURCE_LIMITS_JSON` | Optional extra resource limits such as GPU |
| `KUBE_RUNTIME_CLASS` | Optional RuntimeClass |

Example for an ARM64 node:

```env
DEPLOYMENT_BACKEND=kubernetes
KUBE_CONTEXT=<context>
KUBE_NAMESPACE=mlops
KUBE_DEPLOYMENT_NAME=champion-model
KUBE_DOCKER_REGISTRY=docker.io/<username>
KUBE_DOCKER_PLATFORM=linux/arm64
KUBE_CPU_REQUEST=250m
KUBE_MEM_REQUEST=512Mi
KUBE_CPU_LIMIT=1
KUBE_MEM_LIMIT=1Gi
KUBE_MIN_REPLICAS=1
KUBE_MAX_REPLICAS=1
```

`KUBE_DEPLOYMENT_NAME` is validated rather than silently rewritten. Use lowercase letters, numbers, and hyphens only.

### 5.3 Docker Compose

#### Additional prerequisites

No backend-specific software is required beyond Docker and Docker Compose v2. The configured host ports must be free.

#### Required and important variables

| Variable | Purpose |
| --- | --- |
| `COMPOSE_DOCKER_PLATFORM` | **Required** target image platform |
| `COMPOSE_PROJECT_NAME` | Optional project name; otherwise derived from the model name |
| `COMPOSE_MODEL_PORT` | Host port for `/health`, `/predict`, and `/metrics` |
| `COMPOSE_ENABLE_PROMETHEUS` | Start bundled Prometheus; **defaults to `false`** |
| `COMPOSE_PROMETHEUS_PORT` | Host port for bundled Prometheus |
| `COMPOSE_ENABLE_GRAFANA` | Start bundled Grafana; **defaults to `false`** |
| `COMPOSE_GRAFANA_PORT` | Host port for bundled Grafana |
| `COMPOSE_EXTERNAL_PROMETHEUS_URL` | Required only for bundled Grafana with external Prometheus |

Minimal model-only example:

```env
DEPLOYMENT_BACKEND=docker_compose
COMPOSE_DOCKER_PLATFORM=linux/arm64
COMPOSE_MODEL_PORT=8000
COMPOSE_ENABLE_PROMETHEUS=false
COMPOSE_ENABLE_GRAFANA=false
```

Monitoring services are opt-in:

```text
Prometheus=false, Grafana=false -> model service only
Prometheus=true,  Grafana=false -> model service + bundled Prometheus
Prometheus=true,  Grafana=true  -> model service + bundled Prometheus + Grafana
Prometheus=false, Grafana=true  -> bundled Grafana using COMPOSE_EXTERNAL_PROMETHEUS_URL
```

If a monitoring port is already occupied, either free that host port or select another value such as `COMPOSE_GRAFANA_PORT=3001`.

---

## 6. Validate the Configuration Before Deployment

Run a dry run first:

```bash
python model_deployment/deploy.py --backend <backend> --dry-run
```

Examples:

```bash
python model_deployment/deploy.py --backend aws_sagemaker --dry-run
python model_deployment/deploy.py --backend kubernetes --dry-run
python model_deployment/deploy.py --backend docker_compose --dry-run
```

Configuration errors are reported before infrastructure changes whenever possible.

---

## 7. Deploy

Run the same command without `--dry-run`:

```bash
python model_deployment/deploy.py --backend <backend>
```

The common workflow resolves the model source, prepares the serving image, applies the selected backend, waits for readiness, and configures the backend-specific monitoring path.


---

## 8. Verify the Deployment

### AWS SageMaker

```bash
aws sagemaker describe-endpoint \
  --endpoint-name <endpoint-name> \
  --region <region> \
  --query "{Status:EndpointStatus,FailureReason:FailureReason}"
```

Expected state: `InService`.

### Kubernetes / KServe

```bash
kubectl --context <context> -n <namespace> get inferenceservice,podmonitor
```

To access the predictor locally:

```bash
kubectl --context <context> -n <namespace> \
  port-forward svc/<deployment-name>-inference-predictor 18000:80
```

Then:

```bash
curl http://127.0.0.1:18000/health
curl http://127.0.0.1:18000/metrics
```

### Docker Compose

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics
```

When bundled monitoring is enabled, Prometheus and Grafana are reachable on their configured host ports.

---


## 9. Teardown

Teardown is a dry run by default:

```bash
python -m model_deployment.teardown --backend <backend>
```

Execute removal explicitly:

```bash
python -m model_deployment.teardown --backend <backend> --execute
```

The teardown implementation targets only the selected deployment identity and does not intentionally remove unrelated infrastructure.

---

## 10. Troubleshooting

### Check Docker architecture

```bash
uname -m
docker info --format '{{.Architecture}}'
```

Ensure the selected backend's required `*_DOCKER_PLATFORM` matches the target architecture.

### Check port usage

Linux:

```bash
sudo ss -ltnp
```

Docker-published ports:

```bash
docker ps --format "table {{.Names}}\t{{.Ports}}"
```

### Check Docker Compose state and logs

```bash
docker compose ps -a
docker compose logs --tail 100
```

These commands are useful for deeper investigation after the blueprint has already reported the concrete failure cause.
