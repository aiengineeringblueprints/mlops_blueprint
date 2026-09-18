# Model Deployment

This folder deploys an already registered MLflow model through one common workflow.
Supported environments are **AWS SageMaker**, **Kubernetes/KServe**, and **Docker Compose**.

## 1. File Structure

```text
deploy.py                                 # Repository-level deployment command
.env.example                              # Runtime configuration
model_deployment/
├── main.py                               # Common deployment workflow
├── config.py                             # Reads and validates deployment settings
├── core.py                               # Shared data classes, errors, and naming helpers
├── champion.py                           # Resolves the MLflow version or champion alias
├── teardown.py                           # Teardown only the selected deployment resources
├── container/
│   ├── Dockerfile                        # Shared model-serving image
│   └── containerize.py                   # Downloads the model, builds, and pushes images
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
│   ├── inference.py                      # Input handling and model prediction
│   └── metrics.py                        # Prometheus or no-op metrics
└── test/                                 # Testing for the endpoint working
```

## 2. `.env` Setup

Create the local configuration file from the example:

```bash
cp .env.example .env
```

For every deployment, configure at least:

```env
TRACKING_URI=sqlite:///mlflow.db
MODEL_NAME=champion_model
MODEL_ALIAS=champion
DEPLOYMENT_BACKEND=<backend>
```

Use `MODEL_VERSION` only when an exact registered version should override the alias.
Configure only the section for the backend you are using. Other backend settings are ignored.

Check the configuration before changing infrastructure:

```bash
python deploy.py --backend <backend> --dry-run
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

Set the Kubernetes section in `.env`:

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
python deploy.py --backend kubernetes --dry-run
python deploy.py --backend kubernetes
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

## 4. AWS SageMaker Deployment

### Prerequisites

- Docker is running.
- AWS credentials are available through `.env` or the host environment.
- The SageMaker execution role exists.
- The configured ECR repository path and AWS region belong together.

### Configure

Set the AWS section in `.env`, especially:

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
python deploy.py --backend aws_sagemaker --dry-run
python deploy.py --backend aws_sagemaker
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

## 5. Docker Compose Deployment

### Prerequisites

- Docker is running.
- Docker Compose v2 is available.
- The selected host ports are free.

### Configure

Set the Compose section in `.env`, for example:

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
python deploy.py --backend docker_compose --dry-run
python deploy.py --backend docker_compose
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

## 6. Teardown and Changing Deployment

Teardown is a dry-run by default:

```bash
python -m model_deployment.teardown --backend <backend>
```

To actually remove the selected deployment:

```bash
python -m model_deployment.teardown --backend <backend> --execute
```

To change backend, model version, ports, resources, or monitoring settings:

1. Update `.env`.
2. Run `python deploy.py --backend <backend> --dry-run`.
3. Run `python deploy.py --backend <backend>`.
4. Teardown the previous deployment only when it is no longer required.

The teardown code is scoped to the selected deployment and does not intentionally remove unrelated infrastructure.
