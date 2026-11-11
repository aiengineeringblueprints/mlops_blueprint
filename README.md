# MLOps Blueprint for Model Training, Deployment, and Monitoring

<p align="center">
  <img src="assets/mlops_blueprint.png"
       alt="MLOps Reference Architecture"
       width="900">
</p>
<p align="center">
  <em>Figure: MLOps reference architecture and implementation overview.</em>
</p>

The repository contains three components and their architecture documentation:

## `model_training/`

Supporting workflow for model training and registration.

See [`model_training/README.md`](model_training/README.md) for setup and execution instructions.

## `model_training_small/`

Training variant for small tabular or time-series data. It is repository-based and needs no server: data is versioned with DVC, the pipeline is defined in `dvc.yaml`, and DVCLive records metrics per run.

See [`model_training_small/README.md`](model_training_small/README.md) for setup and execution instructions. The architecture documentation states when to choose this variant over `model_training/`.

## `model_deployment/`

Deployment and monitoring blueprint. It supports deployment from an MLflow-registered model or a standalone local model through AWS SageMaker, Kubernetes/KServe, or Docker Compose.

See [`model_deployment/README.md`](model_deployment/README.md) for prerequisites, configuration, deployment, monitoring, and teardown instructions.

## `docs/`

Architecture documentation of the blueprint: the requirements each part addresses, its components and workflow, the design decisions with their rationale, and known limitations.

- [`docs/training-architecture.md`](docs/training-architecture.md): training, both variants
- [`docs/deployment-architecture.md`](docs/deployment-architecture.md): deployment and monitoring

See [`docs/README.md`](docs/README.md) for an overview.

## Evaluated version

The version evaluated in the paper "AERA: Agent-Based Evaluation of Reference Architectures and Blueprints" is tag [`aera-evaluated`](https://github.com/aiengineeringblueprints/mlops_blueprint/tree/aera-evaluated), archived at [doi:10.5281/zenodo.24137665](https://doi.org/10.5281/zenodo.24137665). Later changes on `main` do not affect the results reported for that version.
