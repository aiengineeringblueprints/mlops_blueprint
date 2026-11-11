# MLOps Blueprint for Model Training, Deployment, and Monitoring

<p align="center">
  <img src="assets/mlops_blueprint.png"
       alt="MLOps Reference Architecture"
       width="900">
</p>
<p align="center">
  <em>Figure: MLOps reference architecture and implementation overview.</em>
</p>

This repository is a technology-specific blueprint for the path from versioned training data to a served and monitored model. It combines a reference architecture with a reference implementation, so that a team can start from working code and adapt it instead of assembling the tools from scratch.

It is meant for data scientists and ML engineers who need a traceable and reproducible training workflow and a repeatable way to deploy and monitor the resulting model, on their own infrastructure or in the cloud.

## How the parts fit together

The blueprint covers two stages of the ML lifecycle, which meet at a model registry:

1. **Training** versions the data, tracks experiments, selects a champion model and registers it in an MLflow model registry.
2. **Deployment** resolves a model from that registry by version or by champion alias, packages it into a serving image, deploys it to one of three targets and connects monitoring.

Each stage can be used on its own. Deployment also accepts a standalone local model file instead of a registry entry.

## Which training variant to use

| Variant | Use it for | Tooling |
|---|---|---|
| `model_training/` | Larger data, such as image data from several hundred MB upwards, several people working on the same data, or a need for auditable data versions | lakeFS on an S3-compatible object store, MLflow for experiment tracking and the model registry |
| `model_training_small/` | Small tabular or time-series files and a single data scientist without server infrastructure | DVC and DVCLive, repository-based and serverless |

The small-data variant keeps everything in the repository and has no model registry. [`docs/training-architecture.md`](docs/training-architecture.md) describes the selection rule and both variants in detail.

## Getting started

1. Clone the repository.
2. Choose the stage you need, and for training the variant that fits your data.
3. Follow the README of that component. Each one lists its prerequisites, its configuration and the commands to run.

The deployment stage needs Python 3.10 or later, Docker and Docker Compose v2; further prerequisites depend on the chosen target. The large-data training variant needs the `lakectl` command-line client, which is not installed with its Python dependencies.

## Repository contents

### `model_training/`

Supporting workflow for model training and registration.

See [`model_training/README.md`](model_training/README.md) for setup and execution instructions.

### `model_training_small/`

Training variant for small tabular or time-series data. It is repository-based and needs no server: data is versioned with DVC, the pipeline is defined in `dvc.yaml`, and DVCLive records metrics per run.

See [`model_training_small/README.md`](model_training_small/README.md) for setup and execution instructions. The architecture documentation states when to choose this variant over `model_training/`.

### `model_deployment/`

Deployment and monitoring blueprint. It supports deployment from an MLflow-registered model or a standalone local model through AWS SageMaker, Kubernetes/KServe, or Docker Compose.

See [`model_deployment/README.md`](model_deployment/README.md) for prerequisites, configuration, deployment, monitoring, and teardown instructions.

### `docs/`

Architecture documentation of the blueprint: the requirements each part addresses, its components and workflow, the design decisions with their rationale, and known limitations.

- [`docs/training-architecture.md`](docs/training-architecture.md): training, both variants
- [`docs/deployment-architecture.md`](docs/deployment-architecture.md): deployment and monitoring

See [`docs/README.md`](docs/README.md) for an overview.

## Origin

The blueprint was developed in the AI Engineering research group at Ansbach University of Applied Sciences. The training stage is based on a bachelor's thesis that searched scientific and grey literature systematically (44 sources), took the most frequently named tools and compared them against a catalogue of 20 requirements for data versioning, experiment tracking and model management. The deployment stage is based on a master's thesis that followed design science research: a systematic review of 307 publications led to nine functional and ten non-functional requirements, which were consolidated into the architectural responsibilities the implementation realises.

The requirement catalogues with the source of each requirement, the design decisions with their rationale and the known limitations are documented in [`docs/`](docs/README.md).

## Evaluated version

The version evaluated in the paper "AERA: Agent-Based Evaluation of Reference Architectures and Blueprints" is tag [`aera-evaluated`](https://github.com/aiengineeringblueprints/mlops_blueprint/tree/aera-evaluated), archived at [doi:10.5281/zenodo.24137665](https://doi.org/10.5281/zenodo.24137665). Later changes on `main` do not affect the results reported for that version.

## License and citation

The blueprint is released under the MIT License, see [`LICENSE`](LICENSE).

To cite the evaluated version, use its archived release: [doi:10.5281/zenodo.24137665](https://doi.org/10.5281/zenodo.24137665).
