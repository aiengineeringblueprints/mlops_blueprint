# Application Example: MLflow, LakeFS

This repository serves as a reference for using LakeFS and MLflow. As an application example, a model for image-based anomaly detection is trained. Due to limited storage capacity, this repository contains only a small amount of sample data. To access the complete dataset, please contact Tobias Saur.


## Local Training Environment

Training is self-contained under `model_training/`. Install only its dependencies and create its own environment file from the repository root:

```bash
python -m pip install -r model_training/requirements.txt
cp model_training/.env.example model_training/.env
```

The training scripts load `model_training/.env` explicitly. Training data and generated `model/` and `metrics/` outputs are resolved relative to the `model_training/` directory, so the scripts can be launched from the repository root.
`lakectl` is an external command-line prerequisite for the LakeFS helper scripts; it is not installed by `model_training/requirements.txt`.

Typical commands are:

```bash
python model_training/setup_lakefs.py
python model_training/prepare_lakefs_branch.py
python model_training/data_augmentation.py
python model_training/train.py
python model_training/mlflow_registry.py
```

The manual LakeFS shell commands in the sections below assume `cd model_training` first, because they use the relative `data/` path.

## Run the whole workflow locally

The infrastructure (SeaweedFS, LakeFS, MLflow server) is one Compose stack. The bundled
SeaweedFS S3 store runs only under the `local-s3` profile:

    cd model_training/infra && docker compose --profile local-s3 up -d --wait

`lakectl` reads its endpoint and credentials from `model_training/.env` (`LAKECTL_*`), so no
`lakectl config` is needed. Licences: SeaweedFS Apache-2.0, LakeFS 1.86.0 Apache-2.0 (last
release before the Business Source License), MLflow Apache-2.0.
The complete sequence, from raw data to an exported champion model:

    python model_training/setup_lakefs.py            # bucket, repository, initial data commit
    python model_training/prepare_lakefs_branch.py   # branch "augmented", local checkout
    python model_training/data_augmentation.py       # 5 augmented images per original
    python model_training/commit_lakefs.py           # commit the data version (GIT_PUSH=true also pushes)
    python model_training/train.py                   # MLflow run, champion tag by validation_accuracy
    python model_training/mlflow_registry.py         # register the champion, alias "champion"
    python model_training/export_champion.py         # output/model.keras, classes.json, MODEL_CARD.md

`model_training/test/smoke_train.sh` runs exactly this sequence in a temporary copy and
checks the result through the MLflow REST API.

## Using an existing S3

To point the stack at an existing S3-compatible service (AWS, a company endpoint) instead of
the bundled SeaweedFS, set `S3_*` in both `infra/.env` and `model_training/.env`
(`S3_ENDPOINT`/`S3_HOST_ENDPOINT`, `S3_REGION`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`,
`S3_BUCKET`, and, if the presigned URLs LakeFS/`lakectl` hand out need a different host,
`S3_PRESIGNED_ENDPOINT`). Leave `S3_ENDPOINT` empty to select AWS. Start the stack without the
`local-s3` profile:

    cd model_training/infra && docker compose up -d --wait

`setup_lakefs.py` creates the bucket if it is missing (`s3:CreateBucket` on the configured
credentials); if company policy withholds that permission, pre-create the bucket out of band
and grant only read/write/list on it instead.

## Infrastructure Setup

Follow steps 0 and 1 from https://github.com/aiengineeringblueprints/lakefs_demo to set up LakeFS and an S3-compatible store.

## Setup of the S3 Bucket and LakeFS Repository (automated in setup_lakefs.py)

1. Create the S3 bucket (SeaweedFS or any other S3-compatible endpoint), idempotently, via boto3.

2. Create LakeFS repository:

```bash
lakectl repo create lakefs://anomaly-detection-data s3://anomaly-detection/anomaly-detection-data
```

3. Link the local data folder to the LakeFS branch path:

```bash
lakectl local init lakefs://anomaly-detection-data/main/data data
```

4. Commit the initial version in LakeFS; `lakectl` uploads the local files itself, so no
   separate S3 client is needed on the host:

```bash
lakectl local commit data -m "Commit initial data version"
```

## Preparation of a new branch for data augmentation (automated in prepare_lakefs_branch.py)

1. Create a new LakeFS branch:

```bash
lakectl branch create lakefs://anomaly-detection-data/augmented --source lakefs://anomaly-detection-data/main
```

2. Checkout to the newly created LakeFS branch:

```bash
lakectl local checkout data --ref lakefs://anomaly-detection-data/augmented --yes
```

## Performing Data Augmentation

Run the data augmentation (here: data_augmentation.py)

## Commit the augmented data to LakeFS (automated in commit_lakefs.py)

1. Commit the new data state to LakeFS:

```bash
lakectl local commit data -m "Adding some augmented pictures"
```

2. Optional: Merge the newly created LakeFS branch back to main:

```bash
lakectl merge lakefs://anomaly-detection-data/augmented lakefs://anomaly-detection-data/main
```

```bash
lakectl local checkout data --ref lakefs://anomaly-detection-data/main --yes
```

3. Commit the current data state or its reference to Git:

```bash
git add .
git commit -m "Adding some augmented pictures"
git push
```

## Setup and Start of MLflow

```bash
mlflow server --port 5000
```

## To reproduce the state of a previous experiment:

1. Determine the Git commit via MLflow:

```bash
git checkout <commit-hash>
```

2. Checkout to the linked version of the data:

```bash
lakectl local checkout data
```

## Model Deployment

If you want to deploy registered MLflow model, follow the dedicated deployment guide:

[`model_deployment/README.md`](../model_deployment/README.md)

It contains the `.env` setup, Kubernetes/KServe, AWS SageMaker, Docker Compose, testing, and teardown steps.
