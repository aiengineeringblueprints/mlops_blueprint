# Application Example: MLflow, LakeFS

This repository serves as a reference for using LakeFS and MLflow. As an application example, a model for image-based anomaly detection is trained. Due to limited storage capacity, this repository contains only a small amount of sample data. To access the complete dataset, please contact Tobias Saur.

## Infrastructure Setup

Follow steps 0 and 1 from https://github.com/aiengineeringblueprints/lakefs_demo to set up LakeFS and MinIO.

## Setup of the MinIO Bucket and LakeFS Repository (automated in setup_lakefs.py)

1. Create MinIO bucket:

```bash
mc mb localminio/anomaly-detection
```

2. Create LakeFS repository:

```bash
lakectl repo create lakefs://anomaly-detection-data s3://anomaly-detection
```

3. Transfer local dataset to MinIO bucket:

```bash
mc cp ./data/ lakefs/anomaly-detection-data/main/data --recursive
```

4. Initialize local LakeFS repository:

```bash
lakectl local init lakefs://anomaly-detection-data/main/data data
```

5. Commit the initial version in LakeFS:

```bash
lakectl commit lakefs://anomaly-detection-data/main -m "Commit initial data version"
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

[`model_deployment/README.md`](model_deployment/README.md)

It contains the `.env` setup, Kubernetes/KServe, AWS SageMaker, Docker Compose, testing, and teardown steps.
