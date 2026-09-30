# Small-data training variant (DVC, DVCLive)

The blueprint's second training variant for small tabular or time-series data: everything is
repository-based and serverless. Data is versioned with DVC, the pipeline is defined in
`dvc.yaml`, parameters live in `params.yaml`, and DVCLive records metrics and plots per run.
The reference application is the Kaggle predictive-maintenance dataset
(https://www.kaggle.com/datasets/hiimanshuagarwal/predictive-maintenance-dataset): daily device
telemetry with a rare `failure` target.

Origin: BA Saur 2026 (`Anwendungsbeispiel_DVC`). Added 2026-09-30 for the agent evaluation:
the preprocess stage persists its scaler (`models/scaler.joblib`) and `predict.py` scores raw rows,
so the variant has an inference path.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc init
dvc add data/raw/predictive_maintenance_dataset.csv
git add . && git commit -m "DVC: raw data tracked"
```

## Pipeline

Three stages in `dvc.yaml`:

1. **preprocess**: reads `data/raw/predictive_maintenance_dataset.csv`, splits by row into train,
   validation and test, standardises the nine metric columns (scaler saved to
   `models/scaler.joblib`), optionally oversamples the training split (SMOTE), writes
   `data/preprocessed/preprocessed_data.npz`.
2. **train**: trains a small dense network with class weights and early stopping; DVCLive writes
   `dvclive/metrics.json` and plots; the model goes to `models/trained_nn_model.keras`.
3. **evaluate**: precision-recall and ROC curves and confusion matrices on the test split
   (`results/`).

Run the whole pipeline (only stages whose inputs changed are executed again):

```bash
dvc repro
```

## Parameters (`params.yaml`)

Hyperparameters are defined centrally in `params.yaml` and read by the stage scripts
(`preprocess.*`, `train.epochs`, `train.batch_size`, `train.learning_rate`,
`train.use_class_weights`, `evaluate.decision_threshold`).

## Experiments

Compare runs with different hyperparameters and keep the best:

```bash
dvc exp run --name baseline
dvc exp run --name more-epochs --set-param train.epochs=15
dvc exp show            # parameters and metrics (val_prc: PR-AUC on the validation split) side by side
dvc exp apply more-epochs   # write the chosen experiment's params, model and metrics into the workspace
```

## Prediction

```bash
python predict.py --input new_rows.csv --output predictions.csv
```

`new_rows.csv` has the raw columns (`date`, `device`, `metric1` ... `metric9`); the output repeats
`date` and `device` and adds `failure_probability`, one row per input row in input order. The
script applies the persisted scaler and the trained model from `models/`.
