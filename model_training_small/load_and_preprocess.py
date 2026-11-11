# Using the tensorflow guide for imbalanced data as reference: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

import numpy as np
import pandas as pd
import os
import yaml
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

# The fitted scaler and the feature order, persisted for inference (predict.py).
SCALER_PATH = "models/scaler.joblib"

# Load data from csv file to pandas DataFrame
def load_data(file_path):
    data = pd.read_csv(file_path)
    print(f"Data loaded from {file_path} with shape {data.shape}")
    return data

def preprocess_data(
    df,
    smote_strategy=0.01,
    test_size=0.3,
    val_size=0.3,
    random_state=42,
    save_path="data/preprocessed/preprocessed_data.npz",
):
    # Datum in numerische Werte umwandeln
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # Zielvariable
    y = df["failure"]

    # Features auswählen: alle numerischen Spalten außer failure
    x = df.drop(columns=["failure", "date", "device", "year", "month", "day"])

    # Split in Trainings- und Testdaten
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y)

    # Split in Training und Validierung
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train)

    # Verteilung der Klassen in Trainings-, Validierungs- und Testdaten
    print(f'Average class probability in training set:   {y_train.mean():.4f}')
    print(f'Average class probability in validation set: {y_val.mean():.4f}')
    print(f'Average class probability in test set:       {y_test.mean():.4f}')

    # Skalierung der Features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    # Persist the fitted scaler with the feature order, so that inference can reproduce the
    # preprocessing exactly (the raw data has 9 metric columns; date and device are not features).
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump({"scaler": scaler, "features": list(x.columns)}, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    if smote_strategy is not None:
        print(f"Applying SMOTE with sampling_strategy={smote_strategy} to training data.")
        # Anwendung von SMOTE auf die Trainingsdaten
        smote = SMOTE(sampling_strategy=smote_strategy, random_state=42, k_neighbors=3)
        x_train_scaled, y_train = smote.fit_resample(x_train_scaled, y_train)

    # Shape der Datensätze ausgeben
    print('Training labels shape:', y_train.shape)
    print('Validation labels shape:', y_val.shape)
    print('Test labels shape:', y_test.shape)

    print('Training features shape:', x_train_scaled.shape)
    print('Validation features shape:', x_val_scaled.shape)
    print('Test features shape:', x_test_scaled.shape)

    # Vorverarbeitete Daten speichern
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path,
                        x_train=x_train_scaled,
                        x_val=x_val_scaled,
                        x_test=x_test_scaled,
                        y_train=y_train,
                        y_val=y_val,
                        y_test=y_test)
    print(f"Preprocessed data saved to {save_path}")

    return x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test

if __name__ == "__main__":
    # Parameter aus params.yaml laden
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    pre_params = params.get("preprocess", {})
    smote_strategy = pre_params.get("smote_strategy", 0.5)
    test_size = pre_params.get("test_size", 0.3)
    val_size = pre_params.get("val_size", 0.3)
    random_state = pre_params.get("random_state", 42)

    raw_data = load_data("data/raw/predictive_maintenance_dataset.csv")
    # Smote strategy = None disables SMOTE
    x_train, x_val, x_test, y_train, y_val, y_test = preprocess_data(
        raw_data,
        smote_strategy=smote_strategy,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        save_path="data/preprocessed/preprocessed_data.npz",
    )