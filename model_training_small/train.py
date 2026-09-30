# Using the tensorflow guide for imbalanced data as reference: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

import os
import yaml

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
from dvclive.keras import DVCLiveCallback

def train_model(
    data_path="data/preprocessed/preprocessed_data.npz",
    output_bias=None,
    use_class_weights=True,
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
):

    # Laden der vorverarbeiteten Daten aus der NPZ-Datei
    with np.load(data_path, allow_pickle=False) as data:
        x_train = data['x_train']
        y_train = data['y_train'].reshape(-1, 1)
        x_val = data['x_val']
        y_val = data['y_val'].reshape(-1, 1)
        _ = data['x_test']
        _ = data['y_test'].reshape(-1, 1)

    # Definieren der Metriken
    metrics = [
      keras.metrics.BinaryCrossentropy(name='cross entropy'),
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR')]

    # Initialer Bias für die Output-Schicht
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    # Berechnen der Class Weights
    weight_for_0 = (1 / np.sum(y_train == 0)) * (len(y_train)) / 2.0
    weight_for_1 = (1 / np.sum(y_train == 1)) * (len(y_train)) / 2.0
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print(f"Class weights: {class_weights}")

    # Definition des Modells
    model = keras.Sequential([
      keras.layers.Dense(
          16, activation='relu',
          input_shape=(x_train.shape[-1],)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias)])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)
    
    # Early Stopping Callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_prc',
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)
    
    model.summary()

    # DVC Live Callback hinzufügen (Standardkonfiguration, schreibt nach ./dvclive)
    live_callback = DVCLiveCallback()

    if use_class_weights:
        print("Using class weights for training.")
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, live_callback],
            class_weight=class_weights,
            verbose=2)
    else:
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping, live_callback],
            verbose=2)

    # Speichern des trainierten Modells
    os.makedirs('models', exist_ok=True)
    model.save('models/trained_nn_model.keras')

    # Visualisierung der Trainingshistorie
    metrics_for_training_plot = ['cross entropy', 'accuracy', 'precision', 'recall', 'auc', 'prc']
    plt.figure(figsize=(12, 8))
    for n, metric in enumerate(metrics_for_training_plot):
        plt.subplot(3, 2, n + 1)
        plt.plot(history.history[metric], label='Train')
        plt.plot(history.history['val_' + metric], label='Validation')
        plt.title(metric)
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.xticks(range(len(history.history[metric])), range(1, len(history.history[metric]) + 1))
        plt.legend()
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_metrics.png')

    return model, history

if __name__ == "__main__":
    # Parameter aus params.yaml laden
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    train_params = params.get("train", {})
    data_path = train_params.get("data_path", "data/preprocessed/preprocessed_data.npz")
    epochs = train_params.get("epochs", 100)
    batch_size = train_params.get("batch_size", 32)
    learning_rate = train_params.get("learning_rate", 0.001)
    use_class_weights = train_params.get("use_class_weights", True)

    model, history = train_model(
        data_path=data_path,
        output_bias=None,
        use_class_weights=use_class_weights,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )