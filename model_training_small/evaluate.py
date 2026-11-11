# Using the tensorflow guide for imbalanced data as reference: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc, average_precision_score

def evaluate_model(data_path="data/preprocessed/preprocessed_data.npz", model_path="models/trained_nn_model.keras", decision_threshold=0.5):

    # Laden der Testdaten
    with np.load(data_path, allow_pickle=False) as data:
        _ = data['x_train']
        _ = data['y_train'].reshape(-1, 1)
        _ = data['x_val']
        _ = data['y_val'].reshape(-1, 1)
        x_test = data['x_test']
        y_test = data['y_test'].reshape(-1, 1)

    # Laden des trainierten Modells
    model = keras.models.load_model(model_path)

    
    print(f"Evaluating with decision threshold: {decision_threshold}")

    # Vorhersagen treffen (Wahrscheinlichkeiten)
    y_pred_proba = model.predict(x_test).flatten()
    y_pred = (y_pred_proba > decision_threshold).astype(int)
    
    # PR AUC und ROC AUC berechnen
    pr_auc = average_precision_score(y_test, y_pred_proba)
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Confusion Matrix und Precision-Recall Kurve berechnen
    cm = confusion_matrix(y_test, y_pred)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)

    
    print(f"PR AUC (Precision-Recall): {pr_auc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Detaillierter Classification Report
    print("=" * 60)
    print("Classification Report:")
    print("=" * 60)
    print(classification_report(y_test, y_pred, 
                                target_names=['No Failure', 'Failure'],
                                digits=4))
    
    # Grafiken erstellen (bestehender Einzel-Threshold-Plot)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fig.suptitle(f'Model Evaluation (Decision Threshold = {decision_threshold})', fontsize=16)
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Failure', 'Failure'],
                yticklabels=['No Failure', 'Failure'],
                ax=axes[0, 0])
    axes[0, 0].set_title(f'Confusion Matrix (PR AUC={pr_auc:.3f}, ROC AUC={roc_auc:.3f})')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # Precision-Recall Kurve
    axes[0, 1].plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[0, 1].axhline(y=np.sum(y_test) / len(y_test), color='navy', linestyle='--', lw=2, label='Baseline (No Skill)')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].grid(alpha=0.3)
    
    # ROC Kurve
    axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Baseline (No Skill)')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(alpha=0.3)
    
    # Leere Grafik (für spätere Nutzung)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/model_evaluation_nn.png', dpi=300)

    # Zusätzlicher Plot: mehrere Confusion-Matrizen für verschiedene Thresholds
    thresholds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    thresholds = np.array(thresholds)
    n = len(thresholds)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    # Vereinheitlichen auf 1D-Array für einfaches Indexing
    if nrows * ncols == 1:
        axes_flat = np.array([axes2])
    else:
        axes_flat = axes2.flatten()

    for i, thr in enumerate(thresholds):
        y_pred_thr = (y_pred_proba > thr).astype(int)
        cm_thr = confusion_matrix(y_test, y_pred_thr)

        sns.heatmap(cm_thr, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Failure', 'Failure'],
                    yticklabels=['No Failure', 'Failure'],
                    ax=axes_flat[i])
        axes_flat[i].set_title(f'Threshold = {thr:.2f}')
        axes_flat[i].set_xlabel('Predicted')
        axes_flat[i].set_ylabel('True')

    # Falls Subplots übrig bleiben, ausblenden
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    outname = f"results/model_confusion_matrices_thresholds_{thresholds.min():.2f}_to_{thresholds.max():.2f}.png"
    plt.savefig(outname, dpi=300)

if __name__ == "__main__":
    # Parameter aus params.yaml laden
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    eval_params = params.get("evaluate", {})
    data_path = eval_params.get("data_path", "data/preprocessed/preprocessed_data.npz")
    model_path = eval_params.get("model_path", "models/trained_nn_model.keras")
    decision_threshold = eval_params.get("decision_threshold", 0.5)

    evaluate_model(
        data_path=data_path,
        model_path=model_path,
        decision_threshold=decision_threshold,
    )