import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


# plot confusion matrix
def plot_confusion_matrix(test_labels, predicted_labels, save_path):
    confusion_matrix_predicted = confusion_matrix(test_labels, predicted_labels)
    label_words = np.unique(np.concatenate((test_labels, predicted_labels)))
    plt.figure(figsize=(10, 8))
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    sns.heatmap(
        confusion_matrix_predicted,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=label_words,
        yticklabels=label_words,
    )
    # Adjust the labels
    plt.xticks(rotation=45)  # Rotates the x-axis labels
    plt.yticks(rotation=45)  # Rotates the y-axis labels
    plt.tight_layout()  # Adjusts the figure parameters to avoid overlap
    plt.subplots_adjust(bottom=0.2)  # Adjusts the bottom margin of the figure
    plt.savefig(save_path)
