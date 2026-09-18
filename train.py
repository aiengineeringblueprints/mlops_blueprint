import os
from dotenv import load_dotenv
import numpy as np
from helper_functions import load_images_from_directory
from train_seitlich import train_cnn
from plot_metrics import plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import mlflow
import mlflow.tensorflow
from mlflow.data.meta_dataset import MetaDataset
from mlflow.tracking import MlflowClient
from mlflow.data.http_dataset_source import HTTPDatasetSource
import logging
import tensorflow as tf
import yaml

# Load environment variables from the .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch variables from .env, fallback to default values if not found
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
CHAMPION_METRIC = os.getenv("CHAMPION_METRIC", "validation_accuracy")
TRACKING_URI = os.getenv("TRACKING_URI", "sqlite:///mlflow.db")

# TensorFlow setup
logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
mlflow.set_tracking_uri(TRACKING_URI)

# MLFlow: Enable autologging for TensorFlow/Keras.
mlflow.tensorflow.autolog(log_datasets=False)

# Sets (or creates) the MLflow experiment for all runs in this script
mlflow.set_experiment(EXPERIMENT_NAME)

# Client for registry/tagging operations
client = MlflowClient()

# Path to the data
train_directory = "./data/train"
val_directory = "./data/val"
test_directory = "./data/test"

# Parameters:
image_size = (224, 224)
epoch = 10
batch_size = 32
num_classes = len(os.listdir(train_directory))

# MLflow: Start the tracking run
with mlflow.start_run() as run:
    # Extract information about LakeFS reference
    lakefs_ref_path = os.path.join("data", ".lakefs_ref.yaml")
    repo = None
    commit = None
    if os.path.exists(lakefs_ref_path):
        with open(lakefs_ref_path, "r") as f:
            ref = yaml.safe_load(f) or {}
        src = ref.get("src", "")
        commit = ref.get("at_head") or ref.get("at_head_commit_id")
        if isinstance(src, str) and src.startswith("lakefs://"):
            repo = src[len("lakefs://") :].split("/")[0]

    if repo and commit:
        dataset_source_url = f"lakefs://{repo}/{commit}"
    else:
        dataset_source_url = None

    # Logging the used dataset in MLflow
    dataset_source = HTTPDatasetSource(url=dataset_source_url)
    mlflow_dataset = MetaDataset(source=dataset_source, name="Augmentierte Bilddaten")
    mlflow.log_input(mlflow_dataset, context="training")

    # Load and prepare data
    train_images, train_labels = load_images_from_directory(train_directory, image_size)
    val_images, val_labels = load_images_from_directory(val_directory, image_size)
    test_images, test_labels = load_images_from_directory(test_directory, image_size)

    # Normalize the image data
    train_images = train_images.astype("float32") / 255.0
    val_images = val_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    # Initialize the OneHotEncoder
    os.makedirs("./model", exist_ok=True)
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    train_labels = one_hot_encoder.fit_transform(train_labels.reshape(-1, 1))
    val_labels = one_hot_encoder.transform(val_labels.reshape(-1, 1))
    test_labels = one_hot_encoder.transform(test_labels.reshape(-1, 1))

    # Preparation for training
    X_train, y_train = train_images, train_labels
    X_val, y_val = val_images, val_labels
    X_test, y_test = test_images, test_labels

    # Check the split
    logger.info(f"Training data: {X_train.shape} {y_train.shape}")
    logger.info(f"Validation data: {X_val.shape} {y_val.shape}")
    logger.info(f"Test data: {X_test.shape} {y_test.shape}")

    # Train CNN
    cnn_model = train_cnn(
        epoch, batch_size, image_size, num_classes, X_train, y_train, X_val, y_val
    )

    # Independent validation evaluation used for champion selection.
    cnn_val_loss, cnn_val_accuracy = cnn_model.evaluate(X_val, y_val, verbose=0)
    validation_metrics = {
        "validation_loss": cnn_val_loss,
        "validation_accuracy": cnn_val_accuracy,
    }
    mlflow.log_metrics(validation_metrics)
    logger.info(f"Validation Loss CNN: {cnn_val_loss}")
    logger.info(f"Validation Accuracy CNN: {cnn_val_accuracy}")

    # Refresh run metrics to capture Keras validation logs for CHAMPION_METRIC.
    current_run_metrics = client.get_run(run.info.run_id).data.metrics
    current_exp_score = current_run_metrics.get(CHAMPION_METRIC)
    if current_exp_score is None:
        raise RuntimeError(
            "CHAMPION-SELECTION BLOCKER: No independent validation metric "
            f"named '{CHAMPION_METRIC}' exists in the current run."
        )

    current_champion_runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string="tags.champion = 'true'",
        order_by=["attributes.start_time DESC"],
        max_results=100,
        output_format="list",
    )

    best_score = float("-inf")
    current_champion = None
    for champion_run in current_champion_runs:
        score = champion_run.data.metrics.get(CHAMPION_METRIC)
        if score is None:
            raise RuntimeError(
                "CHAMPION-SELECTION BLOCKER: Existing champion run "
                f"{champion_run.info.run_id} does not contain validation metric "
                f"'{CHAMPION_METRIC}'."
            )
        if score > best_score:
            best_score = score
            current_champion = champion_run

    if current_exp_score > best_score:
        logger.info(
            f"New Champion-Run ({CHAMPION_METRIC}): "
            f"{current_exp_score:.4f} > {best_score:.4f}"
        )
        # Normalize to exactly one champion tag.
        for champion_run in current_champion_runs:
            client.delete_tag(champion_run.info.run_id, "champion")
        client.set_tag(run.info.run_id, "champion", "true")
        client.set_tag(run.info.run_id, "champion_metric", CHAMPION_METRIC)
    else:
        logger.info(
            f"No new champion ({CHAMPION_METRIC}): "
            f"{current_exp_score:.4f} <= {best_score:.4f}"
        )
        # Keep the champion's validation metric provenance and normalize tags.
        for champion_run in current_champion_runs:
            if (
                current_champion is None
                or champion_run.info.run_id != current_champion.info.run_id
            ):
                client.delete_tag(champion_run.info.run_id, "champion")
        if current_champion is not None:
            client.set_tag(current_champion.info.run_id, "champion", "true")
            client.set_tag(
                current_champion.info.run_id, "champion_metric", CHAMPION_METRIC
            )

    cnn_test_probs = cnn_model.predict(X_test)
    cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Loss CNN: {cnn_test_loss}")
    logger.info(f"Test Accuracy CNN: {cnn_test_accuracy}")

    cnn_test_pred = one_hot_encoder.inverse_transform(cnn_test_probs)
    # Output of the confusion matrix
    y_test_labels = one_hot_encoder.inverse_transform(y_test)
    os.makedirs("./metrics", exist_ok=True)
    plot_confusion_matrix(
        y_test_labels, cnn_test_pred, save_path="./metrics/cnn_confusion_matrix.png"
    )

    # Binary classification evaluation
    y_test_binary = np.where(y_test_labels == "Standard-Schraube", 0, 1)
    cnn_test_pred_binary = np.where(cnn_test_pred == "Standard-Schraube", 0, 1)
    cnn_test_accuracy_binary = np.mean(y_test_binary == cnn_test_pred_binary)
    logger.info(f"Test Accuracy CNN Binary: {cnn_test_accuracy_binary}")
    # MLFlow: Logging custom test metrics
    mlflow.log_metrics(
        {
            "test_loss": cnn_test_loss,
            "test_accuracy": cnn_test_accuracy,
            "test_accuracy_binary": cnn_test_accuracy_binary,
        }
    )
    # MLFlow: Append confusion matrix as an artifact to the run
    mlflow.log_artifacts("./metrics", artifact_path="metrics_plots")
