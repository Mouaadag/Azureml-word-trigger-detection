import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import json
from azure.ai.ml import Input, Output
import logging
import mlflow
import mlflow.pyfunc
import os
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: Input(type="uri_folder"),
    test_data: Input(type="uri_folder"),
    output_metrics: Output(type="uri_file"),
    experiment_name: str = "trigger-word-detection",
) -> str:
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        try:
            model_path = model_path if isinstance(model_path, str) else model_path.path
            test_data_path = test_data if isinstance(test_data, str) else test_data.path
            output_metrics_path = (
                output_metrics
                if isinstance(output_metrics, str)
                else output_metrics.path
            )

            logger.info(f"Loading model from: {model_path}")
            model = mlflow.pyfunc.load_model(model_path)

            logger.info(f"Loading test data from: {test_data_path}")
            X_test = np.load(os.path.join(test_data_path, "X_test.npy"))
            y_test = np.load(os.path.join(test_data_path, "y_test.npy"))

            logger.info(
                f"Test data shapes - X_test: {X_test.shape}, y_test: {y_test.shape}"
            )

            logger.info("Performing predictions")
            y_pred = model.predict(X_test)
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()

            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred_classes)),
                "precision": float(precision_score(y_test, y_pred_classes)),
                "recall": float(recall_score(y_test, y_pred_classes)),
                "f1_score": float(f1_score(y_test, y_pred_classes)),
            }
            mlflow.log_dict(metrics, "metrics.json")

            cm = confusion_matrix(y_test, y_pred_classes)
            metrics["confusion_matrix"] = cm.tolist()

            logger.info(f"Saving metrics to: {output_metrics_path}")
            with open(output_metrics_path, "w") as f:
                json.dump(metrics, f)

            logger.info("Evaluation metrics:")
            for key, value in metrics.items():
                if key != "confusion_matrix":
                    logger.info(f"{key}: {value}")
                    mlflow.log_metric(key, value)

            logger.info(f"Confusion Matrix:\n{cm}")
            mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")

            # Log the evaluation dataset
            logger.info("Logging test data as artifacts")
            mlflow.log_artifact(os.path.join(test_data_path, "X_test.npy"), "test_data")
            mlflow.log_artifact(os.path.join(test_data_path, "y_test.npy"), "test_data")

            logger.info(f"Evaluation completed. Metrics saved to {output_metrics_path}")

        except Exception as e:
            logger.error(f"An error occurred during evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    return output_metrics_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_metrics", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="trigger-word-detection")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        output_path = evaluate_model(
            args.model_path, args.test_data, args.output_metrics, args.experiment_name
        )
        print(f"Evaluation completed. Metrics saved to {output_path}")
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        raise
