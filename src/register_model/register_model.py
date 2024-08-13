import argparse
import json
import mlflow
import tensorflow as tf
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import ManagedIdentityCredential
from azure.core.exceptions import (
    ClientAuthenticationError,
)
import logging
from datetime import datetime
import time
import os

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_ml_client(max_retries=3, retry_delay=5):
    subscription_id = "<subscription_id>"
    resource_group = "<resource_group>"
    workspace_name = "<workspace_name>"

    # Client ID of your user-assigned managed identity
    user_assigned_identity_client_id = "<user_assigned_identity_client_id>"

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Attempt {attempt + 1}: Authenticating using User-Assigned Managed Identity"
            )
            credential = ManagedIdentityCredential(
                client_id=user_assigned_identity_client_id
            )
            ml_client = MLClient(
                credential=credential,
                subscription_id=subscription_id,
                resource_group_name=resource_group,
                workspace_name=workspace_name,
            )
            # Test the client with a simple operation
            ml_client.workspaces.get(name=workspace_name)
            logger.info(
                "Successfully authenticated using User-Assigned Managed Identity"
            )
            return ml_client
        except Exception as e:
            logger.warning(
                f"User-Assigned Managed Identity authentication failed: {str(e)}"
            )

            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise ClientAuthenticationError(
                    "Authentication failed after multiple attempts"
                ) from e

    raise ClientAuthenticationError("Authentication failed after multiple attempts")


def register_model(
    model_path: str,
    evaluation_results: str,
    registered_model: str,
    model_name: str = "trigger_word_detection_model",
    experiment_name: str = "trigger-word-detection",
):
    ml_client = get_ml_client()
    mlflow.set_experiment(experiment_name)
    accuracy_threshold = 0.1
    with mlflow.start_run():
        try:
            logger.info(f"Loading evaluation results from {evaluation_results}")
            with open(evaluation_results, "r") as f:
                metrics = json.load(f)

            accuracy = metrics.get("accuracy", 0.0)

            for key, value in metrics.items():
                if key != "confusion_matrix":
                    mlflow.log_metric(key, value)
                    logger.info(f"Logged metric: {key} = {value}")

            if accuracy <= accuracy_threshold:
                logger.info(
                    f"Model accuracy {accuracy} is not greater than {accuracy_threshold}. Model will not be registered."
                )
                return None

            # Load the model using MLflow
            loaded_model = mlflow.keras.load_model(model_path)
            logger.info(f"Loaded MLflow model from {model_path}")

            # Log the model with MLflow
            mlflow_model_info = mlflow.keras.log_model(
                loaded_model, "model", registered_model_name=model_name
            )
            logger.info(
                f"Model logged with MLflow. Model URI: {mlflow_model_info.model_uri}"
            )
            # Get the latest version of the registered model
            mlflow_client = mlflow.tracking.MlflowClient()
            registered_model = mlflow_client.get_registered_model(model_name)
            latest_versions = registered_model.latest_versions
            mlflow_version = latest_versions[-1].version if latest_versions else None

            # Set tags for the MLflow model
            mlflow_client = mlflow.tracking.MlflowClient()
            mlflow_client.set_registered_model_tag(
                model_name,
                "accuracy",
                str(accuracy),
            )
            mlflow_client.set_registered_model_tag(
                model_name, "passed_threshold", "true"
            )

            # Register the model with Azure ML
            # Prepare tags
            tags = {
                "accuracy": str(accuracy),
                "passed_threshold": "true",
                "framework": "Keras",
                "framework_version": tf.__version__,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "registered with": "Azure ML",
            }
            logger.info(f"Registering model with Azure ML: {model_name}")
            try:
                model = Model(
                    path=mlflow_model_info.model_uri,
                    name=model_name,
                    description="Trigger word detection model",
                    type="mlflow_model",
                    tags=tags,
                )
                azure_registered_model = ml_client.models.create_or_update(model)
                logger.info(
                    f"Model registered with Azure ML. Name: {azure_registered_model.name}, Version: {azure_registered_model.version}"
                )
            except Exception as e:
                logger.error(f"Failed to register model with Azure ML: {str(e)}")
                raise

            model_info = {
                "registered": True,
                "name": model_name,
                "mlflow_version": mlflow_version,
                "azure_ml_version": azure_registered_model.version,
                "run_id": mlflow.active_run().info.run_id,
                "accuracy": accuracy,
                "passed_threshold": "true",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            mlflow.log_dict(model_info, "model_info.json")
            logger.info(f"Model info saved to model_info.json")
            return registered_model

        except Exception as e:
            logger.error(
                f"An error occurred during model registration: {str(e)}", exc_info=True
            )
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--evaluation_results", type=str, required=True)
    parser.add_argument("--registered_model", type=str, required=True)
    parser.add_argument(
        "--model_name", type=str, default="trigger_word_detection_model"
    )
    parser.add_argument("--experiment_name", type=str, default="trigger-word-detection")
    args = parser.parse_args()

    try:
        registered_model_path = register_model(
            args.model_path,
            args.evaluation_results,
            args.registered_model,
            args.model_name,
            args.experiment_name,
        )
        if registered_model_path:
            logger.info(
                f"Model registration process completed. Results saved to {registered_model_path}"
            )
        else:
            logger.info(
                "Model registration was not performed due to accuracy not exceeding {accuracy_threshold}."
            )
    except Exception as e:
        logger.error(f"Model registration failed: {str(e)}", exc_info=True)
        raise
