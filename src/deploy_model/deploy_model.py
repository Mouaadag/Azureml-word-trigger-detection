from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
)
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
import argparse
import json
import logging
import time
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_ml_client(max_retries=3, retry_delay=5):
    subscription_id = "<subscription_id>"
    resource_group = "<resource_group>"
    workspace_name = "<workspace_name>"
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
                raise


def main(args):
    # Connect to the workspace
    ml_client = get_ml_client()

    # Create or update the endpoint
    endpoint = ManagedOnlineEndpoint(
        name=args.endpoint_name,
        description="Endpoint for trigger word detection",
        auth_mode="key",
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    # Get the model
    model = ml_client.models.get(name=args.model, version="3")
    logger.info(f"Using model {model.name}, version {model.version}")

    # Create the deployment
    deployment = ManagedOnlineDeployment(
        name=args.deployment_name,
        endpoint_name=args.endpoint_name,
        model=model,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()

    # Update traffic allocation
    endpoint = ml_client.online_endpoints.get(args.endpoint_name)
    endpoint.traffic = {args.deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    # Get the scoring URI
    endpoint = ml_client.online_endpoints.get(args.endpoint_name)
    scoring_uri = endpoint.scoring_uri

    # Save deployment details
    deployment_details = {
        "endpoint_name": args.endpoint_name,
        "deployment_name": args.deployment_name,
        "model_name": model.name,
        "model_version": model.version,
        "scoring_uri": scoring_uri,
    }
    mlflow.log_dict(deployment_details, "deployment_details.json")

    logger.info(f"Deployment completed. Details saved to {args.deployment_details}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--endpoint_name", type=str, required=True)
    parser.add_argument("--deployment_name", type=str, required=True)
    parser.add_argument("--instance_type", type=str, required=True)
    parser.add_argument("--instance_count", type=int, required=True)
    parser.add_argument("--deployment_details", type=str, required=True)
    args = parser.parse_args()
    main(args)
