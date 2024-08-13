from azure.ai.ml import MLClient, Input, load_component, dsl
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
import logging
from datetime import datetime

# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
# )
logger = logging.getLogger(__name__)


def main():
    # Connect to the workspace
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
        logger.info("Successfully authenticated with DefaultAzureCredential")
    except Exception as ex:
        credential = InteractiveBrowserCredential()
        logger.info("Successfully authenticated with InteractiveBrowserCredential")

    try:
        ml_client = MLClient.from_config(credential=credential)
        workspace_info = ml_client.workspaces.get(ml_client.workspace_name)
        logger.info(f"Successfully connected to Workspace: {workspace_info.name}")
    except Exception as ex:
        logger.error(f"Error accessing workspace: {str(ex)}")
        raise

    # Define compute cluster
    compute_name = "word-trigger-deploy-cluster"
    try:
        compute_cluster = ml_client.compute.get(compute_name)
        logger.info(f"Found existing compute cluster: {compute_name}")
    except Exception:
        logger.info(f"Creating a new compute cluster: {compute_name}")
        compute_cluster = AmlCompute(
            name=compute_name,
            type="amlcompute",
            size="Standard_D2as_v4",
            min_instances=0,
            max_instances=2,
            idle_time_before_scale_down=120,
            tier="Dedicated",
        )
        ml_client.compute.begin_create_or_update(compute_cluster).result()
        logger.info(f"The cluster {compute_name} is created")

    # Load deployment component
    deploy_model = load_component(source="config/deploy_model.yaml")

    @dsl.pipeline(
        name="trigger_word_deployment_pipeline",
        description="Pipeline for deploying trigger word detection model",
        default_compute=compute_name,
    )
    def deployment_pipeline():
        deployment_job = deploy_model(
            model="trigger_word_detection_model",
            endpoint_name="trigger-word-detection-gqvuz",
            deployment_name="trigger-word-detection-model-1",
            instance_type="Standard_D2as_v4",
            instance_count=1,
        )
        return {"deployment_details": deployment_job.outputs.deployment_details}

    pipeline_job = deployment_pipeline()

    # Add tags for versioning and tracking
    pipeline_job.tags["deployment_version"] = "v1.0"
    pipeline_job.tags["deployment_date"] = datetime.now().strftime("%Y-%m-%d")

    # Set pipeline-level settings
    pipeline_job.settings.default_compute = compute_name
    pipeline_job.settings.continue_on_step_failure = False
    pipeline_job.settings.force_rerun = True
    pipeline_job.display_name = "deployment_pipeline"

    # Submit the pipeline
    try:
        pipeline_job = ml_client.jobs.create_or_update(
            pipeline_job, experiment_name="trigger-word-deployment"
        )
        logger.info(f"Deployment pipeline job submitted. Job name: {pipeline_job.name}")
    except Exception as ex:
        logger.error(f"Error submitting deployment pipeline job: {ex}")
        raise

    # Wait for the pipeline to complete
    try:
        ml_client.jobs.stream(pipeline_job.name)
    except Exception as ex:
        logger.error(f"Error streaming deployment pipeline logs: {ex}")

    logger.info("Deployment pipeline completed.")

    # Retrieve and print job details
    job_details = ml_client.jobs.get(pipeline_job.name)
    logger.info(f"Job status: {job_details.status}")

    if job_details.status == "Completed":
        outputs = job_details.outputs
        logger.info(f"Deployment details: {outputs['deployment_details']}")
    else:
        logger.error(
            "Deployment pipeline did not complete successfully. Check the logs for more details."
        )


if __name__ == "__main__":
    main()
