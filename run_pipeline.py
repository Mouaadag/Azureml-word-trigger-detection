# run_pipeline.py

from azure.ai.ml import MLClient, Input, load_component, dsl
from azure.ai.ml.entities import AmlCompute, Environment, BuildContext
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.keyvault.secrets import SecretClient
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def main():
    # Connect to the workspace
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
        logger.info("Successfully authenticated with DefaultAzureCredential")
        print("Successfully authenticated with DefaultAzureCredential")
    except Exception as ex:
        credential = InteractiveBrowserCredential()
        logger.info("Successfully authenticated with InteractiveBrowserCredential")
        print("Successfully authenticated with InteractiveBrowserCredential")

    try:
        ml_client = MLClient.from_config(credential=credential)
        workspace_info = ml_client.workspaces.get(ml_client.workspace_name)
        logger.info(f"Successfully connected to Workspace: {workspace_info.name}")
    except Exception as ex:
        logger.error(f"Error accessing workspace: {str(ex)}")
        raise

    # Define compute cluster
    compute_name = "word-trigger-cluster"
    try:
        compute_cluster = ml_client.compute.get(compute_name)
        logger.info(f"Found existing compute cluster: {compute_name}")
        print(f"Found existing compute cluster: {compute_name}")
    except Exception:
        logger.info(f"Creating a new compute cluster: {compute_name}")
        compute_cluster = AmlCompute(
            name=compute_name,
            type="amlcompute",
            size="Standard_E4ds_v4",
            min_instances=0,
            max_instances=2,
            idle_time_before_scale_down=120,
            tier="Dedicated",
        )
        ml_client.compute.begin_create_or_update(compute_cluster).result()
        print(f"The cluster {compute_name} is created")

    # Define and register environment
    env_name = "trigger-word-detection-environment"
    conda_file = "config/conda.yaml"
    try:
        environment = ml_client.environments.get(name=env_name, version="14")
        logger.info(
            f"Found existing environment: {env_name}, version: {environment.version}"
        )
        print(f"Found existing environment: {env_name}, version: {environment.version}")

    except Exception as e:
        print(f"No existing environment {env_name} found. Creating a new one.")

        # Créer l'environnement à partir du fichier conda.yaml
        new_environment = Environment(
            name=env_name,
            conda_file=conda_file,
            image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
        )

        # Enregistrer l'environnement dans le workspace
        ml_client.environments.create_or_update(new_environment)
        print(f"Created and registered new environment: {env_name}")

    # Load components
    load_data = load_component(source="config/load_data.yaml")
    extract_features = load_component(source="config/extract_features.yaml")
    prepare_dataset = load_component(source="config/prepare_dataset.yaml")
    split_dataset = load_component(source="config/split_data.yaml")
    train_model = load_component(source="config/training.yaml")
    evaluate_model = load_component(source="config/evaluation.yaml")
    register_model = load_component(source="config/register_model.yaml")

    # Get data assets
    positive_data_asset = ml_client.data.get("positives_samples", version="1")
    negative_data_asset = ml_client.data.get("negatives_samples", version="1")
    data_asset_neg = ml_client.data.get("negatives_samples_test", version="1")
    data_asset_oos = ml_client.data.get("positives_samples_test", version="1")

    # Get the Key Vault associated with the workspace
    workspace = ml_client.workspaces.get(ml_client.workspace_name)
    key_vault_uri = workspace.key_vault

    print(f"Original Key Vault URI: {key_vault_uri}")
    secret_client = SecretClient(
        vault_url="https://my-key-vault.vault.azure.net/", credential=credential
    )

    # Retrieve the secret
    secret_name = "<secret_name>"
    try:
        secret = secret_client.get_secret(secret_name)
        print(f"Successfully retrieved secret: {secret_name}")
    except Exception as e:
        print(f"Error retrieving secret: {str(e)}")
        raise

    @dsl.pipeline(
        name="trigger_word_detection_pipeline",
        description="End-to-end pipeline for trigger word detection",
        default_compute=compute_name,
    )
    def trigger_word_pipeline(
        positive_dir: Input(type=AssetTypes.URI_FOLDER),
        negative_dir: Input(type=AssetTypes.URI_FOLDER),
    ):
        load_data_job = load_data(
            positive_dir=positive_dir,
            negative_dir=negative_dir,
            storage_key=secret.value,
        )
        logger.info("Load data step completed.")

        extract_features_job = extract_features(
            input_file=load_data_job.outputs.output_file,
            sample_rate=16000,
            n_mfcc=13,
            max_pad_len=215,
            storage_key=secret.value,
        )
        logger.info("Extract features step completed.")

        prepare_dataset_job = prepare_dataset(
            input_file=extract_features_job.outputs.output_file
        )
        logger.info("Prepare dataset step completed.")

        split_data_job = split_dataset(
            input_dir=prepare_dataset_job.outputs.output_dir,
            test_size=0.01,
            val_size=0.01,
        )
        logger.info("Split dataset step completed.")

        train_model_job = train_model(
            input_dir=split_data_job.outputs.output_dir,
            epochs=50,
            batch_size=32,
            experiment_name="trigger-word-detection",
        )
        logger.info("Train model step completed.")

        evaluate_model_job = evaluate_model(
            model_path=train_model_job.outputs.output_model,
            test_data=split_data_job.outputs.output_dir,
            experiment_name="trigger-word-detection",
        )
        logger.info("Evaluate model step completed.")

        register_model_job = register_model(
            model_path=train_model_job.outputs.output_model,
            model_name="trigger_word_detection_model",
            evaluation_results=evaluate_model_job.outputs.output_metrics,
            experiment_name="trigger-word-detection",
        )
        logger.info("Register model step completed.")

        return {
            "trained_model": train_model_job.outputs.output_model,
            "evaluation_results": evaluate_model_job.outputs.output_metrics,
            "registered_model": register_model_job.outputs.registered_model,
        }

    pipeline_job = trigger_word_pipeline(
        positive_dir=Input(
            type="uri_folder",
            path=positive_data_asset.path,
        ),
        negative_dir=Input(
            type="uri_folder",
            path=negative_data_asset.path,
        ),
    )
    # Add tags for versioning and tracking
    pipeline_job.tags["experiment_version"] = "v1.0"
    pipeline_job.tags["data_version"] = datetime.now().strftime("%Y-%m-%d")
    # Set pipeline-level settings
    pipeline_job.settings.default_compute = compute_name
    pipeline_job.settings.default_datastore = "workspaceblobstore"
    pipeline_job.settings.continue_on_step_failure = False
    pipeline_job.settings.default_environment = f"azureml:{env_name}:14"
    # pipeline_job.settings.ForceRerun = False
    pipeline_job.settings.force_rerun = False
    pipeline_job.display_name = "train_pipeline"

    # Submit the pipeline
    try:
        pipeline_job = ml_client.jobs.create_or_update(
            pipeline_job, experiment_name="trigger-word-detection"
        )
        logger.info(f"Pipeline job submitted. Job name: {pipeline_job.name}")
        print(f"Pipeline job submitted. Job name: {pipeline_job.name}")
    except Exception as ex:
        logger.error(f"Error submitting pipeline job: {ex}")
        print(f"Error submitting pipeline job: {ex}")
        raise

    # Wait for the pipeline to complete
    try:
        ml_client.jobs.stream(pipeline_job.name)
    except Exception as ex:
        logger.error(f"Error streaming pipeline logs: {ex}")
        print(f"Error streaming pipeline logs: {ex}")

    logger.info("Pipeline completed.")
    print("Pipeline completed.")

    # Retrieve and print job details
    job_details = ml_client.jobs.get(pipeline_job.name)
    logger.info(f"Job status: {job_details.status}")
    print(f"Job status: {job_details.status}")

    if job_details.status == "Completed":
        outputs = job_details.outputs
        logger.info(f"Trained model path: {outputs['trained_model']}")
        logger.info(f"Evaluation results: {outputs['evaluation_results']}")
        logger.info(f"Registered model info: {outputs['registered_model']}")
        print(f"Trained model path: {outputs['trained_model']}")
        print(f"Evaluation results: {outputs['evaluation_results']}")
        print(f"Registered model info: {outputs['registered_model']}")
    else:
        logger.error(
            "Pipeline did not complete successfully. Check the logs for more details."
        )
        print(
            "Pipeline did not complete successfully. Check the logs for more details."
        )


if __name__ == "__main__":
    main()
