import argparse
import json
import logging
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml import Input, Output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_paths(storage_key: str, samples_dir: str):
    account_name = "<account_name>"
    container_name = "<container_name>"

    logger.info(f"Connecting to storage account: {account_name}")

    account_url = f"https://{account_name}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(
        account_url=account_url, credential=storage_key
    )
    logger.info("Successfully authenticated using provided storage key")

    container_client = blob_service_client.get_container_client(container_name)

    logger.info(f"Full samples_dir path: {samples_dir}")

    # Determine the blob prefix based on the input path
    if "INPUT_positive_dir" in samples_dir:
        blob_prefix = "UI/2024-07-22_211805_UTC/positives_samples/"
    elif "INPUT_negative_dir" in samples_dir:
        blob_prefix = "UI/2024-07-23_002207_UTC/negatives_samples/"

    else:
        logger.error(f"Unexpected samples_dir: {samples_dir}")
        return []

    logger.info(f"Searching for MP3 files in blob storage with prefix: {blob_prefix}")

    try:
        all_blobs = list(container_client.list_blobs(name_starts_with=blob_prefix))
        logger.info(f"All blobs in {blob_prefix}:")
        for blob in all_blobs:
            logger.info(f"  {blob.name}")

        mp3_paths = [
            "azureml://datastores/workspaceblobstore/paths/" + blob.name
            for blob in all_blobs
            if blob.name.lower().endswith(".mp3")
        ]
        logger.info(f"Found {len(mp3_paths)} MP3 files")

        for mp3_file in mp3_paths:
            logger.info(f"MP3 file found: {mp3_file}")

    except ResourceNotFoundError:
        logger.error(f"Container '{container_name}' not found in the storage account.")
        mp3_paths = []
    except Exception as e:
        logger.error(f"An error occurred while listing blobs: {str(e)}")
        mp3_paths = []

    return mp3_paths


def load_data(
    positive_dir: Input(type="uri_folder"),
    negative_dir: Input(type="uri_folder"),
    storage_key: Input(type="string"),
    output_file: Output(type="uri_file"),
):
    logger.info(f"Positive directory: {positive_dir}")
    logger.info(f"Negative directory: {negative_dir}")
    print(f"Positive directory: {positive_dir}")
    print(f"Negative directory: {negative_dir}")

    try:
        positive_files = get_paths(storage_key, positive_dir)
        negative_files = get_paths(storage_key, negative_dir)

        logger.info(
            f"Found {len(positive_files)} positive files and {len(negative_files)} negative files"
        )

        if not positive_files:
            logger.warning("No valid positive audio files found")
        if not negative_files:
            logger.warning("No valid negative audio files found")

        result = {
            "positive_files": positive_files,
            "negative_files": negative_files,
        }

        with open(output_file, "w") as json_file:
            json.dump(result, json_file, indent=2)

        logger.info(f"Data successfully saved to {output_file}")
        print(f"Data successfully saved to {output_file}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_dir", type=str, required=True)
    parser.add_argument("--negative_dir", type=str, required=True)
    parser.add_argument("--storage_key", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    load_data(args.positive_dir, args.negative_dir, args.storage_key, args.output_file)
