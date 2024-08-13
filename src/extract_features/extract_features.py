import json
import logging
import os
import numpy as np
import librosa
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError
from azure.ai.ml import Input, Output
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure Blob Storage settings
ACCOUNT_NAME = "<account_name>"
CONTAINER_NAME = "<container_name>"


def load_json_data(json_file_path):
    with open(json_file_path, "r") as f:
        return json.load(f)


def download_and_process_audio(
    blob_client, blob_name, sample_rate, n_mfcc, max_pad_len
):
    try:
        # Download the blob content into memory
        stream = io.BytesIO()
        blob_client.download_blob().readinto(stream)
        stream.seek(0)

        # Load and process audio directly from memory
        audio, sr = librosa.load(stream, sr=sample_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs = mfccs.T

        if mfccs.shape[0] > max_pad_len:
            mfccs = mfccs[:max_pad_len, :]
        else:
            pad_width = ((0, max_pad_len - mfccs.shape[0]), (0, 0))
            mfccs = np.pad(mfccs, pad_width, mode="constant")

        logger.info(f"Processed file: {blob_name}")
        return mfccs
    except Exception as e:
        logger.error(f"Error processing file {blob_name}: {str(e)}")
        return None


def extract_features(
    container_client,
    file_paths,
    sample_rate,
    n_mfcc,
    max_pad_len,
    max_workers=6,
    batch_size=500,
):
    features = []
    total_files = len(file_paths)

    with tqdm(total=total_files, desc="Extracting features") as pbar:
        for i in range(0, total_files, batch_size):
            batch = file_paths[i : i + batch_size]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_blob = {
                    executor.submit(
                        download_and_process_audio,
                        container_client.get_blob_client(
                            blob_name.split("paths/", 1)[1]
                        ),
                        blob_name.split("paths/", 1)[1],
                        sample_rate,
                        n_mfcc,
                        max_pad_len,
                    ): blob_name
                    for blob_name in batch
                }

                for future in as_completed(future_to_blob):
                    blob_name = future_to_blob[future]
                    try:
                        mfccs = future.result()
                        if mfccs is not None:
                            features.append(mfccs)
                    except Exception as e:
                        logger.error(f"Error processing file {blob_name}: {str(e)}")
                    finally:
                        pbar.update(1)

            # Log progress after each batch
            logger.info(
                f"Processed {min(i+batch_size, total_files)} out of {total_files} files"
            )

    return np.array(features)


def main(
    input_file: Input(type="uri_file"),
    output_file: Output(type="uri_file"),
    storage_key: Input(type="string"),
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    max_pad_len: int = 215,
):
    data = load_json_data(input_file)
    account_url = f"https://{ACCOUNT_NAME}.blob.core.windows.net"
    blob_service_client = BlobServiceClient(
        account_url=account_url, credential=storage_key
    )
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    positive_features = extract_features(
        container_client, data["positive_files"], sample_rate, n_mfcc, max_pad_len
    )
    negative_features = extract_features(
        container_client, data["negative_files"], sample_rate, n_mfcc, max_pad_len
    )

    logger.info(f"Positive features shape: {positive_features.shape}")
    logger.info(f"Negative features shape: {negative_features.shape}")

    output_data = {
        "positive_features": positive_features.tolist(),
        "negative_features": negative_features.tolist(),
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f)

    logger.info(f"Features saved to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--storage_key", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mfcc", type=int, default=13)
    parser.add_argument("--max_pad_len", type=int, default=215)
    args = parser.parse_args()

    main(
        args.input_file,
        args.output_file,
        args.storage_key,
        args.sample_rate,
        args.n_mfcc,
        args.max_pad_len,
    )
