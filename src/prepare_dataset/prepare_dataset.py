import argparse
import numpy as np
import json
from azure.ai.ml import Input, Output
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_dataset(
    input_file: Input(type="uri_file"), output_dir: Output(type="uri_folder")
):
    logger.info(f"Loading input file: {input_file}")
    with open(input_file, "r") as f:
        features = json.load(f)

    positive_features = np.array(features["positive_features"])
    negative_features = np.array(features["negative_features"])

    X = np.concatenate((positive_features, negative_features), axis=0)
    y = np.concatenate(
        (
            np.ones(len(positive_features)),
            np.zeros(len(negative_features)),
        ),
        axis=0,
    )

    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    logger.info(
        f"Positive samples: {np.sum(y)}, Negative samples: {len(y) - np.sum(y)}"
    )

    # Save X and y separately as .npy files
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)

    logger.info(f"Data saved to {output_dir}")

    # Save a metadata file with shape information
    metadata = {
        "X_shape": X.shape,
        "y_shape": y.shape,
        "positive_samples": int(np.sum(y)),
        "negative_samples": int(len(y) - np.sum(y)),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    logger.info("Metadata saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    prepare_dataset(args.input_file, args.output_dir)
