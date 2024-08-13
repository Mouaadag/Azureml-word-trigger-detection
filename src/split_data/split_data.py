import argparse
import numpy as np
import json
from azure.ai.ml import Input, Output
import logging
import os
from sklearn.model_selection import train_test_split
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_data(
    input_dir: Input(type="uri_folder"),
    output_dir: Output(type="uri_folder"),
    random_state: int = 42,
    test_size: float = 0.01,
    val_size: float = 0.01,
):
    logger.info(f"Loading input data from: {input_dir}")

    # Load X and y
    X = np.load(os.path.join(input_dir, "X.npy"))
    y = np.load(os.path.join(input_dir, "y.npy"))

    logger.info(f"Loaded X shape: {X.shape}")
    logger.info(f"Loaded y shape: {y.shape}")

    # Split the data into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Split the train+val set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_size / (1 - test_size),  # Adjust for the remaining data
        random_state=random_state,
        stratify=y_train_val,
    )

    logger.info(f"Train set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    logger.info(f"Validation set shape: X_val {X_val.shape}, y_val {y_val.shape}")
    logger.info(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")

    # Save the split datasets
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)

    logger.info(f"Split data saved to {output_dir}")

    # Save metadata
    metadata = {
        "X_train_shape": X_train.shape,
        "y_train_shape": y_train.shape,
        "X_val_shape": X_val.shape,
        "y_val_shape": y_val.shape,
        "X_test_shape": X_test.shape,
        "y_test_shape": y_test.shape,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "random_state": random_state,
    }

    with open(os.path.join(output_dir, "split_metadata.json"), "w") as f:
        json.dump(metadata, f)
    mlflow.log_dict(metadata, "split_metadata.json")
    logger.info("Split metadata saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.01)
    parser.add_argument("--val_size", type=float, default=0.01)
    args = parser.parse_args()

    split_data(
        args.input_dir,
        args.output_dir,
        args.random_state,
        args.test_size,
        args.val_size,
    )
