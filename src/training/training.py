import argparse
import numpy as np
import mlflow
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Dropout,
    BatchNormalization,
    GRU,
    Dense,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    LearningRateScheduler,
)
from azure.ai.ml import Input as AzureInput, Output as AzureOutput
import logging
import os
import traceback
from mlflow.models import infer_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=3, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, kernel_size=3, activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.4)(x)

    x = Conv1D(256, kernel_size=3, activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.4)(x)

    x = GRU(128, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = GRU(64)(x)
    x = Dropout(0.5)(x)

    x = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(0.01))(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 5:
        lr *= 0.5
    if epoch > 20:
        lr *= 0.5
    if epoch > 30:
        lr *= 0.5
    return lr


def train_model(
    input_dir: AzureInput(type="uri_folder"),
    output_model: AzureOutput(type="mlflow_model"),
    epochs: int = 50,
    batch_size: int = 32,
    experiment_name: str = "trigger-word-detection",
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        try:
            input_dir_path = input_dir if isinstance(input_dir, str) else input_dir.path
            output_model_path = (
                output_model if isinstance(output_model, str) else output_model.path
            )

            X_train = np.load(os.path.join(input_dir_path, "X_train.npy"))
            y_train = np.load(os.path.join(input_dir_path, "y_train.npy"))
            X_val = np.load(os.path.join(input_dir_path, "X_val.npy"))
            y_val = np.load(os.path.join(input_dir_path, "y_val.npy"))

            logger.info(
                f"Loaded data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}"
            )

            input_shape = (X_train.shape[1], X_train.shape[2])
            model = build_model(input_shape)

            initial_learning_rate = 1e-3
            optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
            model.compile(
                optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
            )

            model.summary(print_fn=logger.info)

            callbacks = [
                EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6
                ),
            ]

            mlflow.log_params(
                {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "initial_lr": initial_learning_rate,
                    "input_shape": input_shape,
                }
            )

            history = model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
            )

            # Log metrics
            for epoch, (loss, acc, val_loss, val_acc) in enumerate(
                zip(
                    history.history["loss"],
                    history.history["accuracy"],
                    history.history["val_loss"],
                    history.history["val_accuracy"],
                )
            ):
                mlflow.log_metrics(
                    {
                        "loss": loss,
                        "accuracy": acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                    },
                    step=epoch,
                )

            # Infer the model signature
            signature = infer_signature(X_val, model.predict(X_val))

            # Log and save the model with MLflow
            mlflow.keras.log_model(model, "model", signature=signature)
            mlflow.keras.save_model(model, output_model_path)

            logger.info(
                f"Training completed. Model saved with MLflow at {output_model_path}"
            )

        except Exception as e:
            logger.error(f"An error occurred during training: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    return output_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory with split data"
    )
    parser.add_argument(
        "--output_model", type=str, required=True, help="Output path for saved model"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="trigger-word-detection",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    try:
        final_model_path = train_model(
            input_dir=args.input_dir,
            output_model=args.output_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            experiment_name=args.experiment_name,
        )
        print(f"Training completed. Model saved to {final_model_path}")
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        logger.error(traceback.format_exc())
        raise
