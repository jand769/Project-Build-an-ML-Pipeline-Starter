#!/usr/bin/env python
"""
This step tests the provided regression model against the test dataset.
"""
import argparse
import logging
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Test the regression model and log metrics.
    """
    run = wandb.init(job_type="test_model")
    run.config.update(vars(args))

    logger.info("Downloading artifacts")

    # Download model artifact
    model_local_path = run.use_artifact(args.mlflow_model).download()

    # Download test dataset artifact
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    # Read the test dataset
    logger.info("Loading test dataset")
    test_df = pd.read_csv(test_dataset_path)
    y_test = test_df.pop("price")
    X_test = test_df

    # Load the model
    logger.info("Loading model and performing inference on test set")
    model = mlflow.sklearn.load_model(model_local_path)
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    logger.info(f"MAE: {mae}")
    logger.info(f"R2: {r2}")
    run.summary["mae"] = mae
    run.summary["r2"] = r2

    logger.info("Testing completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a regression model against the test dataset")

    parser.add_argument(
        "--mlflow_model",
        type=str,
        required=True,
        help="The input MLflow model (e.g., 'model:prod')",
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        required=True,
        help="The input test dataset artifact",
    )

    args = parser.parse_args()

    go(args)
