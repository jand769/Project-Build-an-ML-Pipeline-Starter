#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" tag, and tests it against the test dataset.
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
    # Initialize WandB
    run = wandb.init(job_type="test_model")
    run.config.update(vars(args))

    logger.info("Downloading artifacts")

    # Download the MLflow model artifact
    model_local_path = run.use_artifact(args.mlflow_model).download()

    # Download the test dataset artifact
    test_dataset_path = run.use_artifact(args.test_dataset).file()

    # Load the test dataset
    logger.info("Loading test dataset")
    test_df = pd.read_csv(test_dataset_path)
    y_test = test_df.pop("price")
    X_test = test_df

    # Load the model
    logger.info("Loading model and performing inference on test set")
    model = mlflow.sklearn.load_model(model_local_path)
    y_pred = model.predict(X_test)

    # Calculate metrics
    logger.info("Calculating metrics")
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics to WandB
    logger.info(f"MAE: {mae}")
    logger.info(f"R2: {r2}")
    run.summary["mae"] = mae
    run.summary["r2"] = r2

    logger.info("Testing completed successfully")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Test the provided model against the test dataset")

    parser.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLflow model (e.g., 'model:prod')",
        required=True,
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        help="Test dataset artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
