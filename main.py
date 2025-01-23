#!/usr/bin/env python
"""
Pipeline execution script
"""

import mlflow
import hydra
from omegaconf import DictConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_name="config", config_path=".")
def go(config: DictConfig):
    """
    Executes the pipeline steps as configured in the config.yaml file
    """

    # List of steps to execute
    steps_to_execute = config.main.steps.split(",") if config.main.steps != "all" else [
        "download",
        "basic_cleaning",
        "data_check",
        "data_split",
        "train_random_forest",
        "test_regression_model",
    ]

    # Define all steps
    if "download" in steps_to_execute:
        _ = mlflow.run(
            config.main.components_repository + "/get_data",
            "main",
            parameters={
                "sample": config.etl.sample,
                "artifact_name": "sample.csv",
                "artifact_type": "raw_data",
                "artifact_description": "Raw dataset",
            },
        )

    if "basic_cleaning" in steps_to_execute:
        _ = mlflow.run(
            config.main.components_repository + "/basic_cleaning",
            "main",
            parameters={
                "input_artifact": config.basic_cleaning.input_artifact,
                "output_artifact": config.basic_cleaning.output_artifact,
                "output_type": config.basic_cleaning.output_type,
                "output_description": config.basic_cleaning.output_description,
                "min_price": config.basic_cleaning.min_price,
                "max_price": config.basic_cleaning.max_price,
            },
        )

    if "data_check" in steps_to_execute:
        _ = mlflow.run(
            config.main.components_repository + "/data_check",
            "main",
            parameters={
                "input_artifact": config.data_check.input_artifact,
                "reference_artifact": config.data_check.reference_artifact,
                "kl_threshold": config.data_check.kl_threshold,
                "min_price": config.data_check.min_price,
                "max_price": config.data_check.max_price,
            },
        )

    if "data_split" in steps_to_execute:
        _ = mlflow.run(
            config.main.components_repository + "/data_split",
            "main",
            parameters={
                "input_artifact": config.data_split.input_artifact,
                "test_size": config.modeling.test_size,
                "random_seed": config.modeling.random_seed,
                "stratify_by": config.modeling.stratify_by,
            },
        )

    if "train_random_forest" in steps_to_execute:
        _ = mlflow.run(
            config.main.components_repository + "/train_random_forest",
            "main",
            parameters={
                "trainval_artifact": config.modeling.trainval_artifact,
                "val_size": config.modeling.val_size,
                "random_seed": config.modeling.random_seed,
                "stratify_by": config.modeling.stratify_by,
                "rf_config": config.modeling.rf_config,
                "max_tfidf_features": config.modeling.max_tfidf_features,
                "output_artifact": config.modeling.output_artifact,
            },
        )

    if "test_regression_model" in steps_to_execute:
        logger.info("Running test_regression_model step")
        _ = mlflow.run(
            config.main.components_repository + "/test_regression_model",
            "main",
            parameters={
                "mlflow_model": "random_forest_export:prod",  # Production model alias
                "test_dataset": "test_data.csv:latest",  # Correct key for test dataset
            },
        )


if __name__ == "__main__":
    go()
