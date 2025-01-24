#!/usr/bin/env python
"""
Pipeline execution script
"""

import mlflow
import hydra
from omegaconf import DictConfig
import os
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

    # Log the steps to be executed
    logger.info(f"Steps to execute: {steps_to_execute}")

    # Define all steps
    if "download" in steps_to_execute:
        logger.info("Running the download step")
        try:
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
        except Exception as e:
            logger.error(f"Error in download step: {e}")
            raise

    if "basic_cleaning" in steps_to_execute:
        logger.info("Running the basic_cleaning step")
        try:
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
        except Exception as e:
            logger.error(f"Error in basic_cleaning step: {e}")
            raise

    if "data_check" in steps_to_execute:
        logger.info("Running the data_check step")
        try:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                entry_point="main",
                parameters={
                    "csv": "clean_sample1.csv:latest",
                    "ref": "clean_sample1.csv:reference",
                    "kl_threshold": config.data_check.kl_threshold,
                    "min_price": config.etl.min_price,
                    "max_price": config.etl.max_price,
                },
            )
        except Exception as e:
            logger.error(f"Error in data_check step: {e}")
            raise

    if "data_split" in steps_to_execute:
        logger.info("Running the data_split step")
        try:
            _ = mlflow.run(
                config.main.components_repository + "/train_val_test_split",
                "main",
                parameters={
                    "input": config.data_split.input_artifact,  # Ensure this matches W&B artifact
                    "test_size": config.modeling.test_size,
                    "random_seed": config.modeling.random_seed,
                    "stratify_by": config.modeling.stratify_by,
                },
            )
        except Exception as e:
            logger.error(f"Error in data_split step: {e}")
            raise

    if "train_random_forest" in steps_to_execute:
        logger.info("Running the train_random_forest step")
        try:
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
        except Exception as e:
            logger.error(f"Error in train_random_forest step: {e}")
            raise

    if "test_regression_model" in steps_to_execute:
        logger.info("Running the test_regression_model step")
        try:
            _ = mlflow.run(
                config.main.components_repository + "/test_regression_model",
                "main",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest",
                },
            )
        except Exception as e:
            logger.error(f"Error in test_regression_model step: {e}")
            raise


if __name__ == "__main__":
    go()
