import os
import json
import mlflow
import hydra
from omegaconf import DictConfig
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Steps in the pipeline
_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    "test_model",
]

@hydra.main(config_path=".", config_name="config", version_base="1.2")
def go(config: DictConfig):
    logger.info("Pipeline started with the following configuration:")
    logger.info(config)

    # Determine which steps to execute
    steps_to_execute = config.main.steps.split(",") if config.main.steps != "all" else _steps
    logger.info(f"Steps to execute: {steps_to_execute}")

    # Get the original working directory
    root_path = hydra.utils.get_original_cwd()
    logger.info(f"Original working directory: {root_path}")

    if "download" in steps_to_execute:
        logger.info("Running 'download' step...")
        _ = mlflow.run(
            uri=f"{config.main.components_repository}/get_data",
            entry_point="main",
            parameters={
                "sample": config.etl.sample,
                "artifact_name": "sample2.csv",
                "artifact_type": "raw_data",
                "artifact_description": "Raw dataset from source",
            },
        )
        logger.info("'download' step completed.")

    if "basic_cleaning" in steps_to_execute:
        logger.info("Running 'basic_cleaning' step...")
        _ = mlflow.run(
            uri=os.path.join(root_path, "src", "basic_cleaning"),
            entry_point="main",
            parameters={
                "input_artifact": config.basic_cleaning.input_artifact,
                "output_artifact": "clean_sample2.csv",
                "output_type": "cleaned_sample",
                "output_description": "Dataset cleaned of outliers and invalid geolocations",
                "min_price": config.etl.min_price,
                "max_price": config.etl.max_price,
            },
        )
        logger.info("'basic_cleaning' step completed.")

    if "data_check" in steps_to_execute:
        logger.info("Running 'data_check' step...")
        _ = mlflow.run(
            uri=os.path.join(root_path, "src", "data_check"),
            entry_point="main",
            parameters={
                "csv": "clean_sample2.csv:latest", 
                "ref": "clean_sample2.csv:reference",
                "kl_threshold": config.data_check.kl_threshold,
                "min_price": config.etl.min_price,
                "max_price": config.etl.max_price,
            },
        )
        logger.info("'data_check' step completed.")

    if "data_split" in steps_to_execute:
        logger.info("Running 'data_split' step...")
        _ = mlflow.run(
            uri=f"{config.main.components_repository}/train_val_test_split",
            entry_point="main",
            parameters={
                "input": "clean_sample2.csv:latest",
                "test_size": config.modeling.test_size,
                "random_seed": config.modeling.random_seed,
                "stratify_by": config.modeling.stratify_by,
            },
        )
        logger.info("'data_split' step completed.")

    if "train_random_forest" in steps_to_execute:
        logger.info("Running 'train_random_forest' step...")
        rf_config_path = os.path.join(root_path, "rf_config.json")
        with open(rf_config_path, "w") as fp:
            json.dump(config.modeling.random_forest, fp)

        _ = mlflow.run(
            uri=os.path.join(root_path, "src", "train_random_forest"),
            entry_point="main",
            parameters={
                "trainval_artifact": "trainval_data.csv:latest",
                "rf_config": rf_config_path,
                "output_artifact": "random_forest_export",
            },
        )
        logger.info("'train_random_forest' step completed.")

    if "test_model" in steps_to_execute:
        logger.info("Running 'test_model' step...")
        _ = mlflow.run(
            uri=os.path.join(root_path, "src", "test_model"),
            entry_point="main",
            parameters={
                "mlflow_model": config.test_model.mlflow_model,
                "test_dataset": config.test_model.test_data,
            },
        )
        logger.info("'test_model' step completed.")

    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    go()
