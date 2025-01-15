import os
import mlflow
import hydra
from omegaconf import DictConfig
import hydra.utils
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(config_name="config")
def go(config: DictConfig):

    logger.info("Starting the pipeline...")
    logger.info(f"Config: {config}")

    # Determine steps to execute
    steps_to_execute = config.main.steps.split(",")

    if "all" in steps_to_execute:
        steps_to_execute = [
            "download_file",
            "basic_cleaning",
            "data_check",
            "data_split",
            "train_random_forest",
            "test_model",
        ]
    logger.info(f"Steps to execute: {steps_to_execute}")

    # Execute steps
    if "download_file" in steps_to_execute:
        logger.info("Running 'download_file' step...")
        _ = mlflow.run(
            uri=config.main.components_repository + "/get_data",
            parameters={
                "sample": config.etl.sample,
                "artifact_name": config.etl.sample,
                "artifact_type": "raw_data",
                "artifact_description": "Raw dataset",
            },
        )
        logger.info("'download_file' step completed.")

    if "basic_cleaning" in steps_to_execute:
        logger.info("Running 'basic_cleaning' step...")
        _ = mlflow.run(
            uri=os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
            parameters={
                "input_artifact": config.basic_cleaning.input_artifact,
                "output_artifact": config.basic_cleaning.output_artifact,
                "output_type": config.basic_cleaning.output_type,
                "output_description": config.basic_cleaning.output_description,
                "min_price": config.etl.min_price,
                "max_price": config.etl.max_price,
            },
        )
        logger.info("'basic_cleaning' step completed.")

    if "data_check" in steps_to_execute:
        logger.info("Running 'data_check' step...")
        _ = mlflow.run(
            uri=os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
            parameters={
                "input_artifact": config.data_check.input_artifact,
                "kl_threshold": config.data_check.kl_threshold,
                "min_correlation": config.data_check.min_correlation,
            },
        )
        logger.info("'data_check' step completed.")

    if "data_split" in steps_to_execute:
        logger.info("Running 'data_split' step...")
        _ = mlflow.run(
            uri=os.path.join(hydra.utils.get_original_cwd(), "src", "data_split"),
            parameters={
                "input_artifact": config.data_split.input_artifact,
                "test_size": config.modeling.test_size,
                "random_seed": config.modeling.random_seed,
                "stratify_by": config.modeling.stratify_by,
            },
        )
        logger.info("'data_split' step completed.")

    if "train_random_forest" in steps_to_execute:
        logger.info("Running 'train_random_forest' step...")
        _ = mlflow.run(
            uri=os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
            parameters={
                "train_data": config.train_random_forest.train_data,
                "model_config": config.train_random_forest.model_config,
                "output_artifact": config.train_random_forest.output_artifact,
            },
        )
        logger.info("'train_random_forest' step completed.")

    if "test_model" in steps_to_execute:
        logger.info("Running 'test_model' step...")
        _ = mlflow.run(
            uri=os.path.join(hydra.utils.get_original_cwd(), "src", "test_model"),
            parameters={
                "mlflow_model": config.test_model.mlflow_model,
                "test_data": config.test_model.test_data,
            },
        )
        logger.info("'test_model' step completed.")

    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    go()
