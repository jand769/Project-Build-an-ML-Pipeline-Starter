import os
import json
import mlflow
import tempfile
import hydra
from omegaconf import DictConfig
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define pipeline steps
_steps = [
    "download",
    "basic_cleaning",
]

@hydra.main(config_path=".", config_name="config", version_base="1.2")
def go(config: DictConfig):
    logger.info("Pipeline started with the following configuration:")
    logger.info(config)

    # Set W&B project environment variables
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # steps to execute
    steps_to_execute = config.main.steps.split(",") if config.main.steps != "all" else _steps
    logger.info(f"Steps to execute: {steps_to_execute}")

    # root working directory
    root_path = hydra.utils.get_original_cwd()
    logger.info(f"Root working directory: {root_path}")

    # Temporary directory for intermediate artifacts if needed
    with tempfile.TemporaryDirectory() as tmp_dir:
        if "download" in steps_to_execute:
            logger.info("Running 'download' step...")
            mlflow.run(
                uri=f"{config.main.components_repository}/get_data",
                entry_point="main",
                parameters={
                    "sample": config.etl.sample,
                    "artifact_name": "sample1.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw dataset from source",
                },
            )
            logger.info("'download' step completed.")

        if "basic_cleaning" in steps_to_execute:
            logger.info("Running 'basic_cleaning' step...")
            mlflow.run(
                uri=os.path.join(root_path, "src", "basic_cleaning"),
                entry_point="main",
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

    logger.info("Pipeline execution completed successfully!")


if __name__ == "__main__":
    go()
