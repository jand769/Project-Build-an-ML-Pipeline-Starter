import os
import mlflow
import tempfile
import json
import hydra
from omegaconf import DictConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the pipeline steps
_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
]

@hydra.main(config_name="config", config_path=".", version_base="1.2")
def go(config: DictConfig):
    """
    Executes the pipeline steps as configured in the config.yaml file.
    """
    # Set W&B project environment variables
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Determine the steps to execute
    steps_to_execute = (
        config["main"]["steps"].split(",") if config["main"]["steps"] != "all" else _steps
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        if "download" in steps_to_execute:
            logger.info("Running 'download' step")
            mlflow.run(
                uri=f"{config['main']['components_repository']}/get_data",
                entry_point="main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw dataset from source",
                },
            )

        if "basic_cleaning" in steps_to_execute:
            logger.info("Running 'basic_cleaning' step")
            mlflow.run(
                uri=os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                entry_point="main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample1.csv",
                    "output_type": "cleaned_data",
                    "output_description": "Cleaned dataset with outliers removed",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_check" in steps_to_execute:
            logger.info("Running 'data_check' step")
            mlflow.run(
                uri=os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                entry_point="main",
                parameters={
                    "csv": "clean_sample1.csv:latest",
                    "ref": "clean_sample1.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_split" in steps_to_execute:
            logger.info("Running 'data_split' step")
            mlflow.run(
                uri=f"{config['main']['components_repository']}/train_val_test_split",
                entry_point="main",
                parameters={
                    "input": "clean_sample1.csv:reference",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                },
            )

        if "train_random_forest" in steps_to_execute:
            logger.info("Running 'train_random_forest' step")
            rf_config_path = os.path.abspath("rf_config.json")
            with open(rf_config_path, "w") as fp:
                json.dump(dict(config["modeling"]["random_forest"]), fp)

            mlflow.run(
                uri=os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
                entry_point="main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config_path,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": config["modeling"]["output_artifact"],
                },
            )

if __name__ == "__main__":
    go()
