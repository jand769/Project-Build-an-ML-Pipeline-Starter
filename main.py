import mlflow
import hydra
from omegaconf import DictConfig

@hydra.main(config_path=".", config_name="config")  # Adjust config path if needed
def go(config: DictConfig):
    steps_to_run = config.get("steps", "").split(",")

    if "download" in steps_to_run:
        _ = mlflow.run(
            uri="components/get_data",
            entry_point="main",
            parameters={
                "sample": config["data"]["sample"],
                "artifact_name": config["data"]["artifact_name"],
                "artifact_type": config["data"]["artifact_type"],
                "artifact_description": config["data"]["artifact_description"],
            },
        )

    if "basic_cleaning" in steps_to_run:
        _ = mlflow.run(
            uri="components/basic_cleaning",
            entry_point="main",
            parameters={
                "input_artifact": config["basic_cleaning"]["input_artifact"],
                "output_artifact": config["basic_cleaning"]["output_artifact"],
                "output_type": config["basic_cleaning"]["output_type"],
                "output_description": config["basic_cleaning"]["output_description"],
                "min_price": config["basic_cleaning"]["min_price"],
                "max_price": config["basic_cleaning"]["max_price"],
            },
        )

    if "data_check" in steps_to_run:
        _ = mlflow.run(
            uri="components/data_check",
            entry_point="main",
            parameters={
                "csv": config["data_check"]["csv"],
                "ref": config["data_check"]["ref"],
                "kl_threshold": config["data_check"]["kl_threshold"],
            },
        )

    if "data_split" in steps_to_run:
        _ = mlflow.run(
            uri="components/train_val_test_split",
            entry_point="main",
            parameters={
                "input": config["data_split"]["input"],
                "test_size": config["data_split"]["test_size"],
                "stratify_by": config["data_split"]["stratify_by"],
            },
        )

    if "train_random_forest" in steps_to_run:
        _ = mlflow.run(
            uri="components/train_random_forest",
            entry_point="main",
            parameters={
                "trainval_artifact": config["train_random_forest"]["trainval_artifact"],
                "val_size": config["train_random_forest"]["val_size"],
                "random_seed": config["train_random_forest"]["random_seed"],
                "stratify_by": config["train_random_forest"]["stratify_by"],
                "max_tfidf_features": config["train_random_forest"]["max_tfidf_features"],
                "n_estimators": config["train_random_forest"]["n_estimators"],
                "max_depth": config["train_random_forest"]["max_depth"],
                "min_samples_split": config["train_random_forest"]["min_samples_split"],
                "min_samples_leaf": config["train_random_forest"]["min_samples_leaf"],
            },
        )

    # Add the test_regression_model step
    if "test_regression_model" in steps_to_run:
        _ = mlflow.run(
            uri="components/test_regression_model",
            entry_point="main",
            parameters={
                "mlflow_model": "random_forest_export:prod",
                "test_artifact": "test_data.csv:latest",
            },
        )


if __name__ == "__main__":
    go()
