import os
import mlflow
import hydra
from omegaconf import DictConfig
import hydra.utils

@hydra.main(config_name="config")
def go(config: DictConfig):
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

    if "download_file" in steps_to_execute:
        _ = mlflow.run(
            uri=config.main.components_repository + "/get_data",
            parameters={
                "sample": config.etl.sample,
                "artifact_name": config.etl.sample,
                "artifact_type": "raw_data",
                "artifact_description": "Raw dataset",
            },
        )

    if "basic_cleaning" in steps_to_execute:
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

    if "data_check" in steps_to_execute:
        _ = mlflow.run(
            uri=os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
            parameters={"kl_threshold": config.data_check.kl_threshold},
        )

    if "data_split" in steps_to_execute:
        _ = mlflow.run(
            uri=os.path.join(hydra.utils.get_original_cwd(), "src", "data_split"),
            parameters={
                "test_size": config.modeling.test_size,
                "random_seed": config.modeling.random_seed,
                "stratify_by": config.modeling.stratify_by,
            },
        )

    if "train_random_forest" in steps_to_execute:
        _ = mlflow.run(
            uri=os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
            parameters={
                "max_tfidf_features": config.modeling.max_tfidf_features,
                "random_forest": config.modeling.random_forest,
            },
        )

    if "test_model" in steps_to_execute:
        _ = mlflow.run(
            uri=os.path.join(hydra.utils.get_original_cwd(), "src", "test_model"),
            parameters={},
        )

if __name__ == "__main__":
    go()
