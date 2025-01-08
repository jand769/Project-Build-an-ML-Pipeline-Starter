import hydra
from omegaconf import DictConfig
import mlflow

@hydra.main(config_name="config")
def main(config: DictConfig):
    steps = config['main']['steps']
    selected_steps = steps.split(",")

    # Download step
    if "download" in selected_steps:
        _ = mlflow.run(
            uri="components/get_data",
            entry_point="main",
            parameters={
                "sample_artifact": config["etl"]["sample_artifact"],
                "artifact_name": config["etl"]["artifact_name"],
                "artifact_type": config["etl"]["artifact_type"],
                "artifact_description": config["etl"]["artifact_description"],
            },
        )

    # Basic cleaning step
    if "basic_cleaning" in selected_steps:
        _ = mlflow.run(
            uri="src/basic_cleaning",
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


if __name__ == "__main__":
    main()
