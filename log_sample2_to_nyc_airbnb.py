import wandb

# Initialize a new run in the 'nyc_airbnb' project
run = wandb.init(project="nyc_airbnb", job_type="log_artifact")

# Log the artifact
artifact = wandb.Artifact(
    name="sample2.csv",  # Name of the artifact
    type="raw_data",  # Type of the artifact
    description="Second sample dataset for nyc_airbnb",  # Description
)

artifact.add_file("/workspace/Project-Build-an-ML-Pipeline-Starter/components/get_data/data/sample2.csv")
run.log_artifact(artifact)


run.finish()
