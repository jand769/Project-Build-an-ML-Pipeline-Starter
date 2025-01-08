import wandb

# Initialize a W&B run
run = wandb.init(project="nyc_airbnb", job_type="upload_file")

# Create an artifact
artifact = wandb.Artifact(
    name="sample1.csv",
    type="raw_data",
    description="Sample Airbnb dataset for cleaning",
)

# Add the file to the artifact
artifact.add_file("components/get_data/data/sample1.csv")

# Log the artifact to W&B
run.log_artifact(artifact)
run.finish()
