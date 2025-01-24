import wandb

# Initialize a W&B run
run = wandb.init(project="nyc_airbnb", entity="jand769-western-governors-university")

# Create an artifact
artifact = wandb.Artifact("clean_sample1", type="dataset")

# Add the correct file path
artifact.add_file("/absolute/path/to/clean_sample1.csv")  # Replace with the actual path to your CSV file

# Log the artifact
run.log_artifact(artifact)

# Finish the run
run.finish()
