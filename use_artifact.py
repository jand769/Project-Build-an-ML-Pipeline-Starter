import wandb
import os
import pandas as pd

# Initialize a W&B run
run = wandb.init(
    project="nyc_airbnb", 
    entity="jand769-western-governors-university", 
    job_type="data_tests"
)

# Fetch the artifact
artifact = run.use_artifact(
    'jand769-western-governors-university/nyc_airbnb/clean_sample1.csv:v0', 
    type='cleaned_data'
)

# Download the artifact and get the directory path
artifact_dir = artifact.download()
print(f"Artifact downloaded to: {artifact_dir}")

# Construct the full path to the CSV file
file_path = os.path.join(artifact_dir, "clean_sample1.csv")

# Read the CSV file into a Pandas DataFrame
try:
    data = pd.read_csv(file_path)
    print(f"Loaded dataset:\n{data.head()}")
except FileNotFoundError:
    print(f"File not found: {file_path}. Please check the artifact contents.")

# Mark the run as finished
run.finish()
