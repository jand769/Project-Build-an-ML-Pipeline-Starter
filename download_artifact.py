import wandb

# Initialize the API
api = wandb.Api()

project_name = "nyc_airbnb"
entity_name = "jand769-western-governors-university"
artifact_name = "clean_sample1.csv"  # Replace with the artifact name you're interested in
artifact_version = "v0"  # Replace with the specific version

try:
    # Fetch the artifact
    artifact = api.artifact(f"{entity_name}/{project_name}/{artifact_name}:{artifact_version}")
    print(f"Downloading artifact: {artifact.name}, version: {artifact.version}")
    download_dir = artifact.download()
    print(f"Artifact downloaded to: {download_dir}")
except Exception as e:
    print(f"Error downloading artifact: {e}")
