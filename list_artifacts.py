import wandb

# Initialize the API
api = wandb.Api()

project_name = "nyc_airbnb"
entity_name = "jand769-western-governors-university"

try:
    # Fetch all runs in the project
    runs = api.runs(path=f"{entity_name}/{project_name}")

    print(f"Artifacts in project '{project_name}':")
    for run in runs:
        logged_artifacts = list(run.logged_artifacts())
        if logged_artifacts:
            print(f"\nRun: {run.name} ({run.id})")
            for artifact in logged_artifacts:
                if artifact.type not in ["wandb-history", "wandb-events"]:
                    print(f"  - Artifact: {artifact.name}")
                    print(f"    Type: {artifact.type}")
                    print(f"    Version: {artifact.version}")
                    print(f"    Size: {artifact.size or 'N/A'} bytes")
                    # Fix for get_path
                    try:
                        artifact_link = f"https://wandb.ai/{entity_name}/{project_name}/artifacts/{artifact.type}/{artifact.name}/{artifact.version}"
                        print(f"    Artifact Link: {artifact_link}")
                    except Exception as e:
                        print(f"    Could not retrieve artifact details: {e}")
except Exception as e:
    print(f"Error retrieving artifacts: {e}")
