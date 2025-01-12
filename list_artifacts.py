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
        print(f"\nRun: {run.name} ({run.id})")
        for artifact in run.logged_artifacts():
            print(f"  - Artifact: {artifact.name}")
            print(f"    Type: {artifact.type}")
            print(f"    Version: {artifact.version}")
            # Skip linking artifacts of incompatible types
            if artifact.type in ["wandb-history", "wandb-events"]:
                print("    (Cannot generate link for this artifact type)")
                continue
            try:
                print(f"    Artifact Path: {artifact.metadata.get('path', 'N/A')}")
            except Exception as e:
                print(f"    Could not retrieve artifact details: {e}")
except Exception as e:
    print(f"Error retrieving artifacts: {e}")
