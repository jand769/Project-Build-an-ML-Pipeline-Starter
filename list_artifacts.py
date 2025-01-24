import wandb

api = wandb.Api()
project = "nyc_airbnb"
entity = "jand769-western-governors-university"

# Fetch all artifacts in the project
artifacts = api.artifacts(entity=entity, project=project)
for artifact in artifacts:
    print(artifact.name, artifact.aliases)
