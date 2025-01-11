import wandb

# Initialize a run
run = wandb.init(project="nyc_airbnb")

# Fetch the artifact
artifact = run.use_artifact("clean_sample1.csv:latest")

# Add the alias 'reference'
artifact.aliases.append("reference")
artifact.save()

print("Alias 'reference' added successfully.")
