import wandb

# Step 1: Initialize a W&B run
run = wandb.init(
    project="nyc_airbnb",  # Replace with your W&B project name
    job_type="tag_alias"   # Description of this run
)

# Step 2: Fetch the artifact
artifact = run.use_artifact("random_forest_export:v4")  # Replace with your artifact name and version

# Step 3: Add an alias
artifact.aliases.append("prod")  # Add "prod" alias

# Step 4: Save the updated artifact with the alias
artifact.save()

# Step 5: Finish the run
run.finish()

print("Artifact successfully tagged with the alias 'prod'.")
