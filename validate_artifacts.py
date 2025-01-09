import wandb

run = wandb.init(job_type="data_tests", project="nyc_airbnb")
artifact = run.use_artifact("clean_sample1.csv:latest")
print(f"Artifact file path: {artifact.file()}")
