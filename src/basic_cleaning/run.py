import argparse
import pandas as pd
import wandb

def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download the input artifact
    artifact_local_path = wandb.use_artifact(args.input_artifact).file()
    print(f"Input artifact path: {artifact_local_path}")

    # Read the dataset
    df = pd.read_csv(artifact_local_path)
    print(f"Dataset shape before cleaning: {df.shape}")

    
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    print(f"Dataset shape after geolocation filtering: {df.shape}")

    # Save the cleaned dataset
    df.to_csv(args.output_artifact, index=False)
    print(f"Cleaned dataset saved to: {args.output_artifact}")

    # Log the cleaned dataset as an artifact
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)
    print(f"Cleaned dataset logged as an artifact: {args.output_artifact}")

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic cleaning of the dataset")
    parser.add_argument("--input_artifact", type=str, required=True, help="Input artifact to clean")
    parser.add_argument("--output_artifact", type=str, required=True, help="Name of the cleaned artifact")
    parser.add_argument("--output_type", type=str, required=True, help="Type of the cleaned artifact")
    parser.add_argument("--output_description", type=str, required=True, help="Description of the cleaned artifact")
    args = parser.parse_args()
    go(args)
