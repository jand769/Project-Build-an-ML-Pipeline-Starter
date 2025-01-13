import argparse
import pandas as pd
import wandb


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Fetch the input artifact
    artifact_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_path)

    # Filter data based on price range
    df = df[df['price'].between(args.min_price, args.max_price)].copy()

    # Convert 'last_review' to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Filter for valid geolocation
    # Added the geolocation filter as a separate step
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save the cleaned data
    df.to_csv("clean_sample.csv", index=False)

    # Create and log the output artifact
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")  # Ensure output file matches this name
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Airbnb data")

    # Add input arguments for the cleaning process
    parser.add_argument("--input_artifact", type=str, required=True, help="Input artifact")
    parser.add_argument("--output_artifact", type=str, required=True, help="Output artifact")
    parser.add_argument("--output_type", type=str, required=True, help="Output type")
    parser.add_argument("--output_description", type=str, required=True, help="Output description")
    parser.add_argument("--min_price", type=float, required=True, help="Minimum price")
    parser.add_argument("--max_price", type=float, required=True, help="Maximum price")

    # Parse arguments and call the main function
    args = parser.parse_args()
    go(args)
