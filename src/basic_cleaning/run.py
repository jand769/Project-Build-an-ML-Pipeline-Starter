#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Perform basic data cleaning and log the cleaned dataset as a new artifact.
    """
    # Initialize W&B run
    run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning")
    run.config.update(args)

    # Download the input artifact
    logger.info(f"Downloading input artifact: {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    
    # Read the input data
    logger.info(f"Reading input data from {artifact_local_path}")
    df = pd.read_csv(artifact_local_path)

    # Drop outliers based on price
    logger.info(f"Filtering data based on price range: {args.min_price} - {args.max_price}")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert `last_review` to datetime
    logger.info("Converting 'last_review' to datetime format")
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

    # Filter based on valid geographical boundaries
    logger.info("Filtering data based on geographical boundaries")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save the cleaned data to a new file
    output_file = "clean_sample.csv"
    logger.info(f"Saving cleaned data to {output_file}")
    df.to_csv(output_file, index=False)

    # Log the cleaned data as a new W&B artifact
    logger.info("Logging cleaned data to W&B as a new artifact")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_file)
    run.log_artifact(artifact)

    logger.info("Data cleaning and logging completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price to consider",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price to consider",
        required=True,
    )

    args = parser.parse_args()

    go(args)
