#!/usr/bin/env python

"""
Basic data cleaning script for the NYC Airbnb dataset.
This script fetches the input artifact, performs cleaning, and logs the cleaned artifact.
"""

import argparse
import logging
import pandas as pd
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Main function to perform data cleaning and artifact logging.
    """
    logger.info("Initializing W&B run...")
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Fetch the input artifact
    logger.info(f"Fetching input artifact: {args.input_artifact}")
    artifact = run.use_artifact(args.input_artifact)
    artifact_local_path = artifact.file()

    # Load the data
    logger.info(f"Reading data from {artifact_local_path}")
    df = pd.read_csv(artifact_local_path)

    # Filter data based on price
    logger.info(f"Filtering rows with prices between {args.min_price} and {args.max_price}")
    df = df[df["price"].between(args.min_price, args.max_price)].copy()

    # Convert last_review to datetime
    logger.info("Converting 'last_review' column to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # Remove invalid geolocations
    logger.info("Removing rows with invalid geolocations")
    df = df[
        df["longitude"].between(-74.25, -73.50)
        & df["latitude"].between(40.5, 41.2)
    ].copy()

    # Save cleaned data to a CSV file
    output_file = "clean_sample.csv"
    logger.info(f"Saving cleaned data to {output_file}")
    df.to_csv(output_file, index=False)

    # Log the cleaned dataset as a new artifact
    logger.info("Logging the cleaned dataset as an artifact")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_file)
    run.log_artifact(artifact)

    logger.info("Cleaning process completed and artifact logged successfully.")
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic Data Cleaning for NYC Airbnb Dataset")

    parser.add_argument(
        "--input_artifact",
        type=str,
        required=True,
        help="Name of the input artifact to clean (e.g., 'sample2.csv:latest')",
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        required=True,
        help="Name of the output cleaned artifact (e.g., 'clean_sample2.csv')",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        required=True,
        help="Type of the output artifact (e.g., 'clean_sample')",
    )
    parser.add_argument(
        "--output_description",
        type=str,
        required=True,
        help="Description of the output artifact",
    )
    parser.add_argument(
        "--min_price",
        type=float,
        required=True,
        help="Minimum price to include in the dataset",
    )
    parser.add_argument(
        "--max_price",
        type=float,
        required=True,
        help="Maximum price to include in the dataset",
    )

    args = parser.parse_args()
    go(args)
