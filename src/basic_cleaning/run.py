#!/usr/bin/env python

import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # Initialize a W&B run in the correct project
    logger.info("Initializing W&B run...")
    run = wandb.init(project="nyc_airbnb", job_type="basic_cleaning")
    run.config.update(args)

    # Fetch the artifact
    logger.info(f"Fetching artifact {args.input_artifact}...")
    artifact = run.use_artifact(args.input_artifact)
    artifact_local_path = artifact.file()

    # Load the data
    logger.info(f"Reading data from {artifact_local_path}...")
    df = pd.read_csv(artifact_local_path)

    # Perform basic cleaning
    logger.info("Cleaning the data...")
    df = df[df['price'].between(args.min_price, args.max_price)].copy()
    logger.info(f"Filtered rows with prices between {args.min_price} and {args.max_price}.")

    logger.info("Converting 'last_review' to datetime...")
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Removing rows with invalid geolocations...")
    df = df[df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)].copy()

    # Save the cleaned data
    output_file = "clean_sample.csv"
    logger.info(f"Saving cleaned data to {output_file}...")
    df.to_csv(output_file, index=False)

    # Log the cleaned data as an artifact
    logger.info("Logging cleaned data as an artifact...")
    output_artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    output_artifact.add_file(output_file)
    run.log_artifact(output_artifact)

    logger.info("Artifact logged successfully.")
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic Data Cleaning for NYC Airbnb Dataset")

    parser.add_argument(
        "--input_artifact",
        type=str,
        required=True,
        help="Name of the input artifact to clean (e.g., 'sample2.csv:latest')"
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        required=True,
        help="Name of the output cleaned artifact (e.g., 'clean_sample2.csv')"
    )
    parser.add_argument(
        "--output_type",
        type=str,
        required=True,
        help="Type of the output artifact (e.g., 'clean_sample')"
    )
    parser.add_argument(
        "--output_description",
        type=str,
        required=True,
        help="Description of the output artifact"
    )
    parser.add_argument(
        "--min_price",
        type=float,
        required=True,
        help="Minimum price to include in the dataset"
    )
    parser.add_argument(
        "--max_price",
        type=float,
        required=True,
        help="Maximum price to include in the dataset"
    )

    args = parser.parse_args()
    go(args)
