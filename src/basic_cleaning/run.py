#!/usr/bin/env python
"""
Basic cleaning script for the Airbnb dataset.
"""
import argparse
import logging
import wandb
import pandas as pd

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def clean_data(input_path, min_price, max_price):
    """
    Cleans the input dataset based on price and geographical bounds.

    Args:
        input_path (str): Path to the input CSV file.
        min_price (float): Minimum price to filter rows.
        max_price (float): Maximum price to filter rows.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logger.info(f"Loading dataset from {input_path}")
    df = pd.read_csv(input_path)

    # Filter rows based on price
    logger.info(f"Filtering rows with price between {min_price} and {max_price}")
    df = df[df["price"].between(min_price, max_price)].copy()

    # Convert last_review to datetime
    logger.info("Converting 'last_review' column to datetime format")
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    # Remove invalid geolocations
    logger.info("Filtering rows with valid longitude and latitude ranges")
    df = df[
        df["longitude"].between(-74.25, -73.50)
        & df["latitude"].between(40.5, 41.2)
    ].copy()

    return df


def go(args):
    """
    Main function to execute the data cleaning process and log the artifact.
    """
    logger.info("Starting W&B run for basic cleaning")
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Fetch input artifact
    logger.info(f"Fetching input artifact: {args.input_artifact}")
    artifact = run.use_artifact(args.input_artifact)
    artifact_local_path = artifact.file()

    # Clean data
    df = clean_data(
        input_path=artifact_local_path,
        min_price=args.min_price,
        max_price=args.max_price,
    )

    # Save cleaned data to a new file
    output_file = "clean_sample1.csv"  # Updated to clean_sample1.csv
    logger.info(f"Saving cleaned dataset to {output_file}")
    df.to_csv(output_file, index=False)

    # Log cleaned dataset as a new artifact
    logger.info(f"Logging cleaned dataset as artifact: {args.output_artifact}")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_file)
    run.log_artifact(artifact)

    logger.info("Cleaning process completed and artifact logged successfully.")
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic Data Cleaning for Airbnb Dataset")

    parser.add_argument(
        "--input_artifact",
        type=str,
        required=True,
        help="Name of the input artifact (e.g., 'sample1.csv:latest')",
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        required=True,
        help="Name of the output artifact (e.g., 'clean_sample1.csv')",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        required=True,
        help="Type of the output artifact (e.g., 'cleaned_data')",
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
