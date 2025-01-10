#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact.
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# DO NOT MODIFY
def go(args):
    # Initialize W&B run
    run = wandb.init(
        project="nyc_airbnb",
        entity="jand769-western-governors-university",
        group="cleaning",
        job_type="basic_cleaning",
        save_code=True
    )
    run.config.update(args)

    # Log fetching artifact information
    logger.info(f"Fetching artifact: {args.input_artifact}")
    try:
        artifact_local_path = run.use_artifact(args.input_artifact).file()
    except Exception as e:
        logger.error(f"Error fetching artifact: {e}")
        raise e

    # Load the dataset
    logger.info("Reading input artifact...")
    df = pd.read_csv(artifact_local_path)

    # Drop outliers
    logger.info("Dropping outliers...")
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Converting last_review to datetime...")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Remove rows outside NYC boundaries
    logger.info("Filtering for NYC boundaries...")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save cleaned data
    output_file = "clean_sample1.csv"
    logger.info(f"Saving cleaned data to {output_file}...")
    df.to_csv(output_file, index=False)

    # Log the cleaned data artifact
    logger.info(f"Logging artifact: {args.output_artifact}")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_file)
    run.log_artifact(artifact)

    logger.info("Cleaned data uploaded successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact to be cleaned",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Output artifact for cleaned data",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output dataset",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output dataset",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum house price to be considered",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum house price to be considered",
        required=True
    )

    args = parser.parse_args()

    go(args)
