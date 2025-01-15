#!/usr/bin/env python
"""
Clean raw data by removing outliers and invalid geolocations, and log the cleaned data as an artifact.
"""

import argparse
import logging
import wandb
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def clean_data(input_path, output_path, min_price, max_price):

    logger.info("Reading data from %s", input_path)
    df = pd.read_csv(input_path)

    logger.info("Removing price outliers...")
    df = df[df["price"].between(min_price, max_price)].copy()

    logger.info("Removing invalid geolocations...")
    idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("Saving cleaned data to %s", output_path)
    df.to_csv(output_path, index=False)


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(vars(args))

    # Download input artifact and load it
    logger.info("Downloading artifact: %s", args.input_artifact)
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Perform cleaning
    clean_data(
        input_path=artifact_local_path,
        output_path="clean_sample.csv",
        min_price=args.min_price,
        max_price=args.max_price,
    )

    # Log the cleaned data as a new artifact
    logger.info("Logging cleaned data as artifact...")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    logger.info("Cleaned data logged as artifact.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform basic data cleaning")
  
    parser.add_argument(
        "--input_artifact", 
        type=str, 
        required=True, 
        help="Input artifact to clean (from W&B)"
    )

    parser.add_argument(
        "--output_artifact", 
        type=str, 
        required=True, 
        help="Name of the output cleaned artifact"
    )

    parser.add_argument(
        "--output_type", 
        type=str, 
        required=True, 
        help="Type of the output artifact"
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
        help="Minimum price for filtering data"
    )

    parser.add_argument(
        "--max_price", 
        type=float, 
        required=True, 
        help="Maximum price for filtering data"
    )

    args = parser.parse_args()
    go(args)
