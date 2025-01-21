#!/usr/bin/env python
"""
Basic cleaning script for the Airbnb dataset.
"""
import argparse
import logging
import os
import wandb
import pandas as pd

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    # Initialize W&B run
    run = wandb.init(project="nyc_airbnb", group="cleaning", save_code=True)
    run.config.update(args)

    # Fetch the input artifact
    logger.info(f"Fetching input artifact: {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    if not os.path.exists(artifact_local_path):
        logger.error(f"Artifact not found: {artifact_local_path}")
        return

    # Load dataset
    logger.info("Loading dataset for cleaning...")
    df = pd.read_csv(artifact_local_path)

    # Apply cleaning steps
    logger.info("Applying cleaning steps...")
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    df['last_review'] = pd.to_datetime(df['last_review'])
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save cleaned dataset
    output_file = "clean_sample.csv"
    logger.info(f"Saving cleaned dataset to {output_file}")
    df.to_csv(output_file, index=False)

    # Log the cleaned dataset as an artifact
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(output_file)
    run.log_artifact(artifact)
    logger.info(f"Artifact {args.output_artifact} logged successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic data cleaning for Airbnb dataset")
  
    parser.add_argument("--input_artifact", type=str, help="Input artifact name", required=True)
    parser.add_argument("--output_artifact", type=str, help="Output artifact name", required=True)
    parser.add_argument("--output_type", type=str, help="Type of the output artifact", required=True)
    parser.add_argument("--output_description", type=str, help="Description of the output artifact", required=True)
    parser.add_argument("--min_price", type=float, help="Minimum price filter", required=True)
    parser.add_argument("--max_price", type=float, help="Maximum price filter", required=True)

    args = parser.parse_args()

    go(args)
