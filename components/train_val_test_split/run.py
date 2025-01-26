#!/usr/bin/env python
"""
This script splits the provided dataframe into train-validation and test sets.
"""

import argparse
import logging
import pandas as pd
import wandb
import tempfile
from sklearn.model_selection import train_test_split
from wandb_utils import log_artifact  # Importing the log_artifact utility function

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Main function to split the dataset into train-validation and test sets.
    """
    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)

    # Fetch the input artifact
    logger.info(f"Fetching artifact {args.input}")
    artifact_local_path = run.use_artifact(args.input).file()

    # Load the dataset
    logger.info("Loading dataset")
    df = pd.read_csv(artifact_local_path)

    # Perform train-validation and test split
    logger.info("Splitting dataset into train-validation and test sets")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != "none" else None,
    )

    # Save and log the splits
    for split, name in zip([trainval, test], ["trainval", "test"]):
        logger.info(f"Uploading {name}_data.csv")
        with tempfile.NamedTemporaryFile("w", delete=False) as fp:
            split.to_csv(fp.name, index=False)
            log_artifact(
                name=f"{name}_data.csv",
                type=f"{name}_data",
                description=f"{name} split of the dataset",
                path=fp.name,
                run=run,
                aliases=["latest", "reference"] if name == "trainval" else ["latest"],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train-validation and test sets")

    parser.add_argument("input", type=str, help="Input artifact to split")
    parser.add_argument("test_size", type=float, help="Fraction of the dataset to use for the test set")
    parser.add_argument(
        "--random_seed", type=int, help="Seed for the random number generator", default=42, required=False
    )
    parser.add_argument(
        "--stratify_by", type=str, help="Column to use for stratification", default="none", required=False
    )

    args = parser.parse_args()
    go(args)
