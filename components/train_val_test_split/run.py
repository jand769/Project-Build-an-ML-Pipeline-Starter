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
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    """
    Main function to split the dataset into train-validation and test sets.
    """
    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)

    logger.info(f"Fetching artifact {args.input}")
    artifact_local_path = run.use_artifact(args.input).file()

    # Load the dataset
    df = pd.read_csv(artifact_local_path)

    # Split into train-validation and test sets
    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    # Save and log trainval and test datasets
    for df_split, split_name in zip([trainval, test], ['trainval', 'test']):
        logger.info(f"Uploading {split_name}_data.csv dataset")
        with tempfile.NamedTemporaryFile("w", delete=False) as fp:
            df_split.to_csv(fp.name, index=False)
            log_artifact(
                f"{split_name}_data.csv",
                f"{split_name}_data",
                f"{split_name} split of dataset",
                fp.name,
                run,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train-validation and test sets")

    parser.add_argument("input", type=str, help="Input artifact to split")
    parser.add_argument(
        "test_size", type=float,
        help="Size of the test split. Fraction of the dataset, or number of items"
    )
    parser.add_argument(
        "--random_seed", type=int,
        help="Seed for random number generator",
        default=42,
        required=False
    )
    parser.add_argument(
        "--stratify_by", type=str,
        help="Column to use for stratification",
        default='none',
        required=False
    )

    args = parser.parse_args()
    go(args)
