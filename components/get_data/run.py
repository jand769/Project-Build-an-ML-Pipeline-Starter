#!/usr/bin/env python
"""
This script uploads a local sample file as an artifact to Weights & Biases (W&B).
"""

import argparse
import logging
import os
import wandb
from wandb_utils.log_artifact import log_artifact  # Assuming this is available

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def go(args):
    """
    Main execution function for uploading a sample file as an artifact.
    """
    # Initialize W&B run
    run = wandb.init(job_type="download_file")
    run.config.update(vars(args))

    # Log information
    logger.info(f"Processing sample: {args.sample}")
    logger.info(f"Uploading artifact: {args.artifact_name}")

    # Upload the artifact
    try:
        artifact_path = os.path.join("data", args.sample)
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"The specified sample file does not exist: {artifact_path}")
        
        log_artifact(
            args.artifact_name,
            args.artifact_type,
            args.artifact_description,
            artifact_path,
            run,
        )
        logger.info(f"Artifact {args.artifact_name} uploaded successfully.")
    except Exception as e:
        logger.error(f"Failed to upload artifact: {e}")
        raise


if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Upload a local sample file as an artifact to W&B.")

    parser.add_argument(
        "sample",
        type=str,
        help="Name of the sample file to upload (must exist in the 'data' directory).",
    )

    parser.add_argument(
        "artifact_name",
        type=str,
        help="Name for the output artifact to be created in W&B.",
    )

    parser.add_argument(
        "artifact_type",
        type=str,
        help="Type of the output artifact (e.g., 'raw_data').",
    )

    parser.add_argument(
        "artifact_description",
        type=str,
        help="A brief description of the artifact being uploaded.",
    )

    args = parser.parse_args()

    go(args)
