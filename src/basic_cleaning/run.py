import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info(f"Fetching artifact: {args.input_artifact}")
    artifact_path = run.use_artifact(args.input_artifact).file()

    logger.info(f"Reading dataset from {artifact_path}")
    df = pd.read_csv(artifact_path)

    logger.info(f"Filtering rows by price between {args.min_price} and {args.max_price}")
    df = df[df["price"].between(args.min_price, args.max_price)].copy()

    logger.info("Filtering rows based on longitude and latitude boundaries")
    idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("Saving cleaned dataset")
    df.to_csv(args.output_artifact, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw Airbnb data")

    parser.add_argument("--input_artifact", type=str, required=True, help="Input artifact name")
    parser.add_argument("--output_artifact", type=str, required=True, help="Output artifact name")
    parser.add_argument("--output_type", type=str, required=True, help="Output artifact type")
    parser.add_argument("--output_description", type=str, required=True, help="Output artifact description")
    parser.add_argument("--min_price", type=float, required=True, help="Minimum price filter")
    parser.add_argument("--max_price", type=float, required=True, help="Maximum price filter")

    args = parser.parse_args()
    go(args)
