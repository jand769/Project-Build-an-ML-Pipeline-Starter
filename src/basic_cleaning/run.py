import argparse
import pandas as pd
import wandb

def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    artifact_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_path)

    df = df[df['price'].between(args.min_price, args.max_price)].copy()
    df['last_review'] = pd.to_datetime(df['last_review'])
    df = df[df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)].copy()

    df.to_csv("clean_sample.csv", index=False)

    artifact = wandb.Artifact(
    args.output_artifact,
    type=args.output_type,
    description=args.output_description,
    )
    artifact.add_file("clean_sample1.csv")  # Output file must match this name
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Airbnb data")
    parser.add_argument("--input_artifact", type=str, required=True, help="Input artifact")
    parser.add_argument("--output_artifact", type=str, required=True, help="Output artifact")
    parser.add_argument("--output_type", type=str, required=True, help="Output type")
    parser.add_argument("--output_description", type=str, required=True, help="Output description")
    parser.add_argument("--min_price", type=float, required=True, help="Minimum price")
    parser.add_argument("--max_price", type=float, required=True, help="Maximum price")

    args = parser.parse_args()
    go(args)
