import argparse
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_data(input_path, output_path, min_price, max_price):
    logger.info("Reading data from %s", input_path)
    df = pd.read_csv(input_path)

    # Drop rows with missing prices or out of bounds
    logger.info("Cleaning data: Removing price outliers...")
    df = df[df["price"].between(min_price, max_price)].copy()

    # Additional cleaning: drop rows with invalid geolocation
    logger.info("Cleaning data: Removing invalid geolocations...")
    idx = df["longitude"].between(-74.25, -73.50) & df["latitude"].between(40.5, 41.2)
    df = df[idx].copy()

    # Save cleaned data
    logger.info("Saving cleaned data to %s", output_path)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw data")
    parser.add_argument("input_artifact", type=str, help="Input CSV file")
    parser.add_argument("output_artifact", type=str, help="Output cleaned CSV file")
    parser.add_argument("min_price", type=float, help="Minimum price to include")
    parser.add_argument("max_price", type=float, help="Maximum price to include")
    parser.add_argument("--output_type", type=str, help="Type of the output artifact", required=True)
    parser.add_argument("--output_description", type=str, help="Description of the output artifact", required=True)

    args = parser.parse_args()

    clean_data(
        input_path=args.input_artifact,
        output_path=args.output_artifact,
        min_price=args.min_price,
        max_price=args.max_price,
    )

    logger.info("Output type: %s", args.output_type)
    logger.info("Output description: %s", args.output_description)
