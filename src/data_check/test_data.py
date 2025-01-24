import argparse
import pandas as pd
import scipy.stats
import wandb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_column_names(data):
    """
    Ensure the dataset has the expected columns in the correct order.
    """
    expected_columns = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]
    assert list(data.columns) == expected_columns, "Column names do not match!"


def test_neighborhood_names(data):
    """
    Check that all neighborhoods in the dataset are known and valid.
    """
    known_neighborhoods = {"Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"}
    assert set(data["neighbourhood_group"].unique()) == known_neighborhoods, "Unknown neighborhood names!"


def test_proper_boundaries(data):
    """
    Ensure proper longitude and latitude boundaries for New York City.
    """
    assert data["longitude"].between(-74.25, -73.50).all(), "Longitude out of bounds!"
    assert data["latitude"].between(40.5, 41.2).all(), "Latitude out of bounds!"


def test_similar_neigh_distrib(data, ref_data, kl_threshold):
    """
    Compare the KL divergence of the neighborhood distribution with the reference.
    """
    dist1 = data["neighbourhood_group"].value_counts(normalize=True).sort_index()
    dist2 = ref_data["neighbourhood_group"].value_counts(normalize=True).sort_index()
    kl_divergence = scipy.stats.entropy(dist1, dist2)
    assert kl_divergence < kl_threshold, f"KL divergence {kl_divergence} exceeds threshold!"


def test_row_count(data):
    """
    Ensure the dataset contains a reasonable number of rows.
    """
    assert 15000 < data.shape[0] < 1000000, "Row count is out of range!"


def test_price_range(data, min_price, max_price):
    """
    Ensure that all prices are within the specified range.
    """
    assert data["price"].between(min_price, max_price).all(), "Prices out of range!"


def main(args):
    """
    Runs data checks such as column names, neighborhood names, boundaries, and KL divergence.
    """
    run = wandb.init(job_type="data_check")
    logger.info(f"Fetching data artifact: {args.csv}")
    data_path = run.use_artifact(args.csv).file()
    ref_path = run.use_artifact(args.ref).file()

    data = pd.read_csv(data_path)
    ref_data = pd.read_csv(ref_path)

    # Run tests
    logger.info("Running tests on the dataset...")
    test_column_names(data)
    test_neighborhood_names(data)
    test_proper_boundaries(data)
    test_similar_neigh_distrib(data, ref_data, args.kl_threshold)
    test_row_count(data)
    test_price_range(data, args.min_price, args.max_price)
    logger.info("All tests passed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Check Step")
    parser.add_argument("--csv", type=str, help="Input data artifact name in W&B")
    parser.add_argument("--ref", type=str, help="Reference data artifact name in W&B")
    parser.add_argument("--kl_threshold", type=float, help="KL divergence threshold")
    parser.add_argument("--min_price", type=float, help="Minimum price")
    parser.add_argument("--max_price", type=float, help="Maximum price")

    args = parser.parse_args()
    main(args)
