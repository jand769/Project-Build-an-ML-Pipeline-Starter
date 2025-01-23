import pandas as pd
import numpy as np
import scipy.stats

# Add the necessary imports for testing
def test_column_names(data):
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
    assert list(expected_columns) == list(data.columns.values)


def test_neighborhood_names(data):
    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
    neighborhoods = set(data["neighbourhood_group"].unique())

    # Check that all neighborhoods in the data are known
    assert set(known_names) == neighborhoods


def test_proper_boundaries(data):
    """Ensure proper longitude and latitude boundaries for New York City."""
    assert data["longitude"].between(-74.25, -73.50).all()
    assert data["latitude"].between(40.5, 41.2).all()


def test_similar_neigh_distrib(data, ref_data, kl_threshold):
    """Compare the KL divergence of the neighborhood distribution with the reference."""
    dist1 = data["neighbourhood_group"].value_counts().sort_index()
    dist2 = ref_data["neighbourhood_group"].value_counts().sort_index()
    kl_divergence = scipy.stats.entropy(dist1, dist2, base=2)
    assert kl_divergence < kl_threshold


########################################################
# Implement here test_row_count and test_price_range   #
########################################################

def test_row_count(data):
    """Ensure that the dataset contains a reasonable number of rows."""
    assert 15000 < data.shape[0] < 1000000


def test_price_range(data, min_price, max_price):
    """Ensure that all prices are within the specified range."""
    assert data["price"].between(min_price, max_price).all()
