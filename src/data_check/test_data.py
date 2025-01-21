import pandas as pd
import pytest
import os
import scipy.stats
import wandb

# Constants for testing
EXPECTED_COLUMNS = [
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

KNOWN_NEIGHBORHOODS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

def test_column_names(data):
    """Test if the dataset contains the expected columns."""
    actual_columns = list(data.columns)
    assert EXPECTED_COLUMNS == actual_columns, f"Expected {EXPECTED_COLUMNS}, but got {actual_columns}"


def test_neighborhood_names(data):
    """Test if the dataset contains the known neighborhood groups."""
    actual_neighborhoods = set(data["neighbourhood_group"].unique())
    assert set(KNOWN_NEIGHBORHOODS) == actual_neighborhoods, (
        f"Expected neighborhoods {KNOWN_NEIGHBORHOODS}, but got {actual_neighborhoods}"
    )


def test_proper_boundaries(data):
    """Test if all longitude and latitude values are within the proper boundaries."""
    is_valid = data["longitude"].between(-74.25, -73.50) & data["latitude"].between(40.5, 41.2)
    assert is_valid.all(), "Found invalid geolocation(s) in the dataset."


def test_similar_neigh_distrib(data, ref_data, kl_threshold):
    """Test if the neighborhood group distribution is similar between the current and reference datasets."""
    dist1 = data["neighbourhood_group"].value_counts(normalize=True).sort_index()
    dist2 = ref_data["neighbourhood_group"].value_counts(normalize=True).sort_index()
    kl_divergence = scipy.stats.entropy(dist1, dist2, base=2)
    assert kl_divergence < kl_threshold, f"KL divergence {kl_divergence} exceeds threshold {kl_threshold}"


def test_row_count(data):
    """Test if the dataset has an acceptable number of rows."""
    num_rows = data.shape[0]
    assert 15000 < num_rows < 1000000, f"Row count {num_rows} is outside the acceptable range."


def test_price_range(data, min_price, max_price):
    """Test if all prices are within the acceptable range."""
    is_in_range = data["price"].between(min_price, max_price)
    assert is_in_range.all(), f"Found prices outside the range {min_price}-{max_price}."


@pytest.fixture(scope="session")
def data(request):
    """Fixture to load the data artifact into a DataFrame."""
    artifact_name = request.config.getoption("--csv")
    if not artifact_name:
        pytest.fail("You must provide the '--csv' argument with the dataset artifact name.")
    run = wandb.init(job_type="data_tests", resume=True)
    artifact = run.use_artifact(artifact_name)
    artifact_path = artifact.download()

    # Ensure artifact points to a file, not a directory
    if os.path.isdir(artifact_path):
        artifact_path = os.path.join(artifact_path, os.listdir(artifact_path)[0])

    return pd.read_csv(artifact_path)


@pytest.fixture(scope="session")
def ref_data(request):
    """Fixture to load the reference data artifact into a DataFrame."""
    artifact_name = request.config.getoption("--ref")
    if not artifact_name:
        pytest.fail("You must provide the '--ref' argument with the reference artifact name.")
    run = wandb.init(job_type="data_tests", resume=True)
    try:
        artifact = run.use_artifact(artifact_name)
        artifact_path = artifact.download()

        # Ensure artifact points to a file, not a directory
        if os.path.isdir(artifact_path):
            artifact_path = os.path.join(artifact_path, os.listdir(artifact_path)[0])

        return pd.read_csv(artifact_path)
    except Exception as e:
        pytest.fail(f"Error fetching reference artifact '{artifact_name}': {e}")


@pytest.fixture(scope="session")
def kl_threshold(request):
    """Fixture to retrieve the KL divergence threshold."""
    kl = request.config.getoption("--kl_threshold")
    if not kl:
        pytest.fail("You must provide the '--kl_threshold' argument.")
    return float(kl)


@pytest.fixture(scope="session")
def min_price(request):
    """Fixture to retrieve the minimum acceptable price."""
    min_p = request.config.getoption("--min_price")
    if not min_p:
        pytest.fail("You must provide the '--min_price' argument.")
    return float(min_p)


@pytest.fixture(scope="session")
def max_price(request):
    """Fixture to retrieve the maximum acceptable price."""
    max_p = request.config.getoption("--max_price")
    if not max_p:
        pytest.fail("You must provide the '--max_price' argument.")
    return float(max_p)
