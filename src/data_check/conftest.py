import pytest
import pandas as pd
import wandb
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

def pytest_addoption(parser):
    """
    Adds command-line options to pytest for parameterizing the tests.
    """
    parser.addoption("--csv", action="store", help="Name of the input data artifact in W&B")
    parser.addoption("--ref", action="store", help="Name of the reference data artifact in W&B")
    parser.addoption("--kl_threshold", action="store", help="Threshold for the KL divergence test")
    parser.addoption("--min_price", action="store", help="Minimum acceptable price")
    parser.addoption("--max_price", action="store", help="Maximum acceptable price")

@pytest.fixture(scope="session")
def data(request):
    """
    Pytest fixture to fetch and load the input data artifact as a DataFrame.
    """
    artifact_name = request.config.option.csv
    if not artifact_name:
        pytest.fail("You must provide the '--csv' option to specify the data artifact")

    logger.info(f"Fetching data artifact: {artifact_name}")
    try:
        run = wandb.init(project="nyc_airbnb", entity="jand769-western-governors-university", job_type="data_tests", resume=True)
        data_path = run.use_artifact(artifact_name).file()
        logger.info(f"Fetched data artifact from path: {data_path}")
    except wandb.errors.CommError as e:
        logger.error(f"W&B Communication Error: {e}")
        pytest.fail(f"Failed to fetch data artifact: {e}")
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        pytest.fail(f"Failed to fetch data artifact: {e}")
    finally:
        run.finish()

    return pd.read_csv(data_path)

@pytest.fixture(scope="session")
def ref_data(request):
    """
    Pytest fixture to fetch and load the reference data artifact as a DataFrame.
    """
    artifact_name = request.config.option.ref
    if not artifact_name:
        pytest.fail("You must provide the '--ref' option to specify the reference artifact")

    logger.info(f"Fetching reference artifact: {artifact_name}")
    try:
        run = wandb.init(project="nyc_airbnb", entity="jand769-western-governors-university", job_type="data_tests", resume=True)
        data_path = run.use_artifact(artifact_name).file()
        logger.info(f"Fetched reference artifact from path: {data_path}")
    except wandb.errors.CommError as e:
        logger.error(f"W&B Communication Error: {e}")
        pytest.fail(f"Failed to fetch reference artifact: {e}")
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        pytest.fail(f"Failed to fetch reference artifact: {e}")
    finally:
        run.finish()

    return pd.read_csv(data_path)

@pytest.fixture(scope="session")
def kl_threshold(request):
    """
    Pytest fixture to retrieve the KL divergence threshold.
    """
    kl_threshold = request.config.option.kl_threshold
    if kl_threshold is None:
        pytest.fail("You must provide '--kl_threshold' for the KL divergence test")

    try:
        return float(kl_threshold)
    except ValueError:
        pytest.fail("The provided KL threshold must be a float")

@pytest.fixture(scope="session")
def min_price(request):
    """
    Pytest fixture to retrieve the minimum price for filtering.
    """
    min_price = request.config.option.min_price
    if min_price is None:
        pytest.fail("You must provide '--min_price' for price filtering")

    try:
        return float(min_price)
    except ValueError:
        pytest.fail("The provided minimum price must be a float")

@pytest.fixture(scope="session")
def max_price(request):
    """
    Pytest fixture to retrieve the maximum price for filtering.
    """
    max_price = request.config.option.max_price
    if max_price is None:
        pytest.fail("You must provide '--max_price' for price filtering")

    try:
        return float(max_price)
    except ValueError:
        pytest.fail("The provided maximum price must be a float")
