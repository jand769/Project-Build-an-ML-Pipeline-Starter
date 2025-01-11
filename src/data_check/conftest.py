import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


@pytest.fixture(scope="session")
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    artifact_name = request.config.option.csv
    print(f"Fetching data artifact: {artifact_name}")

    try:
        data_path = run.use_artifact(artifact_name).file()
        print(f"Fetched data artifact: {data_path}")
    except Exception as e:
        print(f"Error fetching data artifact: {e}")
        raise e

    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope="session")
def ref_data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    artifact_name = request.config.option.ref
    print(f"Fetching reference artifact: {artifact_name}")

    try:
        data_path = run.use_artifact(artifact_name).file()
        print(f"Fetched reference artifact: {data_path}")
    except Exception as e:
        print(f"Error fetching reference artifact: {e}")
        raise e

    df = pd.read_csv(data_path)
    return df


@pytest.fixture(scope="session")
def kl_threshold(request):
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)


@pytest.fixture(scope="session")
def min_price(request):
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)


@pytest.fixture(scope="session")
def max_price(request):
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)
