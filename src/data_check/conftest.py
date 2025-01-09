import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)
    data_path = run.use_artifact(request.config.option.csv).file()
    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")
    return pd.read_csv(data_path)


@pytest.fixture(scope='session')
def ref_data(request):
    run = wandb.init(job_type="data_tests", resume=True)
    data_path = run.use_artifact(request.config.option.ref).file()
    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")
    return pd.read_csv(data_path)


@pytest.fixture(scope='session')
def kl_threshold(request):
    return float(request.config.option.kl_threshold)


@pytest.fixture(scope='session')
def min_price(request):
    return float(request.config.option.min_price)


@pytest.fixture(scope='session')
def max_price(request):
    return float(request.config.option.max_price)
