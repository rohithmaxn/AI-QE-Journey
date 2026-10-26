import pytest
from config import ENVIRONMENT
from config import AI_SCORE_THRESHOLD
from config import API_BASE_URL
from clients.api_client import APIClient
from test_data_loader import load_test_cases
from ai_report import generate_report


@pytest.fixture
def api_client():
    """Fixture to provide an instance of the APIClient for tests."""
    return APIClient(API_BASE_URL)


@pytest.fixture
def api_response():
    """Fixture to provide a response from the API for tests."""
    print("SETUP: Calling API")

    client = APIClient(API_BASE_URL)
    response = client.get_todo(1)

    yield response

    print("CLEANUP: Closing API response")
    response.close()

@pytest.fixture(scope="session", autouse=True)
def generate_ai_report():
    """Fixture to generate an AI evaluation report after all tests have run."""
    test_cases = load_test_cases()
    generate_report(test_cases)

@pytest.fixture
def test_cases():
    """Fixture to provide test cases for evaluation."""
    return load_test_cases()

@pytest.fixture(scope="session")
def test_environment():
    """Fixture to set up and tear down the test environment."""
    print("SETUP: Creating test environment")

    environment = {
        "name": ENVIRONMENT,
        "threshold": AI_SCORE_THRESHOLD
    }

    yield environment

    print("CLEANUP: Destroying test environment")

def test_environment_name():
    """Test to verify the environment name."""
    print("TEST: environment name")


def test_environment_threshold():
    """Test to verify the environment threshold."""
    print("TEST: environment threshold")


def test_environment_type():
    """Test to verify the environment type."""
    print("TEST: environment type")

def test_cleanup(api_response):
    """Test to ensure that the API response is cleaned up properly."""
    print("TEST: Using API response")

    assert api_response.status_code == 200
