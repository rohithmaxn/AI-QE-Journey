import json
from pathlib import Path


TEST_DATA_PATH = (
    Path(__file__).parent
    / "test_data"
    / "ai_test_cases.json"
)


def load_test_cases():
    """Load test cases from the JSON file."""
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)
