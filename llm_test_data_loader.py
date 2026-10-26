import json
from pathlib import Path


LLM_TEST_DATA_PATH = (
    Path(__file__).parent
    / "test_data"
    / "llm_evaluation_cases.json"
)


def load_llm_test_cases():
    """
    Load LLM evaluation test cases from the centralized JSON dataset.

    Returns:
        list: A list of LLM evaluation test-case dictionaries.
    """
    with open(LLM_TEST_DATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)
