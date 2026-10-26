import json
from pathlib import Path


REFERENCE_LABELS_PATH = (
    Path(__file__).parent
    / "test_data"
    / "llm_reference_labels.json"
)


def load_reference_labels():
    """
    Load independent reference labels for LLM evaluation.

    Returns:
        list: Reference-label dictionaries.
    """
    with open(
        REFERENCE_LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)
