
from llm_test_data_loader import load_llm_test_cases
from llm_reference_loader import load_reference_labels


def test_groundedness_matches_context_applicability():
    test_cases = load_llm_test_cases()

    for case in test_cases:
        context = case.get("context", "")
        groundedness = case["evaluation_criteria"]["groundedness"]

        if not context.strip():
            assert groundedness is None, (
                f"{case['test_id']}: empty context requires "
                "groundedness to be None"
            )
        else:
            assert isinstance(groundedness, bool), (
                f"{case['test_id']}: supplied context requires "
                "Boolean groundedness"
            )


def test_reference_groundedness_matches_context_applicability():
    test_cases = load_llm_test_cases()
    reference_labels = load_reference_labels()

    references_by_id = {
        label["test_id"]: label
        for label in reference_labels
    }

    assert len(references_by_id) == len(reference_labels), (
        "Reference-label IDs must be unique"
    )

    assert set(references_by_id) == {
        case["test_id"] for case in test_cases
    }, "Every evaluation case must have exactly one reference label"

    for case in test_cases:
        test_id = case["test_id"]
        context = case.get("context", "")

        groundedness = references_by_id[test_id][
            "reference_dimensions"
        ]["groundedness"]

        if not context.strip():
            assert groundedness is None, (
                f"{test_id}: empty context requires reference "
                "groundedness to be None"
            )
        else:
            assert isinstance(groundedness, bool), (
                f"{test_id}: supplied context requires Boolean "
                "reference groundedness"
            )
