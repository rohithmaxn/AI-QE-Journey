
from judge_quality_metrics import calculate_dimension_failure_metrics
import pytest

def test_dimension_metrics_exclude_groundedness_na():
    references = [
        {
            "test_id": "CASE_A",
            "reference_dimensions": {
                "correctness": True,
                "groundedness": None,
                "relevance": True,
                "instruction_following": True,
            },
        },
        {
            "test_id": "CASE_B",
            "reference_dimensions": {
                "correctness": False,
                "groundedness": False,
                "relevance": True,
                "instruction_following": True,
            },
        },
    ]

    records = [
        {
            "test_id": "CASE_A",
            "correctness": True,
            "groundedness": None,
            "relevance": True,
            "instruction_following": True,
        },
        {
            "test_id": "CASE_B",
            "correctness": False,
            "groundedness": False,
            # Deliberately simulate one incorrect relevance judgment.
            "relevance": False,
            "instruction_following": True,
        },
    ]

    result = calculate_dimension_failure_metrics(
        references,
        records,
    )

    groundedness = result["groundedness"]

    assert groundedness["applicable"] == 1
    assert groundedness["not_applicable"] == 1
    assert groundedness["true_positives"] == 1
    assert groundedness["f1_score"] == 1.0

    relevance = result["relevance"]

    assert relevance["applicable"] == 2
    assert relevance["false_positives"] == 1
    assert relevance["precision"] == 0.0
    assert relevance["recall"] is None
    assert relevance["f1_score"] == 0.0


def _valid_dimension_test_data():
    references = [
        {
            "test_id": "CASE_A",
            "reference_dimensions": {
                "correctness": True,
                "groundedness": None,
                "relevance": True,
                "instruction_following": True,
            },
        },
        {
            "test_id": "CASE_B",
            "reference_dimensions": {
                "correctness": False,
                "groundedness": False,
                "relevance": True,
                "instruction_following": True,
            },
        },
    ]

    records = [
        {
            "test_id": "CASE_A",
            "correctness": True,
            "groundedness": None,
            "relevance": True,
            "instruction_following": True,
        },
        {
            "test_id": "CASE_B",
            "correctness": False,
            "groundedness": False,
            "relevance": True,
            "instruction_following": True,
        },
    ]

    return references, records


def test_dimension_metrics_reject_missing_reference_dimension():
    references, records = _valid_dimension_test_data()
    references[0]["reference_dimensions"].pop("relevance")

    with pytest.raises(
        ValueError,
        match="Missing reference dimension 'relevance'",
    ):
        calculate_dimension_failure_metrics(references, records)


def test_dimension_metrics_reject_invalid_reference_value():
    references, records = _valid_dimension_test_data()
    references[0]["reference_dimensions"]["correctness"] = "true"

    with pytest.raises(
        ValueError,
        match="must be boolean",
    ):
        calculate_dimension_failure_metrics(references, records)


def test_dimension_metrics_reject_invalid_record_value():
    references, records = _valid_dimension_test_data()
    records[1]["groundedness"] = "false"

    with pytest.raises(
        ValueError,
        match="must be boolean",
    ):
        calculate_dimension_failure_metrics(references, records)


def test_dimension_metrics_reject_missing_record_dimension():
    references, records = _valid_dimension_test_data()
    records[0].pop("relevance")

    with pytest.raises(
        ValueError,
        match="Evaluation dimension 'relevance' must be boolean",
    ):
        calculate_dimension_failure_metrics(references, records)


def test_dimension_metrics_reject_duplicate_record_ids():
    references, records = _valid_dimension_test_data()
    records.append(records[0].copy())

    with pytest.raises(
        ValueError,
        match="Duplicate record test_id",
    ):
        calculate_dimension_failure_metrics(references, records)
