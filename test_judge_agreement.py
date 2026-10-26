
from judge_agreement import calculate_judge_agreement


def test_judge_agreement_all_match():
    test_cases = [
        {"test_id": "T001", "reference_status": "PASS"},
        {"test_id": "T002", "reference_status": "FAIL"},
    ]

    records = [
        {"test_id": "T001", "status": "PASS"},
        {"test_id": "T002", "status": "FAIL"},
    ]

    result = calculate_judge_agreement(test_cases, records)

    assert result["total"] == 2
    assert result["agreements"] == 2
    assert result["disagreements"] == 0
    assert result["agreement_rate"] == 100.0


def test_judge_agreement_with_mismatch():
    test_cases = [
        {"test_id": "T001", "reference_status": "PASS"},
        {"test_id": "T002", "reference_status": "FAIL"},
    ]

    records = [
        {"test_id": "T001", "status": "PASS"},
        {"test_id": "T002", "status": "PASS"},
    ]

    result = calculate_judge_agreement(test_cases, records)

    assert result["total"] == 2
    assert result["agreements"] == 1
    assert result["disagreements"] == 1
    assert result["agreement_rate"] == 50.0
    assert result["mismatches"] == [
        {
            "test_id": "T002",
            "expected_status": "FAIL",
            "actual_status": "PASS",
        }
    ]


def test_missing_evaluation_record_counts_as_mismatch():
    test_cases = [
        {"test_id": "T001", "reference_status": "PASS"},
        {"test_id": "T002", "reference_status": "FAIL"},
    ]

    records = [
        {"test_id": "T001", "status": "PASS"},
    ]

    result = calculate_judge_agreement(test_cases, records)

    assert result["total"] == 2
    assert result["agreements"] == 1
    assert result["disagreements"] == 1
    assert result["agreement_rate"] == 50.0
    assert result["mismatches"][0]["actual_status"] is None


def test_empty_judge_agreement():
    result = calculate_judge_agreement([], [])

    assert result["total"] == 0
    assert result["agreements"] == 0
    assert result["disagreements"] == 0
    assert result["agreement_rate"] == 0.0
    assert result["mismatches"] == []


def test_dimension_agreement_all_match():
    dimensions = {
        "correctness": True,
        "groundedness": True,
        "relevance": True,
        "instruction_following": True,
    }

    reference_labels = [{
        "test_id": "T001",
        "reference_status": "PASS",
        "reference_dimensions": dimensions,
    }]

    records = [{
        "test_id": "T001",
        "status": "PASS",
        **dimensions,
    }]

    result = calculate_judge_agreement(
        reference_labels,
        records,
    )

    assert result["agreement_rate"] == 100.0

    for dimension in dimensions:
        stats = result["dimension_agreement"][dimension]
        assert stats["total"] == 1
        assert stats["agreements"] == 1
        assert stats["disagreements"] == 0
        assert stats["agreement_rate"] == 100.0


def test_dimension_mismatch_detected_even_when_status_matches():
    reference_dimensions = {
        "correctness": False,
        "groundedness": False,
        "relevance": True,
        "instruction_following": True,
    }

    reference_labels = [{
        "test_id": "T002",
        "reference_status": "FAIL",
        "reference_dimensions": reference_dimensions,
    }]

    records = [{
        "test_id": "T002",
        "status": "FAIL",
        "correctness": False,
        "groundedness": True,
        "relevance": True,
        "instruction_following": True,
    }]

    result = calculate_judge_agreement(
        reference_labels,
        records,
    )

    # Overall status agrees, but one dimension is wrong.
    assert result["agreement_rate"] == 100.0
    assert result["disagreements"] == 0

    groundedness = result["dimension_agreement"]["groundedness"]

    assert groundedness["agreements"] == 0
    assert groundedness["disagreements"] == 1
    assert groundedness["agreement_rate"] == 0.0
    assert groundedness["mismatches"] == [{
        "test_id": "T002",
        "expected": False,
        "actual": True,
    }]


def test_groundedness_na_is_excluded_from_agreement_denominator():
    reference_labels = [
        {
            "test_id": "CASE_NA",
            "reference_status": "PASS",
            "reference_dimensions": {
                "correctness": True,
                "groundedness": None,
                "relevance": True,
                "instruction_following": True,
            },
        },
        {
            "test_id": "CASE_GROUNDED",
            "reference_status": "FAIL",
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
            "test_id": "CASE_NA",
            "status": "PASS",
            "correctness": True,
            "groundedness": None,
            "relevance": True,
            "instruction_following": True,
        },
        {
            "test_id": "CASE_GROUNDED",
            "status": "FAIL",
            "correctness": False,
            "groundedness": False,
            "relevance": True,
            "instruction_following": True,
        },
    ]

    result = calculate_judge_agreement(
        reference_labels,
        records,
    )

    groundedness = result["dimension_agreement"]["groundedness"]

    assert groundedness["total"] == 1
    assert groundedness["agreements"] == 1
    assert groundedness["disagreements"] == 0
    assert groundedness["agreement_rate"] == 100.0
