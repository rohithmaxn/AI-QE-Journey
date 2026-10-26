
from unittest import result

import pytest

from judge_quality_metrics import calculate_failure_metrics
from judge_quality_metrics import calculate_metrics_from_records


def test_perfect_predictions():
    result = calculate_failure_metrics(
        ["FAIL", "PASS"],
        ["FAIL", "PASS"]
    )

    assert result["true_positives"] == 1
    assert result["true_negatives"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1_score"] == 1.0


def test_partial_predictions():
    result = calculate_failure_metrics(
        ["FAIL", "FAIL", "PASS", "PASS"],
        ["FAIL", "PASS", "FAIL", "PASS"]
    )

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["true_negatives"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1_score"] == 0.5


def test_no_failures_predicted():
    result = calculate_failure_metrics(
        ["FAIL", "FAIL"],
        ["PASS", "PASS"]
    )

    assert result["precision"] is None
    assert result["recall"] == 0.0
    assert result["f1_score"] == 0.0

def test_empty_inputs():
    result = calculate_failure_metrics([], [])

    assert result["precision"] is None
    assert result["recall"] is None
    assert result["f1_score"] is None


def test_different_list_lengths_raise_error():
    with pytest.raises(ValueError):
        calculate_failure_metrics(
            ["FAIL", "PASS"],
            ["FAIL"]
        )


def test_invalid_status_raises_error():
    with pytest.raises(ValueError):
        calculate_failure_metrics(
            ["INVALID"],
            ["FAIL"]
        )

from judge_quality_metrics import (
    calculate_failure_metrics,
    calculate_metrics_from_records,
)

def test_metrics_from_reference_labels_and_records():
    references = [
        {"test_id": "A", "reference_status": "FAIL"},
        {"test_id": "B", "reference_status": "FAIL"},
        {"test_id": "C", "reference_status": "PASS"},
        {"test_id": "D", "reference_status": "PASS"},
    ]

    # Deliberately place predictions in a different order.
    records = [
        {"test_id": "D", "status": "PASS"},
        {"test_id": "B", "status": "PASS"},
        {"test_id": "A", "status": "FAIL"},
        {"test_id": "C", "status": "FAIL"},
    ]

    result = calculate_metrics_from_records(references, records)

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["true_negatives"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1_score"] == 0.5


def test_metrics_reject_missing_records():
    references = [
        {"test_id": "A", "reference_status": "FAIL"},
        {"test_id": "B", "reference_status": "PASS"},
    ]

    records = [{"test_id": "A", "status": "FAIL"}]

    with pytest.raises(ValueError, match="Test IDs do not match"):
        calculate_metrics_from_records(references, records)


def test_metrics_reject_duplicate_reference_ids():
    references = [
        {"test_id": "A", "reference_status": "FAIL"},
        {"test_id": "A", "reference_status": "PASS"},
    ]

    records = [{"test_id": "A", "status": "FAIL"}]

    with pytest.raises(ValueError, match="Duplicate reference test_id"):
        calculate_metrics_from_records(references, records)

def test_no_failures_exist():
    result = calculate_failure_metrics(
        ["PASS", "PASS"],
        ["PASS", "PASS"],
    )

    assert result["true_positives"] == 0
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["true_negatives"] == 2

    assert result["precision"] is None
    assert result["recall"] is None
    assert result["f1_score"] is None


def test_metrics_reject_duplicate_record_ids():
    references = [
        {"test_id": "A", "reference_status": "FAIL"},
    ]

    records = [
        {"test_id": "A", "status": "FAIL"},
        {"test_id": "A", "status": "PASS"},
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate record test_id",
    ):
        calculate_metrics_from_records(references, records)


def test_metrics_reject_record_without_test_id():
    references = [
        {"test_id": "A", "reference_status": "FAIL"},
    ]

    records = [
        {"status": "FAIL"},
    ]

    with pytest.raises(
        ValueError,
        match="valid test_id",
    ):
        calculate_metrics_from_records(references, records)


def test_metrics_reject_missing_record_status():
    references = [
        {"test_id": "A", "reference_status": "FAIL"},
    ]

    records = [
        {"test_id": "A"},
    ]

    with pytest.raises(
        ValueError,
        match="Missing status",
    ):
        calculate_metrics_from_records(references, records)


def test_metrics_reject_invalid_reference_status():
    references = [
        {"test_id": "A", "reference_status": "UNKNOWN"},
    ]

    records = [
        {"test_id": "A", "status": "FAIL"},
    ]

    with pytest.raises(
        ValueError,
        match="Statuses must be PASS or FAIL",
    ):
        calculate_metrics_from_records(references, records)
