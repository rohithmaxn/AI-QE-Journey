import pytest

from evaluator import evaluate_test
from test_data_loader import load_test_cases
from ai_report import calculate_metrics, REPORT_PATH
from validators.api_validators import validate_todo_structure
from validators.api_validators import validate_todo_types
from validators.api_validators import validate_todo_values
from config import AI_SCORE_THRESHOLD

# Load AI evaluation data through the centralized loader
test_cases = load_test_cases()


# ---------------------------------------------------------
# API TESTS
# ---------------------------------------------------------

def test_api_status_code(api_response):
    """Test to verify the API response status code."""
    print("TEST: Checking status code")
    assert api_response.status_code == 200


def test_api_response_structure(api_response):
    """Test to verify the structure of the API response."""
    data = api_response.json()

    assert validate_todo_structure(data)


def test_api_response_types(api_response):
    """Test to verify the data types in the API response."""
    data = api_response.json()

    assert validate_todo_types(data)


def test_api_response_values(api_response):
    """Test to verify the values in the API response."""
    data = api_response.json()

    assert validate_todo_values(data)


# ---------------------------------------------------------
# AI EVALUATION TESTS
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "test_case",
    test_cases,
    ids=[case["test_id"] for case in test_cases]
)

def test_ai_evaluation(test_case):
    """Test to evaluate AI-generated responses against expected outcomes."""
    status, reason = evaluate_test(test_case)

    assert status == test_case["expected_status"]
    assert reason == test_case["expected_reason"]


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def test_calculate_metrics():
    """Test to verify the calculation of metrics based on AI evaluations."""
    total, passed, failed, pass_rate = calculate_metrics(test_cases)

    assert total == 8
    assert passed == 3
    assert failed == 5
    assert pass_rate == 37.5


# ---------------------------------------------------------
# FAILURE ANALYSIS
# ---------------------------------------------------------

def test_failure_analysis():
    """Test to analyze failures in AI evaluations."""
    failed_tests = []
    score_failures = 0
    hallucination_failures = 0

    for test_case in test_cases:
        status, reason = evaluate_test(test_case)

        if status == "FAIL":
            failed_tests.append(
                f"{test_case['test_id']} - {reason}"
            )

            if test_case["score"] < AI_SCORE_THRESHOLD:
                score_failures += 1

            if test_case["hallucination"]:
                hallucination_failures += 1

    assert score_failures == 2
    assert hallucination_failures == 3
    assert len(failed_tests) == 5

    assert failed_tests[0] == (
        "LLM002 - Score below threshold"
    )


# ---------------------------------------------------------
# VALIDATION TESTS
# ---------------------------------------------------------

def test_missing_score():
    """Test to verify behavior when the score is missing in the test case."""
    test_case = {
        "hallucination": False
    }

    status, reason = evaluate_test(test_case)

    assert status == "FAIL"
    assert reason == "Missing score"


def test_missing_hallucination():
    """Test to verify behavior when the hallucination field is missing in the test case."""
    test_case = {
        "score": 0.90
    }

    status, reason = evaluate_test(test_case)

    assert status == "FAIL"
    assert reason == "Missing hallucination field"


def test_invalid_score_type():
    """Test to verify behavior when the score is of an invalid type in the test case."""
    test_case = {
        "score": "0.90",
        "hallucination": False
    }

    status, reason = evaluate_test(test_case)

    assert status == "FAIL"
    assert reason == "Invalid score type"


def test_invalid_hallucination_type():
    """Test to verify behavior when the hallucination field is of an invalid type in the test case."""
    test_case = {
        "score": 0.90,
        "hallucination": "false"
    }

    status, reason = evaluate_test(test_case)

    assert status == "FAIL"
    assert reason == "Invalid hallucination type"


# ---------------------------------------------------------
# SCORE BOUNDARY TESTS
# ---------------------------------------------------------

@pytest.mark.parametrize(
    "score,hallucination,expected_status",
    [
        (0.79, False, "FAIL"),
        (0.80, False, "PASS"),
        (0.81, False, "PASS"),
        (0.80, True, "FAIL"),
    ]
)
def test_score_boundary(
    score,
    hallucination,
    expected_status
):
    test_case = {
        "score": score,
        "hallucination": hallucination
    }

    status, reason = evaluate_test(test_case)

    assert status == expected_status


# ---------------------------------------------------------
# REPORT TESTS
# ---------------------------------------------------------

def test_report_file_exists():
    """Test to verify that the AI evaluation report file exists."""
    assert REPORT_PATH.exists()


def test_report_content():
    """Test to verify the content of the AI evaluation report."""
    with open(
        REPORT_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        content = file.read()

    assert "AI-QE EVALUATION REPORT" in content
    assert "Total Tests          : 8" in content
    assert "Passed               : 3" in content
    assert "Failed               : 5" in content
    assert "Pass Rate            : 37.50%" in content
    assert "Hallucination Count  : 3" in content
    assert "Hallucination Rate   : 37.50%" in content
    assert "LLM002 - Score below threshold" in content
    assert "LLM003 - Hallucination detected" in content
