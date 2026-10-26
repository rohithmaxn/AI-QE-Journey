import pytest

from llm_test_data_loader import load_llm_test_cases
from llm_evaluator import evaluate_llm_response


# Load the LLM evaluation dataset through the centralized loader.
test_cases = load_llm_test_cases()


@pytest.mark.parametrize(
    "test_case",
    test_cases,
    ids=[case["test_id"] for case in test_cases]
)
def test_llm_evaluation_status(test_case):
    """
    Verify that each LLM response produces the expected PASS/FAIL status.
    """

    status, reason = evaluate_llm_response(test_case)

    assert status == test_case["expected_status"]
    assert isinstance(reason, str)
    assert reason.strip()


def test_llm_test_cases_have_unique_ids():
    """
    Verify that every LLM test case has a unique test ID.
    """

    test_ids = [case["test_id"] for case in test_cases]

    assert len(test_ids) == len(set(test_ids))

def test_llm_evaluator_missing_required_field():
    test_case = {
        "test_id": "INVALID_001",
        "category": "factual_correctness",
        "question": "What is the capital of France?",
        "context": "",
        "actual_response": "The capital of France is Paris.",
        "evaluation_criteria": {
            "correctness": True,
            "groundedness": True,
            "relevance": True,
            "instruction_following": True
        }
    }

    status, reason = evaluate_llm_response(test_case)

    assert status == "FAIL"
    assert "Missing fields" in reason


def test_llm_evaluator_missing_actual_response():
    test_case = {
        "test_id": "INVALID_002",
        "category": "factual_correctness",
        "question": "What is the capital of France?",
        "context": "",
        "expected_answer": "Paris",
        "evaluation_criteria": {
            "correctness": True,
            "groundedness": True,
            "relevance": True,
            "instruction_following": True
        }
    }

    status, reason = evaluate_llm_response(test_case)

    assert status == "FAIL"
    assert "Missing fields" in reason


def test_llm_evaluator_missing_evaluation_criteria():
    test_case = {
        "test_id": "INVALID_003",
        "category": "factual_correctness",
        "question": "What is the capital of France?",
        "context": "",
        "expected_answer": "Paris",
        "actual_response": "The capital of France is Paris."
    }

    status, reason = evaluate_llm_response(test_case)

    assert status == "FAIL"
    assert "Missing fields" in reason
