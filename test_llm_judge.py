import pytest
import json
from llm_test_data_loader import load_llm_test_cases
from llm_judge import evaluate_with_judge

from llm_judge import (
    build_judge_prompt,
    parse_judge_result,
    evaluate_with_judge,
)


def test_build_judge_prompt():
    test_case = {
        "question": "What is the capital of France?",
        "context": "",
        "expected_answer": "Paris",
        "actual_response": "The capital of France is Paris.",
    }

    prompt = build_judge_prompt(test_case)

    assert "What is the capital of France?" in prompt
    assert "Paris" in prompt
    assert "The capital of France is Paris." in prompt
    assert "correctness" in prompt
    assert "groundedness" in prompt
    assert "relevance" in prompt
    assert "instruction_following" in prompt


def test_parse_valid_judge_result():
    raw_result = """
    {
        "correctness": true,
        "groundedness": true,
        "relevance": true,
        "instruction_following": true,
        "reason": "Response is correct and relevant."
    }
    """

    result = parse_judge_result(raw_result)

    assert result["correctness"] is True
    assert result["groundedness"] is True
    assert result["relevance"] is True
    assert result["instruction_following"] is True
    assert isinstance(result["reason"], str)


def test_parse_invalid_json():
    raw_result = "This is not JSON"

    with pytest.raises(
        ValueError,
        match="Judge response is not valid JSON"
    ):
        parse_judge_result(raw_result)


def test_parse_missing_field():
    raw_result = """
    {
        "correctness": true,
        "groundedness": true,
        "relevance": true,
        "instruction_following": true
    }
    """

    with pytest.raises(
        ValueError,
        match="Judge response missing fields"
    ):
        parse_judge_result(raw_result)


def test_parse_invalid_boolean_type():
    raw_result = """
    {
        "correctness": "true",
        "groundedness": true,
        "relevance": true,
        "instruction_following": true,
        "reason": "Invalid correctness type."
    }
    """

    with pytest.raises(
        ValueError,
        match="must be boolean"
    ):
        parse_judge_result(raw_result)


def test_evaluate_with_mock_judge_pass():
    test_case = {
        "question": "What is the capital of France?",
        "context": "",
        "expected_answer": "Paris",
        "actual_response": "The capital of France is Paris.",
    }

    def mock_judge(prompt):
        return """
        {
            "correctness": true,
            "groundedness": true,
            "relevance": true,
            "instruction_following": true,
            "reason": "Response passed all quality checks."
        }
        """

    status, reason = evaluate_with_judge(
        test_case,
        mock_judge
    )

    assert status == "PASS"
    assert reason == "Response passed all quality checks."


def test_evaluate_with_mock_judge_fail():
    test_case = {
        "question": "What is the capital of France?",
        "context": "",
        "expected_answer": "Paris",
        "actual_response": "The capital of France is Berlin.",
    }

    def mock_judge(prompt):
        return """
        {
            "correctness": false,
            "groundedness": false,
            "relevance": true,
            "instruction_following": true,
            "reason": "The factual answer is incorrect."
        }
        """

    status, reason = evaluate_with_judge(
        test_case,
        mock_judge
    )

    assert status == "FAIL"
    assert reason == "The factual answer is incorrect."

# Load the real LLM evaluation dataset.
test_cases = load_llm_test_cases()



def create_mock_judge(test_case):
    """
    Create a deterministic mock judge based on the expected
    evaluation criteria in the test data.

    JSON serialization correctly converts Python None to JSON null.
    """

    criteria = test_case["evaluation_criteria"]
    expected_status = test_case["expected_status"]

    if expected_status == "PASS":
        reason = "Mock judge: all quality criteria passed."
    else:
        reason = (
            f"Mock judge: expected failure for "
            f"{test_case['test_id']}."
        )

    result = {
        "correctness": criteria["correctness"],
        "groundedness": criteria["groundedness"],
        "relevance": criteria["relevance"],
        "instruction_following": criteria["instruction_following"],
        "reason": reason,
    }

    def mock_judge(prompt):
        return json.dumps(result)

    return mock_judge


@pytest.mark.parametrize(
    "test_case",
    test_cases,
    ids=[case["test_id"] for case in test_cases]
)
def test_llm_judge_with_evaluation_cases(test_case):
    """
    Verify that every real LLM evaluation case can flow through
    the judge abstraction and produce the expected status.
    """

    mock_judge = create_mock_judge(test_case)

    status, reason = evaluate_with_judge(
        test_case,
        mock_judge
    )

    assert status == test_case["expected_status"]
    assert isinstance(reason, str)
    assert reason.strip()


def test_parse_non_string_response():
    with pytest.raises(
        ValueError,
        match="Judge response must be a JSON string",
    ):
        parse_judge_result(None)


@pytest.mark.parametrize(
    "raw_result",
    ['null', '[]', 'true', '"text"'],
)
def test_parse_non_object_json(raw_result):
    with pytest.raises(
        ValueError,
        match="Judge response must be a JSON object",
    ):
        parse_judge_result(raw_result)


def test_parse_invalid_groundedness_type():
    raw_result = """
    {
        "correctness": true,
        "groundedness": "null",
        "relevance": true,
        "instruction_following": true,
        "reason": "Invalid groundedness."
    }
    """

    with pytest.raises(
        ValueError,
        match="groundedness.*must be boolean or null",
    ):
        parse_judge_result(raw_result)


def test_parse_invalid_reason_type():
    raw_result = """
    {
        "correctness": true,
        "groundedness": null,
        "relevance": true,
        "instruction_following": true,
        "reason": 123
    }
    """

    with pytest.raises(
        ValueError,
        match="Judge field 'reason' must be a string",
    ):
        parse_judge_result(raw_result)


def test_parse_unexpected_field():
    raw_result = """
    {
        "correctness": true,
        "groundedness": null,
        "relevance": true,
        "instruction_following": true,
        "reason": "Valid response.",
        "extra_field": "unexpected"
    }
    """

    with pytest.raises(
        ValueError,
        match="Judge response contains unexpected fields",
    ):
        parse_judge_result(raw_result)
