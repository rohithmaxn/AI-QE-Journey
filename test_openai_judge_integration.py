from types import SimpleNamespace

from llm_judge import evaluate_with_judge
from openai_judge import OpenAIJudge


def create_fake_openai_client(
    correctness=True,
    groundedness=True,
    relevance=True,
    instruction_following=True,
    reason="All checks passed."
):
    """
    Create a fake OpenAI client that behaves like the
    Responses API for our tests.
    """

    class FakeResponses:
        def create(self, **kwargs):
            output = (
                "{"
                f'"correctness": {str(correctness).lower()}, '
                f'"groundedness": {str(groundedness).lower()}, '
                f'"relevance": {str(relevance).lower()}, '
                f'"instruction_following": {str(instruction_following).lower()}, '
                f'"reason": "{reason}"'
                "}"
            )

            return SimpleNamespace(
                output_text=output
            )

    return SimpleNamespace(
        responses=FakeResponses()
    )


def test_openai_judge_integrates_with_evaluate_with_judge():
    test_case = {
        "question": "What is the capital of France?",
        "context": "",
        "expected_answer": "Paris",
        "actual_response": "The capital of France is Paris.",
    }

    fake_client = create_fake_openai_client()

    judge = OpenAIJudge(
        model="test-model",
        client=fake_client
    )

    status, reason = evaluate_with_judge(
        test_case,
        judge
    )

    assert status == "PASS"
    assert reason == "All checks passed."


def test_openai_judge_integrates_failure():
    test_case = {
        "question": "What is the capital of France?",
        "context": "",
        "expected_answer": "Paris",
        "actual_response": "The capital of France is Berlin.",
    }

    fake_client = create_fake_openai_client(
        correctness=False,
        groundedness=False,
        relevance=True,
        instruction_following=True,
        reason="The factual answer is incorrect."
    )

    judge = OpenAIJudge(
        model="test-model",
        client=fake_client
    )

    status, reason = evaluate_with_judge(
        test_case,
        judge
    )

    assert status == "FAIL"
    assert reason == "The factual answer is incorrect."
