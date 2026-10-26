from types import SimpleNamespace

from llm_evaluation_runner import run_llm_evaluation


def create_test_case():
    return {
        "test_id": "LLM_RESP_TEST",
        "category": "factual_correctness",
        "question": "What is the capital of France?",
        "context": "",
        "expected_answer": "Paris",
        "actual_response": "The capital of France is Paris.",
        "evaluation_criteria": {
            "correctness": True,
            "groundedness": True,
            "relevance": True,
            "instruction_following": True
        },
        "expected_reason": "Mock judge evaluation for LLM_RESP_TEST."
    }


def test_runner_uses_mock_provider():
    test_case = create_test_case()

    status, reason = run_llm_evaluation(
        test_case,
        provider="MOCK"
    )

    assert status == "PASS"
    assert reason == "Mock judge evaluation for LLM_RESP_TEST."


def test_runner_uses_openai_provider(monkeypatch):
    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text=(
                    '{"correctness": true, '
                    '"groundedness": true, '
                    '"relevance": true, '
                    '"instruction_following": true, '
                    '"reason": "OpenAI mock integration passed."}'
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    class FakeOpenAIJudge:
        def __init__(self):
            self.client = fake_client
            self.model = "test-model"

        def __call__(self, prompt):
            return self.client.responses.create(
                model=self.model,
                input=prompt
            ).output_text

    monkeypatch.setattr(
        "llm_evaluation_runner.get_judge",
        lambda provider=None, test_case=None: FakeOpenAIJudge()
    )

    test_case = create_test_case()

    status, reason = run_llm_evaluation(
        test_case,
        provider="OPENAI"
    )

    assert status == "PASS"
    assert reason == "OpenAI mock integration passed."


def test_runner_provider_override():
    test_case = create_test_case()

    status, reason = run_llm_evaluation(
        test_case,
        provider="mock"
    )

    assert status == "PASS"
    assert reason == "Mock judge evaluation for LLM_RESP_TEST."
