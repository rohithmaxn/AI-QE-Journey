import pytest
from types import SimpleNamespace
from openai_judge import OpenAIJudge, JudgeServiceError

def test_openai_judge_sends_structured_request():
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)

            return SimpleNamespace(
                output_text=(
                    '{"correctness": true, '
                    '"groundedness": true, '
                    '"relevance": true, '
                    '"instruction_following": true, '
                    '"reason": "All checks passed."}'
                )
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    judge = OpenAIJudge(
        model="test-model",
        client=fake_client
    )

    prompt = "Evaluate this LLM response."

    result = judge(prompt)

    assert result == (
        '{"correctness": true, '
        '"groundedness": true, '
        '"relevance": true, '
        '"instruction_following": true, '
        '"reason": "All checks passed."}'
    )

    assert captured["model"] == "test-model"
    assert captured["input"] == prompt


def test_openai_judge_uses_json_schema():
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)

            return SimpleNamespace(
                output_text="{}"
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    judge = OpenAIJudge(
        model="test-model",
        client=fake_client
    )

    judge("Test prompt")

    text_config = captured["text"]
    response_format = text_config["format"]

    assert response_format["type"] == "json_schema"
    assert response_format["name"] == "llm_evaluation"
    assert response_format["strict"] is True

    schema = response_format["schema"]

    assert schema["type"] == "object"
    assert "correctness" in schema["properties"]
    assert "groundedness" in schema["properties"]
    assert "relevance" in schema["properties"]
    assert "instruction_following" in schema["properties"]
    assert "reason" in schema["properties"]


def test_openai_judge_default_model():
    class FakeResponses:
        def create(self, **kwargs):
            return SimpleNamespace(
                output_text="{}"
            )

    fake_client = SimpleNamespace(
        responses=FakeResponses()
    )

    judge = OpenAIJudge(
        client=fake_client
    )

    assert judge.model is not None
    assert isinstance(judge.model, str)
    assert judge.model.strip()

def test_openai_judge_handles_insufficient_quota(monkeypatch):
    class FakeRateLimitError(Exception):
        code = "insufficient_quota"

    def fake_create(**kwargs):
        raise FakeRateLimitError(
            "You have no credits remaining."
        )

    class FakeResponses:
        create = staticmethod(fake_create)

    fake_client = type(
        "FakeClient",
        (),
        {
            "responses": FakeResponses()
        }
    )()

    monkeypatch.setattr(
        "openai_judge.RateLimitError",
        FakeRateLimitError
    )

    judge = OpenAIJudge(
        model="test-model",
        client=fake_client
    )

    with pytest.raises(
        JudgeServiceError,
        match="API quota is exhausted"
    ):
        judge("Test prompt")


def test_openai_judge_handles_non_quota_rate_limit(monkeypatch):
    class FakeRateLimitError(Exception):
        code = "rate_limit_exceeded"

    def fake_create(**kwargs):
        raise FakeRateLimitError("Request rate exceeded.")

    class FakeResponses:
        create = staticmethod(fake_create)

    fake_client = type(
        "FakeClient",
        (),
        {"responses": FakeResponses()},
    )()

    monkeypatch.setattr(
        "openai_judge.RateLimitError",
        FakeRateLimitError,
    )

    judge = OpenAIJudge(
        model="test-model",
        client=fake_client,
    )

    with pytest.raises(
        JudgeServiceError,
        match="rate limit exceeded",
    ):
        judge("Test prompt")


def test_openai_judge_handles_general_api_error(monkeypatch):
    class FakeAPIError(Exception):
        pass

    def fake_create(**kwargs):
        raise FakeAPIError("Simulated API failure.")

    class FakeResponses:
        create = staticmethod(fake_create)

    fake_client = type(
        "FakeClient",
        (),
        {"responses": FakeResponses()},
    )()

    monkeypatch.setattr(
        "openai_judge.APIError",
        FakeAPIError,
    )

    judge = OpenAIJudge(
        model="test-model",
        client=fake_client,
    )

    with pytest.raises(
        JudgeServiceError,
        match="OpenAI judge request failed",
    ):
        judge("Test prompt")


def test_openai_judge_preserves_explicit_falsey_client():
    class FalseyClient:
        def __bool__(self):
            return False

    fake_client = FalseyClient()

    judge = OpenAIJudge(
        model="test-model",
        client=fake_client,
    )

    assert judge.client is fake_client
