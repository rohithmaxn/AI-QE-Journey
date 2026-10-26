import pytest

from judge_factory import get_judge
from openai_judge import OpenAIJudge

def test_get_mock_judge():
    judge = get_judge("MOCK")

    result = judge("Test prompt")

    assert isinstance(result, str)
    assert "correctness" in result
    assert "groundedness" in result
    assert "relevance" in result
    assert "instruction_following" in result


def test_get_openai_judge(monkeypatch):
    fake_client = object()

    monkeypatch.setattr(
        "openai_judge.OpenAI",
        lambda: fake_client
    )

    judge = get_judge("OPENAI")

    assert isinstance(judge, OpenAIJudge)
    assert judge.client is fake_client


def test_provider_is_case_insensitive():
    judge = get_judge("mock")

    assert callable(judge)


def test_unsupported_provider():
    with pytest.raises(
        ValueError,
        match="Unsupported judge provider"
    ):
        get_judge("UNKNOWN")
