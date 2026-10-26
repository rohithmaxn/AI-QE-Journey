from config import JUDGE_PROVIDER
from openai_judge import OpenAIJudge
import json

class MockJudge:
    """
    Deterministic offline judge used for testing and development.
    """

    def __init__(self, test_case=None):
        self.test_case = test_case
        self.model = "mock"

    def __call__(self, prompt):
        """Return a deterministic result for offline testing."""

        if self.test_case is None:
            result = {
                "correctness": True,
                "groundedness": True,
                "relevance": True,
                "instruction_following": True,
                "reason": "Mock judge evaluation passed.",
            }
        else:
            criteria = self.test_case["evaluation_criteria"]

            result = {
                "correctness": criteria["correctness"],
                "groundedness": criteria["groundedness"],
                "relevance": criteria["relevance"],
                "instruction_following": criteria["instruction_following"],
                "reason": self.test_case.get(
                    "expected_reason",
                    f"Mock evaluation for {self.test_case['test_id']}.",
                ),
            }

        return json.dumps(result)


def get_judge(provider=None, test_case=None):
    """
    Return the configured judge implementation.
    """

    selected_provider = (
        provider or JUDGE_PROVIDER
    ).upper()

    if selected_provider == "OPENAI":
        return OpenAIJudge()

    if selected_provider == "MOCK":
        return MockJudge(test_case)

    raise ValueError(
        f"Unsupported judge provider: {selected_provider}"
    )
