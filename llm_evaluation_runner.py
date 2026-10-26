from judge_factory import get_judge
from llm_judge import evaluate_with_judge


def run_llm_evaluation(test_case, provider=None):
    """
    Run an LLM evaluation using the configured judge provider.

    Args:
        test_case: LLM evaluation test case.
        provider: Optional provider override.

    Returns:
        tuple: (status, reason)
    """

    judge = get_judge(
        provider,
        test_case=test_case
    )

    return evaluate_with_judge(
        test_case,
        judge
    )
