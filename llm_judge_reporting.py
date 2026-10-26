def build_judge_record(
    test_case,
    judge_result,
    provider,
    model
):
    """
    Build a standardized AI-QE judge result record.

    Args:
        test_case: Original LLM evaluation test case.
        judge_result: Parsed result returned by the judge.
        provider: Judge provider name.
        model: Judge model name.

    Returns:
        dict: Standardized evaluation record.
    """

    quality_checks = [
        judge_result["correctness"],
        judge_result["relevance"],
        judge_result["instruction_following"],
    ]

    # Groundedness is optional when it is not applicable.
    groundedness = judge_result["groundedness"]

    if groundedness is not None:
        quality_checks.append(groundedness)

    status = "PASS" if all(quality_checks) else "FAIL"

    return {
        "test_id": test_case["test_id"],
        "provider": provider,
        "model": model,
        "correctness": judge_result["correctness"],
        "groundedness": judge_result["groundedness"],
        "relevance": judge_result["relevance"],
        "instruction_following": judge_result[
            "instruction_following"
        ],
        "status": status,
        "reason": judge_result["reason"],
    }
