from llm_judge_reporting import build_judge_record


def test_build_judge_record_pass():
    test_case = {
        "test_id": "LLM_RESP_001"
    }

    judge_result = {
        "correctness": True,
        "groundedness": True,
        "relevance": True,
        "instruction_following": True,
        "reason": "All quality checks passed."
    }

    record = build_judge_record(
        test_case=test_case,
        judge_result=judge_result,
        provider="MOCK",
        model="test-model"
    )

    assert record["test_id"] == "LLM_RESP_001"
    assert record["provider"] == "MOCK"
    assert record["model"] == "test-model"
    assert record["correctness"] is True
    assert record["groundedness"] is True
    assert record["relevance"] is True
    assert record["instruction_following"] is True
    assert record["status"] == "PASS"
    assert record["reason"] == "All quality checks passed."


def test_build_judge_record_fail():
    test_case = {
        "test_id": "LLM_RESP_002"
    }

    judge_result = {
        "correctness": False,
        "groundedness": False,
        "relevance": True,
        "instruction_following": True,
        "reason": "The factual answer is incorrect."
    }

    record = build_judge_record(
        test_case=test_case,
        judge_result=judge_result,
        provider="OPENAI",
        model="test-model"
    )

    assert record["test_id"] == "LLM_RESP_002"
    assert record["provider"] == "OPENAI"
    assert record["status"] == "FAIL"
    assert record["reason"] == "The factual answer is incorrect."
