from llm_judge_execution import (
    execute_llm_judge_report,
)


def test_execute_all_llm_cases_with_mock():
    report = execute_llm_judge_report(
        provider="MOCK"
    )

    assert report["metrics"]["total"] == 8
    assert report["metrics"]["passed"] == 4
    assert report["metrics"]["failed"] == 4

    assert report["metrics"]["correctness_failures"] == 3
    assert report["metrics"]["groundedness_failures"] == 2
    assert report["metrics"]["relevance_failures"] == 0
    assert report["metrics"]["instruction_following_failures"] == 2
    assert report["metrics"]["groundedness_applicable"] == 4
    assert report["metrics"]["groundedness_not_applicable"] == 4


def test_execute_report_contains_all_test_ids():
    report = execute_llm_judge_report(
        provider="MOCK"
    )

    test_ids = [
        record["test_id"]
        for record in report["records"]
    ]

    assert len(test_ids) == 8

    assert test_ids == [
        "LLM_RESP_001",
        "LLM_RESP_002",
        "LLM_RESP_003",
        "LLM_RESP_004",
        "LLM_RESP_005",
        "LLM_RESP_006",
        "LLM_RESP_007",
        "LLM_RESP_008",
    ]
