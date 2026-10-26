from llm_judge_report import (
    calculate_judge_metrics,
    generate_judge_report,
    REPORT_PATH,
)


def test_calculate_judge_metrics():
    records = [
        {
            "test_id": "LLM_RESP_001",
            "status": "PASS",
            "correctness": True,
            "groundedness": True,
            "relevance": True,
            "instruction_following": True,
        },
        {
            "test_id": "LLM_RESP_002",
            "status": "FAIL",
            "correctness": False,
            "groundedness": False,
            "relevance": True,
            "instruction_following": True,
        },
    ]

    metrics = calculate_judge_metrics(records)

    assert metrics["total"] == 2
    assert metrics["passed"] == 1
    assert metrics["failed"] == 1
    assert metrics["pass_rate"] == 50.0

    assert metrics["correctness_failures"] == 1
    assert metrics["groundedness_failures"] == 1
    assert metrics["relevance_failures"] == 0
    assert metrics["instruction_following_failures"] == 0


def test_calculate_judge_metrics_empty():
    metrics = calculate_judge_metrics([])

    assert metrics["total"] == 0
    assert metrics["passed"] == 0
    assert metrics["failed"] == 0
    assert metrics["pass_rate"] == 0.0


def test_generate_judge_report(tmp_path, monkeypatch):
    records = [
        {
            "test_id": "LLM_RESP_001",
            "status": "PASS",
            "correctness": True,
            "groundedness": True,
            "relevance": True,
            "instruction_following": True,
            "reason": "All checks passed.",
        }
    ]

    report_path = tmp_path / "llm_judge_report.json"

    monkeypatch.setattr(
        "llm_judge_report.REPORT_PATH",
        report_path
    )

    report = generate_judge_report(records)

    assert report["metrics"]["total"] == 1
    assert report["metrics"]["passed"] == 1
    assert report["metrics"]["failed"] == 0
    assert report["metrics"]["pass_rate"] == 100.0

    assert len(report["records"]) == 1
    assert report["records"][0]["test_id"] == "LLM_RESP_001"

    assert report_path.exists()


def test_generate_judge_report_contains_metrics_and_records(
    tmp_path,
    monkeypatch
):
    records = [
        {
            "test_id": "LLM_RESP_001",
            "status": "PASS",
            "correctness": True,
            "groundedness": True,
            "relevance": True,
            "instruction_following": True,
            "reason": "Passed.",
        }
    ]

    report_path = tmp_path / "llm_judge_report.json"

    monkeypatch.setattr(
        "llm_judge_report.REPORT_PATH",
        report_path
    )

    generate_judge_report(records)

    import json

    with open(
        report_path,
        "r",
        encoding="utf-8"
    ) as file:
        saved_report = json.load(file)

    assert "metrics" in saved_report
    assert "records" in saved_report
    assert saved_report["metrics"]["total"] == 1
    assert saved_report["records"][0]["test_id"] == (
        "LLM_RESP_001"
    )
