from llm_judge_text_report import (
    generate_text_report,
    TEXT_REPORT_PATH,
)


def test_generate_text_report():
    report = {
        "metrics": {
            "total": 2,
            "passed": 1,
            "failed": 1,
            "pass_rate": 50.0,
            "correctness_failures": 1,
            "groundedness_failures": 1,
            "relevance_failures": 0,
            "instruction_following_failures": 0,
        },
        "records": [
            {
                "test_id": "LLM_RESP_001",
                "status": "PASS",
                "reason": "All checks passed.",
            },
            {
                "test_id": "LLM_RESP_002",
                "status": "FAIL",
                "reason": "Incorrect factual answer",
            },
        ],
    }

    content = generate_text_report(report)

    assert "LLM JUDGE EVALUATION REPORT" in content
    assert "Total Tests                    : 2" in content
    assert "Passed                         : 1" in content
    assert "Failed                         : 1" in content
    assert "Pass Rate                      : 50.00%" in content
    assert "Correctness Failures          : 1" in content
    assert "Groundedness Failures         : 1" in content
    assert "LLM_RESP_002 - Incorrect factual answer" in content


def test_text_report_file_exists():
    report = {
        "metrics": {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
            "correctness_failures": 0,
            "groundedness_failures": 0,
            "relevance_failures": 0,
            "instruction_following_failures": 0,
        },
        "records": [],
    }

    generate_text_report(report)

    assert TEXT_REPORT_PATH.exists()


def test_text_report_when_no_failures():
    report = {
        "metrics": {
            "total": 1,
            "passed": 1,
            "failed": 0,
            "pass_rate": 100.0,
            "correctness_failures": 0,
            "groundedness_failures": 0,
            "relevance_failures": 0,
            "instruction_following_failures": 0,
        },
        "records": [
            {
                "test_id": "LLM_RESP_001",
                "status": "PASS",
                "reason": "All checks passed.",
            }
        ],
    }

    content = generate_text_report(report)

    assert "FAILED TESTS" in content
    assert "None" in content


def test_text_report_formats_undefined_metrics_as_na(
    monkeypatch,
    tmp_path,
):
    import llm_judge_text_report

    monkeypatch.setattr(
        llm_judge_text_report,
        "TEXT_REPORT_PATH",
        tmp_path / "report.txt",
    )

    report = {
        "metrics": {
            "total": 3,
            "passed": 3,
            "failed": 0,
            "pass_rate": 100.0,
            "correctness_failures": 0,
            "groundedness_failures": 0,
            "groundedness_applicable": 3,
            "groundedness_not_applicable": 0,
            "relevance_failures": 0,
            "instruction_following_failures": 0,
        },
        "records": [],
        "failure_metrics": {
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_negatives": 3,
            "precision": None,
            "recall": None,
            "f1_score": None,
        },
        "dimension_failure_metrics": {
            "relevance": {
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "true_negatives": 3,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "applicable": 3,
                "not_applicable": 0,
            }
        },
    }

    content = llm_judge_text_report.generate_text_report(report)

    assert "OVERALL FAILURE-DETECTION METRICS" in content
    assert "PER-DIMENSION FAILURE-DETECTION METRICS" in content
    assert "Precision                       : N/A" in content
    assert "Recall                          : N/A" in content
    assert "F1-score                        : N/A" in content
    assert "Relevance" in content
