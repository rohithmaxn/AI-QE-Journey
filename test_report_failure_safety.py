
import pytest

import llm_judge_report
import llm_judge_text_report
from llm_judge_execution import execute_llm_judge_report
from openai_judge import JudgeServiceError


def test_openai_failure_does_not_overwrite_existing_reports(
    monkeypatch,
    tmp_path,
):
    json_path = tmp_path / "llm_judge_report.json"
    text_path = tmp_path / "llm_judge_report.txt"

    original_json = '{"marker": "existing-json-report"}'
    original_text = "Existing text report must be preserved."

    json_path.write_text(original_json, encoding="utf-8")
    text_path.write_text(original_text, encoding="utf-8")

    # Redirect report output to isolated temporary files.
    monkeypatch.setattr(
        llm_judge_report,
        "REPORT_PATH",
        json_path,
    )
    monkeypatch.setattr(
        llm_judge_text_report,
        "TEXT_REPORT_PATH",
        text_path,
    )

    def fake_get_judge(provider=None, test_case=None):
        def failing_judge(prompt):
            raise JudgeServiceError("Simulated OpenAI failure")

        return failing_judge

    # Replace the provider factory so no live API call can occur.
    monkeypatch.setattr(
        "llm_judge_execution.get_judge",
        fake_get_judge,
    )

    with pytest.raises(
        JudgeServiceError,
        match="Simulated OpenAI failure",
    ):
        execute_llm_judge_report(provider="OPENAI")

    # Both pre-existing reports must remain unchanged.
    assert json_path.read_text(encoding="utf-8") == original_json
    assert text_path.read_text(encoding="utf-8") == original_text
