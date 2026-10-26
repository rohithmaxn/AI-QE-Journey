
import json
from pathlib import Path

from llm_judge_text_report import generate_text_report


REPORT_PATH = (
    Path(__file__).parent
    / "llm_judge_report.json"
)


def calculate_judge_metrics(records):
    """Calculate summary metrics from judge result records."""

    total = len(records)

    passed = sum(
        1
        for record in records
        if record["status"] == "PASS"
    )

    failed = total - passed

    correctness_failures = sum(
        1 for record in records
        if not record["correctness"]
    )

    groundedness_failures = sum(
        1
        for record in records
        if record["groundedness"] is False
    )

    groundedness_applicable = sum(
        1
        for record in records
        if record["groundedness"] is not None
    )

    groundedness_not_applicable = sum(
        1
        for record in records
        if record["groundedness"] is None
    )

    relevance_failures = sum(
        1 for record in records
        if not record["relevance"]
    )

    instruction_following_failures = sum(
        1 for record in records
        if not record["instruction_following"]
    )

    pass_rate = (
        (passed / total) * 100
        if total > 0
        else 0.0
    )

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(pass_rate, 2),
        "correctness_failures": correctness_failures,
        "groundedness_failures": groundedness_failures,
        "groundedness_applicable": groundedness_applicable,
        "groundedness_not_applicable": groundedness_not_applicable,
        "relevance_failures": relevance_failures,
        "instruction_following_failures": instruction_following_failures,
    }


def generate_judge_report(
    records,
    agreement=None,
    quality_metrics=None,
    dimension_metrics=None,
):
    """Generate JSON and text reports for the LLM judge."""

    metrics = calculate_judge_metrics(records)

    report = {
        "metrics": metrics,
        "records": records,
    }

    if agreement is not None:
        report["agreement"] = agreement

    if quality_metrics is not None:
        report["failure_metrics"] = quality_metrics

    if dimension_metrics is not None:
        report["dimension_failure_metrics"] = dimension_metrics

    # Save the complete report after adding all optional metrics.
    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    # Generate the human-readable report from the same data.
    generate_text_report(report)

    return report
