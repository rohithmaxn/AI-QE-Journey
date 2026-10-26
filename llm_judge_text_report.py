
from pathlib import Path


TEXT_REPORT_PATH = (
    Path(__file__).parent / "llm_judge_report.txt"
)


def format_metric(value):
    """Format undefined metrics as N/A and numeric metrics consistently."""

    if value is None:
        return "N/A"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.4f}"

    return str(value)


def generate_text_report(report):
    """
    Generate a human-readable text report from judge results.

    Optional failure_metrics, dimension_failure_metrics,
    and agreement sections are included when available.
    """

    metrics = report["metrics"]
    records = report["records"]

    lines = [
        "LLM JUDGE EVALUATION REPORT",
        "============================",
        "",
        f"Total Tests                    : {metrics['total']}",
        f"Passed                         : {metrics['passed']}",
        f"Failed                         : {metrics['failed']}",
        f"Pass Rate                      : {metrics['pass_rate']:.2f}%",
        "",
        "QUALITY DIMENSIONS",
        "------------------",
        (
            "Correctness Failures          : "
            f"{metrics['correctness_failures']}"
        ),
        (
            "Groundedness Failures         : "
            f"{metrics['groundedness_failures']}"
        ),
        (
            "Groundedness Applicable       : "
            f"{metrics.get('groundedness_applicable', 'N/A')}"
        ),
        (
            "Groundedness Not Applicable  : "
            f"{metrics.get('groundedness_not_applicable', 'N/A')}"
        ),
        (
            "Relevance Failures            : "
            f"{metrics['relevance_failures']}"
        ),
        (
            "Instruction-Following Failures: "
            f"{metrics['instruction_following_failures']}"
        ),
    ]

    # Overall failure-detection metrics.
    failure_metrics = report.get("failure_metrics")

    if failure_metrics is not None:
        lines.extend([
            "",
            "OVERALL FAILURE-DETECTION METRICS",
            "---------------------------------",
            (
                "True Positives                  : "
                f"{failure_metrics.get('true_positives', 'N/A')}"
            ),
            (
                "False Positives                 : "
                f"{failure_metrics.get('false_positives', 'N/A')}"
            ),
            (
                "False Negatives                 : "
                f"{failure_metrics.get('false_negatives', 'N/A')}"
            ),
            (
                "True Negatives                  : "
                f"{failure_metrics.get('true_negatives', 'N/A')}"
            ),
            (
                "Precision                       : "
                f"{format_metric(failure_metrics.get('precision'))}"
            ),
            (
                "Recall                          : "
                f"{format_metric(failure_metrics.get('recall'))}"
            ),
            (
                "F1-score                        : "
                f"{format_metric(failure_metrics.get('f1_score'))}"
            ),
        ])

    # Per-dimension failure-detection metrics.
    dimension_metrics = report.get("dimension_failure_metrics")

    if dimension_metrics is not None:
        lines.extend([
            "",
            "PER-DIMENSION FAILURE-DETECTION METRICS",
            "---------------------------------------",
        ])

        for dimension, values in dimension_metrics.items():
            dimension_name = dimension.replace("_", " ").title()

            lines.extend([
                "",
                dimension_name,
                "-" * len(dimension_name),
                (
                    "  True Positives                : "
                    f"{values.get('true_positives', 'N/A')}"
                ),
                (
                    "  False Positives               : "
                    f"{values.get('false_positives', 'N/A')}"
                ),
                (
                    "  False Negatives               : "
                    f"{values.get('false_negatives', 'N/A')}"
                ),
                (
                    "  True Negatives                : "
                    f"{values.get('true_negatives', 'N/A')}"
                ),
                (
                    "  Precision                     : "
                    f"{format_metric(values.get('precision'))}"
                ),
                (
                    "  Recall                        : "
                    f"{format_metric(values.get('recall'))}"
                ),
                (
                    "  F1-score                      : "
                    f"{format_metric(values.get('f1_score'))}"
                ),
                (
                    "  Applicable                    : "
                    f"{values.get('applicable', 'N/A')}"
                ),
                (
                    "  Not Applicable                : "
                    f"{values.get('not_applicable', 'N/A')}"
                ),
            ])

    # Agreement metrics.
    agreement = report.get("agreement")

    if agreement is not None:
        lines.extend([
            "",
            "JUDGE AGREEMENT",
            "---------------",
            f"Cases Compared  : {agreement['total']}",
            f"Matches         : {agreement['agreements']}",
            f"Mismatches      : {agreement['disagreements']}",
            f"Agreement Rate  : {agreement['agreement_rate']:.2f}%",
        ])

        if agreement["mismatches"]:
            lines.extend([
                "",
                "AGREEMENT MISMATCHES",
                "--------------------",
            ])

            for mismatch in agreement["mismatches"]:
                lines.append(
                    f"{mismatch['test_id']} - "
                    f"expected={mismatch['expected_status']}, "
                    f"actual={mismatch['actual_status']}"
                )

    # Failed evaluation records.
    lines.extend([
        "",
        "FAILED TESTS",
        "------------",
    ])

    failed_records = [
        record
        for record in records
        if record["status"] == "FAIL"
    ]

    if not failed_records:
        lines.append("None")
    else:
        for record in failed_records:
            lines.append(
                f"{record['test_id']} - {record['reason']}"
            )

    report_content = "\n".join(lines)

    with open(
        TEXT_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report_content)

    return report_content
