from pathlib import Path
from evaluator import evaluate_test

REPORT_PATH = (
    Path(__file__).parent / "ai_qe_results.txt"
)


def calculate_metrics(evaluations):
    """Calculate metrics based on the evaluations of test cases."""
    total = len(evaluations)

    passed = 0
    failed = 0

    for test_case in evaluations:
        status, reason = evaluate_test(test_case)

        if status == "PASS":
            passed += 1
        else:
            failed += 1

    pass_rate = (
        (passed / total) * 100
        if total > 0
        else 0.0
    )

    return total, passed, failed, pass_rate


def generate_report(evaluations):
    """Generate a report based on the evaluations of test cases."""
    total, passed, failed, pass_rate = calculate_metrics(
        evaluations
    )

    hallucination_count = sum(
        1
        for test_case in evaluations
        if test_case["hallucination"]
    )

    hallucination_rate = (
        (hallucination_count / total) * 100
        if total > 0
        else 0.0
    )

    failure_reasons = {}
    failed_tests = []

    for test_case in evaluations:
        status, reason = evaluate_test(test_case)

        if status == "FAIL":
            failure_reasons[reason] = (
                failure_reasons.get(reason, 0) + 1
            )

            failed_tests.append(
                f"{test_case['test_id']} - {reason}"
            )

    report_lines = [
        "AI-QE EVALUATION REPORT",
        "=======================",
        f"Total Tests          : {total}",
        f"Passed               : {passed}",
        f"Failed               : {failed}",
        f"Pass Rate            : {pass_rate:.2f}%",
        f"Hallucination Count  : {hallucination_count}",
        f"Hallucination Rate   : {hallucination_rate:.2f}%",
        "",
        "FAILURE REASONS",
        "---------------",
    ]

    for reason, count in failure_reasons.items():
        report_lines.append(
            f"{reason:<22}: {count}"
        )

    report_lines.extend([
        "",
        "FAILED TESTS",
        "---------------",
    ])

    report_lines.extend(failed_tests)

    report_content = "\n".join(report_lines)

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report_content)

    print(report_content)

    return report_content
