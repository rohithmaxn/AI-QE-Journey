from config import JUDGE_PROVIDER
from judge_factory import get_judge
from llm_judge import (
    build_judge_prompt,
    parse_judge_result,
)
from llm_judge_report import generate_judge_report
from llm_judge_reporting import build_judge_record
from llm_test_data_loader import load_llm_test_cases
from judge_agreement import calculate_judge_agreement
from llm_reference_loader import load_reference_labels
from judge_quality_metrics import (
    calculate_metrics_from_records,
    calculate_dimension_failure_metrics,
)


def execute_llm_judge_report(provider=None):
    """
    Execute all LLM evaluation cases and generate
    a complete judge report.

    Returns:
        dict: Complete report containing metrics and records.
    """

    selected_provider = (
        provider or JUDGE_PROVIDER
    ).upper()

    test_cases = load_llm_test_cases()
    records = []

    for test_case in test_cases:
        judge = get_judge(
            selected_provider,
            test_case=test_case
        )

        prompt = build_judge_prompt(test_case)

        raw_result = judge(prompt)

        judge_result = parse_judge_result(
            raw_result
        )

        model = getattr(
            judge,
            "model",
            selected_provider
        )

        record = build_judge_record(
            test_case=test_case,
            judge_result=judge_result,
            provider=selected_provider,
            model=model
        )

        records.append(record)

    reference_labels = load_reference_labels()

    agreement = calculate_judge_agreement(
        reference_labels,
        records
    )

    quality_metrics = calculate_metrics_from_records(
        reference_labels,
        records,
    )

    dimension_metrics = calculate_dimension_failure_metrics(
    reference_labels,
    records,
    )

    return generate_judge_report(
    records,
    agreement=agreement,
    quality_metrics=quality_metrics,
    dimension_metrics=dimension_metrics,
)
