
def calculate_failure_metrics(
    reference_statuses: list[str],
    predicted_statuses: list[str],
) -> dict:
    """Measure how accurately a judge identifies failed responses.

    FAIL is the positive class.

    Undefined metrics return None:
    - Precision: no failures were predicted.
    - Recall: no actual failures exist.
    - F1-score: neither actual nor predicted failures exist.
    """

    if not isinstance(reference_statuses, list) or not isinstance(
        predicted_statuses, list
    ):
        raise ValueError("Reference and predicted statuses must be lists.")

    if len(reference_statuses) != len(predicted_statuses):
        raise ValueError(
            "Reference and predicted lists must have equal lengths."
        )

    valid_statuses = {"PASS", "FAIL"}

    if not all(
        isinstance(status, str) and status in valid_statuses
        for status in reference_statuses + predicted_statuses
    ):
        raise ValueError("Statuses must be PASS or FAIL.")

    true_positives = sum(
        ref == "FAIL" and pred == "FAIL"
        for ref, pred in zip(reference_statuses, predicted_statuses)
    )

    false_positives = sum(
        ref == "PASS" and pred == "FAIL"
        for ref, pred in zip(reference_statuses, predicted_statuses)
    )

    false_negatives = sum(
        ref == "FAIL" and pred == "PASS"
        for ref, pred in zip(reference_statuses, predicted_statuses)
    )

    true_negatives = sum(
        ref == "PASS" and pred == "PASS"
        for ref, pred in zip(reference_statuses, predicted_statuses)
    )

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives

    f1_denominator = (
        2 * true_positives
        + false_positives
        + false_negatives
    )

    precision = (
        round(true_positives / precision_denominator, 4)
        if precision_denominator > 0
        else None
    )

    recall = (
        round(true_positives / recall_denominator, 4)
        if recall_denominator > 0
        else None
    )

    f1_score = (
        round(2 * true_positives / f1_denominator, 4)
        if f1_denominator > 0
        else None
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def calculate_metrics_from_records(
    reference_labels: list[dict],
    records: list[dict],
) -> dict:
    """Compare judge predictions against reference labels by test ID."""

    if not isinstance(reference_labels, list) or not isinstance(
        records, list
    ):
        raise ValueError("Reference labels and records must be lists.")

    reference_by_id = {}

    for label in reference_labels:
        if not isinstance(label, dict):
            raise ValueError("Each reference label must be a dictionary.")

        test_id = label.get("test_id")

        if not isinstance(test_id, str) or not test_id.strip():
            raise ValueError(
                "Each reference label must have a valid test_id."
            )

        if test_id in reference_by_id:
            raise ValueError(f"Duplicate reference test_id: {test_id}")

        if "reference_status" not in label:
            raise ValueError(
                f"Missing reference_status for test_id: {test_id}"
            )

        reference_by_id[test_id] = label["reference_status"]

    records_by_id = {}

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each evaluation record must be a dictionary.")

        test_id = record.get("test_id")

        if not isinstance(test_id, str) or not test_id.strip():
            raise ValueError(
                "Each evaluation record must have a valid test_id."
            )

        if test_id in records_by_id:
            raise ValueError(f"Duplicate record test_id: {test_id}")

        if "status" not in record:
            raise ValueError(f"Missing status for test_id: {test_id}")

        records_by_id[test_id] = record["status"]

    reference_ids = set(reference_by_id)
    record_ids = set(records_by_id)

    if reference_ids != record_ids:
        missing = sorted(reference_ids - record_ids)
        unexpected = sorted(record_ids - reference_ids)

        raise ValueError(
            f"Test IDs do not match. "
            f"Missing records: {missing}; "
            f"unexpected records: {unexpected}"
        )

    ordered_ids = list(reference_by_id)

    return calculate_failure_metrics(
        [reference_by_id[test_id] for test_id in ordered_ids],
        [records_by_id[test_id] for test_id in ordered_ids],
    )


QUALITY_DIMENSIONS = (
    "correctness",
    "groundedness",
    "relevance",
    "instruction_following",
)


def calculate_dimension_failure_metrics(
    reference_labels: list[dict],
    records: list[dict],
) -> dict:
    """Calculate failure-detection metrics for each quality dimension.

    Groundedness values marked None in the reference labels are
    excluded from that dimension's metrics.
    """

    if not isinstance(reference_labels, list) or not isinstance(
        records, list
    ):
        raise ValueError("Reference labels and records must be lists.")

    reference_by_id = {}

    for label in reference_labels:
        if not isinstance(label, dict):
            raise ValueError("Each reference label must be a dictionary.")

        test_id = label.get("test_id")

        if not isinstance(test_id, str) or not test_id.strip():
            raise ValueError(
                "Each reference label must have a valid test_id."
            )

        if test_id in reference_by_id:
            raise ValueError(f"Duplicate reference test_id: {test_id}")

        reference_by_id[test_id] = label

    records_by_id = {}

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each evaluation record must be a dictionary.")

        test_id = record.get("test_id")

        if not isinstance(test_id, str) or not test_id.strip():
            raise ValueError(
                "Each evaluation record must have a valid test_id."
            )

        if test_id in records_by_id:
            raise ValueError(f"Duplicate record test_id: {test_id}")

        records_by_id[test_id] = record

    if set(reference_by_id) != set(records_by_id):
        missing = sorted(set(reference_by_id) - set(records_by_id))
        unexpected = sorted(set(records_by_id) - set(reference_by_id))

        raise ValueError(
            f"Reference and evaluation record IDs must match. "
            f"Missing records: {missing}; unexpected records: {unexpected}"
        )

    dimension_results = {}

    for dimension in QUALITY_DIMENSIONS:
        reference_statuses = []
        predicted_statuses = []
        not_applicable = 0

        for test_id, label in reference_by_id.items():
            dimensions = label.get("reference_dimensions", {})

            if not isinstance(dimensions, dict):
                raise ValueError(
                    f"Reference dimensions must be a dictionary for {test_id}"
                )

            if dimension not in dimensions:
                raise ValueError(
                    f"Missing reference dimension '{dimension}' "
                    f"for {test_id}"
                )

            expected = dimensions[dimension]

            if expected is None:
                if dimension != "groundedness":
                    raise ValueError(
                        f"Only groundedness may be None: {test_id}"
                    )

                not_applicable += 1
                continue

            if not isinstance(expected, bool):
                raise ValueError(
                    f"Reference dimension '{dimension}' "
                    f"must be boolean for {test_id}"
                )

            actual = records_by_id[test_id].get(dimension)

            if not isinstance(actual, bool):
                raise ValueError(
                    f"Evaluation dimension '{dimension}' "
                    f"must be boolean for {test_id}"
                )

            # A failed quality dimension is the positive class.
            reference_statuses.append(
                "PASS" if expected else "FAIL"
            )

            predicted_statuses.append(
                "PASS" if actual else "FAIL"
            )

        # Calculate metrics independently for each dimension.
        metrics = calculate_failure_metrics(
            reference_statuses,
            predicted_statuses,
        )

        dimension_results[dimension] = {
            **metrics,
            "applicable": len(reference_statuses),
            "not_applicable": not_applicable,
        }

    return dimension_results
