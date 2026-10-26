
QUALITY_DIMENSIONS = (
    "correctness",
    "groundedness",
    "relevance",
    "instruction_following",
)


def calculate_judge_agreement(reference_labels, records):
    """
    Compare judge results against reference labels.

    Reference dimensions marked None are not applicable and
    are excluded from that dimension's agreement denominator.
    """

    reference_ids = [
        label["test_id"] for label in reference_labels
    ]
    record_ids = [
        record["test_id"] for record in records
    ]

    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("Reference-label IDs must be unique")

    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Evaluation record IDs must be unique")

    unknown_ids = set(record_ids) - set(reference_ids)

    if unknown_ids:
        raise ValueError(
            f"Unexpected evaluation record IDs: {sorted(unknown_ids)}"
        )

    records_by_id = {
        record["test_id"]: record
        for record in records
    }

    agreements = 0
    mismatches = []

    dimension_metrics = {
        dimension: {
            "total": 0,
            "agreements": 0,
            "disagreements": 0,
            "agreement_rate": 0.0,
            "mismatches": [],
        }
        for dimension in QUALITY_DIMENSIONS
    }

    for label in reference_labels:
        test_id = label["test_id"]
        reference_status = label["reference_status"]
        record = records_by_id.get(test_id)

        actual_status = (
            record["status"] if record is not None else None
        )

        # Overall PASS/FAIL agreement.
        if actual_status == reference_status:
            agreements += 1
        else:
            mismatches.append({
                "test_id": test_id,
                "expected_status": reference_status,
                "actual_status": actual_status,
            })

        reference_dimensions = label.get(
            "reference_dimensions", {}
        )

        for dimension in QUALITY_DIMENSIONS:
            if dimension not in reference_dimensions:
                continue

            expected_value = reference_dimensions[dimension]

            if expected_value is None:
                if dimension != "groundedness":
                    raise ValueError(
                        f"Only groundedness may be None "
                        f"in reference labels: {test_id}"
                    )

                # Not applicable: exclude from this dimension's totals.
                continue

            if not isinstance(expected_value, bool):
                raise ValueError(
                    f"Reference dimension '{dimension}' "
                    f"must be boolean or None for {test_id}"
                )

            actual_value = (
                record.get(dimension)
                if record is not None
                else None
            )

            if (
                actual_value is not None
                and not isinstance(actual_value, bool)
            ):
                raise ValueError(
                    f"Evaluation dimension '{dimension}' "
                    f"must be boolean or None for {test_id}"
                )

            stats = dimension_metrics[dimension]
            stats["total"] += 1

            if actual_value is expected_value:
                stats["agreements"] += 1
            else:
                stats["disagreements"] += 1
                stats["mismatches"].append({
                    "test_id": test_id,
                    "expected": expected_value,
                    "actual": actual_value,
                })

    total = len(reference_labels)

    agreement_rate = (
        agreements / total * 100
        if total
        else 0.0
    )

    for stats in dimension_metrics.values():
        dimension_total = stats["total"]

        stats["agreement_rate"] = round(
            stats["agreements"] / dimension_total * 100
            if dimension_total
            else 0.0,
            2,
        )

    return {
        "total": total,
        "agreements": agreements,
        "disagreements": len(mismatches),
        "agreement_rate": round(agreement_rate, 2),
        "mismatches": mismatches,
        "dimension_agreement": dimension_metrics,
    }
