import re


def _normalize_text(text):
    """Normalize text for deterministic comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _calculate_word_overlap(expected, actual):
    """
    Calculate the proportion of expected words
    that appear in the actual response.
    """

    expected_words = set(_normalize_text(expected))
    actual_words = set(_normalize_text(actual))

    if not expected_words:
        return 0.0

    return len(expected_words & actual_words) / len(expected_words)

def _evaluate_ambiguity_handling(test_case):
    """
    Evaluate whether the response handles an ambiguous request safely.

    A good response should:
    1. Ask for clarification.
    2. Avoid claiming that an action was already performed.
    """

    actual_response = test_case["actual_response"].strip().lower()

    asks_for_clarification = "?" in actual_response

    claims_reset_action = bool(
        re.search(
            r"\b(i have|i've)\s+reset\b"
            r"|\breset your\b",
            actual_response
        )
    )

    return (
        asks_for_clarification
        and not claims_reset_action
    )

def _evaluate_correctness(test_case):
    """
    Determine whether the actual response is consistent
    with the expected answer.
    """

    category = test_case["category"]

    if category == "ambiguity_handling":
        return _evaluate_ambiguity_handling(test_case)

    expected_answer = test_case["expected_answer"]
    actual_response = test_case["actual_response"]

    overlap = _calculate_word_overlap(
        expected_answer,
        actual_response
    )

    return overlap >= 0.50


def _evaluate_groundedness(test_case):
    """
    Determine whether the response is grounded in the
    expected answer and supplied context.
    """

    category = test_case["category"]

    if category == "ambiguity_handling":
        return _evaluate_ambiguity_handling(test_case)

    expected_answer = test_case["expected_answer"]
    actual_response = test_case["actual_response"]

    overlap = _calculate_word_overlap(
        expected_answer,
        actual_response
    )

    return overlap >= 0.50


def _evaluate_relevance(test_case):
    """
    Determine whether the response is relevant to the question.
    """

    category = test_case["category"]
    actual_response = test_case["actual_response"].strip()

    if not actual_response:
        return False

    # Ambiguous requests require a clarification response.
    if category == "ambiguity_handling":
        return "?" in actual_response

    return True


def _evaluate_instruction_following(test_case):
    """
    Evaluate explicit instructions in the user's question.
    """

    question = test_case["question"].lower()
    actual_response = test_case["actual_response"].strip()

    # ---------------------------------------------------------
    # Exactly three items
    # ---------------------------------------------------------

    if "exactly three" in question:
        numbered_items = [
            line
            for line in actual_response.splitlines()
            if re.match(r"^\s*\d+[\.\)]", line)
        ]

        return len(numbered_items) == 3

    # ---------------------------------------------------------
    # One sentence
    # ---------------------------------------------------------

    if "one sentence" in question:
        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"[.!?]+",
                actual_response
            )
            if sentence.strip()
        ]

        return len(sentences) == 1

    # ---------------------------------------------------------
    # Ambiguous request
    # ---------------------------------------------------------

    if test_case["category"] == "ambiguity_handling":
        return "?" in actual_response

    return True


def evaluate_llm_response(test_case):
    """
    Evaluate a single LLM response across all quality dimensions.

    Returns:
        tuple:
            ("PASS", reason)
            or
            ("FAIL", reason)
    """

    required_fields = {
        "test_id",
        "category",
        "question",
        "context",
        "expected_answer",
        "actual_response",
        "evaluation_criteria",
    }

    missing_fields = required_fields - test_case.keys()

    if missing_fields:
        return (
            "FAIL",
            f"Missing fields: {', '.join(sorted(missing_fields))}"
        )

    category = test_case["category"]

    correctness = _evaluate_correctness(test_case)
    groundedness = _evaluate_groundedness(test_case)
    relevance = _evaluate_relevance(test_case)
    instruction_following = _evaluate_instruction_following(
        test_case
    )

    # ---------------------------------------------------------
    # Determine overall result
    # ---------------------------------------------------------

    if (
        correctness
        and groundedness
        and relevance
        and instruction_following
    ):
        return "PASS", "LLM response quality criteria met"

    # ---------------------------------------------------------
    # Failure reasons
    # ---------------------------------------------------------

    if category == "factual_correctness":
        return "FAIL", "Incorrect factual answer"

    if category == "groundedness":
        return "FAIL", "Response contradicts the supplied context"

    if category == "instruction_following":
        return (
            "FAIL",
            "Response does not follow the requested instruction"
        )

    if category == "ambiguity_handling":
        return (
            "FAIL",
            "Response assumes missing information and claims "
            "an action without sufficient context"
        )

    return "FAIL", "LLM response quality criteria not met"
