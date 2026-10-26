from config import AI_SCORE_THRESHOLD
def evaluate_test(test_case):
    """Evaluate a single test case based on its score and hallucination status."""
    if "score" not in test_case:
        return "FAIL", "Missing score"

    if "hallucination" not in test_case:
        return "FAIL", "Missing hallucination field"

    if isinstance(test_case["score"], bool):
        return "FAIL", "Invalid score type"

    if not isinstance(test_case["score"], (int, float)):
        return "FAIL", "Invalid score type"

    if not isinstance(test_case["hallucination"], bool):
        return "FAIL", "Invalid hallucination type"

    if test_case["score"] < AI_SCORE_THRESHOLD and test_case["hallucination"]:
        return "FAIL", "Score below threshold + Hallucination detected"

    if test_case["score"] < AI_SCORE_THRESHOLD:
        return "FAIL", "Score below threshold"

    if test_case["hallucination"]:
        return "FAIL", "Hallucination detected"

    return "PASS", "Quality threshold met"
