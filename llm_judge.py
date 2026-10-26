
import json


REQUIRED_JUDGE_FIELDS = {
    "correctness",
    "groundedness",
    "relevance",
    "instruction_following",
    "reason",
}


def build_judge_prompt(test_case):
    """
    Build the prompt used to evaluate an LLM response.
    """

    return f"""
You are an AI Quality Engineer evaluating an LLM response.

Evaluate the response using these dimensions:

1. correctness
2. groundedness
3. relevance
4. instruction_following

Groundedness rules:
- Evaluate groundedness against the supplied CONTEXT.
- If CONTEXT is empty or contains only whitespace,
  return null for groundedness because there is no supplied
  context against which to assess grounding.
- If CONTEXT is provided, return true when the response is
  supported by the context.
- If the response makes unsupported claims or contradicts
  the supplied context, return false for groundedness.
- Use null only for groundedness. The other three dimensions
  must always be Boolean values: true or false.

Return ONLY valid JSON using this structure:

{{
    "correctness": true,
    "groundedness": true,
    "relevance": true,
    "instruction_following": true,
    "reason": "Brief explanation"
}}

When CONTEXT is empty, use null instead of true or false
for groundedness.

QUESTION:
{test_case["question"]}

CONTEXT:
{test_case["context"]}

EXPECTED ANSWER:
{test_case["expected_answer"]}

ACTUAL RESPONSE:
{test_case["actual_response"]}
""".strip()



def parse_judge_result(raw_result):
    """
    Parse and validate a structured judge response.

    Raises:
        ValueError: If the input is not valid JSON or the response
                    does not conform to the expected schema.
    """

    if not isinstance(raw_result, str):
        raise ValueError("Judge response must be a JSON string")

    try:
        result = json.loads(raw_result)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Judge response is not valid JSON"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError("Judge response must be a JSON object")

    missing_fields = REQUIRED_JUDGE_FIELDS - result.keys()

    if missing_fields:
        raise ValueError(
            f"Judge response missing fields: "
            f"{', '.join(sorted(missing_fields))}"
        )

    unexpected_fields = set(result) - REQUIRED_JUDGE_FIELDS

    if unexpected_fields:
        raise ValueError(
            f"Judge response contains unexpected fields: "
            f"{', '.join(sorted(unexpected_fields))}"
        )

    boolean_fields = {
        "correctness",
        "relevance",
        "instruction_following",
    }

    for field in boolean_fields:
        if not isinstance(result[field], bool):
            raise ValueError(
                f"Judge field '{field}' must be boolean"
            )

    groundedness = result["groundedness"]

    if (
        groundedness is not None
        and not isinstance(groundedness, bool)
    ):
        raise ValueError(
            "Judge field 'groundedness' must be boolean or null"
        )

    if not isinstance(result["reason"], str):
        raise ValueError(
            "Judge field 'reason' must be a string"
        )

    return result



def evaluate_with_judge(test_case, judge):
    """
    Evaluate a test case using an injected judge.

    Args:
        test_case: LLM evaluation test case.
        judge: Callable that accepts a prompt and returns
               the judge's raw JSON response.

    Returns:
        tuple: (status, reason)
    """

    prompt = build_judge_prompt(test_case)

    raw_result = judge(prompt)

    result = parse_judge_result(raw_result)

    quality_checks = [
        result["correctness"],
        result["relevance"],
        result["instruction_following"],
    ]

    # Groundedness is excluded from the decision when it is N/A.
    if result["groundedness"] is not None:
        quality_checks.append(result["groundedness"])

    if all(quality_checks):
        return "PASS", result["reason"]

    return "FAIL", result["reason"]
