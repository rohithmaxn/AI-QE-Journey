
import os

from openai import APIError, OpenAI, RateLimitError


DEFAULT_MODEL = os.getenv(
    "OPENAI_JUDGE_MODEL",
    "gpt-5",
)


class JudgeServiceError(RuntimeError):
    """Raised when the LLM judge service cannot complete an evaluation."""


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {
            "type": "boolean",
        },
        "groundedness": {
            "type": ["boolean", "null"],
        },
        "relevance": {
            "type": "boolean",
        },
        "instruction_following": {
            "type": "boolean",
        },
        "reason": {
            "type": "string",
        },
    },
    "required": [
        "correctness",
        "groundedness",
        "relevance",
        "instruction_following",
        "reason",
    ],
    "additionalProperties": False,
}


class OpenAIJudge:
    """Adapter around the OpenAI Responses API."""

    def __init__(self, model=None, client=None):
        self.model = model or DEFAULT_MODEL
        self.client = client if client is not None else OpenAI()

    def __call__(self, prompt):
        """
        Send the prompt and return the structured JSON response.

        Service exceptions are translated to JudgeServiceError.
        """

        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "llm_evaluation",
                        "strict": True,
                        "schema": JUDGE_SCHEMA,
                    }
                },
            )

            return response.output_text

        except RateLimitError as exc:
            error_code = getattr(exc, "code", None)

            if error_code == "insufficient_quota":
                raise JudgeServiceError(
                    "OpenAI judge unavailable: API quota is exhausted. "
                    "Add API credits and try again."
                ) from exc

            raise JudgeServiceError(
                "OpenAI judge unavailable: rate limit exceeded."
            ) from exc

        except APIError as exc:
            raise JudgeServiceError(
                "OpenAI judge request failed. Check API credentials, "
                "model access, connectivity, and request settings."
            ) from exc
