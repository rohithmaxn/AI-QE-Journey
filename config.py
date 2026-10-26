import os
#handles environment configuration

API_MAX_RETRIES = 3
API_RETRY_DELAY = 1
API_RETRY_METHODS = ("GET",)
API_RETRY_STATUS_CODES = (503,)

ENVIRONMENTS = {
    "DEV": {
        "API_BASE_URL": "https://jsonplaceholder.typicode.com",
        "AI_SCORE_THRESHOLD": 0.75
    },

    "QA": {
        "API_BASE_URL": "https://jsonplaceholder.typicode.com",
        "AI_SCORE_THRESHOLD": 0.80
    },

    "STAGING": {
        "API_BASE_URL": "https://jsonplaceholder.typicode.com",
        "AI_SCORE_THRESHOLD": 0.85
    }
}

ENVIRONMENT = os.getenv("ENVIRONMENT", "QA")
API_BASE_URL = ENVIRONMENTS[ENVIRONMENT]["API_BASE_URL"]
AI_SCORE_THRESHOLD = ENVIRONMENTS[ENVIRONMENT]["AI_SCORE_THRESHOLD"]

JUDGE_PROVIDER = os.getenv(
    "JUDGE_PROVIDER",
    "MOCK"
).upper()
