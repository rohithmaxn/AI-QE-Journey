# AI Quality Engineering (AI-QE) Framework

A Python-based framework for testing APIs and evaluating the quality of Large Language Model (LLM) responses. The project combines traditional test automation with structured AI response evaluation, reference-label comparison, and automated reporting.

## Project Overview

AI-generated responses can vary between runs, so validating them with exact text comparisons is often insufficient. This project explores a more structured approach: evaluating responses against defined quality criteria and measuring results through repeatable automated tests.

The framework supports offline development using a deterministic MOCK judge and includes an OpenAI judge adapter for future live evaluation.

## Key Features

### API Test Automation
- Reusable REST API client.
- HTTP request and response validation.
- Response structure, data type, and expected-value checks.
- Error handling, logging, timeout handling, and retry logic.

### LLM Response Evaluation
- Configurable MOCK and OpenAI judge providers.
- Structured JSON judge responses.
- Evaluation across four quality dimensions:
  - Correctness
  - Groundedness
  - Relevance
  - Instruction following
- Groundedness marked as not applicable when no supporting context is supplied.
- Reference-label comparison with overall and per-dimension agreement metrics.

### Quality Metrics and Reporting
- Pass/fail summaries and pass rates.
- Confusion-matrix counts: true positives, false positives, false negatives, and true negatives.
- Precision, recall, and F1-score for detecting failed responses.
- Per-dimension failure-detection metrics.
- Explicit handling of undefined metrics.
- Machine-readable JSON reports and human-readable text reports.

### Automated Testing
- Python and pytest-based regression suite.
- Tests for malformed JSON, invalid fields, incorrect data types, duplicate IDs, and missing records.
- Offline tests for OpenAI adapter behavior and service-error handling.
- Regression tests for N/A groundedness and report preservation after a simulated service failure.

## Technology Stack

- Python
- pytest
- requests
- OpenAI Python SDK
- JSON
- REST API testing
- Automated evaluation and reporting

## Project Structure

```text
AI-QE-Journey/
├── clients/
│   └── api_client.py
├── validators/
│   └── api_validators.py
├── test_data/
│   ├── ai_test_cases.json
│   ├── llm_evaluation_cases.json
│   └── llm_reference_labels.json
├── config.py
├── evaluator.py
├── judge_factory.py
├── judge_agreement.py
├── judge_quality_metrics.py
├── llm_evaluator.py
├── llm_judge.py
├── llm_judge_execution.py
├── llm_judge_reporting.py
├── llm_judge_report.py
├── llm_judge_text_report.py
├── openai_judge.py
├── requirements.txt
├── .gitignore
└── test_*.py
```

The test files exercise the framework's API validation, judge integration, metric calculations, reference-label agreement, and reporting behavior.

## Setup

### Prerequisites

- Python 3.14 or a compatible Python version.
- pip.

### 1. Clone the repository

After publishing the project to GitHub, clone it using:

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-QE-Journey
```

Replace the placeholder with your repository URL.

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## Run the Tests

Execute the complete regression suite:

```powershell
python -m pytest -q
```

**Locally verified status:** 143 tests passed in the most recent full regression run.

Test counts may change as the framework develops. Run the tests in your own environment to verify its current status.

## Run an Offline LLM Evaluation

The MOCK provider allows the evaluation pipeline to be exercised without making live API calls.

From the project directory, run:

```powershell
python -c "from llm_judge_execution import execute_llm_judge_report; r = execute_llm_judge_report(provider='MOCK'); print(r['metrics'])"
```

The current dataset contains eight evaluation cases. The expected overall result is four passing and four failing cases.

The execution generates JSON and text reports:

- `llm_judge_report.json`
- `llm_judge_report.txt`

Generated reports are excluded from version control by default. They can be inspected locally after an evaluation run.

## Evaluation Methodology

The framework evaluates responses using predefined criteria and compares judge predictions against reference labels.

Groundedness is treated as not applicable when no supporting context is supplied. Such cases are excluded from groundedness-specific metrics.

Precision, recall, and F1-score measure failure detection, with FAIL treated as the positive class. When a metric cannot be calculated meaningfully, the framework represents it as undefined rather than reporting a misleading zero.

## Current Validation Status and Limitations

The current framework has been tested offline using pytest and a deterministic MOCK judge.

The MOCK judge uses predefined evaluation criteria to produce repeatable results. Its agreement and metric scores validate the evaluation pipeline, not real-world LLM judge accuracy.

The OpenAI judge adapter is implemented and its error handling has been tested with simulated responses. A successful live-model evaluation against independently reviewed reference labels has not yet been validated.

Live evaluation requires valid API credentials, appropriate model access, and available API quota. No API key should be committed to the repository.

## Future Improvements

- Validate real LLM judge decisions against independently reviewed reference labels.
- Expand the evaluation dataset with additional edge cases.
- Add GitHub Actions for continuous integration.
- Compare judge behavior across model versions and repeated runs.
- Expand evaluation coverage for hallucinations, ambiguity, and groundedness.

## Purpose

This project is part of a practical learning journey toward AI Quality Engineering. It demonstrates the development of reusable automation, structured LLM evaluation, quality metrics, test reporting, and regression safeguards using Python and pytest.

[![AI-QE Tests](https://github.com/rohithmaxn/AI-QE-Journey/actions/workflows/ci.yml/badge.svg)](https://github.com/rohithmaxn/AI-QE-Journey/actions/workflows/ci.yml)