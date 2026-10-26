import logging
import time

import requests

from config import (
    API_MAX_RETRIES,
    API_RETRY_DELAY,
    API_RETRY_METHODS,
    API_RETRY_STATUS_CODES,
)


logger = logging.getLogger(__name__)


class APIClient:

    def __init__(self, base_url):
        """Initialize the API client with the base URL and default headers."""
        self.base_url = base_url.rstrip("/")
        self.default_headers = {
            "Content-Type": "application/json"
        }

    def _should_retry(self, method, status_code=None, exception=None):
        """Determine if the request should be retried based on the method, status code, or exception."""
        method = method.upper()

        if method not in API_RETRY_METHODS:
            return False

        if status_code in API_RETRY_STATUS_CODES:
            return True

        if isinstance(
            exception,
            (requests.Timeout, requests.ConnectionError)
        ):
            return True

        return False

    def _request(self, method, endpoint, data=None, headers=None):
        """Send an HTTP request to the specified endpoint with retries for certain conditions."""
        url = f"{self.base_url}{endpoint}"

        request_headers = self.default_headers.copy()

        if headers:
            request_headers.update(headers)

        method = method.upper()
        attempts = 0

        while True:
            """Attempt to send the request, retrying on certain conditions."""
            try:
                logger.info(
                    "Sending %s request to %s",
                    method,
                    url
                )

                response = requests.request(
                    method=method,
                    url=url,
                    json=data,
                    headers=request_headers,
                    timeout=10
                )

                logger.info(
                    "%s %s → %s",
                    method,
                    url,
                    response.status_code
                )

                # Retry only configured HTTP status codes
                # for configured HTTP methods.
                if self._should_retry(
                    method,
                    status_code=response.status_code
                ):
                    attempts += 1

                    if attempts > API_MAX_RETRIES:
                        raise RuntimeError(
                            f"{method} request failed with status "
                            f"{response.status_code} after "
                            f"{API_MAX_RETRIES} retries: {url}"
                        )

                    logger.warning(
                        "Retrying %s %s (attempt %s)",
                        method,
                        url,
                        attempts
                    )

                    time.sleep(API_RETRY_DELAY)
                    continue

                response.raise_for_status()
                return response

            except requests.Timeout as e:
                if not self._should_retry(
                    method,
                    exception=e
                ):
                    raise RuntimeError(
                        f"{method} request timed out: {url}"
                    ) from e

                attempts += 1

                if attempts > API_MAX_RETRIES:
                    raise RuntimeError(
                        f"{method} request timed out after "
                        f"{API_MAX_RETRIES} retries: {url}"
                    ) from e

                logger.warning(
                    "Retrying %s %s (attempt %s)",
                    method,
                    url,
                    attempts
                )

                time.sleep(API_RETRY_DELAY)

            except requests.ConnectionError as e:
                if not self._should_retry(
                    method,
                    exception=e
                ):
                    raise RuntimeError(
                        f"{method} request connection failed: {url}"
                    ) from e

                attempts += 1

                if attempts > API_MAX_RETRIES:
                    raise RuntimeError(
                        f"{method} request connection failed after "
                        f"{API_MAX_RETRIES} retries: {url}"
                    ) from e

                logger.warning(
                    "Retrying %s %s (attempt %s)",
                    method,
                    url,
                    attempts
                )

                time.sleep(API_RETRY_DELAY)

            except requests.RequestException as e:
                raise RuntimeError(
                    f"{method} request failed for {url}: {e}"
                ) from e

    def get(self, endpoint, headers=None):
        """Send a GET request to the specified endpoint with optional headers."""
        return self._request(
            "GET",
            endpoint,
            headers=headers
        )

    def post(self, endpoint, data=None, headers=None):
        """Send a POST request to the specified endpoint with optional data and headers."""
        return self._request(
            "POST",
            endpoint,
            data=data,
            headers=headers
        )

    def put(self, endpoint, data=None, headers=None):
        """Send a PUT request to the specified endpoint with optional data and headers."""
        return self._request(
            "PUT",
            endpoint,
            data=data,
            headers=headers
        )

    def delete(self, endpoint, headers=None):
        """Send a DELETE request to the specified endpoint."""
        return self._request(
            "DELETE",
            endpoint,
            headers=headers
        )

    def get_todo(self, todo_id):
        """Fetch a specific todo item by its ID."""
        return self.get(f"/todos/{todo_id}")

    def parse_json(self, response):
        """Parse the JSON content from the response, raising an error if invalid."""
        try:
            return response.json()
        except ValueError as e:
            raise RuntimeError(
                "Response does not contain valid JSON"
            ) from e
