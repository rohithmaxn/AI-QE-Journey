# verifies the API client
from clients.api_client import APIClient
from config import API_BASE_URL
import pytest
import requests


def test_get_todo_by_id(api_client):
    """Test the get_todo method of the APIClient."""
    response = api_client.get("/todos/2")

    assert response.status_code == 200

    data = response.json()
    assert data["id"] == 2

def test_get_generic_endpoint(api_client):
    """Test the generic GET method of the APIClient."""
    response = api_client.get("/todos/2")

    assert response.status_code == 200

    data = response.json()
    assert data["id"] == 2

def test_get_with_custom_header(api_client):
    """Test the GET method with a custom header."""
    client = APIClient(API_BASE_URL)

    response = client.get(
        "/todos/2",
        headers={"X-Test-Header": "AI-QE"}
    )

    assert response.status_code == 200

def test_post_generic_endpoint():
    """Test the generic POST method of the APIClient."""
    client = APIClient(API_BASE_URL)

    payload = {
        "title": "AI-QE Test",
        "completed": False,
        "userId": 1
    }

    response = client.post("/todos", data=payload)

    assert response.status_code == 201

    data = response.json()

    assert "id" in data
    assert data["title"] == payload["title"]
    assert data["completed"] is False
    assert data["userId"] == 1

def test_put_generic_endpoint():
    """Test the generic PUT method of the APIClient."""
    client = APIClient(API_BASE_URL)

    payload = {
        "title": "Updated AI-QE Test",
        "completed": True,
        "userId": 1
    }

    response = client.put(
        "/todos/1",
        data=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == 1
    assert data["title"] == "Updated AI-QE Test"
    assert data["completed"] is True
    assert data["userId"] == 1

def test_delete_generic_endpoint():
    """Test the generic DELETE method of the APIClient."""
    client = APIClient(API_BASE_URL)

    response = client.delete("/todos/1")

    assert response.status_code in (200, 204)

def test_parse_json():
    """Test the JSON parsing method of the APIClient."""
    client = APIClient(API_BASE_URL)

    response = client.get("/todos/2")
    data = client.parse_json(response)

    assert isinstance(data, dict)
    assert data["id"] == 2
    assert data["title"] == "quis ut nam facilis et officia qui"

def test_parse_invalid_json():
    """Test the JSON parsing method with invalid JSON."""
    client = APIClient(API_BASE_URL)

    response = client.get("/todos/2")

    class FakeResponse:
        def json(self):
            raise ValueError("Invalid JSON")

    fake_response = FakeResponse()

    with pytest.raises(
        RuntimeError,
        match="Response does not contain valid JSON"
    ):
        client.parse_json(fake_response)

def test_retry_on_timeout(monkeypatch):
    """Test that the APIClient retries on timeout for GET requests."""
    client = APIClient(API_BASE_URL)

    attempts = 0

    def fake_request(*args, **kwargs):
        """Simulate a timeout for the first two attempts and succeed on the third."""
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            raise requests.Timeout("Simulated timeout")

        response = requests.Response()
        response.status_code = 200
        return response

    monkeypatch.setattr(
        requests,
        "request",
        fake_request
    )

    monkeypatch.setattr(
        "clients.api_client.time.sleep",
        lambda _: None
    )

    response = client.get("/todos/2")

    assert attempts == 3
    assert isinstance(response, requests.Response)

def test_retry_exhaustion(monkeypatch):
    """Test that the APIClient raises an error after exhausting retries on timeout."""
    client = APIClient(API_BASE_URL)

    attempts = 0

    def fake_request(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise requests.Timeout("Simulated timeout")

    monkeypatch.setattr(
        requests,
        "request",
        fake_request
    )

    monkeypatch.setattr(
        "clients.api_client.time.sleep",
        lambda _: None
    )

    with pytest.raises(
        RuntimeError,
        match="request timed out"
    ):
        client.get("/todos/2")

    assert attempts == 4

def test_post_timeout_is_not_retried(monkeypatch):
    """Test that the APIClient does not retry POST requests on timeout."""
    client = APIClient(API_BASE_URL)

    attempts = 0

    def fake_request(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise requests.Timeout("Simulated timeout")

    monkeypatch.setattr(
        requests,
        "request",
        fake_request
    )

    with pytest.raises(
        RuntimeError,
        match="POST request timed out"
    ):
        client.post(
            "/todos",
            data={"title": "AI-QE Test"}
        )

    assert attempts == 1

def test_retry_on_connection_error(monkeypatch):
    """Test that the APIClient retries on connection error for GET requests."""
    client = APIClient(API_BASE_URL)

    attempts = 0

    def fake_request(*args, **kwargs):
        """Simulate a connection error for the first two attempts and succeed on the third."""
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            raise requests.ConnectionError("Simulated connection failure")

        response = requests.Response()
        response.status_code = 200
        return response

    monkeypatch.setattr(
        requests,
        "request",
        fake_request
    )

    monkeypatch.setattr(
        "clients.api_client.time.sleep",
        lambda _: None
    )

    response = client.get("/todos/2")

    assert attempts == 3
    assert isinstance(response, requests.Response)

def test_404_is_not_retried(monkeypatch):
    """Test that the APIClient does not retry on 404 Not Found responses."""
    client = APIClient(API_BASE_URL)

    attempts = 0

    def fake_request(*args, **kwargs):
        """Simulate a 404 Not Found response."""
        nonlocal attempts
        attempts += 1

        response = requests.Response()
        response.status_code = 404
        return response

    monkeypatch.setattr(
        requests,
        "request",
        fake_request
    )

    with pytest.raises(
        RuntimeError,
        match="GET request failed"
    ):
        client.get("/todos/999999")

    assert attempts == 1

def test_503_is_retried(monkeypatch):
    """Test that the APIClient retries on 503 Service Unavailable responses."""
    client = APIClient(API_BASE_URL)

    attempts = 0

    def fake_request(*args, **kwargs):
        """Simulate a 503 Service Unavailable response for the first two attempts and succeed on the third."""
        nonlocal attempts
        attempts += 1

        response = requests.Response()

        if attempts < 3:
            response.status_code = 503
        else:
            response.status_code = 200

        return response

    monkeypatch.setattr(
        requests,
        "request",
        fake_request
    )

    monkeypatch.setattr(
        "clients.api_client.time.sleep",
        lambda _: None
    )

    response = client.get("/todos/2")

    assert response.status_code == 200
    assert attempts == 3

def test_should_retry_timeout_for_get():
    """Test that the APIClient should retry on timeout for GET requests."""
    client = APIClient(API_BASE_URL)

    result = client._should_retry(
        "GET",
        exception=requests.Timeout()
    )

    assert result is True

def test_should_not_retry_timeout_for_post():
    """Test that the APIClient should not retry on timeout for POST requests."""
    client = APIClient(API_BASE_URL)

    result = client._should_retry(
        "POST",
        exception=requests.Timeout()
    )

    assert result is False

def test_should_retry_503_for_get():
    """Test that the APIClient should retry on 503 Service Unavailable for GET requests."""
    client = APIClient(API_BASE_URL)

    result = client._should_retry(
        "GET",
        status_code=503
    )

    assert result is True

def test_should_not_retry_404_for_get():
    """Test that the APIClient should not retry on 404 Not Found for GET requests."""
    client = APIClient(API_BASE_URL)

    result = client._should_retry(
        "GET",
        status_code=404
    )

    assert result is False
