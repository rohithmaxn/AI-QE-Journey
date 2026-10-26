import pytest

from validators.api_validators import (
    validate_todo_structure,
    validate_todo_types,
    validate_todo_values,
)


def test_validate_todo_structure():
    """Test the validate_todo_structure function."""
    data = {
        "id": 1,
        "title": "delectus aut autem",
        "completed": False,
    }

    assert validate_todo_structure(data) is True


def test_validate_todo_types():
    """"""
    data = {
        "id": 1,
        "title": "delectus aut autem",
        "completed": False,
    }

    assert validate_todo_types(data) is True


def test_validate_todo_values():
    """"""
    data = {
        "id": 1,
        "title": "delectus aut autem",
        "completed": False,
    }

    assert validate_todo_values(data) is True


def test_validate_todo_structure_missing_field():
    """Test the validate_todo_structure function with missing fields."""
    data = {
        "id": 1,
        "title": "delectus aut autem",
    }

    assert validate_todo_structure(data) is False


def test_validate_todo_values_wrong_value():
    """Test the validate_todo_values function with incorrect values."""
    data = {
        "id": 2,
        "title": "delectus aut autem",
        "completed": False,
    }

    assert validate_todo_values(data) is False


def test_validate_todo_values_missing_field():
    """Test the validate_todo_values function with missing fields."""
    data = {
        "id": 1,
        "title": "delectus aut autem",
    }

    assert validate_todo_values(data) is False


@pytest.mark.parametrize(
    "data",
    [
        {
            "id": "1",
            "title": "delectus aut autem",
            "completed": False,
        },
        {
            "id": 1,
            "title": "delectus aut autem",
        },
    ],
    ids=[
        "invalid_id_type",
        "missing_field",
    ],
)
def test_validate_todo_types_invalid_data(data):
    """Test the validate_todo_types function with invalid data."""
    assert validate_todo_types(data) is False
