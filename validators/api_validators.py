def validate_todo_structure(data):
    """Validate that the data has the required structure for a todo item."""
    required_fields = {
        "id",
        "title",
        "completed"
    }

    return required_fields.issubset(data.keys())


def validate_todo_types(data):
    """Validate that the data has the correct types for a todo item."""
    required_fields = {
        "id",
        "title",
        "completed"
    }

    if not required_fields.issubset(data.keys()):
        return False

    return (
        isinstance(data["id"], int)
        and isinstance(data["title"], str)
        and isinstance(data["completed"], bool)
    )


def validate_todo_values(data):
    """Validate that the data has the expected values for a specific todo item."""
    required_fields = {
        "id",
        "title",
        "completed"
    }

    if not required_fields.issubset(data.keys()):
        return False

    return (
        data["id"] == 1
        and data["title"] == "delectus aut autem"
        and data["completed"] is False
    )
