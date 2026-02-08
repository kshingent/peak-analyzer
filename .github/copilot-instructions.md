# Python Coding Standards

- Do not use deprecated type hints from the `typing` module (e.g., `List`, `Dict`, `Tuple`).
- Use built-in collection types (e.g., `list`, `dict`, `tuple`) for generics (available in Python 3.9+).
- Use the pipe operator (`|`) for Union types instead of `Union[A, B]` (e.g., `str | int`).
- Use `A | None` instead of `Optional[A]`.