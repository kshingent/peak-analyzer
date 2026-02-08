# Python Coding Standards

## Type Hinting (Python 3.9+)

* Do not use deprecated type hints from the `typing` module (e.g., `List`, `Dict`, `Tuple`).
* Use built-in collection types (e.g., `list`, `dict`, `tuple`) for generics.
* Use the pipe operator (`|`) for Union types instead of `Union[A, B]` (e.g., `str | int`).
* Use `A | None` instead of `Optional[A]`.

## Implementation Principles

* **Avoid Over-Engineering**:
* Strictly implement only what is requested.
* Do not implement extensions; instead, document suggested expansion paths in comments.
* **Do not add unnecessary `if` branches** to handle hypothetical input variations that were not specified.


* **Library Usage**:
* Prioritize using standard external libraries (e.g., `numpy`, `scipy`) for efficient computation rather than implementing custom logic or loops.


* **Backward Compatibility**: Do not use aliases for imports or functions that could break backward compatibility or lead to ambiguity.
* **Error Handling**: Do not suppress exceptions with unnecessary `try-except` blocks. Only catch specific exceptions that can be handled appropriately.