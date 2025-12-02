"""Tasks package: contains task generators such as maze_task."""

try:
    from . import maze_task  # noqa: F401
except ModuleNotFoundError:
    # Some tasks rely on extra submodules (e.g., maze_dataset). Allow importing
    # lightweight tasks without forcing every optional dependency to be present.
    maze_task = None

__all__ = ["maze_task"] if maze_task is not None else []
