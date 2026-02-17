from pathlib import Path
from typing import Optional


def select_latest_run(task_dir: Path) -> Optional[Path]:
    """
    Select the most recent run directory under a task directory.
    
    Supports both:
    - Flat structure: task_dir itself contains video/ and question/ folders
    - Nested structure: task_dir contains run subdirectories (backward compatibility)

    Args:
        task_dir: Path to the task directory.

    Returns:
        The run directory (task_dir itself or newest subdirectory), or None if none exist.
    """
    # Check if task_dir itself is a flat output structure (has video/ folder)
    if (task_dir / "video").exists() and (task_dir / "video").is_dir():
        return task_dir
    
    # Otherwise, look for run subdirectories (backward compatibility)
    run_dirs = [p for p in task_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        return None

    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0]

