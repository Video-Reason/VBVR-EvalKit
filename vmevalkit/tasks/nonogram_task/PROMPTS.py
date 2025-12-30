"""
Prompt templates for the Nonogram task.
"""

from __future__ import annotations

# Single comprehensive prompt template
PROMPT_TEMPLATE = (
    "Solve this nonogram puzzle by filling in the grid cells according to the row and column hints. "
    "The numbers on the left indicate the lengths of consecutive filled blocks in each row, "
    "and the numbers on top indicate the lengths of consecutive filled blocks in each column. "
    "Fill in the cells to reveal the hidden pattern. "
    "Keep the camera view fixed in the top-down perspective and maintain the grid structure unchanged. "
    "Stop the video when all cells are correctly filled and the complete pattern is revealed."
)


def get_prompt() -> str:
    """Generate prompt for nonogram task."""
    return PROMPT_TEMPLATE

