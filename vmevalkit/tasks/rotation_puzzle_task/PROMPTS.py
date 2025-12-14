"""
Prompt templates for the Rotation Puzzle task.
"""

from __future__ import annotations

# Single comprehensive prompt template
PROMPT_TEMPLATE = (
    "Solve this rotation puzzle by rotating the four squares to connect the pipe paths. "
    "Each square can be rotated 90 degrees clockwise or counterclockwise. "
    "Rotate the squares so that all pipe paths connect to form a continuous path. "
    "Keep the camera view fixed in the top-down perspective and maintain all square positions unchanged. "
    "Stop the video when all pipes are connected and the puzzle is solved."
)


def get_prompt() -> str:
    """Generate prompt for rotation puzzle task."""
    return PROMPT_TEMPLATE

