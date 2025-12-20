"""
Prompt templates for the Symmetry Completion task.
"""

from __future__ import annotations

PROMPT_TEMPLATE = (
    "Complete this pattern by filling in the missing grid cells on the right side. "
    "Observe the left half of the pattern and recognize that it should be mirrored to create a symmetric pattern. "
    "Fill in the right half by mirroring the left half across the vertical center line. "
    "Keep the camera view fixed in the top-down perspective and maintain all existing cells unchanged. "
    "Stop the video when the symmetric pattern is fully completed."
)


def get_prompt() -> str:
    """Returns the formatted prompt for the Symmetry Completion task."""
    return PROMPT_TEMPLATE

