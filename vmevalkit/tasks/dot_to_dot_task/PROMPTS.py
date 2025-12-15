"""
Prompt templates for the Dot to Dot task.
"""

from __future__ import annotations

# Single comprehensive prompt template
PROMPT_TEMPLATE = (
    "Complete this dot-to-dot puzzle by connecting the numbered dots in numerical order from 1 to {max_number}. "
    "Starting from dot 1, draw a continuous line connecting each dot to the next number in sequence. "
    "Draw smooth lines between consecutive dots without lifting the pen or teleporting. "
    "Keep the camera view fixed in the top-down perspective and maintain all dot positions unchanged. "
    "Stop the video when all dots are connected and the complete pattern is fully revealed."
)


def get_prompt(max_number: int) -> str:
    """Generate prompt for dot-to-dot task."""
    return PROMPT_TEMPLATE.format(max_number=max_number)

