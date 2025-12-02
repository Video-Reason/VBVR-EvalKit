"""
Prompt templates for the Shape Sorter task.
"""

from __future__ import annotations

from typing import Iterable, List


PROMPT_TEMPLATES: List[str] = [
    (
        "Move each colored shape card from the left staging area into its matching outline on the right. "
        "Keep the camera fixed in the top-down view, move only one card at a time, and slide the cards smoothly without teleportation. "
        "{shape_summary} "
        "Stop the video when every outline is filled exactly."
    ),
    (
        "Solve the flat shape sorter puzzle exactly as shown. "
        "Starting from the unsolved first frame, drag the colored cards across the board and place them into the matching outlines on the right. "
        "{shape_summary} "
        "Keep the board orientation unchanged and end once all outlines are packed tightly."
    ),
]


def format_shape_summary(shape_labels: Iterable[str]) -> str:
    labels = [label for label in shape_labels if label]
    if not labels:
        return ""
    if len(labels) == 1:
        return f"Match the {labels[0]} card to its outline."
    if len(labels) == 2:
        return f"Match the {labels[0]} card first, followed by the {labels[1]} card."
    body = ", ".join(labels[:-1])
    return f"Match the {body}, and finally the {labels[-1]} card."

