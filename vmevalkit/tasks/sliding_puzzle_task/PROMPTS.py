"""Centralized prompts for Sliding Puzzle Task."""

# Single prompt with dynamic move count and constraints
PROMPTS = [
    "Complete this sliding puzzle. The goal is to arrange the numbered tiles in grid order "
    "(filling each row from left to right, with rows from top to bottom), "
    "with the empty space at the bottom-right corner.\n\n"
    "Rules: Only tiles adjacent to the empty space can be moved. Slide one tile per move into the empty space.\n\n"
    "Complete in exactly {num_moves} move{plural}.\n\n"
    "Do not make extra moves. Keep the camera view fixed and maintain the grid structure unchanged.",
]

DEFAULT_PROMPT_INDEX = 0

