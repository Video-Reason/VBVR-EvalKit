"""
Prompts for Tetris Reasoning Tasks

This file centralizes all prompts used for Tetris tasks.
Modify prompts here to experiment with different instruction styles.
"""

PROMPTS = [
    (
        "Given a {n}x{n} Tetris map, determine whether any complete line(s) will be "
        "cleared after a new block locks. Difficulty: {difficulty}. "
        "If lines are cleared, simulate the elimination process and provide the final map. "
        "IMPORTANT (for visualization): when a line is cleared, animate that row by briefly "
        "flashing or highlighting the cleared cells and then removing them in-place — do NOT "
        "create new blocks or use an upward 'flip' animation. After removal, let blocks above "
        "fall straight down to fill the emptied cells. For simple cases, show a short local "
        "flash-and-disappear animation focused only on the cleared line(s). Also include a "
        "brief textual description of the animation steps (e.g., 'row flashes then disappears, "
        "blocks above fall down')."
    )
]

# Default prompt index to use
DEFAULT_PROMPT_INDEX = 0