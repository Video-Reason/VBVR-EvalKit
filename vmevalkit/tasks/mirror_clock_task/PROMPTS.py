"""
Prompts for Mirror Clock Reasoning Tasks

This file centralizes all prompts used for mirror clock tasks.
Prompts use {time_delta} placeholder for dynamic time insertion.
"""

# Prompts with {time_delta} placeholder
# Example: {time_delta} will be replaced with "2 hours", "1 hour and 30 minutes", etc.
PROMPTS = [
    "This is a mirrored clock. If the original clock moves forward by {time_delta}, what time will it show?",
    "The image shows a horizontally flipped clock. After {time_delta} passes on the original clock, what will be the new time?",
    "This mirror-reflected clock needs to advance {time_delta}. Show what the original clock will display after this time passes.",
    "From this mirrored clock, determine the original time, then add {time_delta}. What is the result?",
]


# Difficulty-specific prompts (optional)
# Note: Difficulty is based on visual precision needed, not calculation complexity
DIFFICULTY_SPECIFIC_PROMPTS = {
    'easy': [
        "This horizontally flipped clock shows an hour position. Reverse the flip to show the original.",
        "The clock is mirrored at an exact hour. Show what it looks like unflipped.",
    ],
    'medium': [
        "This mirrored clock has hands at common positions. Reverse the horizontal flip to reveal the original.",
        "Flip this mirror-reflected clock back to show its original appearance.",
    ],
    'hard': [
        "This horizontally flipped clock shows precise hand positions. Reverse the flip to reveal the original clock.",
        "This mirror-reflected clock requires careful attention to hand angles. Show the original by reversing the flip.",
    ]
}


# Default prompt index to use
DEFAULT_PROMPT_INDEX = 0
