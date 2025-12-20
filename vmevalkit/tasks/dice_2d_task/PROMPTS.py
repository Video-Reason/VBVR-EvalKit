"""
Prompts for Dice Opposite Face Reasoning Tasks

This file centralizes all prompts used for dice reasoning tasks.
Prompts use {shown_number} placeholder for dynamic content insertion.
"""

# Prompts with {shown_number} placeholder
# Example: {shown_number} will be replaced with "3", "5", etc.
PROMPTS = [
    "This dice face shows {shown_number} dots. What number is on the opposite face?",
    "The dice displays {shown_number}. Determine the number on the opposite side.",
    "A standard dice shows {shown_number} on one face. What is on the face directly opposite to it?",
    "On this dice, one face has {shown_number} dots. Show the opposite face.",
]


# Difficulty-specific prompts (optional, for future use)
DIFFICULTY_SPECIFIC_PROMPTS = {
    'easy': [
        "This dice shows {shown_number}. What is on the opposite face?",
        "The face displays {shown_number} dots. Show the opposite side.",
    ],
    'medium': [
        "Using the rule that opposite faces sum to 7, if one face shows {shown_number}, what is the opposite?",
        "On a standard dice showing {shown_number}, determine the opposite face value.",
    ],
    'hard': [
        "Given that opposite faces of a dice always sum to 7, and one face shows {shown_number}, what number appears on the opposite face?",
        "Apply the opposite faces rule: if {shown_number} is visible, calculate the hidden opposite face.",
    ]
}


# Default prompt index to use
DEFAULT_PROMPT_INDEX = 0
