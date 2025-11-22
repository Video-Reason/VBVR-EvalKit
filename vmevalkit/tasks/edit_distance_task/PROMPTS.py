"""
Prompts for Edit Distance Tasks

This file centralizes all prompts used for edit distance calculation tasks.
The model must compute the Levenshtein distance between two strings and
display the result in the answer box.
"""

# Standard prompts for edit distance calculation
PROMPTS = [
    "Calculate the edit distance between the two strings shown and display the number in the answer box.",
    "Compute the Levenshtein distance between String_A and String_B, then show the result in the answer box.",
    "Find the minimum number of single-character edits (insertions, deletions, or substitutions) needed to change String_A into String_B, and display this number in the answer box.",
    "Determine the edit distance between the top string and the bottom string, then animate the calculated number into the answer box.",
]

# Default prompt index to use
DEFAULT_PROMPT_INDEX = 0

