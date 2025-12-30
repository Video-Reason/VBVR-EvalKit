"""
Prompts for Sequence Completion Reasoning Tasks

This file centralizes all prompts used for sequence completion reasoning tasks.
Each type has a prompt template that describes the sequence pattern.
"""

# Type 1: Arithmetic Sequence
TYPE_1_PROMPT = "This is a sequence. Observe its pattern, then replace the question mark with the correct number to complete the arithmetic sequence."

# Type 2: Geometric Sequence
TYPE_2_PROMPT = "This is a sequence. Observe its pattern, then replace the question mark with the correct number to complete the geometric sequence."

# Type 3: Power Sequence
TYPE_3_PROMPT = "This is a sequence. Observe its pattern, then replace the question mark with the correct number to complete the square sequence."

# Type 4: Fibonacci Sequence
TYPE_4_PROMPT = "This is a sequence. Observe its pattern, then replace the question mark with the correct number to complete the Fibonacci sequence."

# Type 5: Shape Cycle
TYPE_5_PROMPT = "This is a sequence with a repeating cycle of shapes. Observe the pattern, then replace the question mark with the correct shape to complete the shape cycle."

# Type 6: Color Cycle
TYPE_6_PROMPT = "This is a sequence with a repeating cycle of colors. Observe the pattern, then replace the question mark with the correct color to complete the color cycle."

# Type 7: Direction Cycle
TYPE_7_PROMPT = "This is a sequence with a repeating cycle of directions. Observe the pattern, then replace the question mark with the correct direction to complete the direction cycle."

# Type 8: Mixed Sequence (Color + Shape only)
TYPE_8_PROMPT = "This is a sequence with a repeating cycle of mixed elements (combining color and shape). Observe the pattern, then replace the question mark with the correct element to complete the mixed sequence."

# All Type prompts organized by type index (1-8)
TYPE_PROMPTS = {
    1: TYPE_1_PROMPT,
    2: TYPE_2_PROMPT,
    3: TYPE_3_PROMPT,
    4: TYPE_4_PROMPT,
    5: TYPE_5_PROMPT,
    6: TYPE_6_PROMPT,
    7: TYPE_7_PROMPT,
    8: TYPE_8_PROMPT
}

