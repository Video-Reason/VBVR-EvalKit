"""
Prompts for Light Sequence Reasoning Tasks

This file centralizes all prompts used for light sequence reasoning tasks.
Each Type has 2 question templates that are used for all light counts (4, 6, 8, 10).
"""

# Scene description prefix (common to all prompts)
SCENE_PREFIX = "The scene contains a horizontal row of lights. Some lights are on, and some lights are off.\n\nYour task is: "

# Effect description suffix (common to all prompts)
EFFECT_SUFFIX = "\n\nYou can turn lights on and off simultaneously or in any order. The appearance of lights turning on or off should match the visual style of the lights in the initial frame. Focus on achieving the final state; the lights can change directly without showing gradual transitions."

# Type 1: Single Point Localization
TYPE_1_PROMPTS = [
    SCENE_PREFIX + "Ensure the {position_desc} light is on, and all other lights are off." + EFFECT_SUFFIX,
    SCENE_PREFIX + "Ensure the {position_desc} light is on, and all other lights are off." + EFFECT_SUFFIX
]

# Type 2: Multiple Point Localization
TYPE_2_PROMPTS = [
    SCENE_PREFIX + "Ensure the {position_desc} lights are on, and all other lights are off." + EFFECT_SUFFIX,
    SCENE_PREFIX + "Ensure the {position_desc} lights are on, and all other lights are off." + EFFECT_SUFFIX
]

# Type 3: Mathematical Pattern
TYPE_3_PROMPTS = [
    SCENE_PREFIX + "Ensure all lights at odd positions (counting from left to right) are on, and all other lights are off." + EFFECT_SUFFIX,
    SCENE_PREFIX + "Ensure all lights at even positions (counting from left to right) are on, and all other lights are off." + EFFECT_SUFFIX
]

# Type 4: Spatial Range
TYPE_4_PROMPTS = [
    SCENE_PREFIX + "Ensure the left half of the lights are on, and all other lights are off." + EFFECT_SUFFIX,
    SCENE_PREFIX + "Ensure the right half of the lights are on, and all other lights are off." + EFFECT_SUFFIX
]

# Type 5: Continuous Sequence
TYPE_5_PROMPTS = [
    SCENE_PREFIX + "Ensure the lights from {start_desc} to {end_desc} (counting from left to right) are on, and all other lights are off." + EFFECT_SUFFIX,
    SCENE_PREFIX + "Ensure the lights from {start_desc} to {end_desc} (counting from left to right) are on, and all other lights are off." + EFFECT_SUFFIX
]

# Type 6: Relative Position
TYPE_6_PROMPTS = [
    SCENE_PREFIX + "Ensure the leftmost 2 lights are on, and all other lights are off." + EFFECT_SUFFIX,
    SCENE_PREFIX + "Ensure the rightmost 2 lights are on, and all other lights are off." + EFFECT_SUFFIX
]

# All Type prompts organized by type index (1-6)
TYPE_PROMPTS = {
    1: TYPE_1_PROMPTS,
    2: TYPE_2_PROMPTS,
    3: TYPE_3_PROMPTS,
    4: TYPE_4_PROMPTS,
    5: TYPE_5_PROMPTS,
    6: TYPE_6_PROMPTS
}

# Default prompt index to use (0 or 1 for each type)
DEFAULT_PROMPT_INDEX = {
    1: 0,  # First question: position 1
    2: 0,  # First question: first set of positions
    3: 0,  # First question: odd positions
    4: 0,  # First question: left half
    5: 0,  # First question: first continuous sequence
    6: 0   # First question: leftmost 2
}

