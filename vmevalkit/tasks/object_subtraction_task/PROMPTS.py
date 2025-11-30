"""
Prompts for Object Subtraction Tasks

This file centralizes all prompts used for object subtraction tasks.
Organized by task type (type1-type4).
Modify prompts here to experiment with different instruction styles.
"""

# Type 1: Explicit Specificity - Remove objects by explicit visual attributes
PROMPTS_TYPE1 = [
    "Remove all red objects from the scene. Do not do anything to other objects.",
]

# Type 2: Enumerated Selection - Remove multiple explicitly listed objects
PROMPTS_TYPE2 = [
    "Remove the red cube, the green sphere, and the blue pyramid from the scene. Do not do anything to other objects.",
    "Move the yellow cube, the blue sphere, and the red pyramid out of view. Do not do anything to other objects.",
]

# Type 3: Relational Reference - Remove objects using spatial relations
PROMPTS_TYPE3 = [
    "Remove the three objects on the left side of the screen. Do not do anything to other objects.",
    "Move the two objects farthest from the center out of view. Do not do anything to other objects.",
    "Remove all objects in the upper half of the image. Do not do anything to other objects.",
]

# Type 4: Conceptual Abstraction - Remove objects based on semantic properties
PROMPTS_TYPE4 = [
    "Remove all large objects from the scene. Do not do anything to other objects.",
    "Remove the object that looks different from the others. Do not do anything to other objects.",
    "Remove all objects that do not share the same shape. Do not do anything to other objects.",
]

# Default prompt index to use for each type
DEFAULT_PROMPT_INDEX_TYPE1 = 0
DEFAULT_PROMPT_INDEX_TYPE2 = 0
DEFAULT_PROMPT_INDEX_TYPE3 = 0
DEFAULT_PROMPT_INDEX_TYPE4 = 0



