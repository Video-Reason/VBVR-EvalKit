"""
Prompts for Object Subtraction Tasks

This file centralizes all prompts used for object subtraction tasks.
Organized by cognitive level (L1-L4).
Modify prompts here to experiment with different instruction styles.
"""

# Level 1: Explicit Specificity - Remove objects by explicit visual attributes
PROMPTS_L1 = [
    "Remove all red objects from the scene. Keep all other objects in their exact positions.",
    "Move all blue objects out of view. Do not move any other objects.",
    "Remove all green objects so that only objects of other colors remain in their original positions.",
    "Take away all yellow objects from the scene. Keep all remaining objects stationary.",
]

# Level 2: Enumerated Selection - Remove multiple explicitly listed objects
PROMPTS_L2 = [
    "Remove the red cube, the green sphere, and the blue pyramid from the scene. Keep all other objects fixed in their positions.",
    "Move the yellow cube, the blue sphere, and the red pyramid out of view. Do not move any other objects.",
]

# Level 3: Relational Reference - Remove objects using spatial relations
PROMPTS_L3 = [
    "Remove the three objects on the left side of the screen. Keep all other objects in their exact positions.",
    "Move the two objects farthest from the center out of view. Keep all remaining objects stationary.",
    "Remove all objects in the upper half of the image. Keep objects in the lower half unchanged.",
]

# Level 4: Conceptual Abstraction - Remove objects based on semantic properties
PROMPTS_L4 = [
    "Remove all large objects and keep only the small ones in their original positions.",
    "Remove the object that looks different from the others. Keep all similar objects fixed.",
    "Keep only the objects that share the same shape. Remove all others from the scene.",
]

# Default prompt index to use for each level
DEFAULT_PROMPT_INDEX_L1 = 0
DEFAULT_PROMPT_INDEX_L2 = 0
DEFAULT_PROMPT_INDEX_L3 = 0
DEFAULT_PROMPT_INDEX_L4 = 0

