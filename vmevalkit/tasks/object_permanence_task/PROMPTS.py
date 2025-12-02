"""
Prompts for Object Permanence Tasks

Design: Tell model what objects are in the scene, then ask to move occluder 
from current position to the right until it completely exits the frame.
Final frame should have no occluder, objects remain unchanged.

Unified prompt template that adapts to any number of objects (1, 2, 3, 4, etc.)
"""

# Unified prompt template for any number of objects
# Placeholders:
# - {objects_count_description}: Object count description (e.g., "one", "two", "three")
# - {object_word}: Singular or plural form of "object" (e.g., "object", "objects")
# - {objects_description}: Description of all objects (e.g., "a red cube" or "a red cube and a blue sphere")
# - {objects_reference}: Reference to objects (e.g., "the object", "the objects")
# - {objects_pronoun}: Pronoun for objects (e.g., "it", "them")
PROMPTS = [
    "The scene contains {objects_count_description} 2D {object_word} at fixed positions: {objects_description}.\n\n"
    "A solid opaque gray panel is placed at the left side of the scene.\n\n"
    "The panel is rigid and maintains the same shape, size, color, and orientation for the entire sequence.\n\n"
    "The panel moves horizontally to the right at a steady, continuous speed.\n\n"
    "During its motion, the panel temporarily occludes the {objects_reference} when passing over {objects_pronoun}, without physically interacting with {objects_pronoun}.\n\n"
    "Continue the motion horizontally to the right until the panel has moved completely out of the scene.\n\n"
    "The camera view remains fixed throughout the entire sequence."
]

# Default prompt index
DEFAULT_PROMPT_INDEX = 0

