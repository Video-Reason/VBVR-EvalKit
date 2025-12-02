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
    "The scene contains several stable 2D objects on a flat plane:\n\n"
    "(1) {objects_description}\n\n"
    "(2) A long thin gray rectangle positioned on the left side of the scene.\n\n"
    "Move the long thin gray rectangle horizontally to the right at a steady speed.\n\n"
    "As it moves, the rectangle will pass in front of {objects_reference} on the 2D plane and temporarily occlude {objects_pronoun}, without any physical interaction.\n\n"
    "Continue moving the long thin gray rectangle to the right until it has fully exited the scene.\n\n"
    "The camera view remains fixed for the entire sequence."
]

# Default prompt index
DEFAULT_PROMPT_INDEX = 0

