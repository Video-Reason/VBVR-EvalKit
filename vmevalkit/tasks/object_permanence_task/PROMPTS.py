"""
Prompts for Object Permanence Tasks

Design: Tell model what objects are in the scene, then ask to move occluder 
from current position to the right until it completely exits the frame.
Final frame should have no occluder, objects remain unchanged.

Unified prompt template that adapts to any number of objects (1, 2, 3, 4, etc.)
"""

# Unified prompt template for any number of objects
# Placeholders:
# - {objects_count_intro}: Introduction with object count (e.g., "There is one object" or "There are two objects")
# - {objects_description}: Description of all objects (e.g., "a red cube" or "a red cube and a blue sphere")
# - {objects_reference}: Reference to objects (e.g., "the object", "the objects")
# - {objects_pronoun}: Pronoun for objects (e.g., "it", "them")
PROMPTS = [
    "{objects_count_intro} in the scene: {objects_description}. An opaque gray panel is on the left side. "
    "Move the panel horizontally from left to right. "
    "The panel passes above the {objects_reference} and does not touch or move {objects_pronoun}. "
    "Continue moving the panel from left to right until the panel completely exits the frame. "
    "Keep the camera view fixed."
]

# Default prompt index
DEFAULT_PROMPT_INDEX = 0

