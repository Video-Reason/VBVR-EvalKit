"""
Prompts for Control Panel Animation Tasks

Design: Test VideoModel's ability to generate animations for a control panel with
indicator lights and switches. Each light can display three colors (red, green, blue)
depending on the lever position in its corresponding black control slot.

The control panel has:
- Three indicator lights in a horizontal row
- Each light has a black control slot below it with left, middle, right positions
- A lever that can be in left, middle, or right position
- Lever position determines the light color (left=red, middle=green, right=blue)
"""

# Prompt template for control panel tasks
# Placeholders:
# - {num_lights_description}: Description of number of lights (e.g., "one", "two", "three")
# - {initial_state_description}: Description of initial lever positions and light colors
# - {target_state_description}: Description of target lever positions and light colors
# - {lever_actions}: Specific actions to take (which levers to move and where)
PROMPTS = [
    "The scene contains a control panel with {num_lights_description} arranged horizontally:\n\n"
    "{initial_state_description}\n\n"
    "Each light has a black control slot below it with three positions: left, middle, and right. "
    "A lever in each slot can be positioned at left, middle, or right.\n\n"
    "The lever position determines the light color:\n"
    "- Left position: Red light\n"
    "- Middle position: Green light\n"
    "- Right position: Blue light\n\n"
    "{lever_actions}\n\n"
    "As each lever moves to its target position, the corresponding indicator light should change "
    "to the appropriate color. Generate a smooth animation showing the lever movements and "
    "simultaneous light color changes.\n\n"
    "The camera view remains fixed for the entire sequence."
]

# Default prompt index
DEFAULT_PROMPT_INDEX = 0

