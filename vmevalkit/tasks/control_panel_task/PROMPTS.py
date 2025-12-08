"""
Prompts for Control Panel Animation Tasks

Design: Test VideoModel's ability to observe and reason about control panel systems.
The model must infer the mapping between lever positions and light colors from the initial image,
then generate animations showing lever movements to achieve target colors.

Key design principles:
- Do not tell the model the specific mapping (left=red, middle=green, right=blue)
- Only provide general rules (each position corresponds to a fixed color)
- Model must observe the initial image to infer the mapping
- Each scene generates 3 independent tasks (one for each target color: red, green, blue)
"""

# Prompt template for control panel tasks
# Placeholders:
# - {num_lights}: Number of lights description (e.g., "two", "three", "four")
# - {target_color}: Target color (e.g., "red", "green", "blue")
# - {num_lights_note}: Optional note for 2-light scenarios
# - {duplicate_color_note}: Optional note for scenarios with duplicate colors (currently unused)
# - {inference_note}: Optional note in task objective
PROMPTS = [
    """The scene contains {num_lights} indicator lights. Each light has a unique 
control panel directly below it, and each control panel has a horizontal slider. 
Each light is controlled solely by the slider in its own control panel below it. 
Each slider can be positioned at three discrete positions within its control 
panel: left, middle, or right. Each position corresponds to one of three colors 
(red, green, blue), and all control panels share the same position-to-color 
mapping. The mapping must be inferred from the initial image.{num_lights_note}

Task: Make all lights display {target_color}. Observe the initial image to infer 
the position-to-color mapping and identify which light(s) need to change, then 
move the sliders horizontally to the appropriate positions within their control 
panels. The only way to change a light's color is by moving its slider. Sliders 
move only horizontally and stop at discrete positions. Camera remains fixed."""
]

# Default prompt index
DEFAULT_PROMPT_INDEX = 0
