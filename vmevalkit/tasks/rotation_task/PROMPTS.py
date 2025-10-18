"""
Prompts for 3D Mental Rotation Tasks

This file centralizes all prompts used for 3D mental rotation tasks.
Modify prompts here to experiment with different instruction styles.
"""

# Standardized prompts for rotation tasks (can add variations for experiments)
PROMPTS = [
    # Standard prompt with placeholders for dynamic content
    "A {num_voxels}-block sculpture sits fixed on a table. "
    "First frame: Your camera is tilted at {elev1}° elevation, viewing from {azim1}° azimuth. "
    "Final frame: Your camera remains at {elev2}° elevation, but rotates horizontally to {azim2}° azimuth. This is a 180-degree rotation "
    "Create a smooth video showing the camera's horizontal rotation around the sculpture, and try to maintain the tilted viewing angle throughout.",
    # Future variations can be added here for prompt experiments
]

# Default prompt index to use
DEFAULT_PROMPT_INDEX = 0

