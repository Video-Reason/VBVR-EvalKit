"""
Control Panel Animation Task Module for VMEvalKit

This module provides control panel animation tasks for evaluating video models' ability to:
- Understand the relationship between lever position and light color
- Generate smooth animations of lever movements
- Synchronize lever movements with light color changes
- Reason about control panel state transitions

The control panel tasks test whether models can understand and generate animations for
a control panel where lever positions determine indicator light colors.
"""

from .control_panel_reasoning import (
    ControlPanelTaskPair,
    ControlPanelTaskGenerator,
    create_dataset,
)

__all__ = [
    'ControlPanelTaskPair',
    'ControlPanelTaskGenerator',
    'create_dataset',
]

__version__ = "1.0.0"

