"""
Light Sequence Task Module for VMEvalKit

This module provides light sequence reasoning tasks for evaluating video models' ability to:
- Understand spatial positioning (left, right, nth position)
- Recognize mathematical patterns (odd, even positions)
- Process spatial ranges (left half, right half)
- Handle multiple position specifications
- Demonstrate spatial reasoning through video generation

The light sequence tasks test spatial perception, mathematical reasoning, and instruction following
capabilities in video models.
"""

from .light_sequence_reasoning import (
    LightSequenceTaskPair,
    LightSequenceDataset,
    LightSequenceTaskGenerator,
    LightSequenceGenerator,
    create_dataset,
    render_light_sequence
)

__all__ = [
    'LightSequenceTaskPair',
    'LightSequenceDataset',
    'LightSequenceTaskGenerator',
    'LightSequenceGenerator',
    'create_dataset',
    'render_light_sequence'
]

__version__ = "1.0.0"

