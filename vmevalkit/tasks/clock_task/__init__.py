"""
Clock Task Module for VMEvalKit

This module provides clock time reasoning tasks for evaluating video models' ability to:
- Understand clock time from visual input with hour hand only
- Calculate time after k hours (k from 1 to 24)
- Demonstrate time progression through generated video
- Show temporal reasoning capabilities

The clock tasks test temporal reasoning, arithmetic calculation, and visual understanding
of clock representations in video models.
"""

from .clock_reasoning import (
    ClockTaskPair,
    ClockDataset,
    ClockTaskGenerator,
    create_dataset,
    generate_clock_image
)

__all__ = [
    'ClockTaskPair',
    'ClockDataset', 
    'ClockTaskGenerator',
    'create_dataset',
    'generate_clock_image'
]

__version__ = "1.0.0"

