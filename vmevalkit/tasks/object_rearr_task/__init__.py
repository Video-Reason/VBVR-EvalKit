"""
Object Rearrangement Task Module for VMEvalKit

This module provides object rearrangement reasoning tasks for evaluating video models' ability to:
- Understand spatial relation instructions
- Manipulate objects based on spatial relations (left, right, above, below, etc.)
- Demonstrate spatial reasoning and object manipulation capabilities

The object rearrangement tasks test spatial understanding, relational reasoning, and manipulation
planning in video generation models.
"""

from .object_rearr import (
    ObjectRearrGenerator,
    ObjectSpec,
    create_dataset,
    create_task_pair,
)

__all__ = [
    'ObjectRearrGenerator',
    'ObjectSpec',
    'create_dataset',
    'create_task_pair',
]

