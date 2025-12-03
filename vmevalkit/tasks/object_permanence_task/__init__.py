"""
Object Permanence Task Module for VMEvalKit

This module provides object permanence reasoning tasks for evaluating video models' ability to:
- Understand that objects continue to exist when occluded
- Maintain object properties (position, color, shape) when occluder moves
- Demonstrate object permanence reasoning

The object permanence tasks test whether models understand that objects remain unchanged
when an occluder moves across them, even when the objects are temporarily hidden.
"""

from .object_permanence_reasoning import (
    ObjectPermanenceTaskPair,
    ObjectPermanenceGenerator,
    create_dataset,
)

__all__ = [
    'ObjectPermanenceTaskPair',
    'ObjectPermanenceGenerator',
    'create_dataset',
]

__version__ = "1.0.0"

