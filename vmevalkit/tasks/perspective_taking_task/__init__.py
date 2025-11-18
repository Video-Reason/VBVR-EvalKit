"""
Perspective Taking Task Module for VMEvalKit

This module provides perspective taking reasoning tasks for evaluating video models' ability to:
- Understand spatial relationships between objects and agents
- Transform viewpoints and perspectives
- Reason about what can be seen from different angles
- Generate scenes from a back-facing perspective

The perspective taking tasks test spatial reasoning, viewpoint transformation,
and scene understanding capabilities in video models.
"""

from .perspective_taking_reasoning import (
    create_dataset,
    validate_dataset
)

__all__ = [
    'create_dataset',
    'validate_dataset'
]

__version__ = "1.0.0"

