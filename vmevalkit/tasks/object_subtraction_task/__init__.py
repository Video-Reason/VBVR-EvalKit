"""
Object Subtraction Task Module for VMEvalKit

This module provides object subtraction reasoning tasks for evaluating video models' ability to:
- Understand selective removal instructions
- Remove specific objects while keeping others stationary
- Demonstrate multi-level cognitive reasoning (L1-L4)

The object subtraction tasks test selective attention, inhibitory control, and causal consistency
in video generation models.

Level 1: Explicit Specificity - Remove objects by explicit visual attributes (color, shape)
Level 2: Enumerated Selection - Remove multiple explicitly listed objects
Level 3: Relational Reference - Remove objects using spatial relations
Level 4: Conceptual Abstraction - Remove objects based on semantic properties
"""

from .object_subtraction_reasoning import (
    ObjectSubtractionTaskPair,
    ObjectGenerator,
    SceneRenderer,
    RuleGenerator,
    ObjectSubtractionGenerator,
    create_dataset,
)

__all__ = [
    'ObjectSubtractionTaskPair',
    'ObjectGenerator',
    'SceneRenderer',
    'RuleGenerator',
    'ObjectSubtractionGenerator',
    'create_dataset',
]

__version__ = "1.0.0"

