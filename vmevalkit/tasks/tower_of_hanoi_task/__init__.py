"""Tower of Hanoi task module for VMEvalKit."""

from .tower_of_hanoi_reasoning import (
    HanoiTaskPair,
    HanoiTaskGenerator,
    create_dataset,
)

__all__ = [
    'HanoiTaskPair',
    'HanoiTaskGenerator',
    'create_dataset',
]
