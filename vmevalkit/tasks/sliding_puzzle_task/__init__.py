"""
Sliding Puzzle Task Module for VMEvalKit

This module provides sliding puzzle reasoning tasks for evaluating video models' ability to:
- Understand spatial relationships in 2D grids
- Recognize near-complete puzzle states
- Execute simple sliding moves (1-2 steps)
- Demonstrate visual consistency during tile movement

The sliding puzzle tasks test spatial reasoning, simple planning, and visual consistency
in video generation models. Puzzles are designed to be near-complete, requiring only
1-2 moves to solve, making them suitable for video model evaluation.
"""

from .sliding_puzzle_reasoning import (
    SlidingPuzzleTaskPair,
    SlidingPuzzleGenerator,
    create_dataset,
)

__all__ = [
    'SlidingPuzzleTaskPair',
    'SlidingPuzzleGenerator',
    'create_dataset',
]

__version__ = "1.0.0"

