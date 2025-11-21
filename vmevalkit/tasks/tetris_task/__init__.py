"""
Tetris Task Module for VMEvalKit

Provides Tetris line-clearing task generators and data structures.

Exports:
- TetrisTaskPair: dataclass for task metadata
- TetrisTaskGenerator / TetrisEasyTaskGenerator / TetrisHardTaskGenerator
- create_tetris_task_pair: helper to render a single task to temp files
- create_dataset: convenience function to create a dataset
"""

from .tetris_reasoning import (
    TetrisTaskPair,
    TetrisTaskGenerator,
    TetrisEasyTaskGenerator,
    TetrisHardTaskGenerator,
    create_dataset,
)

__all__ = [
    'TetrisTaskPair',
    'TetrisTaskGenerator',
    'TetrisEasyTaskGenerator',
    'TetrisHardTaskGenerator',
    'create_dataset',
]

__version__ = "0.1.0"
