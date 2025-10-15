"""
Sudoku Task Module for VMEvalKit

This module provides sudoku reasoning tasks for evaluating video models' ability to:
- Understand sudoku puzzles from visual input with MNIST digit styling
- Apply logical sudoku rules and constraints
- Demonstrate solutions through generated video
- Show step-by-step reasoning in puzzle solving

The sudoku tasks test logical reasoning, pattern recognition, constraint satisfaction,
and sequential problem-solving capabilities in video models.
"""

from .sudoku_reasoning import (
    SudokuTaskPair,
    SudokuDataset,
    SudokuTaskGenerator,
    create_dataset,
    generate_sudoku_board_image
)

__all__ = [
    'SudokuTaskPair',
    'SudokuDataset', 
    'SudokuTaskGenerator',
    'create_dataset',
    'generate_sudoku_board_image'
]

__version__ = "1.0.0"
