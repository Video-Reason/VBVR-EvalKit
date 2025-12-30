"""
Sequence Completion Task Module for VMEvalKit

This module provides sequence completion reasoning tasks for evaluating video models' ability to:
- Recognize mathematical patterns (arithmetic, geometric, power, Fibonacci sequences)
- Understand cyclic patterns (shape, color, direction cycles)
- Handle mixed attribute sequences (color+shape combinations)
- Demonstrate pattern recognition and sequence completion through video generation

The sequence completion tasks test pattern recognition, mathematical reasoning, and logical
extrapolation capabilities in video models.
"""

from .sequence_completion_reasoning import (
    SequenceCompletionTaskPair,
    SequenceCompletionDataset,
    SequenceCompletionTaskGenerator,
    SequenceRenderer,
    create_dataset,
    render_sequence
)

__all__ = [
    'SequenceCompletionTaskPair',
    'SequenceCompletionDataset',
    'SequenceCompletionTaskGenerator',
    'SequenceRenderer',
    'create_dataset',
    'render_sequence'
]

__version__ = "1.0.0"

