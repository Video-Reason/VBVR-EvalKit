"""
2D Dice Opposite Face Reasoning Task

Evaluates spatial reasoning and logical deduction through 2D dice face relationships.
Uses 2D rendered dice faces for visual clarity.
"""

from .dice_reasoning import (
    create_dataset,
    create_single_task,
    DiceTaskPair,
    DiceReasoningGenerator,
    DiceTaskGenerator,
    DiceRenderer
)

__all__ = [
    'create_dataset',
    'create_single_task',
    'DiceTaskPair',
    'DiceReasoningGenerator',
    'DiceTaskGenerator',
    'DiceRenderer'
]
