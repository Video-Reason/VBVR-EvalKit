"""
2D Dice Opposite Face Reasoning Task

Evaluates video generation models' visual reasoning ability through dice
opposite face problems with varying complexity based on number of dice.

Difficulty Levels:
- Easy (1 dice): Direct opposite face question
- Medium (2 dice): Visual selection + opposite face
- Hard (3 dice): Multi-dice comparison + opposite face
- Expert (4 dice): Complex selection + opposite face
"""

from .dice_reasoning import (
    create_dataset,
    create_single_task,
    calculate_unique_combinations,
    DiceTaskPair,
    MultiDiceReasoningGenerator,
    DiceTaskGenerator,
    DiceRenderer
)

__all__ = [
    'create_dataset',
    'create_single_task',
    'calculate_unique_combinations',
    'DiceTaskPair',
    'MultiDiceReasoningGenerator',
    'DiceTaskGenerator',
    'DiceRenderer'
]
