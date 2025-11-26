"""
Mirror Clock Task
Mirror reflection and spatial transformation reasoning task module
"""

from .mirror_clock_reasoning import (
    create_dataset,
    create_single_task,
    MirrorClockTaskPair,
    MirrorClockGenerator,
    MirrorClockTaskGenerator,
    ClockRenderer
)

__all__ = [
    'create_dataset',           # Required: main entry function
    'create_single_task',       # Optional: convenience function
    'MirrorClockTaskPair',      # Optional: data structure
    'MirrorClockGenerator',     # Optional: generator class
    'MirrorClockTaskGenerator', # Optional: task generator class
    'ClockRenderer'             # Optional: rendering utility
]

__version__ = '1.0.0'
