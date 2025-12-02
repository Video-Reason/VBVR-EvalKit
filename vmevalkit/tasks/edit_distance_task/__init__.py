"""
Edit Distance Task Module

A visual calculation task where video models must compute the Levenshtein
distance between two strings and display the result in an answer box.

Example:
    >>> from vmevalkit.tasks.edit_distance_task import create_dataset
    >>> dataset = create_dataset(num_samples=15)
    >>> print(f"Created {len(dataset['pairs'])} edit distance tasks")
"""

from .edit_distance_reasoning import create_dataset

__all__ = ['create_dataset']

