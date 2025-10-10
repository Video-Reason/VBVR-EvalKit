"""
Patched maze generation functions to fix KnowWhat maze failures.

The original KnowWhat maze generation fails when there are no valid start/end 
pairs with sufficient distance. This patch provides a more robust implementation.
"""

import numpy as np
import random
from typing import Tuple


# Constants from original KnowWhat code
WALL = 0
PATH = 1
POS = 2
END = 3
MIN_DISTANCE = 2


def distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def init_random_start_end_robust(maze: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Robust version of init_random_start_end that handles edge cases.
    
    Falls back to relaxed constraints if no valid pairs are found with MIN_DISTANCE.
    """
    path_indices = np.argwhere(maze == PATH)
    
    if len(path_indices) < 2:
        raise ValueError("Maze must have at least 2 path positions")
    
    # Try with original minimum distance requirement
    valid_pairs = []
    for p1 in path_indices:
        for p2 in path_indices:
            if distance(tuple(p1), tuple(p2)) >= MIN_DISTANCE:
                valid_pairs.append((tuple(p1), tuple(p2)))
    
    # Fallback strategies if no valid pairs found
    if len(valid_pairs) < k:
        print(f"   Warning: Only {len(valid_pairs)} pairs meet MIN_DISTANCE={MIN_DISTANCE}, trying fallbacks...")
        
        # Fallback 1: Reduce minimum distance requirement
        for min_dist in [1.5, 1.0, 0.5]:
            valid_pairs = []
            for p1 in path_indices:
                for p2 in path_indices:
                    if distance(tuple(p1), tuple(p2)) >= min_dist and not np.array_equal(p1, p2):
                        valid_pairs.append((tuple(p1), tuple(p2)))
            
            if len(valid_pairs) >= k:
                print(f"   Fallback: Using min_distance={min_dist}")
                break
        
        # Fallback 2: Use any two different path positions
        if len(valid_pairs) < k:
            valid_pairs = []
            for i, p1 in enumerate(path_indices):
                for j, p2 in enumerate(path_indices[i+1:], i+1):
                    valid_pairs.append((tuple(p1), tuple(p2)))
            print(f"   Final fallback: Using any different positions ({len(valid_pairs)} pairs)")
    
    if len(valid_pairs) < k:
        raise ValueError(f"Cannot find {k} valid start/end pairs in maze")
    
    # Sample the requested number of pairs
    random_start_ends = random.sample(valid_pairs, k=k)
    random_mazes = []
    
    for start, end in random_start_ends:
        random_maze = np.copy(maze)
        random_maze[start] = POS
        random_maze[end] = END
        random_mazes.append(random_maze)
    
    if k == 1:
        return random_mazes[0]
    
    return random_mazes


def patch_knowwhat_generation():
    """
    Monkey-patch the KnowWhat maze generation to use our robust version.
    """
    try:
        import sys
        import os
        
        # Add KnowWhat to path
        knowwhat_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'submodules', 'KnowWhat')
        if knowwhat_path not in sys.path:
            sys.path.append(knowwhat_path)
        
        # Import and patch
        from core import maze_generator
        maze_generator.init_random_start_end = init_random_start_end_robust
        print("✅ Patched KnowWhat maze generation for robustness")
        
    except ImportError as e:
        print(f"⚠️  Could not patch KnowWhat maze generation: {e}")
        print("   Maze generation may still fail occasionally")
