# Tower of Hanoi Task

Single-move reasoning task. Given any valid disk configuration, demonstrate one optimal move toward solving (all disks on right peg).

## Task Description

**Input**: Image of 3-peg Tower of Hanoi with 2-4 disks in any valid configuration
**Output**: Image after one optimal move toward the goal

**Rules**:
1. Move one disk at a time
2. Only move the top disk of a stack
3. Never place a larger disk on a smaller disk

## State Representation

```
Pegs: Left (0), Middle (1), Right (2, Goal)
State: [[4,3,1], [2], []]  # Lists of disk sizes per peg, bottom to top
```

Disks are numbered by size (1=smallest). The goal is all disks on peg 2, stacked largest to smallest.

## Algorithm

BFS backwards from goal state computes distance-to-goal for all reachable states. An optimal move is any move that reduces this distance by 1.

```python
optimal_moves = find_optimal_moves(state, num_disks)  # Returns list of (from_peg, to_peg, disk)
```

## Difficulty

| Level  | Disks | State Space |
|--------|-------|-------------|
| easy   | 3     | 27 states   |
| medium | 4     | 81 states   |
| hard   | 5     | 243 states  |

## Data Structure

```python
{
    "id": "tower_of_hanoi_0001",
    "prompt": "...",
    "first_image_path": "...",
    "final_image_path": "...",
    "task_category": "TowerOfHanoi",
    "difficulty": "medium",
    "num_disks": 4,
    "hanoi_data": {
        "initial_state": [[4,3], [2], [1]],
        "final_state": [[4,3,1], [2], []],
        "optimal_move": [2, 0, 1],  # from_peg, to_peg, disk
        "all_optimal_moves": [[2, 0, 1]],
        "moves_remaining": 14
    }
}
```

## Usage

```python
from vmevalkit.tasks.tower_of_hanoi_task import create_dataset

dataset = create_dataset(num_samples=50)
```

## Files

```
tower_of_hanoi_task/
├── __init__.py
├── PROMPTS.py
├── tower_of_hanoi_reasoning.py
└── TOWER_OF_HANOI.md
```
