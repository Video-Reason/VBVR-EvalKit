# Tower of Hanoi Task - Implementation Plan

## Overview

Single-move Tower of Hanoi evaluation. The model sees a valid game state and must demonstrate ONE optimal move toward solving the puzzle (moving all disks to the right peg).

**Why single-move?**
- Aligns with other VMEvalKit tasks (chess = 1 move, sudoku = 1 cell)
- Easier to evaluate: did the video show exactly one legal, optimal move?
- Isolates reasoning from video generation length issues

---

## Task Specification

### Problem Definition

**Input (First Frame)**:
- Three pegs (left, middle, right)
- `n` disks in any valid configuration
- Goal is always: move all disks to the **right peg**

**Output (Final Frame)**:
- State after ONE optimal disk move

**Rules**:
1. Move only one disk at a time
2. Only the top disk of any stack can be moved
3. No disk may be placed on a smaller disk

### What "Optimal" Means

From any valid state, there exists an optimal path to the goal. The **optimal next move** is the first move of that shortest path. In some states, multiple moves may be equally optimal.

### Difficulty Levels

| Level | Disks | State Space | Notes |
|-------|-------|-------------|-------|
| Easy | 2 | 9 states | Simple, few options |
| Medium | 3 | 27 states | Standard puzzle |
| Hard | 4 | 81 states | More complex reasoning |

---

## Implementation Checklist

### Files to Create

```
vmevalkit/tasks/tower_of_hanoi_task/
├── __init__.py                    # Export create_dataset
├── tower_of_hanoi_reasoning.py    # Core implementation
├── PROMPTS.py                     # Prompt templates
└── TOWER_OF_HANOI.md              # Task documentation
```

### Files to Modify

1. **`vmevalkit/runner/TASK_CATALOG.py`** - Add task entry
2. **`vmevalkit/eval/gpt4o_eval.py`** - Add scoring guidance
3. **`vmevalkit/eval/internvl.py`** - Add scoring guidance

---

## Technical Design

### Data Structures

```python
@dataclass
class HanoiTaskPair:
    id: str
    prompt: str
    first_image_path: str
    final_image_path: str
    task_category: str              # "TowerOfHanoi"
    difficulty: str                 # "easy", "medium", "hard"
    num_disks: int                  # 2, 3, or 4
    initial_state: List[List[int]] # State before move
    final_state: List[List[int]]   # State after one optimal move
    optimal_move: Tuple[int, int, int]  # (from_peg, to_peg, disk_size)
    all_optimal_moves: List[Tuple] # All equally-optimal moves (if multiple)
    moves_remaining: int           # Moves left to solve after this move
    tower_of_hanoi_data: Dict      # Full metadata
    created_at: str
```

### State Representation

```python
# Pegs as lists of disk sizes (bottom to top)
# Disk sizes: 1 (smallest) to n (largest)
# Pegs: 0 = left, 1 = middle, 2 = right (goal)

# Example: 3 disks, mid-game state
initial_state = [
    [3],        # Left peg - largest disk
    [2, 1],     # Middle peg - two smaller disks
    []          # Right peg (goal) - empty
]

# After one optimal move (move disk 1 to right peg):
final_state = [
    [3],        # Left peg
    [2],        # Middle peg
    [1]         # Right peg - smallest disk moved here
]

optimal_move = (1, 2, 1)  # from middle, to right, disk size 1
```

### Optimal Move Algorithm

```python
def find_optimal_moves(state: List[List[int]], num_disks: int) -> List[Tuple]:
    """
    Find all optimal next moves from current state.
    Returns list of (from_peg, to_peg, disk) tuples.

    Uses BFS to find shortest path to goal, returns all first moves
    that lead to optimal-length solutions.
    """
    from collections import deque

    GOAL_PEG = 2  # Right peg

    def state_to_tuple(s):
        return tuple(tuple(peg) for peg in s)

    def get_valid_moves(s):
        """All legal single-disk moves from state s."""
        moves = []
        s = [list(peg) for peg in s]
        for src in range(3):
            if not s[src]:
                continue
            disk = s[src][-1]
            for dst in range(3):
                if src == dst:
                    continue
                if not s[dst] or s[dst][-1] > disk:
                    moves.append((src, dst, disk))
        return moves

    def apply_move(s, move):
        src, dst, disk = move
        new_state = [list(peg) for peg in s]
        new_state[src] = new_state[src][:-1]
        new_state[dst] = new_state[dst] + [disk]
        return state_to_tuple(new_state)

    # Goal: all disks on right peg
    goal = tuple(
        tuple(range(num_disks, 0, -1)) if i == GOAL_PEG else ()
        for i in range(3)
    )

    start = state_to_tuple(state)
    if start == goal:
        return []  # Already solved

    # BFS to find shortest path length
    queue = deque([(start, 0)])
    dist = {start: 0}

    while queue:
        current, d = queue.popleft()
        for move in get_valid_moves(current):
            next_state = apply_move(current, move)
            if next_state not in dist:
                dist[next_state] = d + 1
                if next_state != goal:
                    queue.append((next_state, d + 1))

    # Find all first moves that lead to optimal path
    goal_dist = dist.get(goal)
    if goal_dist is None:
        return []  # No solution (shouldn't happen)

    optimal_moves = []
    for move in get_valid_moves(start):
        next_state = apply_move(start, move)
        if dist.get(next_state) == goal_dist - 1:
            optimal_moves.append(move)

    return optimal_moves
```

### Image Generation

```python
def create_hanoi_image(state: List[List[int]], num_disks: int,
                       filepath: str, highlight_move: Tuple = None) -> None:
    """
    Create Tower of Hanoi visualization.

    Layout:
    - Three pegs on a base, labeled "Left", "Middle", "Right (Goal)"
    - Disks as colored rectangles (width proportional to size)
    - Right peg subtly highlighted as goal
    - Canvas: 900x600px (figsize=(6,4), dpi=150)
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Colors for disks (distinct per size)
    disk_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    # Peg positions
    peg_x = [1.5, 4.5, 7.5]
    peg_labels = ['Left', 'Middle', 'Right (Goal)']

    # Draw base
    ax.add_patch(Rectangle((0.5, 0), 8, 0.3, color='#8B4513'))

    # Draw pegs
    for x in peg_x:
        ax.add_patch(Rectangle((x - 0.1, 0.3), 0.2, 3, color='#A0522D'))

    # Draw disks
    max_width = 1.4
    for peg_idx, peg in enumerate(state):
        for disk_idx, disk in enumerate(peg):
            width = 0.4 + (disk / num_disks) * max_width
            x = peg_x[peg_idx] - width / 2
            y = 0.3 + disk_idx * 0.4
            color = disk_colors[disk - 1]
            ax.add_patch(Rectangle((x, y), width, 0.35,
                                   color=color, ec='black', lw=1))

    # Highlight goal peg
    ax.add_patch(Rectangle((6.3, 0), 2.4, 3.5,
                           fill=False, ec='green', lw=2, ls='--'))

    # Labels
    for i, (x, label) in enumerate(zip(peg_x, peg_labels)):
        ax.text(x, -0.5, label, ha='center', fontsize=10,
                color='green' if i == 2 else 'black',
                fontweight='bold' if i == 2 else 'normal')

    ax.set_xlim(0, 9)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.savefig(filepath, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
```

---

## Prompt Design

### PROMPTS.py

```python
PROMPTS = [
    "This is a Tower of Hanoi puzzle. The goal is to move all disks to the "
    "right peg (marked as Goal). Rules: (1) Move one disk at a time, "
    "(2) Only the top disk of a stack can be moved, (3) A larger disk "
    "cannot be placed on a smaller disk. Show the next optimal move.",
]

DEFAULT_PROMPT_INDEX = 0
```

### Evaluation Guidance

For `gpt4o_eval.py` and `internvl.py`:

```python
"tower_of_hanoi_task": "Check if exactly ONE disk moved between frames. "
"Verify the move is legal (top disk only, not placed on smaller disk). "
"The expected final frame shows the optimal move - compare disk positions."
```

---

## Task Catalog Entry

```python
'tower_of_hanoi': {
    'name': 'Tower of Hanoi',
    'description': 'Single-move planning and constraint satisfaction',
    'module': 'vmevalkit.tasks.tower_of_hanoi_task',
    'create_function': 'create_dataset',
    'process_dataset': lambda dataset, num_samples: dataset['pairs']
}
```

---

## Generation Logic

### Dataset Generation

```python
def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    """Generate Tower of Hanoi single-move task pairs."""
    pairs = []

    for i in range(num_samples):
        # Distribute difficulty
        difficulty_idx = i % 3
        num_disks = difficulty_idx + 2  # 2, 3, or 4 disks
        difficulty = ["easy", "medium", "hard"][difficulty_idx]

        # Generate random valid state (not already solved)
        initial_state = generate_random_valid_state(num_disks)

        # Find optimal move(s)
        optimal_moves = find_optimal_moves(initial_state, num_disks)

        if not optimal_moves:
            continue  # Skip if already solved (shouldn't happen)

        # Pick one optimal move for the expected final state
        chosen_move = optimal_moves[0]
        final_state = apply_move(initial_state, chosen_move)

        # Calculate remaining moves after this one
        remaining = len(find_optimal_path(final_state, num_disks))

        task = generate_single_task(
            task_id=f"tower_of_hanoi_{i:04d}",
            num_disks=num_disks,
            difficulty=difficulty,
            initial_state=initial_state,
            final_state=final_state,
            optimal_move=chosen_move,
            all_optimal_moves=optimal_moves,
            moves_remaining=remaining
        )
        pairs.append(task)

    return {
        "name": "Tower of Hanoi Single-Move Dataset",
        "pairs": pairs,
        "metadata": {
            "task_type": "single_move",
            "goal_peg": "right (index 2)",
            "total_pairs": len(pairs)
        }
    }
```

### Valid State Generation

```python
def generate_random_valid_state(num_disks: int) -> List[List[int]]:
    """
    Generate a random valid Hanoi state that is NOT the goal.
    """
    GOAL_PEG = 2

    while True:
        state = [[], [], []]

        # Place disks from largest to smallest
        for disk in range(num_disks, 0, -1):
            valid_pegs = [
                p for p in range(3)
                if not state[p] or state[p][-1] > disk
            ]
            peg = random.choice(valid_pegs)
            state[peg].append(disk)

        # Check it's not already solved
        goal = [[], [], list(range(num_disks, 0, -1))]
        if state != goal:
            return state
```

---

## Evaluation Criteria

### What Makes a Move Correct?

1. **Legal**: Only one disk moved, from top of a stack, onto empty peg or larger disk
2. **Optimal**: The move is on a shortest path to the solution

### Scoring Guidance

| Score | Meaning |
|-------|---------|
| 5 | Exactly one optimal move shown correctly |
| 4 | Legal move shown, close to optimal |
| 3 | Legal move shown, but suboptimal |
| 2 | Attempted a move, but illegal (e.g., wrong disk) |
| 1 | No clear move, or multiple moves, or disk teleportation |

---

## Success Criteria

- [ ] `create_dataset(50)` generates 50 valid single-move pairs
- [ ] All difficulty levels (2, 3, 4 disks) represented
- [ ] Initial states are varied (not just standard starts)
- [ ] Optimal moves computed correctly via BFS
- [ ] Images clearly show disk arrangement
- [ ] Final frame shows state after exactly one move
- [ ] Evaluation guidance enables accurate VLM scoring

---

## Implementation Order

1. Create directory structure and `__init__.py`
2. Implement `PROMPTS.py`
3. Implement `tower_of_hanoi_reasoning.py`:
   - State validation
   - Optimal move finder (BFS)
   - Image generation
   - `create_dataset()` function
4. Register in `TASK_CATALOG.py`
5. Add evaluation guidance to eval files
6. Test with small dataset
7. Write `TOWER_OF_HANOI.md` documentation
