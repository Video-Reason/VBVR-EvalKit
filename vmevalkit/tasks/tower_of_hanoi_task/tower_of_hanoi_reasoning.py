"""Tower of Hanoi single-move reasoning task for VMEvalKit."""

import random
import tempfile
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .PROMPTS import PROMPTS

GOAL_PEG = 2  # Right peg is always the goal


@dataclass
class HanoiTaskPair:
    id: str
    prompt: str
    first_image_path: str
    final_image_path: str
    task_category: str
    difficulty: str = ""
    num_disks: int = 3
    hanoi_data: Dict[str, Any] = None
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


def state_key(state: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(p) for p in state)


def get_valid_moves(state) -> List[Tuple[int, int, int]]:
    moves = []
    state = [list(p) for p in state]
    for src in range(3):
        if not state[src]:
            continue
        disk = state[src][-1]
        for dst in range(3):
            if src != dst and (not state[dst] or state[dst][-1] > disk):
                moves.append((src, dst, disk))
    return moves


def apply_move(state, move: Tuple[int, int, int]) -> List[List[int]]:
    src, dst, _ = move
    new_state = [list(p) for p in state]
    disk = new_state[src].pop()
    new_state[dst].append(disk)
    return new_state


def find_optimal_moves(state: List[List[int]], num_disks: int) -> List[Tuple[int, int, int]]:
    """Find all optimal first moves from current state to goal (BFS backwards from goal)."""
    goal = tuple(
        tuple(range(num_disks, 0, -1)) if i == GOAL_PEG else ()
        for i in range(3)
    )
    start = state_key(state)

    if start == goal:
        return []

    # BFS backwards from goal to compute distance-to-goal for all states
    dist_to_goal = {goal: 0}
    queue = deque([goal])

    while queue:
        curr = queue.popleft()
        curr_dist = dist_to_goal[curr]
        for move in get_valid_moves(curr):
            nxt = state_key(apply_move(curr, move))
            if nxt not in dist_to_goal:
                dist_to_goal[nxt] = curr_dist + 1
                queue.append(nxt)

    if start not in dist_to_goal:
        return []

    start_dist = dist_to_goal[start]
    optimal = []
    for move in get_valid_moves(start):
        nxt = state_key(apply_move(start, move))
        if dist_to_goal.get(nxt) == start_dist - 1:
            optimal.append(move)

    return optimal


def generate_random_valid_state(num_disks: int) -> List[List[int]]:
    """Generate a random valid state that isn't already solved."""
    goal = [[], [], list(range(num_disks, 0, -1))]

    while True:
        state = [[], [], []]
        for disk in range(num_disks, 0, -1):
            valid_pegs = [p for p in range(3) if not state[p] or state[p][-1] > disk]
            state[random.choice(valid_pegs)].append(disk)
        if state != goal:
            return state


def create_hanoi_image(state: List[List[int]], num_disks: int, filepath: str) -> None:
    """Create Tower of Hanoi visualization."""
    fig, ax = plt.subplots(figsize=(6, 4))

    disk_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    peg_x = [1.5, 4.5, 7.5]
    peg_labels = ['Left', 'Middle', 'Right (Goal)']

    # Base
    ax.add_patch(Rectangle((0.5, 0), 8, 0.3, color='#8B4513'))

    # Pegs
    for x in peg_x:
        ax.add_patch(Rectangle((x - 0.1, 0.3), 0.2, 3, color='#A0522D'))

    # Disks
    max_width = 1.4
    for peg_idx, peg in enumerate(state):
        for disk_idx, disk in enumerate(peg):
            width = 0.4 + (disk / num_disks) * max_width
            x = peg_x[peg_idx] - width / 2
            y = 0.3 + disk_idx * 0.4
            color = disk_colors[disk - 1]
            ax.add_patch(Rectangle((x, y), width, 0.35, color=color, ec='black', lw=1))

    # Goal highlight
    ax.add_patch(Rectangle((6.3, 0), 2.4, 3.5, fill=False, ec='green', lw=2, ls='--'))

    # Labels
    for i, (x, label) in enumerate(zip(peg_x, peg_labels)):
        ax.text(x, -0.5, label, ha='center', fontsize=10,
                color='green' if i == 2 else 'black',
                fontweight='bold' if i == 2 else 'normal')

    ax.set_xlim(0, 9)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


class HanoiTaskGenerator:
    """Generator for Tower of Hanoi single-move tasks."""

    def generate_single_task(self, task_id: str, difficulty: int = 1) -> HanoiTaskPair:
        num_disks = difficulty + 2
        difficulty_names = ["easy", "medium", "hard"]

        initial_state = generate_random_valid_state(num_disks)
        optimal_moves = find_optimal_moves(initial_state, num_disks)
        chosen_move = optimal_moves[0]
        final_state = apply_move(initial_state, chosen_move)

        temp_dir = tempfile.mkdtemp()
        first_path = Path(temp_dir) / f"{task_id}_first.png"
        final_path = Path(temp_dir) / f"{task_id}_final.png"

        create_hanoi_image(initial_state, num_disks, str(first_path))
        create_hanoi_image(final_state, num_disks, str(final_path))

        moves_remaining = len(find_optimal_moves(final_state, num_disks))

        return HanoiTaskPair(
            id=task_id,
            prompt=PROMPTS[0],
            first_image_path=str(first_path),
            final_image_path=str(final_path),
            task_category="TowerOfHanoi",
            difficulty=difficulty_names[difficulty],
            num_disks=num_disks,
            hanoi_data={
                "initial_state": initial_state,
                "final_state": final_state,
                "optimal_move": list(chosen_move),
                "all_optimal_moves": [list(m) for m in optimal_moves],
                "moves_remaining": moves_remaining
            }
        )

    def generate_dataset(self, num_samples: int = 10) -> Dict[str, Any]:
        difficulties = [0, 1, 2]
        tasks = []

        print(f"Generating {num_samples} Tower of Hanoi tasks...")

        for i in range(num_samples):
            difficulty = difficulties[i % 3]
            task_id = f"tower_of_hanoi_{i:04d}"

            try:
                task = self.generate_single_task(task_id, difficulty)
                tasks.append(task)
                print(f"Generated {i+1}/{num_samples}: {task_id} ({task.difficulty}, {task.num_disks} disks)")
            except Exception as e:
                print(f"Failed {task_id}: {e}")
                continue

        task_dicts = []
        for task in tasks:
            task_dicts.append({
                'id': task.id,
                'prompt': task.prompt,
                'first_image_path': task.first_image_path,
                'final_image_path': task.final_image_path,
                'task_category': task.task_category,
                'difficulty': task.difficulty,
                'num_disks': task.num_disks,
                'hanoi_data': task.hanoi_data,
                'created_at': task.created_at
            })

        return {
            'name': "Tower of Hanoi Single-Move Dataset",
            'description': "Tower of Hanoi single-move tasks for video model evaluation",
            'pairs': task_dicts,
            'metadata': {
                "total_tasks": len(tasks),
                "difficulties": difficulties,
                "generation_date": datetime.now().isoformat(),
                "task_categories": ["TowerOfHanoi"]
            }
        }


def create_dataset(num_samples: int = 10) -> Dict[str, Any]:
    """Main entry point for dataset generation."""
    generator = HanoiTaskGenerator()
    return generator.generate_dataset(num_samples)
