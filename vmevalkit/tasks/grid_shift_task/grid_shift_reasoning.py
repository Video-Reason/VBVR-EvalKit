"""
Grid Shift Task for VMEvalKit.

The task shows a 6x6 grid with three same-colored blocks. The model should
shift all blocks 1-2 steps in a random direction without leaving the grid.
"""

import hashlib
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .PROMPTS import PROMPTS, DEFAULT_PROMPT_INDEX


GRID_SIZE = 6
NUM_BLOCKS = 3
STEP_OPTIONS = (1, 2)
DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
COLORS = ["red", "green", "blue", "yellow", "orange", "purple"]


@dataclass
class GridShiftTaskPair:
    """
    Data structure for grid shift video model evaluation.
    """
    id: str
    prompt: str
    first_image_path: str
    final_image_path: str
    task_category: str = "GridShift"
    grid_shift_data: Dict[str, Any] = None
    difficulty: str = "easy"
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class GridRenderer:
    """Render grid images with square blocks."""

    def __init__(self, grid_size: int = GRID_SIZE, dpi: int = 150, figsize: Tuple[int, int] = (4, 4)):
        self.grid_size = grid_size
        self.dpi = dpi
        self.figsize = figsize

    def render(self, positions: List[Tuple[int, int]], color: str, output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        for i in range(self.grid_size + 1):
            ax.plot([i, i], [0, self.grid_size], "-", linewidth=1, color="#333333")
            ax.plot([0, self.grid_size], [i, i], "-", linewidth=1, color="#333333")

        padding = 0.08
        size = 1 - 2 * padding
        for row, col in positions:
            y = self.grid_size - 1 - row
            rect = patches.Rectangle(
                (col + padding, y + padding),
                size,
                size,
                facecolor=color,
                edgecolor="black",
                linewidth=1.5,
            )
            ax.add_patch(rect)

        plt.tight_layout(pad=0)
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)


class GridShiftGenerator:
    """Generator for grid shift tasks."""

    def __init__(self, grid_size: int = GRID_SIZE, num_blocks: int = NUM_BLOCKS):
        self.grid_size = grid_size
        self.num_blocks = num_blocks
        self.rng = random.Random()
        self.renderer = GridRenderer(grid_size=grid_size)
        self.temp_dir = tempfile.mkdtemp(prefix="grid_shift_")

    def _valid_positions(self, direction: str, steps: int) -> List[Tuple[int, int]]:
        dr, dc = DIRECTIONS[direction]

        if dr == -1:
            row_range = range(steps, self.grid_size)
        elif dr == 1:
            row_range = range(0, self.grid_size - steps)
        else:
            row_range = range(0, self.grid_size)

        if dc == -1:
            col_range = range(steps, self.grid_size)
        elif dc == 1:
            col_range = range(0, self.grid_size - steps)
        else:
            col_range = range(0, self.grid_size)

        return [(r, c) for r in row_range for c in col_range]

    def _difficulty_for_steps(self, steps: int) -> str:
        return "easy" if steps == 1 else "medium"

    def generate_single_task(self, task_id: str, seed: Optional[int] = None) -> GridShiftTaskPair:
        if seed is not None:
            self.rng.seed(seed)

        direction = self.rng.choice(list(DIRECTIONS.keys()))
        steps = self.rng.choice(STEP_OPTIONS)
        color = self.rng.choice(COLORS)

        valid_positions = self._valid_positions(direction, steps)
        if len(valid_positions) < self.num_blocks:
            raise ValueError("Not enough valid positions for the requested shift.")

        positions = self.rng.sample(valid_positions, self.num_blocks)
        dr, dc = DIRECTIONS[direction]
        shifted_positions = [(r + dr * steps, c + dc * steps) for r, c in positions]

        template_index = DEFAULT_PROMPT_INDEX
        if len(PROMPTS) > 1:
            template_index = self.rng.randint(0, len(PROMPTS) - 1)
        step_word = "step" if steps == 1 else "steps"
        prompt = PROMPTS[template_index].format(
            steps=steps,
            step_word=step_word,
            direction=direction,
        )

        first_path = Path(self.temp_dir) / f"{task_id}_first.png"
        final_path = Path(self.temp_dir) / f"{task_id}_final.png"
        self.renderer.render(positions, color, first_path)
        self.renderer.render(shifted_positions, color, final_path)

        task_pair = GridShiftTaskPair(
            id=task_id,
            prompt=prompt,
            first_image_path=str(first_path),
            final_image_path=str(final_path),
            task_category="GridShift",
            grid_shift_data={
                "grid_size": self.grid_size,
                "num_blocks": self.num_blocks,
                "color": color,
                "direction": direction,
                "steps": steps,
                "positions": positions,
                "shifted_positions": shifted_positions,
            },
            difficulty=self._difficulty_for_steps(steps),
            created_at=datetime.now().isoformat(),
        )

        return task_pair


def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    """Create grid shift dataset."""
    print("Creating Grid Shift Dataset")
    print(f"   Total samples: {num_samples}")

    generator = GridShiftGenerator()
    pairs: List[GridShiftTaskPair] = []

    for i in range(num_samples):
        task_id = f"grid_shift_{i:04d}"
        task_hash = int(hashlib.md5(task_id.encode()).hexdigest()[:8], 16)
        seed = task_hash + i
        task_pair = generator.generate_single_task(task_id, seed=seed)
        pairs.append(task_pair)

    pairs_dict = []
    for pair in pairs:
        pairs_dict.append({
            "id": pair.id,
            "prompt": pair.prompt,
            "first_image_path": pair.first_image_path,
            "final_image_path": pair.final_image_path,
            "task_category": pair.task_category,
            "grid_shift_data": pair.grid_shift_data,
            "difficulty": pair.difficulty,
            "created_at": pair.created_at,
        })

    dataset = {
        "name": "grid_shift_tasks",
        "description": "Grid shift tasks for video model evaluation",
        "pairs": pairs_dict,
        "metadata": {
            "total_tasks": len(pairs),
            "grid_size": GRID_SIZE,
            "num_blocks": NUM_BLOCKS,
            "step_options": STEP_OPTIONS,
            "colors": COLORS,
            "generation_date": datetime.now().isoformat(),
        },
        "created_at": datetime.now().isoformat(),
    }

    print("Dataset creation complete")
    print(f"   Total tasks: {len(pairs)}")
    return dataset
