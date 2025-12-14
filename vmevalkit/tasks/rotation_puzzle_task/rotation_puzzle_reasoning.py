"""
Rotation Puzzle task for VMEvalKit.

This task evaluates whether video generation models can solve pipe puzzle problems
by rotating squares to connect pipe paths. Models must rotate the four squares
to form a continuous path through all connected pipes.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Circle, Rectangle

from .PROMPTS import get_prompt

Canvas = Tuple[int, int]
Point = Tuple[float, float]

CANVAS: Canvas = (768, 512)
DPI = 150

# Square and pipe appearance
SQUARE_SIZE = 140
GRID_SPACING = 20
PIPE_WIDTH = 12
SQUARE_BORDER_WIDTH = 2
PIPE_COLOR = "#3b82f6"
PIPE_EDGE_COLOR = "#2563eb"
SQUARE_BORDER_COLOR = "#64748b"
BACKGROUND_COLOR = "#f8fafc"


@dataclass
class SquareSpec:
    """Specification for a rotatable square with pipe pattern."""
    position: Point  # Top-left corner
    initial_angle: int  # Initial rotation: 0, 90, 180, 270
    target_angle: int  # Target rotation to solve puzzle
    pipe_pattern: List[int]  # Connections: [top, right, bottom, left] (0 or 1)
    index: int  # Position in grid: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right

    def get_connections(self, angle: int) -> List[int]:
        """Get pipe connections for a given rotation angle (for path validation)."""
        # When square rotates, the pipe pattern rotates with it
        # This method is used for validation, but actual rendering uses transform
        # Note: top and bottom are swapped in rendering, so we swap them here too
        # Original pattern: [top, right, bottom, left]
        # Swapped pattern: [bottom, right, top, left]
        swapped_pattern = [self.pipe_pattern[2], self.pipe_pattern[1], 
                          self.pipe_pattern[0], self.pipe_pattern[3]]
        
        if angle == 0:
            return swapped_pattern
        elif angle == 90:
            # Rotate clockwise: top->right, right->bottom, bottom->left, left->top
            return [swapped_pattern[3], swapped_pattern[0], 
                   swapped_pattern[1], swapped_pattern[2]]
        elif angle == 180:
            # Rotate 180: top->bottom, right->left, bottom->top, left->right
            return [swapped_pattern[2], swapped_pattern[3],
                   swapped_pattern[0], swapped_pattern[1]]
        elif angle == 270:
            # Rotate counterclockwise: top->left, right->top, bottom->right, left->bottom
            return [swapped_pattern[1], swapped_pattern[2],
                   swapped_pattern[3], swapped_pattern[0]]
        return swapped_pattern


class RotationPuzzleRenderer:
    """Renderer for rotation puzzle frames."""

    def __init__(self, canvas: Canvas = CANVAS, dpi: int = DPI):
        self.canvas = canvas
        self.dpi = dpi

    def render_start(self, squares: Sequence[SquareSpec], path: Path) -> None:
        """Render first frame: squares at initial rotations."""
        fig, ax = self._setup_axes()
        self._draw_background(ax)
        for square in squares:
            self._draw_square(ax, square, square.initial_angle)
        self._finalize(fig, path)

    def render_end(self, squares: Sequence[SquareSpec], path: Path) -> None:
        """Render final frame: squares at target rotations (solved)."""
        fig, ax = self._setup_axes()
        self._draw_background(ax)
        for square in squares:
            self._draw_square(ax, square, square.target_angle)
        self._finalize(fig, path)

    def _setup_axes(self):
        w, h = self.canvas
        fig, ax = plt.subplots(figsize=(w / self.dpi, h / self.dpi), dpi=self.dpi)
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.invert_yaxis()
        return fig, ax

    def _draw_background(self, ax) -> None:
        w, h = self.canvas
        bg = Rectangle((0, 0), w, h, facecolor=BACKGROUND_COLOR, edgecolor="none")
        ax.add_patch(bg)

    def _draw_square(self, ax, square: SquareSpec, angle: int) -> None:
        """Draw a square with pipe pattern at given rotation angle. Square and pipes rotate together."""
        x, y = square.position
        square_center_x = x + SQUARE_SIZE / 2
        square_center_y = y + SQUARE_SIZE / 2

        # Create transformation for rotation around center
        # Note: matplotlib uses degrees, and we need to account for inverted y-axis
        rotation_angle = -angle  # Negative because y-axis is inverted
        t = transforms.Affine2D().rotate_deg_around(square_center_x, square_center_y, rotation_angle) + ax.transData

        # Draw square border (will be rotated)
        square_rect = Rectangle(
            (x, y), SQUARE_SIZE, SQUARE_SIZE,
            linewidth=SQUARE_BORDER_WIDTH,
            edgecolor=SQUARE_BORDER_COLOR,
            facecolor="white",
            alpha=0.9,
            transform=t
        )
        ax.add_patch(square_rect)

        # Get original connections (before rotation - pipes are part of the square)
        # Swap top and bottom: [top, right, bottom, left] -> [bottom, right, top, left]
        original_pattern = square.pipe_pattern
        connections = [original_pattern[2], original_pattern[1], original_pattern[0], original_pattern[3]]
        half_size = SQUARE_SIZE / 2

        # Draw pipe paths based on swapped pattern [bottom, right, top, left]
        # All pipes will be rotated together with the square
        # Top connection (now uses original bottom)
        if connections[0]:
            pipe = Rectangle(
                (square_center_x - PIPE_WIDTH / 2, square_center_y),
                PIPE_WIDTH, half_size,
                facecolor=PIPE_COLOR,
                edgecolor=PIPE_EDGE_COLOR,
                linewidth=1,
                transform=t
            )
            ax.add_patch(pipe)

        # Right connection
        if connections[1]:
            pipe = Rectangle(
                (square_center_x, square_center_y - PIPE_WIDTH / 2),
                half_size, PIPE_WIDTH,
                facecolor=PIPE_COLOR,
                edgecolor=PIPE_EDGE_COLOR,
                linewidth=1,
                transform=t
            )
            ax.add_patch(pipe)

        # Bottom connection (now uses original top)
        if connections[2]:
            pipe = Rectangle(
                (square_center_x - PIPE_WIDTH / 2, square_center_y - half_size),
                PIPE_WIDTH, half_size,
                facecolor=PIPE_COLOR,
                edgecolor=PIPE_EDGE_COLOR,
                linewidth=1,
                transform=t
            )
            ax.add_patch(pipe)

        # Left connection
        if connections[3]:
            pipe = Rectangle(
                (square_center_x - half_size, square_center_y - PIPE_WIDTH / 2),
                half_size, PIPE_WIDTH,
                facecolor=PIPE_COLOR,
                edgecolor=PIPE_EDGE_COLOR,
                linewidth=1,
                transform=t
            )
            ax.add_patch(pipe)

        # Draw center connector (also rotated)
        center = Circle(
            (square_center_x, square_center_y),
            PIPE_WIDTH / 2 + 2,
            facecolor=PIPE_COLOR,
            edgecolor=PIPE_EDGE_COLOR,
            linewidth=1,
            transform=t
        )
        ax.add_patch(center)

    def _finalize(self, fig, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)


class RotationPuzzleGenerator:
    """Generator for rotation puzzle tasks."""

    def __init__(self, canvas: Canvas = CANVAS):
        self.canvas = canvas
        self.renderer = RotationPuzzleRenderer(canvas)
        self.rng = random.Random()
        self.output_root = Path("data/questions/rotation_puzzle_task")
        self._seen_signatures: set[str] = set()

    def generate(
        self,
        task_id: str,
        difficulty: str = "medium",
        seed: Optional[int] = None,
        ensure_unique: bool = True,
    ) -> Dict:
        """Generate a single rotation puzzle task."""
        if seed is not None:
            self.rng.seed(seed)

        squares = self._create_puzzle(difficulty)

        # Check uniqueness
        signature = None
        if ensure_unique:
            signature = self._build_signature(squares)
            max_attempts = 25
            attempts = 0
            while signature in self._seen_signatures and attempts < max_attempts:
                squares = self._create_puzzle(difficulty)
                signature = self._build_signature(squares)
                attempts += 1
            if signature in self._seen_signatures:
                raise RuntimeError("Failed to generate unique Rotation Puzzle sample after multiple attempts.")

        if ensure_unique and signature is not None:
            self._seen_signatures.add(signature)

        question_dir = self.output_root / task_id
        question_dir.mkdir(parents=True, exist_ok=True)
        first_png = question_dir / "first_frame.png"
        final_png = question_dir / "final_frame.png"

        self.renderer.render_start(squares, first_png)
        self.renderer.render_end(squares, final_png)

        prompt = get_prompt()
        (question_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

        metadata = self._build_metadata(task_id, squares, difficulty)
        (question_dir / "question_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "id": task_id,
            "prompt": prompt,
            "first_image_path": str(first_png),
            "final_image_path": str(final_png),
            "task_category": "RotationPuzzle",
            "difficulty": difficulty,
            "rotation_puzzle_data": metadata,
            "created_at": datetime.now().isoformat(),
        }

    def _create_puzzle(self, difficulty: str) -> List[SquareSpec]:
        """Create a solvable rotation puzzle where final state forms a closed square path."""
        w, h = self.canvas
        total_width = 2 * SQUARE_SIZE + GRID_SPACING
        start_x = (w - total_width) / 2
        start_y = (h - total_width) / 2

        positions = [
            (start_x, start_y),  # top-left (index 0)
            (start_x + SQUARE_SIZE + GRID_SPACING, start_y),  # top-right (index 1)
            (start_x, start_y + SQUARE_SIZE + GRID_SPACING),  # bottom-left (index 2)
            (start_x + SQUARE_SIZE + GRID_SPACING, start_y + SQUARE_SIZE + GRID_SPACING),  # bottom-right (index 3)
        ]

        # Define pipe patterns that form a closed square with all openings facing center:
        # Path: 0→1→3→2→0 (clockwise from top-left, going right first)
        # After swapping top/bottom, patterns are:
        # Square 0 (top-left): pattern [0, 1, 1, 0] at 0° → opens right (→) and bottom (↓) toward center
        # Square 1 (top-right): pattern [0, 0, 1, 1] at 0° → opens left (←) and bottom (↓) toward center
        # Square 2 (bottom-left): pattern [1, 1, 0, 0] at 0° → opens right (→) and top (↑) toward center
        # Square 3 (bottom-right): pattern [1, 0, 0, 1] at 0° → opens left (←) and top (↑) toward center
        # This creates a closed square path: 0→1→3→2→0 with all openings facing center
        closed_square_patterns = [
            [0, 1, 1, 0],  # top-left: right-bottom L (at 0° opens toward center: right and bottom)
            [0, 0, 1, 1],  # top-right: bottom-left L (at 0° opens toward center: left and bottom)
            [1, 1, 0, 0],  # bottom-left: top-right L (at 0° opens toward center: right and top)
            [1, 0, 0, 1],  # bottom-right: left-top L (at 0° opens toward center: left and top)
        ]

        # Use the closed square patterns (ensures final state forms a square)
        patterns = closed_square_patterns.copy()

        # Determine target angles for solved state (all openings face center):
        # All squares at 0° so that pipe openings face toward the center
        target_angles = [0, 0, 0, 0]

        # Determine initial angles based on difficulty
        initial_angles = self._get_initial_angles(difficulty, target_angles)

        squares = []
        for i in range(4):
            squares.append(SquareSpec(
                position=positions[i],
                initial_angle=initial_angles[i],
                target_angle=target_angles[i],
                pipe_pattern=patterns[i],
                index=i
            ))

        return squares

    def _get_initial_angles(self, difficulty: str, target_angles: List[int]) -> List[int]:
        """Get initial rotation angles based on difficulty."""
        if difficulty == "easy":
            # Easy: rotate 1-2 squares
            num_rotations = self.rng.randint(1, 2)
        elif difficulty == "hard":
            # Hard: rotate 3-4 squares
            num_rotations = self.rng.randint(3, 4)
        else:  # medium
            # Medium: rotate 2-3 squares
            num_rotations = self.rng.randint(2, 3)

        initial_angles = target_angles.copy()
        squares_to_rotate = self.rng.sample([0, 1, 2, 3], k=num_rotations)

        for idx in squares_to_rotate:
            # Rotate by 90, 180, or 270 degrees
            rotation = self.rng.choice([90, 180, 270])
            initial_angles[idx] = (initial_angles[idx] + rotation) % 360

        return initial_angles

    def _build_metadata(self, task_id: str, squares: Sequence[SquareSpec], difficulty: str) -> Dict:
        """Build metadata dictionary."""
        return {
            "task_id": task_id,
            "domain": "rotation_puzzle_task",
            "difficulty": difficulty,
            "input_type": "image_pair",
            "output_type": "video",
            "canvas_size": {"width": self.canvas[0], "height": self.canvas[1]},
            "camera": {"view": "top_down", "movement": "static"},
            "num_squares": len(squares),
            "squares": [
                {
                    "index": square.index,
                    "position": [round(square.position[0], 2), round(square.position[1], 2)],
                    "initial_angle": square.initial_angle,
                    "target_angle": square.target_angle,
                    "pipe_pattern": square.pipe_pattern,
                }
                for square in squares
            ],
            "created_at": datetime.now().isoformat(),
        }

    def _build_signature(self, squares: Sequence[SquareSpec]) -> str:
        """Create a deterministic signature to avoid duplicates."""
        parts = []
        for square in sorted(squares, key=lambda s: s.index):
            parts.append(f"{square.index}-{square.initial_angle}-{square.target_angle}-{''.join(map(str, square.pipe_pattern))}")
        return "|".join(parts)


def create_dataset(
    num_samples: int = 10,
    difficulties: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, List[Dict]]:
    """
    Entry point used by VMEvalKit runner.
    """
    generator = RotationPuzzleGenerator()
    rng = random.Random(seed)
    diffs = list(difficulties) if difficulties else ["easy", "medium", "hard"]

    pairs = []
    for idx in range(num_samples):
        difficulty = diffs[idx % len(diffs)]
        task_id = f"rotation_puzzle_{idx:04d}"
        pairs.append(
            generator.generate(
                task_id=task_id,
                difficulty=difficulty,
                seed=rng.randint(0, 10_000_000),
            )
        )

    dataset = {
        "name": "rotation_puzzle_tasks",
        "description": f"Rotation puzzle reasoning tasks ({len(pairs)} pairs)",
        "pairs": pairs,
        "metadata": {
            "total_tasks": len(pairs),
            "difficulties": diffs,
            "canvas": CANVAS,
            "created_at": datetime.now().isoformat(),
        },
        "created_at": datetime.now().isoformat(),
    }
    return dataset

