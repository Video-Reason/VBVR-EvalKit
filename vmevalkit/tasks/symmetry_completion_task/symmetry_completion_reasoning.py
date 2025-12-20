"""
Symmetry Completion task for VMEvalKit.

This task evaluates whether video generation models can complete visual patterns
in grids by filling in missing cells based on pattern rules.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .PROMPTS import get_prompt

Canvas = Tuple[int, int]

CANVAS: Canvas = (768, 512)
DPI = 150

# Grid appearance
GRID_LINE_COLOR = "#cbd5e1"
GRID_LINE_WIDTH = 2
FILL_COLOR = "#1e293b"  # Black for filled cells
EMPTY_COLOR = "white"
BACKGROUND_COLOR = "#f8fafc"

# Pattern types - Focus on symmetry patterns
PATTERN_TYPES = [
    "vertical_symmetry",      # Left-right symmetry (main focus)
    "vertical_symmetry_checkerboard",  # Checkerboard with vertical symmetry
    "vertical_symmetry_stripes",       # Stripes with vertical symmetry
    "vertical_symmetry_increment",     # Increment pattern with vertical symmetry
]

# Grid sizes - Must be even numbers for perfect left-right symmetry
GRID_SIZES = [4, 6, 8, 10, 12]


@dataclass
class PatternSpec:
    """Specification for a symmetry completion task."""
    pattern_type: str
    grid_size: int
    full_pattern: np.ndarray  # Complete pattern (0=empty, 1=filled)
    incomplete_pattern: np.ndarray  # Pattern with missing cells (-1=missing)
    missing_positions: List[Tuple[int, int]]  # List of (row, col) missing positions
    difficulty: str

    def get_signature(self) -> str:
        """Create a unique signature for this pattern."""
        missing_str = ",".join(f"{r},{c}" for r, c in sorted(self.missing_positions))
        pattern_str = "".join(str(self.full_pattern[i, j]) for i in range(self.grid_size) for j in range(self.grid_size))
        return f"{self.pattern_type}-{self.grid_size}-{missing_str}-{pattern_str}"


class SymmetryCompletionRenderer:
    """Renderer for symmetry completion frames."""

    def __init__(self, canvas: Canvas = CANVAS, dpi: int = DPI):
        self.canvas = canvas
        self.dpi = dpi

    def render_start(self, pattern: PatternSpec, path: Path) -> None:
        """Render first frame: incomplete pattern."""
        fig, ax = self._setup_axes()
        self._draw_background(ax)
        self._draw_grid(ax, pattern, show_missing=True)
        self._finalize(fig, path)

    def render_end(self, pattern: PatternSpec, path: Path) -> None:
        """Render final frame: complete pattern."""
        fig, ax = self._setup_axes()
        self._draw_background(ax)
        self._draw_grid(ax, pattern, show_missing=False)
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

    def _draw_grid(self, ax, pattern: PatternSpec, show_missing: bool) -> None:
        """Draw the grid with pattern."""
        w, h = self.canvas
        size = pattern.grid_size
        
        # Calculate cell size to fit in canvas
        max_cell_size = min(w, h) * 0.6 / size
        cell_size = max_cell_size
        
        # Center the grid
        grid_width = size * cell_size
        grid_height = size * cell_size
        start_x = (w - grid_width) / 2
        start_y = (h - grid_height) / 2
        
        # Draw cells
        pattern_to_show = pattern.incomplete_pattern if show_missing else pattern.full_pattern
        
        for i in range(size):
            for j in range(size):
                x = start_x + j * cell_size
                y = start_y + i * cell_size
                
                # Grid cell border
                cell = Rectangle((x, y), cell_size, cell_size,
                               facecolor=EMPTY_COLOR, edgecolor=GRID_LINE_COLOR, 
                               linewidth=GRID_LINE_WIDTH)
                ax.add_patch(cell)
                
                # Fill if pattern indicates
                if pattern_to_show[i, j] == 1:
                    fill = Rectangle((x + 2, y + 2), cell_size - 4, cell_size - 4,
                                    facecolor=FILL_COLOR, edgecolor="none")
                    ax.add_patch(fill)
                # Missing cells (only in first frame) are left empty

    def _finalize(self, fig, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)


class SymmetryCompletionGenerator:
    """Generator for symmetry completion tasks."""

    def __init__(self, canvas: Canvas = CANVAS):
        self.canvas = canvas
        self.renderer = SymmetryCompletionRenderer(canvas)
        self.rng = random.Random()
        self.output_root = Path("data/questions/symmetry_completion_task")
        self._seen_signatures: set[str] = set()

    def generate(
        self,
        task_id: str,
        difficulty: str = "medium",
        seed: Optional[int] = None,
        ensure_unique: bool = True,
    ) -> Dict:
        """Generate a single symmetry completion task."""
        if seed is not None:
            self.rng.seed(seed)

        max_attempts = 25
        attempt = 0
        pattern = None
        signature = None

        while attempt < max_attempts:
            pattern = self._create_pattern(difficulty)
            signature = pattern.get_signature()

            if not ensure_unique or signature not in self._seen_signatures:
                break
            attempt += 1
            self.rng.seed(self.rng.randint(0, 10_000_000))
        else:
            if ensure_unique:
                raise RuntimeError("Failed to generate unique Symmetry Completion sample after multiple attempts.")

        if ensure_unique and signature is not None:
            self._seen_signatures.add(signature)

        question_dir = self.output_root / task_id
        question_dir.mkdir(parents=True, exist_ok=True)
        first_png = question_dir / "first_frame.png"
        final_png = question_dir / "final_frame.png"

        self.renderer.render_start(pattern, first_png)
        self.renderer.render_end(pattern, final_png)

        prompt = get_prompt()
        (question_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

        metadata = self._build_metadata(task_id, pattern, difficulty)
        (question_dir / "question_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        return {
            "id": task_id,
            "prompt": prompt,
            "first_image_path": str(first_png),
            "final_image_path": str(final_png),
            "task_category": "SymmetryCompletion",
            "difficulty": difficulty,
            "symmetry_completion_data": metadata,
            "created_at": datetime.now().isoformat(),
        }

    def _create_pattern(self, difficulty: str) -> PatternSpec:
        """Create a symmetry completion task."""
        # Select grid size based on difficulty (block count)
        # Must be even numbers for perfect left-right symmetry
        if difficulty == "easy":
            grid_size = self.rng.choice([4, 6])  # 16-36 blocks (even only)
        elif difficulty == "hard":
            grid_size = self.rng.choice([8, 10])  # 64-100 blocks (even only)
        else:  # medium
            grid_size = self.rng.choice([6, 8])  # 36-64 blocks (even only)

        # Select pattern type (all are vertical symmetry patterns)
        pattern_type = self.rng.choice(PATTERN_TYPES)

        # Generate full symmetric pattern
        full_pattern = self._generate_full_pattern(pattern_type, grid_size)
        
        # Calculate missing cells based on difficulty
        # We remove cells from the right half (or most of right half)
        total_cells = grid_size * grid_size
        mid = grid_size // 2
        
        if difficulty == "easy":
            # Missing 30-40% of right half
            right_half_cells = grid_size * (grid_size - mid)
            missing_range = (int(right_half_cells * 0.3), int(right_half_cells * 0.4))
        elif difficulty == "hard":
            # Missing 70-80% of right half
            right_half_cells = grid_size * (grid_size - mid)
            missing_range = (int(right_half_cells * 0.7), int(right_half_cells * 0.8))
        else:  # medium
            # Missing 50-60% of right half
            right_half_cells = grid_size * (grid_size - mid)
            missing_range = (int(right_half_cells * 0.5), int(right_half_cells * 0.6))
        
        num_missing = self.rng.randint(missing_range[0], missing_range[1])
        
        # Select missing positions from right half (symmetry completion task)
        right_half_positions = [(i, j) for i in range(grid_size) for j in range(mid, grid_size)]
        
        # Select missing positions from right half
        num_missing = min(num_missing, len(right_half_positions))
        missing_positions = self.rng.sample(right_half_positions, k=num_missing)
        
        # Create incomplete pattern: show left half + symmetry axis, hide right half cells
        incomplete_pattern = full_pattern.copy()
        for r, c in missing_positions:
            incomplete_pattern[r, c] = -1  # -1 means missing

        return PatternSpec(
            pattern_type=pattern_type,
            grid_size=grid_size,
            full_pattern=full_pattern,
            incomplete_pattern=incomplete_pattern,
            missing_positions=missing_positions,
            difficulty=difficulty,
        )

    def _generate_full_pattern(self, pattern_type: str, size: int) -> np.ndarray:
        """Generate a complete vertically symmetric pattern.
        
        Note: size must be even for perfect left-right symmetry.
        """
        assert size % 2 == 0, f"Grid size must be even for symmetry, got {size}"
        
        pattern = np.zeros((size, size), dtype=int)
        mid = size // 2  # For even sizes, this perfectly divides left and right
        
        # Generate left half pattern, then mirror to right half

        if pattern_type == "vertical_symmetry":
            # Simple vertical symmetry: random pattern on left, mirror to right
            # Fill left half with random pattern
            for i in range(size):
                for j in range(mid):
                    pattern[i, j] = self.rng.choice([0, 1])
            # Mirror to right half (perfect symmetry for even sizes)
            for i in range(size):
                for j in range(mid, size):
                    mirror_j = size - 1 - j
                    pattern[i, j] = pattern[i, mirror_j]

        elif pattern_type == "vertical_symmetry_checkerboard":
            # Checkerboard with vertical symmetry
            # Fill left half with checkerboard pattern
            for i in range(size):
                for j in range(mid):
                    pattern[i, j] = (i + j) % 2
            # Mirror to right half (perfect symmetry for even sizes)
            for i in range(size):
                for j in range(mid, size):
                    mirror_j = size - 1 - j
                    pattern[i, j] = pattern[i, mirror_j]

        elif pattern_type == "vertical_symmetry_stripes":
            # Horizontal stripes with vertical symmetry
            # Fill left half with horizontal stripes
            for i in range(size):
                fill = i % 2
                for j in range(mid):
                    pattern[i, j] = fill
            # Mirror to right half (perfect symmetry for even sizes)
            for i in range(size):
                for j in range(mid, size):
                    mirror_j = size - 1 - j
                    pattern[i, j] = pattern[i, mirror_j]

        elif pattern_type == "vertical_symmetry_increment":
            # Row increment with vertical symmetry
            # Left half: incrementing pattern
            for i in range(size):
                num_filled = min(i + 1, mid)
                for j in range(num_filled):
                    pattern[i, j] = 1
            # Mirror to right half (perfect symmetry for even sizes)
            for i in range(size):
                for j in range(mid, size):
                    mirror_j = size - 1 - j
                    pattern[i, j] = pattern[i, mirror_j]
        
        # Verify symmetry (sanity check)
        for i in range(size):
            for j in range(mid):
                mirror_j = size - 1 - j
                assert pattern[i, j] == pattern[i, mirror_j], \
                    f"Symmetry violation at ({i}, {j}) vs ({i}, {mirror_j})"

        return pattern

    def _select_missing_positions_old(self, pattern: np.ndarray, num_missing: int, size: int, pattern_type: str) -> List[Tuple[int, int]]:
        """Select positions to remove, ensuring unique solution."""
        # Strategy: Select positions that ensure unique solution
        # For different pattern types, we use different strategies
        
        all_positions = [(i, j) for i in range(size) for j in range(size)]
        
        # Pattern-specific strategies to ensure uniqueness
        if pattern_type in ["row_increment", "col_increment"]:
            # For increment patterns, ensure each row/col has at least one visible cell
            # This ensures we can determine the increment rule
            missing = []
            if pattern_type == "row_increment":
                # Ensure each row has at least 1 visible cell
                for i in range(size):
                    row_cells = [(i, j) for j in range(size)]
                    visible_in_row = [p for p in row_cells if pattern[p[0], p[1]] == 1]
                    if visible_in_row:
                        # Keep at least one visible cell per row
                        keep = self.rng.choice(visible_in_row)
                        row_cells.remove(keep)
                    missing.extend(self.rng.sample(row_cells, k=min(num_missing // size + 1, len(row_cells))))
            else:  # col_increment
                for j in range(size):
                    col_cells = [(i, j) for i in range(size)]
                    visible_in_col = [p for p in col_cells if pattern[p[0], p[1]] == 1]
                    if visible_in_col:
                        keep = self.rng.choice(visible_in_col)
                        col_cells.remove(keep)
                    missing.extend(self.rng.sample(col_cells, k=min(num_missing // size + 1, len(col_cells))))
            # Limit to num_missing
            missing = self.rng.sample(missing, k=min(num_missing, len(missing)))
        elif pattern_type in ["row_position_shift", "col_position_shift"]:
            # For position shift, ensure we can see the shift pattern
            # Keep at least 2-3 rows/cols visible to see the shift
            missing = []
            if pattern_type == "row_position_shift":
                # Keep first and last rows visible, can remove from middle
                keep_rows = {0, size - 1}
                for i in range(size):
                    if i not in keep_rows:
                        row_cells = [(i, j) for j in range(size)]
                        missing.extend(self.rng.sample(row_cells, k=min(2, len(row_cells))))
            else:  # col_position_shift
                keep_cols = {0, size - 1}
                for j in range(size):
                    if j not in keep_cols:
                        col_cells = [(i, j) for i in range(size)]
                        missing.extend(self.rng.sample(col_cells, k=min(2, len(col_cells))))
            missing = self.rng.sample(missing, k=min(num_missing, len(missing)))
        elif pattern_type in ["symmetry_horizontal", "symmetry_vertical"]:
            # For symmetry, ensure we can see the symmetry axis
            # Keep key positions that define symmetry
            missing = []
            if pattern_type == "symmetry_horizontal":
                # Keep top half mostly visible, can remove from bottom
                mid = size // 2
                for i in range(mid + 1, size):
                    row_cells = [(i, j) for j in range(size)]
                    missing.extend(self.rng.sample(row_cells, k=min(size // 2, len(row_cells))))
            else:  # symmetry_vertical
                mid = size // 2
                for j in range(mid + 1, size):
                    col_cells = [(i, j) for i in range(size)]
                    missing.extend(self.rng.sample(col_cells, k=min(size // 2, len(col_cells))))
            missing = self.rng.sample(missing, k=min(num_missing, len(missing)))
        else:
            # For other patterns, use random selection but ensure sufficient information
            # Ensure at least 40-50% of cells are visible to determine pattern
            missing = self.rng.sample(all_positions, k=num_missing)
        
        # Verify uniqueness by checking if pattern can be uniquely determined
        # This is a simplified check - in practice, we'd need to verify against all pattern types
        return missing
    
    def _verify_uniqueness(self, incomplete_pattern: np.ndarray, full_pattern: np.ndarray, 
                          pattern_type: str, size: int) -> bool:
        """Verify that the incomplete pattern has a unique solution."""
        # Count visible cells
        visible_count = np.sum(incomplete_pattern >= 0)
        total_cells = size * size
        visible_ratio = visible_count / total_cells
        
        # Pattern-specific uniqueness requirements
        if pattern_type == "checkerboard":
            # Need to see at least 2 adjacent cells to determine checkerboard pattern
            if visible_ratio < 0.4:
                return False
            # Check if we can see the alternating pattern
            for i in range(size - 1):
                for j in range(size - 1):
                    if (incomplete_pattern[i, j] >= 0 and incomplete_pattern[i, j+1] >= 0 and
                        incomplete_pattern[i+1, j] >= 0):
                        # We can see the checkerboard pattern
                        return True
            return False
        
        elif pattern_type in ["horizontal_stripes", "vertical_stripes"]:
            # Need to see at least 2 rows/cols to determine stripe pattern
            if visible_ratio < 0.4:
                return False
            if pattern_type == "horizontal_stripes":
                # Need at least 2 rows with visible cells
                rows_with_visible = sum(1 for i in range(size) if np.sum(incomplete_pattern[i, :] >= 0) > 0)
                return rows_with_visible >= 2
            else:  # vertical_stripes
                cols_with_visible = sum(1 for j in range(size) if np.sum(incomplete_pattern[:, j] >= 0) > 0)
                return cols_with_visible >= 2
        
        elif pattern_type == "row_increment":
            # Each row must have at least one visible cell to see the increment
            if visible_ratio < 0.4:
                return False
            for i in range(size):
                if np.sum(incomplete_pattern[i, :] >= 0) == 0:
                    return False
                # Need to see the increment: at least one filled cell per row
                if np.sum(incomplete_pattern[i, :] == 1) == 0:
                    return False
            return True
        
        elif pattern_type == "col_increment":
            # Each column must have at least one visible cell to see the increment
            if visible_ratio < 0.4:
                return False
            for j in range(size):
                if np.sum(incomplete_pattern[:, j] >= 0) == 0:
                    return False
                # Need to see the increment: at least one filled cell per column
                if np.sum(incomplete_pattern[:, j] == 1) == 0:
                    return False
            return True
        
        elif pattern_type in ["row_position_shift", "col_position_shift"]:
            # Need to see at least 2-3 rows/cols to see the shift pattern
            if visible_ratio < 0.5:
                return False
            if pattern_type == "row_position_shift":
                rows_with_visible = sum(1 for i in range(size) if np.sum(incomplete_pattern[i, :] >= 0) > 0)
                return rows_with_visible >= 2
            else:  # col_position_shift
                cols_with_visible = sum(1 for j in range(size) if np.sum(incomplete_pattern[:, j] >= 0) > 0)
                return cols_with_visible >= 2
        
        elif pattern_type in ["symmetry_horizontal", "symmetry_vertical"]:
            # Need sufficient information in one half to determine symmetry
            if visible_ratio < 0.5:
                return False
            if pattern_type == "symmetry_horizontal":
                mid = size // 2
                top_visible = np.sum(incomplete_pattern[:mid, :] >= 0)
                return top_visible >= mid * size * 0.4
            else:  # symmetry_vertical
                mid = size // 2
                left_visible = np.sum(incomplete_pattern[:, :mid] >= 0)
                return left_visible >= size * mid * 0.4
        
        elif pattern_type == "diagonal":
            # Need to see at least part of both diagonals
            if visible_ratio < 0.4:
                return False
            main_diag_visible = sum(1 for i in range(size) if incomplete_pattern[i, i] >= 0)
            anti_diag_visible = sum(1 for i in range(size) if incomplete_pattern[i, size-1-i] >= 0)
            return main_diag_visible >= 1 or anti_diag_visible >= 1
        
        elif pattern_type == "border":
            # Need to see at least part of the border
            if visible_ratio < 0.4:
                return False
            border_visible = (np.sum(incomplete_pattern[0, :] >= 0) +
                            np.sum(incomplete_pattern[-1, :] >= 0) +
                            np.sum(incomplete_pattern[:, 0] >= 0) +
                            np.sum(incomplete_pattern[:, -1] >= 0))
            return border_visible >= 4
        
        elif pattern_type == "cross":
            # Need to see at least part of the cross
            if visible_ratio < 0.4:
                return False
            center = size // 2
            cross_visible = (np.sum(incomplete_pattern[center, :] >= 0) +
                           np.sum(incomplete_pattern[:, center] >= 0))
            return cross_visible >= 2
        
        elif pattern_type == "combined_row_col":
            # Need sufficient information to see both row and col patterns
            if visible_ratio < 0.5:
                return False
            # Check if we can see row increment
            rows_with_visible = sum(1 for i in range(size) if np.sum(incomplete_pattern[i, :] >= 0) > 0)
            return rows_with_visible >= 2
        
        else:
            # Default: ensure at least 40% visible for other patterns
            return visible_ratio >= 0.4

    def _build_metadata(self, task_id: str, pattern: PatternSpec, difficulty: str) -> Dict:
        """Build metadata dictionary."""
        return {
            "task_id": task_id,
            "domain": "symmetry_completion_task",
            "difficulty": difficulty,
            "input_type": "image_pair",
            "output_type": "video",
            "canvas_size": {"width": self.canvas[0], "height": self.canvas[1]},
            "camera": {"view": "top_down", "movement": "static"},
            "pattern_type": pattern.pattern_type,
            "grid_size": pattern.grid_size,
            "num_missing": len(pattern.missing_positions),
            "missing_positions": [[r, c] for r, c in pattern.missing_positions],
            "full_pattern": pattern.full_pattern.tolist(),
            "created_at": datetime.now().isoformat(),
        }

    def _build_signature(self, pattern: PatternSpec) -> str:
        """Create a deterministic signature to avoid duplicates."""
        return pattern.get_signature()


def create_dataset(
    num_samples: int = 10,
    difficulties: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, List[Dict]]:
    """
    Entry point used by VMEvalKit runner.
    """
    generator = SymmetryCompletionGenerator()
    rng = random.Random(seed)
    diffs = list(difficulties) if difficulties else ["easy", "medium", "hard"]

    pairs = []
    for idx in range(num_samples):
        difficulty = diffs[idx % len(diffs)]
        task_id = f"symmetry_completion_{idx:04d}"
        pairs.append(
            generator.generate(
                task_id=task_id,
                difficulty=difficulty,
                seed=rng.randint(0, 10_000_000),
            )
        )

    dataset = {
        "name": "symmetry_completion_tasks",
        "description": f"Symmetry completion reasoning tasks ({len(pairs)} pairs)",
        "pairs": pairs,
        "metadata": {
            "total_tasks": len(pairs),
            "difficulties": diffs,
            "canvas": CANVAS,
            "pattern_types": PATTERN_TYPES,
            "grid_sizes": GRID_SIZES,
            "created_at": datetime.now().isoformat(),
        },
        "created_at": datetime.now().isoformat(),
    }
    return dataset

