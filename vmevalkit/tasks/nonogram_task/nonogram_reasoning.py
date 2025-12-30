"""
Nonogram Puzzle task for VMEvalKit.

This task evaluates whether video generation models can solve nonogram puzzles
by filling in grid cells according to row and column hints. Models must use
constraint satisfaction reasoning to determine which cells should be filled.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
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
CELL_SIZE = 35
GRID_LINE_COLOR = "#cbd5e1"
GRID_LINE_WIDTH = 2
FILL_COLOR = "#1e293b"
EMPTY_COLOR = "white"
BACKGROUND_COLOR = "#f8fafc"
HINT_COLOR = "#64748b"


@dataclass
class NonogramPattern:
    """Specification for a nonogram pattern."""
    grid_size: int  # NxN grid
    pattern: np.ndarray  # 2D array: 0=empty, 1=filled
    row_hints: List[List[int]]  # Hints for each row
    col_hints: List[List[int]]  # Hints for each column
    pattern_type: str  # Type of pattern (cross, square, circle, etc.)


class NonogramRenderer:
    """Renderer for nonogram puzzle frames."""

    def __init__(self, canvas: Canvas = CANVAS, dpi: int = DPI):
        self.canvas = canvas
        self.dpi = dpi

    def render_start(self, pattern: NonogramPattern, path: Path) -> None:
        """Render first frame: blank grid with hints."""
        fig, ax = self._setup_axes()
        self._draw_background(ax)
        self._draw_nonogram(ax, pattern, show_solution=False)
        self._finalize(fig, path)

    def render_end(self, pattern: NonogramPattern, path: Path) -> None:
        """Render final frame: complete solution."""
        fig, ax = self._setup_axes()
        self._draw_background(ax)
        self._draw_nonogram(ax, pattern, show_solution=True)
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

    def _draw_nonogram(self, ax, pattern: NonogramPattern, show_solution: bool = False) -> None:
        """Draw nonogram grid with hints."""
        size = pattern.grid_size
        cell_size = CELL_SIZE
        
        # Calculate hint area sizes - ensure enough space
        max_row_hint_len = max(len(",".join(map(str, hints))) for hints in pattern.row_hints)
        max_col_hint_count = max(len(hints) for hints in pattern.col_hints)
        
        # Hint area dimensions
        row_hint_width = max_row_hint_len * 7 + 30  # Enough space for row hints
        col_hint_height = max_col_hint_count * 12 + 20  # Enough space for column hints
        
        # Position grid (centered, with hints outside)
        grid_start_x = row_hint_width + 20
        grid_start_y = col_hint_height + 20
        
        # Draw column hints (above grid, outside)
        for j in range(size):
            x = grid_start_x + j * cell_size + cell_size / 2
            hints = pattern.col_hints[j]
            hint_text = "\n".join(map(str, hints)) if hints != [0] else "0"
            # Position above grid with clear spacing
            ax.text(x, grid_start_y - 10, hint_text, ha="center", va="bottom",
                    fontsize=9, color=HINT_COLOR, weight="bold", family="monospace")
        
        # Draw row hints (left of grid, outside)
        for i in range(size):
            y = grid_start_y + i * cell_size + cell_size / 2
            hints = pattern.row_hints[i]
            hint_text = ",".join(map(str, hints)) if hints != [0] else "0"
            # Position left of grid with clear spacing
            ax.text(grid_start_x - 10, y, hint_text, ha="right", va="center",
                    fontsize=9, color=HINT_COLOR, weight="bold", family="monospace")
        
        # Draw grid cells
        for i in range(size):
            for j in range(size):
                x = grid_start_x + j * cell_size
                y = grid_start_y + i * cell_size
                
                # Grid cell
                cell = Rectangle((x, y), cell_size, cell_size,
                               facecolor=EMPTY_COLOR, edgecolor=GRID_LINE_COLOR,
                               linewidth=GRID_LINE_WIDTH)
                ax.add_patch(cell)
                
                # Fill if solution is shown and pattern indicates
                if show_solution and pattern.pattern[i, j] == 1:
                    fill = Rectangle((x + 2, y + 2), cell_size - 4, cell_size - 4,
                                   facecolor=FILL_COLOR, edgecolor="none")
                    ax.add_patch(fill)

    def _finalize(self, fig, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)


class NonogramGenerator:
    """Generator for nonogram puzzle tasks."""

    def __init__(self, canvas: Canvas = CANVAS):
        self.canvas = canvas
        self.renderer = NonogramRenderer(canvas)
        self.rng = random.Random()
        self.output_root = Path("data/questions/nonogram_task")
        self._seen_signatures: set[str] = set()

    def generate(
        self,
        task_id: str,
        difficulty: str = "medium",
        grid_size: Optional[int] = None,
        seed: Optional[int] = None,
        ensure_unique: bool = True,
    ) -> Dict:
        """Generate a single nonogram puzzle task."""
        if seed is not None:
            self.rng.seed(seed)
        
        size = grid_size or self._grid_size_for_difficulty(difficulty)
        pattern = self._create_pattern(size, difficulty)
        
        # Check uniqueness using full pattern array
        signature = None
        if ensure_unique:
            signature = self._build_signature(pattern)
            max_attempts = 300  # Increased attempts for better uniqueness
            attempts = 0
            original_size = size
            while signature in self._seen_signatures and attempts < max_attempts:
                # Try different approaches to ensure uniqueness
                if attempts % 30 == 0 and attempts > 0:
                    # Every 30 attempts, try a different size
                    size = self._grid_size_for_difficulty(difficulty)
                elif attempts % 10 == 0:
                    # Every 10 attempts, reset size to original
                    size = original_size
                
                # Add some randomness to pattern generation by varying seed
                pattern = self._create_pattern(size, difficulty)
                signature = self._build_signature(pattern)
                attempts += 1
            if signature in self._seen_signatures:
                raise RuntimeError(f"Failed to generate unique Nonogram sample after {max_attempts} attempts.")
        
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
        (question_dir / "question_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "id": task_id,
            "prompt": prompt,
            "first_image_path": str(first_png),
            "final_image_path": str(final_png),
            "task_category": "Nonogram",
            "difficulty": difficulty,
            "nonogram_data": metadata,
            "created_at": datetime.now().isoformat(),
        }

    def _grid_size_for_difficulty(self, difficulty: str) -> int:
        """Determine grid size based on difficulty."""
        if difficulty == "easy":
            return self.rng.choice([5, 6])  # Small grids: 5x5 to 6x6
        if difficulty == "hard":
            return self.rng.choice([12, 15])  # Large grids: 12x12 to 15x15
        return self.rng.choice([7, 8, 10])  # Medium grids: 7x7 to 10x10

    def _create_pattern(self, size: int, difficulty: str) -> NonogramPattern:
        """Generate a nonogram pattern."""
        max_attempts = 50
        attempts = 0
        
        while attempts < max_attempts:
            pattern_type = self._choose_pattern_type(size, difficulty)
            
            if pattern_type == "cross":
                pattern_array = self._generate_cross(size)
            elif pattern_type == "square":
                pattern_array = self._generate_square(size)
            elif pattern_type == "circle":
                pattern_array = self._generate_circle(size)
            elif pattern_type == "checkerboard":
                pattern_array = self._generate_checkerboard(size)
            elif pattern_type == "letter_t":
                pattern_array = self._generate_letter_t(size)
            elif pattern_type == "diagonal":
                pattern_array = self._generate_diagonal(size)
            else:  # random
                pattern_array = self._generate_random(size, difficulty)
            
            # Ensure no completely empty rows or columns
            if self._has_empty_row_or_column(pattern_array):
                attempts += 1
                continue
            
            row_hints, col_hints = self._calculate_hints(pattern_array)
            
            # Ensure no row or column has all zeros in hints
            if self._has_all_zero_hints(row_hints) or self._has_all_zero_hints(col_hints):
                attempts += 1
                continue
            
            return NonogramPattern(
                grid_size=size,
                pattern=pattern_array,
                row_hints=row_hints,
                col_hints=col_hints,
                pattern_type=pattern_type
            )
        
        # If we couldn't generate a valid pattern, add at least one cell to each row/column
        pattern_array = self._ensure_no_empty_rows_columns(pattern_array)
        row_hints, col_hints = self._calculate_hints(pattern_array)
        
        return NonogramPattern(
            grid_size=size,
            pattern=pattern_array,
            row_hints=row_hints,
            col_hints=col_hints,
            pattern_type=pattern_type
        )
    
    def _has_empty_row_or_column(self, pattern: np.ndarray) -> bool:
        """Check if pattern has any completely empty rows or columns."""
        size = pattern.shape[0]
        # Check rows
        for i in range(size):
            if np.sum(pattern[i, :]) == 0:
                return True
        # Check columns
        for j in range(size):
            if np.sum(pattern[:, j]) == 0:
                return True
        return False
    
    def _has_all_zero_hints(self, hints: List[List[int]]) -> bool:
        """Check if all hints are [0] (completely empty)."""
        return all(h == [0] for h in hints)
    
    def _ensure_no_empty_rows_columns(self, pattern: np.ndarray) -> np.ndarray:
        """Ensure no row or column is completely empty by adding at least one cell."""
        size = pattern.shape[0]
        # Check and fix rows
        for i in range(size):
            if np.sum(pattern[i, :]) == 0:
                # Add a random cell in this row
                j = self.rng.randint(0, size - 1)
                pattern[i, j] = 1
        # Check and fix columns
        for j in range(size):
            if np.sum(pattern[:, j]) == 0:
                # Add a random cell in this column
                i = self.rng.randint(0, size - 1)
                pattern[i, j] = 1
        return pattern

    def _choose_pattern_type(self, size: int, difficulty: str) -> str:
        """Choose pattern type based on grid size and difficulty."""
        if difficulty == "easy":
            # Easy: simple, recognizable patterns
            if size <= 6:
                return self.rng.choice(["cross", "square", "diagonal"])
            else:
                return self.rng.choice(["cross", "square", "checkerboard"])
        elif difficulty == "hard":
            # Hard: complex patterns or random
            if size >= 12:
                return self.rng.choice(["circle", "letter_t", "random"])
            else:
                return self.rng.choice(["circle", "letter_t", "checkerboard", "random"])
        else:  # medium
            # Medium: moderate complexity patterns
            if size <= 8:
                return self.rng.choice(["cross", "square", "circle", "checkerboard"])
            else:
                return self.rng.choice(["circle", "letter_t", "checkerboard"])

    def _generate_cross(self, size: int) -> np.ndarray:
        """Generate a cross pattern with optional variations."""
        pattern = np.zeros((size, size), dtype=int)
        center = size // 2
        
        # Add some randomness: sometimes offset the center slightly
        if self.rng.random() < 0.3 and size > 5:
            offset = self.rng.choice([-1, 1])
            center = max(1, min(size - 2, center + offset))
        
        # Vertical line
        pattern[:, center] = 1
        # Horizontal line
        pattern[center, :] = 1
        
        # Occasionally add small variations (10% chance)
        if self.rng.random() < 0.1 and size >= 7:
            # Add a small dot at one corner
            corner = self.rng.choice([(0, 0), (0, size-1), (size-1, 0), (size-1, size-1)])
            pattern[corner[0], corner[1]] = 1
        
        return pattern

    def _generate_square(self, size: int) -> np.ndarray:
        """Generate a hollow square pattern with optional variations."""
        pattern = np.zeros((size, size), dtype=int)
        # Border only
        pattern[0, :] = 1  # Top
        pattern[-1, :] = 1  # Bottom
        pattern[:, 0] = 1  # Left
        pattern[:, -1] = 1  # Right
        
        # Occasionally add a diagonal or inner decoration (15% chance)
        if self.rng.random() < 0.15 and size >= 7:
            if self.rng.random() < 0.5:
                # Add main diagonal
                for i in range(size):
                    pattern[i, i] = 1
            else:
                # Add anti-diagonal
                for i in range(size):
                    pattern[i, size - 1 - i] = 1
        
        return pattern

    def _generate_circle(self, size: int) -> np.ndarray:
        """Generate a circular pattern (approximated) with variations."""
        pattern = np.zeros((size, size), dtype=int)
        center = size / 2
        
        # Vary radius slightly for uniqueness
        base_radius = min(size, size) / 2 - 1
        radius_variation = self.rng.uniform(-0.3, 0.3)
        radius = base_radius + radius_variation
        
        # Vary thickness for uniqueness
        thickness = self.rng.uniform(0.6, 1.0)
        
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                if abs(dist - radius) < thickness:
                    pattern[i, j] = 1
        return pattern

    def _generate_checkerboard(self, size: int) -> np.ndarray:
        """Generate a checkerboard pattern."""
        pattern = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if (i + j) % 2 == 0:
                    pattern[i, j] = 1
        return pattern

    def _generate_letter_t(self, size: int) -> np.ndarray:
        """Generate a letter T pattern with variations."""
        pattern = np.zeros((size, size), dtype=int)
        # Top horizontal bar
        bar_width = max(3, size // 3)
        # Vary bar width slightly
        bar_width += self.rng.choice([-1, 0, 1])
        bar_width = max(2, min(size - 2, bar_width))
        start_col = (size - bar_width) // 2
        pattern[0, start_col:start_col + bar_width] = 1
        
        # Vertical stem - sometimes offset slightly
        stem_col = size // 2
        if self.rng.random() < 0.3 and size > 6:
            stem_col += self.rng.choice([-1, 1])
            stem_col = max(1, min(size - 2, stem_col))
        pattern[:, stem_col] = 1
        return pattern

    def _generate_diagonal(self, size: int) -> np.ndarray:
        """Generate a diagonal line pattern with variations."""
        pattern = np.zeros((size, size), dtype=int)
        
        # Choose diagonal type
        diagonal_type = self.rng.choice(["main", "anti", "both"])
        
        if diagonal_type == "main" or diagonal_type == "both":
            for i in range(size):
                pattern[i, i] = 1
        if diagonal_type == "anti" or diagonal_type == "both":
            for i in range(size):
                pattern[i, size - 1 - i] = 1
        
        return pattern

    def _generate_random(self, size: int, difficulty: str) -> np.ndarray:
        """Generate a random pattern with controlled density and hint complexity."""
        pattern = np.zeros((size, size), dtype=int)
        
        # Density based on difficulty
        if difficulty == "easy":
            density = self.rng.uniform(0.25, 0.35)  # 25-35% filled (simpler hints)
        elif difficulty == "hard":
            density = self.rng.uniform(0.55, 0.70)  # 55-70% filled (complex hints)
        else:
            density = self.rng.uniform(0.40, 0.50)  # 40-50% filled (moderate hints)
        
        # For easy difficulty, prefer patterns with fewer, longer blocks
        # For hard difficulty, allow more fragmented patterns
        if difficulty == "easy":
            # Create patterns with fewer, longer consecutive blocks
            num_blocks = self.rng.randint(2, max(3, size // 2))
            total_filled = 0
            target_filled = int(size * size * density)
            
            for _ in range(num_blocks):
                if total_filled >= target_filled * 0.9:  # Stop if we're close to target
                    break
                # Create a horizontal or vertical block
                if self.rng.random() < 0.5:
                    # Horizontal block
                    row = self.rng.randint(0, size - 1)
                    col_start = self.rng.randint(0, max(1, size - 3))
                    block_length = self.rng.randint(2, min(4, size - col_start))
                    pattern[row, col_start:col_start + block_length] = 1
                    total_filled += block_length
                else:
                    # Vertical block
                    col = self.rng.randint(0, size - 1)
                    row_start = self.rng.randint(0, max(1, size - 3))
                    block_length = self.rng.randint(2, min(4, size - row_start))
                    pattern[row_start:row_start + block_length, col] = 1
                    total_filled += block_length
        else:
            # For medium/hard: more random distribution
            num_filled = int(size * size * density)
            positions = self.rng.sample(range(size * size), num_filled)
            for pos in positions:
                i = pos // size
                j = pos % size
                pattern[i, j] = 1
        
        return pattern

    def _calculate_hints(self, pattern: np.ndarray) -> Tuple[List[List[int]], List[List[int]]]:
        """Calculate row and column hints from pattern."""
        size = pattern.shape[0]
        
        # Row hints
        row_hints = []
        for i in range(size):
            row = pattern[i, :]
            hints = []
            count = 0
            for j in range(size):
                if row[j] == 1:
                    count += 1
                else:
                    if count > 0:
                        hints.append(count)
                        count = 0
            if count > 0:
                hints.append(count)
            row_hints.append(hints if hints else [0])
        
        # Column hints
        col_hints = []
        for j in range(size):
            col = pattern[:, j]
            hints = []
            count = 0
            for i in range(size):
                if col[i] == 1:
                    count += 1
                else:
                    if count > 0:
                        hints.append(count)
                        count = 0
            if count > 0:
                hints.append(count)
            col_hints.append(hints if hints else [0])
        
        return row_hints, col_hints

    def _build_signature(self, pattern: NonogramPattern) -> str:
        """Build a unique signature for the pattern."""
        # Use pattern type, grid size, and pattern array as string for exact matching
        # Convert pattern to a compact string representation for uniqueness
        pattern_str = ''.join(str(int(x)) for x in pattern.pattern.flatten())
        return f"{pattern.pattern_type}_{pattern.grid_size}_{pattern_str}"

    def _build_metadata(self, task_id: str, pattern: NonogramPattern, difficulty: str) -> Dict:
        """Build metadata dictionary."""
        return {
            "task_id": task_id,
            "domain": "nonogram_task",
            "difficulty": difficulty,
            "grid_size": pattern.grid_size,
            "pattern_type": pattern.pattern_type,
            "row_hints": pattern.row_hints,
            "col_hints": pattern.col_hints,
            "filled_cells": int(np.sum(pattern.pattern)),
            "total_cells": pattern.grid_size * pattern.grid_size,
            "fill_percentage": float(np.sum(pattern.pattern)) / (pattern.grid_size * pattern.grid_size) * 100,
            "canvas_size": list(self.canvas),
            "camera": {
                "type": "fixed",
                "view": "top-down",
                "zoom": 1.0
            }
        }


def create_dataset(
    num_samples: int = 50,
    difficulties: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Dict:
    """
    Create a dataset of nonogram puzzle tasks with balanced difficulty distribution.
    
    Args:
        num_samples: Number of tasks to generate
        difficulties: List of difficulty levels (default: ["easy", "medium", "hard"])
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with "name" and "pairs" keys, where "pairs" is a list of task dictionaries
    """
    if difficulties is None:
        difficulties = ["easy", "medium", "hard"]
    
    generator = NonogramGenerator()
    if seed is not None:
        generator.rng.seed(seed)
    
    # Balanced difficulty distribution: 40% easy, 40% medium, 20% hard
    difficulty_weights = {
        "easy": 0.4,
        "medium": 0.4,
        "hard": 0.2
    }
    
    # Normalize weights for available difficulties
    available_weights = {d: difficulty_weights.get(d, 1.0/len(difficulties)) for d in difficulties}
    total_weight = sum(available_weights.values())
    normalized_weights = {d: w/total_weight for d, w in available_weights.items()}
    
    # Create weighted choices
    difficulty_list = list(normalized_weights.keys())
    weight_list = list(normalized_weights.values())
    
    pairs = []
    for i in range(num_samples):
        difficulty = generator.rng.choices(difficulty_list, weights=weight_list)[0]
        task_id = f"nonogram_{i:04d}"
        
        try:
            task_dict = generator.generate(
                task_id=task_id,
                difficulty=difficulty,
                seed=None,  # Use generator's RNG state
                ensure_unique=True
            )
            pairs.append(task_dict)
        except RuntimeError as e:
            print(f"Warning: {e}")
            continue
    
    return {
        "name": "nonogram_task",
        "pairs": pairs
    }

