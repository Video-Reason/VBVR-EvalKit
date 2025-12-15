"""
Dot to Dot Puzzle task for VMEvalKit.

This task evaluates whether video generation models can connect numbered dots
in sequence to reveal a complete image. Models must draw lines between consecutive
dots while maintaining a fixed camera view.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

from .PROMPTS import get_prompt

Canvas = Tuple[int, int]
Point = Tuple[float, float]

CANVAS: Canvas = (768, 512)
DPI = 150

# Dot appearance
DOT_RADIUS = 8
DOT_COLOR = "#2563eb"  # Bright blue for dots
DOT_BORDER_COLOR = "#1e40af"  # Darker blue border
LINE_COLOR = "#dc2626"  # Red for lines (high contrast)
LINE_WIDTH = 3
NUMBER_COLOR = "#ffffff"  # White numbers on blue dots
FONT_SIZE = 4.5  # Smaller font size for numbers in dots (50% of previous size)
FONT_WEIGHT = "bold"


@dataclass
class DotSpec:
    """Specification for a single numbered dot."""
    number: int
    position: Point
    
    @property
    def x(self) -> float:
        return self.position[0]
    
    @property
    def y(self) -> float:
        return self.position[1]


class DotToDotRenderer:
    """Renderer for dot-to-dot puzzle frames."""

    def __init__(self, canvas: Canvas = CANVAS, dpi: int = DPI):
        self.canvas = canvas
        self.dpi = dpi

    def render_start(self, dots: Sequence[DotSpec], path: Path) -> None:
        """Render first frame: dots with numbers, no lines."""
        fig, ax = self._setup_axes()
        self._draw_background(ax)
        for dot in dots:
            self._draw_dot(ax, dot, show_number=True)
        self._finalize(fig, path)

    def render_end(self, dots: Sequence[DotSpec], path: Path) -> None:
        """Render final frame: dots connected with lines."""
        fig, ax = self._setup_axes()
        self._draw_background(ax)
        # Draw lines first (behind dots)
        self._draw_lines(ax, dots)
        # Draw dots on top
        for dot in dots:
            self._draw_dot(ax, dot, show_number=True)
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
        bg = "#f8fafc"  # Light gray background for better contrast
        ax.add_patch(FancyBboxPatch((0, 0), w, h, facecolor=bg, edgecolor="none", boxstyle="round,pad=0"))

    def _draw_dot(self, ax, dot: DotSpec, show_number: bool = True) -> None:
        """Draw a single dot with optional number label."""
        x, y = dot.x, dot.y
        # Draw dot circle with border
        circle = Circle((x, y), DOT_RADIUS, facecolor=DOT_COLOR, edgecolor=DOT_BORDER_COLOR, linewidth=2)
        ax.add_patch(circle)
        
        # Draw number label
        if show_number:
            ax.text(
                x, y,
                str(dot.number),
                fontsize=FONT_SIZE,
                fontweight=FONT_WEIGHT,
                color=NUMBER_COLOR,
                ha="center",
                va="center",
                zorder=10
            )

    def _draw_lines(self, ax, dots: Sequence[DotSpec]) -> None:
        """Draw lines connecting dots in sequence."""
        if len(dots) < 2:
            return
        
        # Sort dots by number to ensure correct order
        sorted_dots = sorted(dots, key=lambda d: d.number)
        
        # Draw lines between consecutive dots
        for i in range(len(sorted_dots) - 1):
            dot1 = sorted_dots[i]
            dot2 = sorted_dots[i + 1]
            ax.plot(
                [dot1.x, dot2.x],
                [dot1.y, dot2.y],
                color=LINE_COLOR,
                linewidth=LINE_WIDTH,
                zorder=1
            )

    def _finalize(self, fig, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)


class DotToDotGenerator:
    """Generator for dot-to-dot puzzle tasks."""

    def __init__(self, canvas: Canvas = CANVAS):
        self.canvas = canvas
        self.renderer = DotToDotRenderer(canvas)
        self.rng = random.Random()
        self.output_root = Path("data/questions/dot_to_dot_task")
        self._seen_signatures: set[str] = set()

    def generate(
        self,
        task_id: str,
        difficulty: str = "medium",
        num_dots: Optional[int] = None,
        seed: Optional[int] = None,
        ensure_unique: bool = True,
    ) -> Dict:
        """Generate a single dot-to-dot puzzle task."""
        if seed is not None:
            self.rng.seed(seed)
        
        count = num_dots or self._dots_for_difficulty(difficulty)
        dots = self._create_dot_pattern(count)
        
        # Check uniqueness
        signature = None
        if ensure_unique:
            signature = self._build_signature(dots)
            max_attempts = 25
            attempts = 0
            while signature in self._seen_signatures and attempts < max_attempts:
                dots = self._create_dot_pattern(count)
                signature = self._build_signature(dots)
                attempts += 1
            if signature in self._seen_signatures:
                raise RuntimeError("Failed to generate unique Dot to Dot sample after multiple attempts.")
        
        if ensure_unique and signature is not None:
            self._seen_signatures.add(signature)

        question_dir = self.output_root / task_id
        question_dir.mkdir(parents=True, exist_ok=True)
        first_png = question_dir / "first_frame.png"
        final_png = question_dir / "final_frame.png"

        self.renderer.render_start(dots, first_png)
        self.renderer.render_end(dots, final_png)

        prompt = self._format_prompt(dots, difficulty)
        (question_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

        metadata = self._build_metadata(task_id, dots, difficulty)
        (question_dir / "question_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "id": task_id,
            "prompt": prompt,
            "first_image_path": str(first_png),
            "final_image_path": str(final_png),
            "task_category": "DotToDot",
            "difficulty": difficulty,
            "dot_to_dot_data": metadata,
            "created_at": datetime.now().isoformat(),
        }

    def _dots_for_difficulty(self, difficulty: str) -> int:
        """Determine number of dots based on difficulty. Maximum 20 dots."""
        if difficulty == "easy":
            return self.rng.randint(5, 10)  # Simple: 5-10 dots
        if difficulty == "hard":
            return self.rng.randint(15, 20)  # Hard: 15-20 dots (max 20)
        return self.rng.randint(10, 15)  # Medium: 10-15 dots

    def _create_dot_pattern(self, num_dots: int) -> List[DotSpec]:
        """Generate a pattern of dots forming a recognizable shape."""
        w, h = self.canvas
        margin = 80
        
        # For simple patterns (10 dots or less), use simpler shapes
        if num_dots <= 10:
            pattern_type = self.rng.choice(["circle", "triangle", "square", "star"])
        else:
            # For more complex patterns, include all shape types
            pattern_type = self.rng.choice(["star", "heart", "circle", "triangle", "square", "spiral"])
        
        dots: List[DotSpec] = []
        
        if pattern_type == "star":
            dots = self._generate_star(num_dots, w, h, margin)
        elif pattern_type == "heart":
            dots = self._generate_heart(num_dots, w, h, margin)
        elif pattern_type == "circle":
            dots = self._generate_circle(num_dots, w, h, margin)
        elif pattern_type == "triangle":
            dots = self._generate_triangle(num_dots, w, h, margin)
        elif pattern_type == "square":
            dots = self._generate_square(num_dots, w, h, margin)
        else:  # spiral
            dots = self._generate_spiral(num_dots, w, h, margin)
        
        # Ensure no overlapping dots and each has unique position
        dots = self._ensure_unique_positions(dots)
        
        return dots
    
    def _ensure_unique_positions(self, dots: List[DotSpec]) -> List[DotSpec]:
        """Ensure each dot has a unique position with minimum distance and within canvas bounds."""
        min_distance = DOT_RADIUS * 3  # Minimum distance between dot centers
        w, h = self.canvas
        padding = DOT_RADIUS + 5  # Padding from canvas edges
        
        unique_dots: List[DotSpec] = []
        used_positions: List[Point] = []
        
        for dot in dots:
            x, y = dot.x, dot.y
            
            # Clamp initial position to canvas bounds
            x = max(padding, min(w - padding, x))
            y = max(padding, min(h - padding, y))
            
            # Check if this position is too close to existing dots
            too_close = False
            for used_x, used_y in used_positions:
                distance = math.sqrt((x - used_x) ** 2 + (y - used_y) ** 2)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                unique_dots.append(DotSpec(number=dot.number, position=(x, y)))
                used_positions.append((x, y))
            else:
                # If too close, try to adjust position slightly
                adjusted = False
                for attempt in range(20):  # More attempts
                    angle = 2 * math.pi * attempt / 20
                    offset = min_distance * 1.2
                    new_x = x + offset * math.cos(angle)
                    new_y = y + offset * math.sin(angle)
                    
                    # Clamp to canvas bounds
                    new_x = max(padding, min(w - padding, new_x))
                    new_y = max(padding, min(h - padding, new_y))
                    
                    # Check if new position is valid (not too close to others)
                    valid = True
                    for used_x, used_y in used_positions:
                        distance = math.sqrt((new_x - used_x) ** 2 + (new_y - used_y) ** 2)
                        if distance < min_distance:
                            valid = False
                            break
                    
                    if valid:
                        # Create new dot with adjusted position
                        adjusted_dot = DotSpec(number=dot.number, position=(new_x, new_y))
                        unique_dots.append(adjusted_dot)
                        used_positions.append((new_x, new_y))
                        adjusted = True
                        break
                
                if not adjusted:
                    # If can't adjust, use original position (clamped to bounds)
                    # This should rarely happen with proper pattern generation
                    clamped_dot = DotSpec(number=dot.number, position=(x, y))
                    unique_dots.append(clamped_dot)
                    used_positions.append((x, y))
        
        # Re-number dots to ensure sequential numbering
        for i, dot in enumerate(unique_dots):
            if dot.number != i + 1:
                unique_dots[i] = DotSpec(number=i + 1, position=dot.position)
        
        return unique_dots

    def _generate_star(self, num_dots: int, w: int, h: int, margin: int) -> List[DotSpec]:
        """Generate a star pattern."""
        center_x, center_y = w / 2, h / 2
        radius = min(w, h) / 2 - margin
        outer_radius = radius
        inner_radius = radius * 0.4
        
        dots = []
        for i in range(num_dots):
            angle = 2 * math.pi * i / num_dots - math.pi / 2
            # Alternate between outer and inner radius for star shape
            r = outer_radius if i % 2 == 0 else inner_radius
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            dots.append(DotSpec(number=i + 1, position=(x, y)))
        return dots

    def _generate_heart(self, num_dots: int, w: int, h: int, margin: int) -> List[DotSpec]:
        """Generate a heart shape pattern."""
        center_x, center_y = w / 2, h / 2
        scale = min(w, h) / 3 - margin / 2
        
        dots = []
        for i in range(num_dots):
            t = 2 * math.pi * i / num_dots
            # Heart parametric equation
            x = center_x + scale * (16 * math.sin(t) ** 3)
            y = center_y - scale * (13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
            dots.append(DotSpec(number=i + 1, position=(x, y)))
        return dots

    def _generate_circle(self, num_dots: int, w: int, h: int, margin: int) -> List[DotSpec]:
        """Generate a circle pattern."""
        center_x, center_y = w / 2, h / 2
        radius = min(w, h) / 2 - margin
        
        dots = []
        for i in range(num_dots):
            angle = 2 * math.pi * i / num_dots
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            dots.append(DotSpec(number=i + 1, position=(x, y)))
        return dots

    def _generate_triangle(self, num_dots: int, w: int, h: int, margin: int) -> List[DotSpec]:
        """Generate a triangle pattern with evenly distributed points."""
        center_x, center_y = w / 2, h / 2
        radius = min(w, h) / 2 - margin
        
        dots = []
        # Distribute points evenly along triangle perimeter
        for i in range(num_dots):
            t = i / num_dots  # Position along perimeter (0 to 1)
            perimeter_length = 3  # Triangle has 3 sides
            side_length = 1.0 / perimeter_length
            
            # Determine which side and position on that side
            side = int(t * perimeter_length)
            side_t = (t * perimeter_length) % 1.0
            
            if side == 0:  # Top side (left to right)
                x = center_x + radius * (side_t - 0.5) * 2
                y = center_y - radius * 0.866
            elif side == 1:  # Bottom right side
                x = center_x + radius * 0.5 - radius * side_t
                y = center_y + radius * 0.866 - radius * side_t * 1.732
            else:  # Bottom left side
                x = center_x - radius * 0.5 + radius * side_t
                y = center_y + radius * 0.866 - radius * side_t * 1.732
            dots.append(DotSpec(number=i + 1, position=(x, y)))
        return dots

    def _generate_square(self, num_dots: int, w: int, h: int, margin: int) -> List[DotSpec]:
        """Generate a square pattern with evenly distributed points."""
        center_x, center_y = w / 2, h / 2
        size = min(w, h) / 2 - margin
        
        dots = []
        # Distribute points evenly along square perimeter
        for i in range(num_dots):
            t = i / num_dots  # Position along perimeter (0 to 1)
            perimeter_length = 4  # Square has 4 sides
            side_length = 1.0 / perimeter_length
            
            # Determine which side and position on that side
            side = int(t * perimeter_length)
            side_t = (t * perimeter_length) % 1.0
            
            if side == 0:  # Top side (left to right)
                x = center_x - size + 2 * size * side_t
                y = center_y - size
            elif side == 1:  # Right side (top to bottom)
                x = center_x + size
                y = center_y - size + 2 * size * side_t
            elif side == 2:  # Bottom side (right to left)
                x = center_x + size - 2 * size * side_t
                y = center_y + size
            else:  # Left side (bottom to top)
                x = center_x - size
                y = center_y + size - 2 * size * side_t
            dots.append(DotSpec(number=i + 1, position=(x, y)))
        return dots

    def _generate_spiral(self, num_dots: int, w: int, h: int, margin: int) -> List[DotSpec]:
        """Generate a spiral pattern."""
        center_x, center_y = w / 2, h / 2
        max_radius = min(w, h) / 2 - margin
        
        dots = []
        for i in range(num_dots):
            t = i / num_dots
            angle = 4 * math.pi * t
            radius = max_radius * t
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            dots.append(DotSpec(number=i + 1, position=(x, y)))
        return dots

    def _format_prompt(self, dots: Sequence[DotSpec], difficulty: str) -> str:
        """Format prompt with dot information."""
        max_number = max(dot.number for dot in dots)
        return get_prompt(max_number)

    def _build_metadata(self, task_id: str, dots: Sequence[DotSpec], difficulty: str) -> Dict:
        """Build metadata dictionary."""
        sorted_dots = sorted(dots, key=lambda d: d.number)
        return {
            "task_id": task_id,
            "domain": "dot_to_dot_task",
            "difficulty": difficulty,
            "input_type": "image_pair",
            "output_type": "video",
            "canvas_size": {"width": self.canvas[0], "height": self.canvas[1]},
            "camera": {"view": "top_down", "movement": "static"},
            "num_dots": len(dots),
            "max_number": max(dot.number for dot in dots),
            "dots": [
                {
                    "number": dot.number,
                    "position": [round(dot.x, 2), round(dot.y, 2)],
                }
                for dot in sorted_dots
            ],
            "created_at": datetime.now().isoformat(),
        }

    def _build_signature(self, dots: Sequence[DotSpec]) -> str:
        """Create a deterministic signature to avoid duplicates."""
        sorted_dots = sorted(dots, key=lambda d: d.number)
        quantized = [
            (dot.number, round(dot.x, 1), round(dot.y, 1))
            for dot in sorted_dots
        ]
        return "|".join(f"{n},{x},{y}" for n, x, y in quantized)


def create_dataset(
    num_samples: int = 10,
    difficulties: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, List[Dict]]:
    """
    Entry point used by VMEvalKit runner.
    """
    generator = DotToDotGenerator()
    rng = random.Random(seed)
    diffs = list(difficulties) if difficulties else ["easy", "medium", "hard"]

    pairs = []
    for idx in range(num_samples):
        difficulty = diffs[idx % len(diffs)]
        task_id = f"dot_to_dot_{idx:04d}"
        pairs.append(
            generator.generate(
                task_id=task_id,
                difficulty=difficulty,
                seed=rng.randint(0, 10_000_000),
            )
        )

    dataset = {
        "name": "dot_to_dot_tasks",
        "description": f"Dot to Dot puzzle reasoning tasks ({len(pairs)} pairs)",
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

