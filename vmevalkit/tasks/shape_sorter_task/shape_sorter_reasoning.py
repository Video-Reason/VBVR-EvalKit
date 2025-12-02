"""
Minimal 2D Shape Sorter task for VMEvalKit.

This implementation focuses on producing simple but clean PNG pairs showing
colored cards on the left and outline slots on the right. It is intentionally
lightweight: no animation logic, just static start/end frames plus prompt and
metadata so the task can be registered like other domains.
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
from matplotlib.patches import Circle, Polygon, Rectangle, RegularPolygon

from .PROMPTS import PROMPT_TEMPLATES, format_shape_summary

Canvas = Tuple[int, int]
Point = Tuple[float, float]

CANVAS: Canvas = (768, 512)
DPI = 150

COLORS = [
    ("red", "#f87171"),
    ("yellow", "#facc15"),
    ("blue", "#60a5fa"),
    ("green", "#4ade80"),
    ("purple", "#c084fc"),
    ("orange", "#fb923c"),
]

SHAPES = ["circle", "square", "triangle", "star", "hexagon", "diamond"]


@dataclass
class ShapeSpec:
    shape: str
    color_name: str
    color_hex: str
    start: Point
    target: Point
    size: float

    def label(self) -> str:
        return f"{self.color_name} {self.shape}"


class ShapeSorterRenderer:
    """Draw the simple left/right layout."""

    def __init__(self, canvas: Canvas = CANVAS, dpi: int = DPI):
        self.canvas = canvas
        self.dpi = dpi

    def render_start(self, specs: Sequence[ShapeSpec], path: Path) -> None:
        fig, ax = self._setup_axes()
        self._draw_layout(ax)
        for spec in specs:
            self._draw_outline(ax, spec.target, spec.size, spec.shape)
        for spec in specs:
            self._draw_shape(ax, spec.start, spec.size, spec.shape, spec.color_hex, filled=True)
        self._finalize(fig, path)

    def render_end(self, specs: Sequence[ShapeSpec], path: Path) -> None:
        fig, ax = self._setup_axes()
        # Keep the same neutral background as the first frame for consistency.
        self._draw_layout(ax, solved=False)
        for spec in specs:
            self._draw_shape(ax, spec.target, spec.size, spec.shape, spec.color_hex, filled=True)
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

    def _draw_layout(self, ax, solved: bool = False) -> None:
        w, h = self.canvas
        bg = "#f8fafc"
        ax.add_patch(Rectangle((0, 0), w, h, facecolor=bg, edgecolor="none"))
        divider = w * 0.5
        ax.add_patch(Rectangle((divider - 4, 40), 8, h - 80, facecolor="#94a3b8", alpha=0.35, edgecolor="none"))

    def _draw_outline(self, ax, center: Point, size: float, shape: str) -> None:
        self._draw_shape(ax, center, size, shape, "#64748b", filled=False, linewidth=3)

    def _draw_shape(
        self,
        ax,
        center: Point,
        size: float,
        shape: str,
        color: str,
        filled: bool,
        linewidth: float = 2.5,
    ) -> None:
        x, y = center
        if shape == "circle":
            patch = Circle((x, y), radius=size / 2, facecolor=color if filled else "none", edgecolor=color, linewidth=linewidth)
        elif shape == "square":
            patch = Rectangle((x - size / 2, y - size / 2), size, size, facecolor=color if filled else "none", edgecolor=color, linewidth=linewidth)
        elif shape == "triangle":
            points = [
                (x, y - size / 2),
                (x - size / 2, y + size / 2),
                (x + size / 2, y + size / 2),
            ]
            patch = Polygon(points, closed=True, facecolor=color if filled else "none", edgecolor=color, linewidth=linewidth)
        elif shape == "hexagon":
            patch = RegularPolygon(
                (x, y),
                numVertices=6,
                radius=size / 2,
                orientation=math.pi / 6,
                facecolor=color if filled else "none",
                edgecolor=color,
                linewidth=linewidth,
            )
        elif shape == "diamond":
            points = [
                (x, y - size / 2),
                (x + size / 2, y),
                (x, y + size / 2),
                (x - size / 2, y),
            ]
            patch = Polygon(points, closed=True, facecolor=color if filled else "none", edgecolor=color, linewidth=linewidth)
        elif shape == "star":
            patch = self._star_patch(center, size / 2, color, filled, linewidth)
        else:
            raise ValueError(f"Unsupported shape type: {shape}")
        ax.add_patch(patch)

    def _star_patch(self, center: Point, radius: float, color: str, filled: bool, linewidth: float):
        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            r = radius if i % 2 == 0 else radius * 0.45
            points.append(
                (center[0] + r * math.cos(angle), center[1] + r * math.sin(angle))
            )
        return Polygon(
            points,
            closed=True,
            facecolor=color if filled else "none",
            edgecolor=color,
            linewidth=linewidth,
        )

    def _finalize(self, fig, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)


class ShapeSorterGenerator:
    """Simple procedural generator."""

    def __init__(self, canvas: Canvas = CANVAS):
        self.canvas = canvas
        self.renderer = ShapeSorterRenderer(canvas)
        self.rng = random.Random()
        self.output_root = Path("data/questions/shape_sorter_task")
        self._seen_signatures: set[str] = set()

    def generate(
        self,
        task_id: str,
        difficulty: str = "medium",
        num_shapes: Optional[int] = None,
        seed: Optional[int] = None,
        ensure_unique: bool = True,
        max_attempts: int = 25,
    ) -> Dict:
        if seed is not None:
            self.rng.seed(seed)

        attempt = 0
        signature = None
        while True:
            attempt += 1
            count = num_shapes or self._shape_count_for_difficulty(difficulty)
            specs, layout_variant = self._create_specs(count)
            signature = self._build_signature(specs, layout_variant)
            if not ensure_unique or signature not in self._seen_signatures:
                break
            if attempt >= max_attempts:
                raise RuntimeError("Failed to generate unique Shape Sorter sample after multiple attempts.")

        if ensure_unique and signature is not None:
            self._seen_signatures.add(signature)

        question_dir = self.output_root / task_id
        question_dir.mkdir(parents=True, exist_ok=True)
        first_png = question_dir / "first_frame.png"
        final_png = question_dir / "final_frame.png"

        self.renderer.render_start(specs, first_png)
        self.renderer.render_end(specs, final_png)

        prompt = self._format_prompt(specs)
        (question_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

        metadata = self._build_metadata(task_id, specs, difficulty, layout_variant)
        (question_dir / "question_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return {
            "id": task_id,
            "prompt": prompt,
            "first_image_path": str(first_png),
            "final_image_path": str(final_png),
            "task_category": "ShapeSorter",
            "difficulty": difficulty,
            "shape_sorter_data": metadata,
            "created_at": datetime.now().isoformat(),
        }

    def _shape_count_for_difficulty(self, difficulty: str) -> int:
        if difficulty == "easy":
            return self.rng.randint(2, 3)
        if difficulty == "hard":
            return self.rng.randint(5, 6)
        return self.rng.randint(3, 5)

    def _create_specs(self, count: int) -> Tuple[List[ShapeSpec], str]:
        layout_variant = self._choose_layout(count)
        cards, card_size = self._generate_positions(
            count=count,
            columns=self._layout_columns(layout_variant, side="cards"),
            x_range=(0.12 * self.canvas[0], 0.4 * self.canvas[0]),
            y_range=(0.18 * self.canvas[1], 0.82 * self.canvas[1]),
            jitter=self._layout_jitter(layout_variant, side="cards"),
        )
        slots, slot_size = self._generate_positions(
            count=count,
            columns=self._layout_columns(layout_variant, side="slots"),
            x_range=(0.6 * self.canvas[0], 0.88 * self.canvas[0]),
            y_range=(0.18 * self.canvas[1], 0.82 * self.canvas[1]),
            jitter=self._layout_jitter(layout_variant, side="slots"),
        )

        size = min(card_size, slot_size)
        shapes = self._sample_shapes(count)
        colors = self._sample_colors(count)

        specs: List[ShapeSpec] = []
        for i in range(count):
            color_name, color_hex = colors[i]
            specs.append(
                ShapeSpec(
                    shape=shapes[i],
                    color_name=color_name,
                    color_hex=color_hex,
                    start=cards[i],
                    target=slots[i],
                    size=size,
                )
            )
        return specs, layout_variant

    def _sample_shapes(self, count: int) -> List[str]:
        if count <= len(SHAPES):
            return self.rng.sample(SHAPES, k=count)
        base = self.rng.sample(SHAPES, k=len(SHAPES))
        base += self.rng.choices(SHAPES, k=count - len(SHAPES))
        return base

    def _sample_colors(self, count: int) -> List[Tuple[str, str]]:
        if count <= len(COLORS):
            return self.rng.sample(COLORS, k=count)
        colors = self.rng.sample(COLORS, k=len(COLORS))
        colors += self.rng.choices(COLORS, k=count - len(COLORS))
        return colors

    def _choose_layout(self, count: int) -> str:
        if count <= 2:
            return "line"
        if count == 3:
            return "staggered"
        if count == 4:
            return "grid"
        return "scatter"

    def _layout_columns(self, layout: str, side: str) -> int:
        base = {
            "line": 1,
            "staggered": 2 if side == "cards" else 1,
            "grid": 2,
            "scatter": 2,
        }
        return base.get(layout, 1)

    def _layout_jitter(self, layout: str, side: str) -> float:
        jitter_map = {
            "line": 0.0,
            "staggered": 0.015 if side == "cards" else 0.0,
            "grid": 0.01,
            "scatter": 0.03 if side == "cards" else 0.015,
        }
        return jitter_map.get(layout, 0.0)

    def _generate_positions(
        self,
        count: int,
        columns: int,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        jitter: float = 0.0,
    ) -> Tuple[List[Point], float]:
        columns = max(1, columns)
        rows = math.ceil(count / columns)
        positions: List[Point] = []
        for idx in range(count):
            row = idx // columns
            col = idx % columns
            x = x_range[0] + (col + 0.5) / columns * (x_range[1] - x_range[0])
            y = y_range[0] + (row + 0.5) / rows * (y_range[1] - y_range[0])
            if jitter > 0:
                x += self.rng.uniform(-jitter, jitter) * (x_range[1] - x_range[0])
                y += self.rng.uniform(-jitter, jitter) * (y_range[1] - y_range[0])
            positions.append((x, y))

        cell_w = (x_range[1] - x_range[0]) / columns
        cell_h = (y_range[1] - y_range[0]) / rows
        size = min(90.0, cell_w * 0.55, cell_h * 0.55)
        return positions, size

    def _format_prompt(self, specs: Sequence[ShapeSpec]) -> str:
        template = self.rng.choice(PROMPT_TEMPLATES)
        summary = format_shape_summary([spec.label() for spec in specs])
        return template.format(shape_summary=summary)

    def _build_metadata(self, task_id: str, specs: Sequence[ShapeSpec], difficulty: str, layout_variant: str) -> Dict:
        return {
            "task_id": task_id,
            "domain": "shape_sorter_task",
            "difficulty": difficulty,
            "input_type": "image_pair",
            "output_type": "video",
            "canvas_size": {"width": self.canvas[0], "height": self.canvas[1]},
            "camera": {"view": "top_down", "movement": "static"},
            "layout_variant": layout_variant,
            "shape_count": len(specs),
            "shape_library": SHAPES,
            "shapes": [
                {
                    "shape": spec.shape,
                    "color": spec.color_name,
                    "start": list(spec.start),
                    "target": list(spec.target),
                    "size": spec.size,
                }
                for spec in specs
            ],
            "colors": sorted({spec.color_name for spec in specs}),
            "created_at": datetime.now().isoformat(),
        }

    def _build_signature(self, specs: Sequence[ShapeSpec], layout_variant: str) -> str:
        """
        Create a deterministic signature for a set of specs so we can avoid duplicates.
        """
        quantized = []
        for spec in specs:
            quantized.append(
                (
                    spec.shape,
                    spec.color_name,
                    round(spec.start[0], 1),
                    round(spec.start[1], 1),
                    round(spec.target[0], 1),
                    round(spec.target[1], 1),
                    round(spec.size, 1),
                )
            )
        quantized.sort()
        return f"{layout_variant}|{len(specs)}|" + "|".join(
            ",".join(map(str, item)) for item in quantized
        )


def create_dataset(
    num_samples: int = 10,
    difficulties: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, List[Dict]]:
    """
    Entry point used by VMEvalKit runner.
    """
    generator = ShapeSorterGenerator()
    rng = random.Random(seed)
    diffs = list(difficulties) if difficulties else ["easy", "medium", "hard"]

    pairs = []
    for idx in range(num_samples):
        difficulty = diffs[idx % len(diffs)]
        task_id = f"shape_sorter_{idx:04d}"
        pairs.append(
            generator.generate(
                task_id=task_id,
                difficulty=difficulty,
                seed=rng.randint(0, 10_000_000),
            )
        )

    dataset = {
        "name": "shape_sorter_tasks",
        "description": f"Simple 2D shape sorter reasoning tasks ({len(pairs)} pairs)",
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

