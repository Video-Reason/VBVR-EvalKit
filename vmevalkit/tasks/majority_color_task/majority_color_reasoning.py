"""
Majority Color Task for VMEvalKit.

The task shows multiple colored shapes. The model should recolor every object
to match the most common color in the scene while keeping positions and shapes
unchanged.
"""

import hashlib
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .PROMPTS import PROMPTS, DEFAULT_PROMPT_INDEX
from ..object_subtraction_task.object_subtraction_reasoning import (
    ObjectGenerator,
    SceneRenderer,
)


@dataclass
class MajorityColorTaskPair:
    """
    Data structure for majority color video model evaluation.
    """
    id: str
    prompt: str
    first_image_path: str
    final_image_path: str
    task_category: str = "MajorityColor"
    majority_color_data: Dict[str, Any] = None
    difficulty: str = "easy"
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class MajorityColorGenerator:
    """
    Generator for majority color replacement tasks.
    """

    DIFFICULTY_CONFIG = {
        "easy": {"num_objects": (6, 8), "num_colors": (3, 3)},
        "medium": {"num_objects": (8, 10), "num_colors": (3, 4)},
        "hard": {"num_objects": (10, 12), "num_colors": (4, 5)},
    }

    DEFAULT_DIFFICULTY_DISTRIBUTION = {
        "easy": 0.4,
        "medium": 0.4,
        "hard": 0.2,
    }

    def __init__(self, canvas_size: Tuple[int, int] = (256, 256)):
        self.canvas_size = canvas_size
        self.object_gen = ObjectGenerator(canvas_size=canvas_size)
        self.renderer = SceneRenderer(canvas_size=canvas_size)
        self.rng = random.Random()
        self.temp_dir = tempfile.mkdtemp(prefix="majority_color_")

        self.num_objects_range = (
            min(v["num_objects"][0] for v in self.DIFFICULTY_CONFIG.values()),
            max(v["num_objects"][1] for v in self.DIFFICULTY_CONFIG.values()),
        )
        self.num_colors_range = (
            min(v["num_colors"][0] for v in self.DIFFICULTY_CONFIG.values()),
            max(v["num_colors"][1] for v in self.DIFFICULTY_CONFIG.values()),
        )

    def _choose_difficulty(self, distribution: Dict[str, float]) -> str:
        total = sum(distribution.values())
        pick = self.rng.random() * total
        cumulative = 0.0
        for name, weight in distribution.items():
            cumulative += weight
            if pick <= cumulative:
                return name
        return "easy"

    def _choose_num_objects_colors(self, difficulty: str) -> Tuple[int, int]:
        config = self.DIFFICULTY_CONFIG[difficulty]
        num_objects = self.rng.randint(*config["num_objects"])
        min_colors, max_colors = config["num_colors"]
        max_colors_for_n = min(max_colors, num_objects - num_objects // 2)
        if max_colors_for_n < min_colors:
            num_objects = max(num_objects, min_colors * 2 - 1)
            max_colors_for_n = min(max_colors, num_objects - num_objects // 2)
        num_colors = self.rng.randint(min_colors, max_colors_for_n)
        return num_objects, num_colors

    def _assign_colors(self, objects: List[Dict[str, Any]], num_colors: int) -> Tuple[str, Dict[str, int]]:
        colors = self.rng.sample(ObjectGenerator.COLORS, num_colors)
        majority_color = self.rng.choice(colors)

        num_objects = len(objects)
        min_majority = num_objects // 2 + 1
        max_majority = num_objects - (num_colors - 1)
        if max_majority < min_majority:
            max_majority = min_majority
        majority_count = self.rng.randint(min_majority, max_majority)

        counts = {color: 1 for color in colors if color != majority_color}
        remaining = num_objects - majority_count - (num_colors - 1)
        non_majority = [c for c in colors if c != majority_color]
        for _ in range(remaining):
            counts[self.rng.choice(non_majority)] += 1
        counts[majority_color] = majority_count

        color_list: List[str] = []
        for color, count in counts.items():
            color_list.extend([color] * count)
        self.rng.shuffle(color_list)

        for obj, color in zip(objects, color_list):
            obj["color"] = color

        if counts[majority_color] <= num_objects // 2:
            raise ValueError("Majority color count is not more than half of objects")

        return majority_color, counts

    def generate_single_task(
        self,
        task_id: str,
        difficulty: Optional[str] = None,
        seed: Optional[int] = None,
        difficulty_distribution: Optional[Dict[str, float]] = None,
    ) -> MajorityColorTaskPair:
        if seed is not None:
            self.rng.seed(seed)

        distribution = difficulty_distribution or self.DEFAULT_DIFFICULTY_DISTRIBUTION
        if difficulty is None:
            difficulty = self._choose_difficulty(distribution)

        num_objects, num_colors = self._choose_num_objects_colors(difficulty)
        objects = self.object_gen.generate_objects(num_objects, seed=seed)

        majority_color, color_counts = self._assign_colors(objects, num_colors)

        initial_objects = [dict(obj) for obj in objects]
        final_objects = [dict(obj) for obj in objects]
        for obj in final_objects:
            obj["color"] = majority_color

        prompt_index = DEFAULT_PROMPT_INDEX
        if len(PROMPTS) > 1:
            prompt_index = self.rng.randint(0, len(PROMPTS) - 1)
        prompt = PROMPTS[prompt_index]

        first_path = Path(self.temp_dir) / f"{task_id}_first.png"
        final_path = Path(self.temp_dir) / f"{task_id}_final.png"
        self.renderer.render_scene(initial_objects, output_path=first_path)
        self.renderer.render_scene(final_objects, output_path=final_path)

        task_pair = MajorityColorTaskPair(
            id=task_id,
            prompt=prompt,
            first_image_path=str(first_path),
            final_image_path=str(final_path),
            task_category="MajorityColor",
            majority_color_data={
                "objects": initial_objects,
                "majority_color": majority_color,
                "color_counts": color_counts,
                "num_objects": num_objects,
                "num_colors": num_colors,
            },
            difficulty=difficulty,
            created_at=datetime.now().isoformat(),
        )

        return task_pair


def create_dataset(
    num_samples: int = 50,
    difficulty_distribution: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Create majority color replacement dataset.
    """
    print("Creating Majority Color Dataset")
    print(f"   Total samples: {num_samples}")

    generator = MajorityColorGenerator()
    distribution = difficulty_distribution or generator.DEFAULT_DIFFICULTY_DISTRIBUTION

    pairs: List[MajorityColorTaskPair] = []
    for i in range(num_samples):
        task_id = f"majority_color_{i:04d}"
        task_hash = int(hashlib.md5(task_id.encode()).hexdigest()[:8], 16)
        seed = task_hash + i
        task_pair = generator.generate_single_task(
            task_id,
            seed=seed,
            difficulty_distribution=distribution,
        )
        pairs.append(task_pair)

    pairs_dict = []
    for pair in pairs:
        pairs_dict.append({
            "id": pair.id,
            "prompt": pair.prompt,
            "first_image_path": pair.first_image_path,
            "final_image_path": pair.final_image_path,
            "task_category": pair.task_category,
            "majority_color_data": pair.majority_color_data,
            "difficulty": pair.difficulty,
            "created_at": pair.created_at,
        })

    dataset = {
        "name": "majority_color_tasks",
        "description": "Majority color replacement tasks for video model evaluation",
        "pairs": pairs_dict,
        "metadata": {
            "total_tasks": len(pairs),
            "canvas_size": generator.canvas_size,
            "num_objects_range": generator.num_objects_range,
            "num_colors_range": generator.num_colors_range,
            "difficulty_distribution": distribution,
            "generation_date": datetime.now().isoformat(),
        },
        "created_at": datetime.now().isoformat(),
    }

    print("Dataset creation complete")
    print(f"   Total tasks: {len(pairs)}")
    return dataset
