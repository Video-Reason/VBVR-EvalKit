#!/usr/bin/env python3
"""
Mirror Clock Reasoning Task for VMEvalKit

Tests spatial reasoning and mirror transformation capabilities by asking models
to determine what time is shown when looking at a clock through a mirror.

Author: VMEvalKit Team
"""

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, time
from PIL import Image, ImageDraw, ImageFont

from .PROMPTS import PROMPTS


@dataclass
class MirrorClockTaskPair:
    """Single mirror clock task pair data structure"""
    id: str
    prompt: str
    first_image_path: str
    final_image_path: str
    domain: str = "mirror_clock"
    task_category: str = "Spatial Reasoning"
    difficulty: str = "medium"

    # Task-specific data
    mirror_clock_data: Optional[Dict[str, Any]] = None

    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return asdict(self)


class ClockRenderer:
    """Clock face rendering utility"""

    def __init__(self, image_size: int = 500):
        """
        Initialize clock renderer

        Args:
            image_size: Size of the clock image (width and height)
        """
        self.image_size = image_size
        self.center = image_size // 2
        self.clock_radius = int(image_size * 0.4)
        self.hour_hand_length = int(self.clock_radius * 0.5)
        self.minute_hand_length = int(self.clock_radius * 0.7)

    def _draw_clock_face(self, draw: ImageDraw.Draw):
        """Draw the basic clock face with numbers"""
        # Draw outer circle
        draw.ellipse(
            [
                self.center - self.clock_radius,
                self.center - self.clock_radius,
                self.center + self.clock_radius,
                self.center + self.clock_radius
            ],
            outline='#333333',
            width=4,
            fill='#ffffff'
        )

        # Try to load a nice font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        except:
            font = ImageFont.load_default()

        # Draw hour numbers
        for hour in range(1, 13):
            angle = math.radians(90 - (hour * 30))  # 12 is at top (90 degrees)
            x = self.center + int((self.clock_radius * 0.75) * math.cos(angle))
            y = self.center - int((self.clock_radius * 0.75) * math.sin(angle))

            hour_str = str(hour)
            bbox = draw.textbbox((0, 0), hour_str, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            draw.text(
                (x - text_width // 2, y - text_height // 2),
                hour_str,
                fill='#333333',
                font=font
            )

        # Draw center dot
        center_dot_radius = 8
        draw.ellipse(
            [
                self.center - center_dot_radius,
                self.center - center_dot_radius,
                self.center + center_dot_radius,
                self.center + center_dot_radius
            ],
            fill='#333333'
        )

    def _draw_hand(self, draw: ImageDraw.Draw, angle_degrees: float,
                   length: int, width: int, color: str):
        """
        Draw a clock hand

        Args:
            angle_degrees: Angle in degrees (0 = 12 o'clock, clockwise)
            length: Length of the hand
            width: Width of the hand
            color: Color of the hand
        """
        # Convert to radians (subtract 90 to make 0 degrees point up)
        angle_rad = math.radians(angle_degrees - 90)

        end_x = self.center + int(length * math.cos(angle_rad))
        end_y = self.center + int(length * math.sin(angle_rad))

        draw.line(
            [self.center, self.center, end_x, end_y],
            fill=color,
            width=width
        )

    def draw_clock(self, hours: int, minutes: int) -> Image.Image:
        """
        Draw a clock showing the specified time

        Args:
            hours: Hour (0-23, will be converted to 12-hour)
            minutes: Minutes (0-59)

        Returns:
            PIL Image of the clock
        """
        # Create image
        img = Image.new('RGB', (self.image_size, self.image_size), color='white')
        draw = ImageDraw.Draw(img)

        # Draw clock face
        self._draw_clock_face(draw)

        # Convert to 12-hour format
        hours_12 = hours % 12

        # Calculate angles
        # Hour hand: moves 30 degrees per hour + 0.5 degrees per minute
        hour_angle = (hours_12 * 30) + (minutes * 0.5)
        # Minute hand: moves 6 degrees per minute
        minute_angle = minutes * 6

        # Draw hands (minute hand first, then hour hand on top)
        self._draw_hand(draw, minute_angle, self.minute_hand_length, 6, '#666666')
        self._draw_hand(draw, hour_angle, self.hour_hand_length, 8, '#333333')

        return img


class MirrorClockGenerator:
    """Mirror clock task generator"""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize generator

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        self.clock_renderer = ClockRenderer()

    def add_time(self, hours: int, minutes: int,
                 add_hours: int, add_minutes: int) -> Tuple[int, int]:
        """
        Add time to a given time

        Args:
            hours: Original hours
            minutes: Original minutes
            add_hours: Hours to add
            add_minutes: Minutes to add

        Returns:
            (new_hours, new_minutes) tuple
        """
        total_minutes = minutes + add_minutes
        extra_hours = total_minutes // 60
        new_minutes = total_minutes % 60

        new_hours = (hours + add_hours + extra_hours) % 24

        return new_hours, new_minutes

    def generate_time_delta(self, difficulty: str = "medium") -> Tuple[int, int]:
        """
        Generate a time delta based on difficulty

        Args:
            difficulty: easy, medium, or hard

        Returns:
            (hours_to_add, minutes_to_add) tuple
        """
        if difficulty == "easy":
            # Only add full hours (1-3 hours)
            return random.randint(1, 3), 0
        elif difficulty == "medium":
            # Add hours with 30-minute intervals
            hours = random.randint(0, 2)
            minutes = random.choice([0, 30])
            # Ensure at least some time is added
            if hours == 0 and minutes == 0:
                hours = 1
            return hours, minutes
        else:  # hard
            # Any time combination (0-3 hours, 0-59 minutes)
            hours = random.randint(0, 3)
            minutes = random.randint(0, 59)
            # Ensure at least some time is added
            if hours == 0 and minutes == 0:
                minutes = random.randint(15, 45)
            return hours, minutes

    def generate_random_time(self, difficulty: str = "medium") -> Tuple[int, int]:
        """
        Generate a random time based on difficulty

        Args:
            difficulty: easy, medium, or hard

        Returns:
            (hours, minutes) tuple
        """
        if difficulty == "easy":
            # Only hour marks (no minutes)
            hours = random.randint(1, 12)
            minutes = 0
        elif difficulty == "medium":
            # 5-minute intervals
            hours = random.randint(1, 12)
            minutes = random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        else:  # hard
            # Any minute
            hours = random.randint(1, 12)
            minutes = random.randint(0, 59)

        return hours, minutes

    def determine_difficulty(self, hours: int, minutes: int) -> str:
        """
        Determine difficulty level based on time

        Args:
            hours: Hours
            minutes: Minutes

        Returns:
            Difficulty level string
        """
        if minutes == 0:
            return "easy"
        elif minutes % 5 == 0:
            return "medium"
        else:
            return "hard"


class MirrorClockTaskGenerator:
    """Mirror clock task generator with file I/O"""

    def __init__(self, data_root: str = "data/questions"):
        """
        Initialize task generator

        Args:
            data_root: Data root directory
        """
        self.data_root = Path(data_root)
        self.task_dir = self.data_root / "mirror_clock_task"
        self.clock_generator = MirrorClockGenerator()
        self.prompts = PROMPTS

    def generate_single_task(self, task_id: str,
                            hours: Optional[int] = None,
                            minutes: Optional[int] = None,
                            add_hours: Optional[int] = None,
                            add_minutes: Optional[int] = None,
                            difficulty: Optional[str] = None) -> MirrorClockTaskPair:
        """
        Generate a single mirror clock task with future time prediction

        Args:
            task_id: Task ID
            hours: Original hours (None for random)
            minutes: Original minutes (None for random)
            add_hours: Hours to add (None for random based on difficulty)
            add_minutes: Minutes to add (None for random based on difficulty)
            difficulty: Difficulty level (None for auto)

        Returns:
            MirrorClockTaskPair object
        """
        # Determine difficulty first if not provided
        if difficulty is None:
            difficulty = random.choice(["easy", "medium", "hard"])

        # Generate or use provided time
        if hours is None or minutes is None:
            hours, minutes = self.clock_generator.generate_random_time(difficulty)

        # Generate time delta if not provided
        if add_hours is None or add_minutes is None:
            add_hours, add_minutes = self.clock_generator.generate_time_delta(difficulty)

        # Calculate future time
        future_hours, future_minutes = self.clock_generator.add_time(
            hours, minutes, add_hours, add_minutes
        )

        # Create task directory
        task_path = self.task_dir / task_id
        task_path.mkdir(parents=True, exist_ok=True)

        # Generate original clock image (current time)
        original_image = self.clock_generator.clock_renderer.draw_clock(hours, minutes)

        # Create mirrored version by flipping horizontally
        mirrored_image = original_image.transpose(Image.FLIP_LEFT_RIGHT)

        # Generate future clock image (answer)
        future_image = self.clock_generator.clock_renderer.draw_clock(
            future_hours, future_minutes
        )

        # Save images
        # first_frame: mirrored clock (current time in mirror)
        # final_frame: future clock (after adding time)
        first_image_path = task_path / "first_frame.png"
        final_image_path = task_path / "final_frame.png"
        mirrored_image.save(first_image_path)
        future_image.save(final_image_path)

        # Format time strings
        def format_time(h: int, m: int) -> str:
            return f"{h % 12 if h % 12 != 0 else 12}:{m:02d}"

        def format_time_delta(h: int, m: int) -> str:
            if h == 0:
                return f"{m} minute{'s' if m != 1 else ''}"
            elif m == 0:
                return f"{h} hour{'s' if h != 1 else ''}"
            else:
                return f"{h} hour{'s' if h != 1 else ''} and {m} minute{'s' if m != 1 else ''}"

        original_time_str = format_time(hours, minutes)
        future_time_str = format_time(future_hours, future_minutes)
        time_delta_str = format_time_delta(add_hours, add_minutes)

        # Select prompt and fill in time_delta placeholder
        prompt_template = random.choice(self.prompts)
        prompt = prompt_template.format(time_delta=time_delta_str)

        # Save prompt
        prompt_path = task_path / "prompt.txt"
        prompt_path.write_text(prompt, encoding='utf-8')

        # Build task-specific data
        task_specific_data = {
            'original_time': {
                'hours': hours,
                'minutes': minutes,
                'formatted': original_time_str
            },
            'time_delta': {
                'hours': add_hours,
                'minutes': add_minutes,
                'formatted': time_delta_str
            },
            'future_time': {
                'hours': future_hours,
                'minutes': future_minutes,
                'formatted': future_time_str
            },
            'difficulty': difficulty,
            'task_type': 'mirror_with_future_prediction'
        }

        # Save metadata
        metadata_path = task_path / "question_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(task_specific_data, f, indent=2, ensure_ascii=False)

        # Create task pair
        task_pair = MirrorClockTaskPair(
            id=task_id,
            prompt=prompt,
            first_image_path=str(first_image_path.relative_to(self.data_root)),
            final_image_path=str(final_image_path.relative_to(self.data_root)),
            difficulty=difficulty,
            mirror_clock_data=task_specific_data
        )

        return task_pair

    def generate_balanced_dataset(self, num_samples: int = 50) -> List[MirrorClockTaskPair]:
        """
        Generate balanced dataset across difficulty levels

        Args:
            num_samples: Total number of samples

        Returns:
            List of task pairs
        """
        difficulties = ["easy", "medium", "hard"]
        samples_per_difficulty = num_samples // len(difficulties)
        remaining = num_samples % len(difficulties)

        task_pairs = []
        task_counter = 0

        for difficulty in difficulties:
            count = samples_per_difficulty + (1 if remaining > 0 else 0)
            remaining -= 1

            for i in range(count):
                task_id = f"mirror_clock_{task_counter:04d}"
                task_pair = self.generate_single_task(
                    task_id, difficulty=difficulty
                )
                task_pairs.append(task_pair)
                task_counter += 1

        return task_pairs


def create_dataset(num_samples: int = 50,
                  data_root: str = "data/questions",
                  balanced: bool = True,
                  seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Create mirror clock task dataset (main entry function)

    This is the standard entry function called by TASK_CATALOG

    Args:
        num_samples: Number of tasks to generate
        data_root: Data root directory
        balanced: Whether to generate balanced dataset across difficulties
        seed: Random seed

    Returns:
        Dataset dictionary containing task pairs
    """
    # Set random seed
    if seed is not None:
        random.seed(seed)

    # Create generator
    generator = MirrorClockTaskGenerator(data_root)
    generator.clock_generator = MirrorClockGenerator(seed)

    # Generate tasks
    if balanced:
        task_pairs = generator.generate_balanced_dataset(num_samples)
    else:
        task_pairs = []
        for i in range(num_samples):
            task_id = f"mirror_clock_{i:04d}"
            task_pair = generator.generate_single_task(task_id)
            task_pairs.append(task_pair)

    # Build dataset
    dataset = {
        "name": "mirror_clock_tasks",
        "description": "Mirror clock reasoning tasks testing spatial transformation and mirror reflection understanding",
        "domain": "mirror_clock",
        "task_category": "Spatial Reasoning",
        "pairs": [pair.to_dict() for pair in task_pairs],
        "created_at": datetime.now().isoformat(),
        "generation_info": {
            "num_samples": len(task_pairs),
            "balanced": balanced,
            "seed": seed,
            "difficulty_levels": ["easy", "medium", "hard"]
        }
    }

    return dataset


# Convenience functions
def create_single_task(task_id: str = "mirror_clock_0000",
                      hours: Optional[int] = None,
                      minutes: Optional[int] = None,
                      add_hours: Optional[int] = None,
                      add_minutes: Optional[int] = None,
                      difficulty: Optional[str] = None,
                      data_root: str = "data/questions") -> MirrorClockTaskPair:
    """
    Create a single task (for debugging/testing)

    Args:
        task_id: Task ID
        hours: Original hours (None for random)
        minutes: Original minutes (None for random)
        add_hours: Hours to add (None for random)
        add_minutes: Minutes to add (None for random)
        difficulty: Difficulty level (None for random)
        data_root: Data root directory

    Returns:
        Single task pair
    """
    generator = MirrorClockTaskGenerator(data_root)
    return generator.generate_single_task(task_id, hours, minutes, add_hours, add_minutes, difficulty)


if __name__ == "__main__":
    # Test code
    print("Testing Mirror Clock Task Generator...")

    # Test single task
    task = create_single_task("test_0001", hours=3, minutes=0, add_hours=2, add_minutes=0)
    print(f"✅ Created task: {task.id}")
    print(f"   Original time: {task.mirror_clock_data['original_time']['formatted']}")
    print(f"   Time delta: {task.mirror_clock_data['time_delta']['formatted']}")
    print(f"   Future time: {task.mirror_clock_data['future_time']['formatted']}")
    print(f"   Difficulty: {task.difficulty}")
    print(f"   Prompt: {task.prompt}")

    # Test dataset generation
    dataset = create_dataset(num_samples=30, seed=42)
    print(f"\n✅ Created dataset with {len(dataset['pairs'])} tasks")
    print(f"   Generation info: {dataset['generation_info']}")

    # Show sample tasks
    print("\nSample tasks:")
    for i, pair in enumerate(dataset['pairs'][:3]):
        data = pair['mirror_clock_data']
        print(f"  {i+1}. {data['original_time']['formatted']} + {data['time_delta']['formatted']} = {data['future_time']['formatted']}")
