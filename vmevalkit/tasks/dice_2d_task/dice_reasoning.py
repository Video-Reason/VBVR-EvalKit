"""
2D Dice Opposite Face Reasoning Task

Evaluates video generation models' visual reasoning ability through dice
opposite face problems with varying complexity based on number of dice.

Difficulty Levels:
- Easy (1 dice): Direct opposite face question
- Medium (2 dice): Visual selection + opposite face
- Hard (3 dice): Multi-dice comparison + opposite face
- Expert (4 dice): Complex selection + opposite face
"""

import random
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from itertools import product
import json

from PIL import Image, ImageDraw, ImageFont


@dataclass
class DiceTaskPair:
    """Data structure for a single dice reasoning task"""
    id: str
    first_frame_path: str
    final_frame_path: str
    prompt: str
    dice_data: Dict[str, Any]
    difficulty: str


class DiceRenderer:
    """Render 2D dice faces - single or multiple"""

    def __init__(self, face_size: int = 200, padding: int = 20):
        self.face_size = face_size
        self.padding = padding

    def _get_dot_positions(self, number: int) -> List[Tuple[int, int]]:
        """Get dot positions for dice face (1-6) in 3x3 grid"""
        patterns = {
            1: [(0, 0)],
            2: [(-1, -1), (1, 1)],
            3: [(-1, -1), (0, 0), (1, 1)],
            4: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
            5: [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)],
            6: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)]
        }
        return patterns.get(number, [])

    def _draw_rounded_rect(self, draw: ImageDraw, bbox: List[int],
                           radius: int, fill: str, outline: str, width: int):
        """Draw rounded rectangle (compatible with older Pillow)"""
        x1, y1, x2, y2 = bbox

        # Try new method first, fall back to manual drawing
        try:
            draw.rounded_rectangle(bbox, radius=radius, fill=fill,
                                  outline=outline, width=width)
        except AttributeError:
            # Manual rounded rectangle for older Pillow
            draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
            draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)
            draw.pieslice([x1, y1, x1 + 2*radius, y1 + 2*radius], 180, 270, fill=fill)
            draw.pieslice([x2 - 2*radius, y1, x2, y1 + 2*radius], 270, 360, fill=fill)
            draw.pieslice([x1, y2 - 2*radius, x1 + 2*radius, y2], 90, 180, fill=fill)
            draw.pieslice([x2 - 2*radius, y2 - 2*radius, x2, y2], 0, 90, fill=fill)
            # Draw outline
            draw.arc([x1, y1, x1 + 2*radius, y1 + 2*radius], 180, 270, fill=outline, width=width)
            draw.arc([x2 - 2*radius, y1, x2, y1 + 2*radius], 270, 360, fill=outline, width=width)
            draw.arc([x1, y2 - 2*radius, x1 + 2*radius, y2], 90, 180, fill=outline, width=width)
            draw.arc([x2 - 2*radius, y2 - 2*radius, x2, y2], 0, 90, fill=outline, width=width)
            draw.line([x1 + radius, y1, x2 - radius, y1], fill=outline, width=width)
            draw.line([x1 + radius, y2, x2 - radius, y2], fill=outline, width=width)
            draw.line([x1, y1 + radius, x1, y2 - radius], fill=outline, width=width)
            draw.line([x2, y1 + radius, x2, y2 - radius], fill=outline, width=width)

    def _draw_single_dice(self, draw: ImageDraw, x: int, y: int,
                          number: int, label: Optional[str] = None):
        """Draw a single dice face at position (x, y)"""
        size = self.face_size
        margin = 15

        # Draw rounded rectangle
        self._draw_rounded_rect(
            draw,
            [x + margin, y + margin, x + size - margin, y + size - margin],
            radius=15,
            fill='#FFFFFF',
            outline='#333333',
            width=3
        )

        # Draw dots
        center_x = x + size // 2
        center_y = y + size // 2
        spacing = size // 4
        dot_radius = size // 18

        for row, col in self._get_dot_positions(number):
            dot_x = center_x + col * spacing
            dot_y = center_y + row * spacing
            draw.ellipse(
                [dot_x - dot_radius, dot_y - dot_radius,
                 dot_x + dot_radius, dot_y + dot_radius],
                fill='#000000'
            )

        # Draw label if provided
        if label:
            label_y = y + size - 8
            draw.text((x + size // 2, label_y), label,
                     fill='#666666', anchor='mm')

    def draw_dice_face(self, number: int) -> Image.Image:
        """Draw a single dice face"""
        img = Image.new('RGB', (self.face_size, self.face_size), color='#F5F5F5')
        draw = ImageDraw.Draw(img)
        self._draw_single_dice(draw, 0, 0, number)
        return img

    def draw_multiple_dice(self, numbers: List[int],
                           labels: Optional[List[str]] = None) -> Image.Image:
        """
        Draw multiple dice faces horizontally

        Args:
            numbers: List of dice values (1-6)
            labels: Optional labels for each dice (A, B, C, ...)

        Returns:
            PIL Image with all dice
        """
        n = len(numbers)
        if n == 1:
            return self.draw_dice_face(numbers[0])

        # Calculate image dimensions
        width = n * self.face_size + (n - 1) * self.padding
        height = self.face_size + 30  # Extra space for labels

        img = Image.new('RGB', (width, height), color='#F5F5F5')
        draw = ImageDraw.Draw(img)

        # Default labels
        if labels is None:
            labels = [chr(65 + i) for i in range(n)]  # A, B, C, D...

        # Draw each dice
        for i, (num, label) in enumerate(zip(numbers, labels)):
            x = i * (self.face_size + self.padding)
            self._draw_single_dice(draw, x, 0, num, label)

        return img


class MultiDiceReasoningGenerator:
    """Generate multi-dice reasoning tasks with varying difficulty"""

    OPPOSITE_MAP = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.renderer = DiceRenderer()

    def get_opposite(self, face: int) -> int:
        """Get opposite face (sum = 7)"""
        return self.OPPOSITE_MAP[face]

    def generate_dice_values(self, num_dice: int,
                             constraint: Optional[str] = None) -> List[int]:
        """
        Generate dice values with optional constraints

        Args:
            num_dice: Number of dice to generate
            constraint: Optional constraint like 'unique', 'has_even', 'has_odd'
        """
        if constraint == 'unique' and num_dice <= 6:
            return random.sample(range(1, 7), num_dice)
        elif constraint == 'has_even':
            values = [random.choice([2, 4, 6])]  # Ensure at least one even
            values.extend([random.randint(1, 6) for _ in range(num_dice - 1)])
            random.shuffle(values)
            return values
        elif constraint == 'has_odd':
            values = [random.choice([1, 3, 5])]  # Ensure at least one odd
            values.extend([random.randint(1, 6) for _ in range(num_dice - 1)])
            random.shuffle(values)
            return values
        else:
            return [random.randint(1, 6) for _ in range(num_dice)]

    def generate_easy_task(self) -> Dict[str, Any]:
        """Generate 1-dice task"""
        shown = random.randint(1, 6)
        answer = self.get_opposite(shown)

        prompts = [
            "What number is on the opposite face of this dice?",
            "Show the face opposite to this one.",
            "This dice shows {shown}. What's on the opposite side?",
            "Find the opposite face of this dice.",
        ]
        prompt = random.choice(prompts).format(shown=shown)

        return {
            'dice_values': [shown],
            'answer': answer,
            'prompt': prompt,
            'reasoning_type': 'direct_opposite'
        }

    def generate_medium_task(self) -> Dict[str, Any]:
        """Generate 2-dice task with selection"""
        dice = self.generate_dice_values(2)

        # Different question types
        task_types = [
            ('larger', max(dice), "Show the opposite of the larger dice."),
            ('smaller', min(dice), "Show the opposite of the smaller dice."),
            ('dice_a', dice[0], "Show the opposite of dice A."),
            ('dice_b', dice[1], "Show the opposite of dice B."),
        ]

        # Add even/odd if applicable
        evens = [d for d in dice if d % 2 == 0]
        odds = [d for d in dice if d % 2 == 1]

        if len(evens) == 1:
            task_types.append(('even', evens[0],
                "Show the opposite of the dice with an even number."))
        if len(odds) == 1:
            task_types.append(('odd', odds[0],
                "Show the opposite of the dice with an odd number."))

        task_type, target, prompt = random.choice(task_types)
        answer = self.get_opposite(target)

        return {
            'dice_values': dice,
            'answer': answer,
            'prompt': prompt,
            'reasoning_type': task_type,
            'target_dice': target
        }

    def generate_hard_task(self) -> Dict[str, Any]:
        """Generate 3-dice task with complex selection"""
        dice = self.generate_dice_values(3, constraint='unique')
        sorted_dice = sorted(dice)

        # For 3 dice: second_largest = second_smallest = middle, so only use 'middle'
        task_types = [
            ('largest', sorted_dice[2], "Show the opposite of the largest dice."),
            ('smallest', sorted_dice[0], "Show the opposite of the smallest dice."),
            ('middle', sorted_dice[1], "Show the opposite of the middle-valued dice."),
            ('dice_a', dice[0], "Show the opposite of dice A."),
            ('dice_b', dice[1], "Show the opposite of dice B."),
            ('dice_c', dice[2], "Show the opposite of dice C."),
        ]

        task_type, target, prompt = random.choice(task_types)
        answer = self.get_opposite(target)

        return {
            'dice_values': dice,
            'answer': answer,
            'prompt': prompt,
            'reasoning_type': task_type,
            'target_dice': target
        }

    def generate_expert_task(self) -> Dict[str, Any]:
        """Generate 4-dice task with complex reasoning"""
        dice = self.generate_dice_values(4, constraint='unique')
        sorted_dice = sorted(dice)

        task_types = [
            ('largest', sorted_dice[3], "Show the opposite of the largest dice."),
            ('smallest', sorted_dice[0], "Show the opposite of the smallest dice."),
            ('second_largest', sorted_dice[2], "Show the opposite of the second largest dice."),
            ('second_smallest', sorted_dice[1], "Show the opposite of the second smallest dice."),
            ('dice_a', dice[0], "Show the opposite of dice A."),
            ('dice_b', dice[1], "Show the opposite of dice B."),
            ('dice_c', dice[2], "Show the opposite of dice C."),
            ('dice_d', dice[3], "Show the opposite of dice D."),
        ]

        task_type, target, prompt = random.choice(task_types)
        answer = self.get_opposite(target)

        return {
            'dice_values': dice,
            'answer': answer,
            'prompt': prompt,
            'reasoning_type': task_type,
            'target_dice': target
        }

    def generate_task(self, difficulty: str) -> Dict[str, Any]:
        """Generate a task based on difficulty"""
        generators = {
            'easy': self.generate_easy_task,
            'medium': self.generate_medium_task,
            'hard': self.generate_hard_task,
            'expert': self.generate_expert_task
        }
        return generators.get(difficulty, self.generate_easy_task)()


class DiceTaskGenerator:
    """Main task generator with multi-dice support"""

    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize generator with optional temp directory.

        Args:
            temp_dir: Directory for temp files. If None, creates a temp directory.
        """
        if temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="dice_task_"))
        else:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.reasoning_generator = MultiDiceReasoningGenerator()
        self.renderer = DiceRenderer()

    def generate_single_task(self, task_id: str,
                            difficulty: Optional[str] = None) -> DiceTaskPair:
        """Generate a single dice reasoning task"""
        if difficulty is None:
            difficulty = random.choice(['easy', 'medium', 'hard', 'expert'])

        # Generate task configuration
        config = self.reasoning_generator.generate_task(difficulty)

        # Generate first frame (input dice)
        if len(config['dice_values']) == 1:
            first_img = self.renderer.draw_dice_face(config['dice_values'][0])
        else:
            first_img = self.renderer.draw_multiple_dice(config['dice_values'])

        # Generate final frame (answer)
        answer_img = self.renderer.draw_dice_face(config['answer'])

        # Save images to temp directory
        first_frame_path = self.temp_dir / f"{task_id}_first.png"
        final_frame_path = self.temp_dir / f"{task_id}_final.png"
        first_img.save(first_frame_path)
        answer_img.save(final_frame_path)

        # Build metadata
        task_metadata = {
            'dice_values': config['dice_values'],
            'num_dice': len(config['dice_values']),
            'answer': config['answer'],
            'difficulty': difficulty,
            'reasoning_type': config['reasoning_type'],
            'target_dice': config.get('target_dice'),
            'rule': 'opposite_faces_sum_to_7'
        }

        return DiceTaskPair(
            id=task_id,
            first_frame_path=str(first_frame_path),
            final_frame_path=str(final_frame_path),
            prompt=config['prompt'],
            dice_data=task_metadata,
            difficulty=difficulty
        )


def calculate_unique_combinations() -> Dict[str, int]:
    """
    Calculate the number of unique task combinations per difficulty

    Returns:
        Dictionary with unique counts per difficulty
    """
    counts = {}

    # Easy: 1 dice × 4 prompts = 6 × 4 = 24
    counts['easy'] = {
        'dice_combinations': 6,  # 1-6
        'prompt_variations': 4,
        'total': 6 * 4
    }

    # Medium: 2 dice × prompt types
    # 6×6 = 36 combinations, ~6 prompt types (larger, smaller, A, B, even, odd)
    counts['medium'] = {
        'dice_combinations': 6 * 6,  # 36
        'prompt_variations': 6,
        'total': 36 * 6  # 216
    }

    # Hard: 3 unique dice × prompt types
    # P(6,3) = 120 arrangements, 6 prompt types (largest, smallest, middle, A, B, C)
    counts['hard'] = {
        'dice_combinations': 6 * 5 * 4,  # P(6,3) = 120
        'prompt_variations': 6,
        'total': 120 * 6  # 720
    }

    # Expert: 4 unique dice × prompt types
    # P(6,4) = 360 arrangements, ~8 prompt types
    counts['expert'] = {
        'dice_combinations': 6 * 5 * 4 * 3,  # P(6,4) = 360
        'prompt_variations': 8,
        'total': 360 * 8  # 2880
    }

    counts['grand_total'] = sum(d['total'] for d in counts.values() if isinstance(d, dict))

    return counts


def create_dataset(num_samples: int = 50,
                   balanced: bool = True,
                   seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Create a dataset of dice reasoning tasks

    Args:
        num_samples: Number of tasks to generate
        balanced: Whether to balance difficulty distribution
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing task pairs and metadata
    """
    if seed is not None:
        random.seed(seed)

    generator = DiceTaskGenerator()
    difficulties = ['easy', 'medium', 'hard', 'expert']

    # Determine difficulty distribution
    if balanced:
        diff_list = []
        samples_per_diff = num_samples // 4
        remainder = num_samples % 4

        for diff in difficulties:
            diff_list.extend([diff] * samples_per_diff)

        for i in range(remainder):
            diff_list.append(difficulties[i])

        random.shuffle(diff_list)
    else:
        diff_list = [random.choice(difficulties) for _ in range(num_samples)]

    # Generate tasks
    tasks = []
    for i, diff in enumerate(diff_list):
        task_id = f"dice_{i:04d}"
        task = generator.generate_single_task(task_id, difficulty=diff)
        tasks.append({
            'id': task.id,
            'first_image_path': task.first_frame_path,
            'final_image_path': task.final_frame_path,
            'prompt': task.prompt,
            'dice_data': task.dice_data,
            'difficulty': task.difficulty
        })

    # Calculate unique combinations
    unique_counts = calculate_unique_combinations()

    dataset = {
        'pairs': tasks,
        'generation_info': {
            'num_samples': num_samples,
            'balanced': balanced,
            'seed': seed,
            'difficulty_distribution': {d: diff_list.count(d) for d in difficulties},
            'unique_combinations': unique_counts
        }
    }

    return dataset


def create_single_task(task_id: str = "dice_0000",
                      difficulty: Optional[str] = None) -> DiceTaskPair:
    """Create a single dice task"""
    generator = DiceTaskGenerator()
    return generator.generate_single_task(task_id, difficulty)


if __name__ == "__main__":
    print("=" * 60)
    print("2D Dice Reasoning Task Generator")
    print("=" * 60)

    # Calculate and display unique combinations
    print("\n📊 Unique Combinations per Difficulty:")
    counts = calculate_unique_combinations()
    for diff in ['easy', 'medium', 'hard', 'expert']:
        info = counts[diff]
        print(f"  {diff.upper():8} | {info['dice_combinations']:4} dice × {info['prompt_variations']} prompts = {info['total']:5} unique")
    print(f"  {'TOTAL':8} | {counts['grand_total']:>20} unique combinations")

    # Test each difficulty
    print("\n🎲 Sample Tasks:")
    for diff in ['easy', 'medium', 'hard', 'expert']:
        task = create_single_task(f"test_{diff}", difficulty=diff)
        data = task.dice_data
        print(f"\n  [{diff.upper()}] {task.id}")
        print(f"    Dice: {data['dice_values']} ({data['num_dice']} dice)")
        print(f"    Answer: {data['answer']}")
        print(f"    Prompt: {task.prompt}")

    # Test dataset
    print("\n📦 Dataset Generation Test:")
    dataset = create_dataset(num_samples=40, balanced=True, seed=42)
    print(f"  Generated {len(dataset['pairs'])} tasks")
    print(f"  Distribution: {dataset['generation_info']['difficulty_distribution']}")
