"""
2D Dice Opposite Face Reasoning Task

This task evaluates spatial reasoning and logical deduction by asking models
to determine the opposite face of a standard dice using the rule that
opposite faces always sum to 7.

Uses 2D rendered dice faces for visual clarity and accessibility.

Task Types:
- Easy: Direct opposite face question (show 1 face)
- Medium: Inference from multiple visible faces
- Hard: Complex spatial reasoning with constraints
"""

import random
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import json

from PIL import Image, ImageDraw

from .PROMPTS import PROMPTS, DEFAULT_PROMPT_INDEX


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
    """Render 2D dice faces showing dot patterns"""

    def __init__(self, face_size=300):
        self.face_size = face_size

    def _get_dot_positions(self, number: int):
        """
        Get dot positions for dice face (1-6)
        Returns positions in a 3x3 grid: -1, 0, 1
        """
        patterns = {
            1: [(0, 0)],
            2: [(-1, -1), (1, 1)],
            3: [(-1, -1), (0, 0), (1, 1)],
            4: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
            5: [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)],
            6: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)]
        }
        return patterns.get(number, [])

    def _draw_dot(self, draw, x, y, radius=12, color='#000000'):
        """Draw a single dot"""
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=color, outline=color)

    def draw_dice_face(self, number: int) -> Image.Image:
        """
        Draw a single dice face showing the number

        Args:
            number: Number to show (1-6)

        Returns:
            PIL Image of the dice face
        """
        # Create image
        img = Image.new('RGB', (self.face_size, self.face_size), color='white')
        draw = ImageDraw.Draw(img)

        # Draw rounded rectangle for dice face
        margin = 30
        radius = 20
        draw.rounded_rectangle(
            [margin, margin, self.face_size - margin, self.face_size - margin],
            radius=radius,
            fill='#FFFFFF',
            outline='#333333',
            width=4
        )

        # Calculate center and spacing
        center = self.face_size // 2
        spacing = self.face_size // 4

        # Get and draw dot pattern
        dot_positions = self._get_dot_positions(number)
        for row, col in dot_positions:
            dot_x = center + col * spacing
            dot_y = center + row * spacing
            self._draw_dot(draw, dot_x, dot_y, radius=14)

        return img


class DiceReasoningGenerator:
    """Generate dice opposite face reasoning tasks"""

    # Standard dice rule: opposite faces sum to 7
    OPPOSITE_MAP = {
        1: 6, 2: 5, 3: 4,
        4: 3, 5: 2, 6: 1
    }

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional random seed"""
        if seed is not None:
            random.seed(seed)
        self.dice_renderer = DiceRenderer()

    def get_opposite_face(self, face: int) -> int:
        """Get the opposite face using the rule: sum = 7"""
        return self.OPPOSITE_MAP.get(face, 7 - face)

    def generate_task_config(self, difficulty: str) -> Dict[str, Any]:
        """
        Generate task configuration based on difficulty

        Args:
            difficulty: 'easy', 'medium', or 'hard'

        Returns:
            Dictionary with task configuration
        """
        if difficulty == "easy":
            # Easy: Show one face, ask for opposite
            shown_face = random.randint(1, 6)
            opposite_face = self.get_opposite_face(shown_face)

            return {
                'type': 'direct_opposite',
                'shown_face': shown_face,
                'answer_face': opposite_face,
                'question_type': 'opposite'
            }

        elif difficulty == "medium":
            # Medium: Show one face, ask about opposite with reasoning
            shown_face = random.randint(1, 6)
            opposite_face = self.get_opposite_face(shown_face)

            return {
                'type': 'opposite_with_rule',
                'shown_face': shown_face,
                'answer_face': opposite_face,
                'question_type': 'opposite_rule',
                'rule': 'opposite faces sum to 7'
            }

        else:  # hard
            # Hard: Given opposite is X, what is shown face?
            answer_face = random.randint(1, 6)
            shown_face = self.get_opposite_face(answer_face)

            return {
                'type': 'reverse_reasoning',
                'shown_face': shown_face,
                'answer_face': answer_face,
                'question_type': 'reverse',
                'constraint': f'opposite face is {shown_face}'
            }


class DiceTaskGenerator:
    """Main task generator following mirror_clock pattern"""

    def __init__(self, data_root: str = "data/questions"):
        self.data_root = Path(data_root)
        self.task_dir = self.data_root / "dice_2d_task"
        self.dice_generator = DiceReasoningGenerator()
        self.prompts = PROMPTS

    def generate_single_task(self,
                            task_id: str,
                            shown_face: Optional[int] = None,
                            difficulty: Optional[str] = None) -> DiceTaskPair:
        """
        Generate a single dice reasoning task

        Args:
            task_id: Unique task identifier
            shown_face: Face to show (None for random)
            difficulty: Task difficulty (None for random)

        Returns:
            DiceTaskPair object
        """
        # Determine difficulty
        if difficulty is None:
            difficulty = random.choice(["easy", "medium", "hard"])

        # Generate task configuration
        if shown_face is not None:
            # Use provided face
            opposite_face = self.dice_generator.get_opposite_face(shown_face)
            config = {
                'type': 'direct_opposite',
                'shown_face': shown_face,
                'answer_face': opposite_face,
                'question_type': 'opposite'
            }
        else:
            config = self.dice_generator.generate_task_config(difficulty)

        # Create task directory
        task_path = self.task_dir / task_id
        task_path.mkdir(parents=True, exist_ok=True)

        # Generate images
        shown_img = self.dice_generator.dice_renderer.draw_dice_face(config['shown_face'])
        answer_img = self.dice_generator.dice_renderer.draw_dice_face(config['answer_face'])

        # Save images
        first_frame_path = task_path / "first_frame.png"
        final_frame_path = task_path / "final_frame.png"
        shown_img.save(first_frame_path)
        answer_img.save(final_frame_path)

        # Generate prompt using template with placeholder
        prompt_template = random.choice(self.prompts)
        prompt = prompt_template.format(
            shown_number=config['shown_face']
        )

        # Save prompt
        prompt_path = task_path / "prompt.txt"
        prompt_path.write_text(prompt, encoding='utf-8')

        # Build metadata
        task_metadata = {
            'shown_face': config['shown_face'],
            'answer_face': config['answer_face'],
            'difficulty': difficulty,
            'task_type': config['type'],
            'rule': 'opposite_faces_sum_to_7'
        }

        # Save metadata
        metadata_path = task_path / "question_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(task_metadata, f, indent=2, ensure_ascii=False)

        # Create task pair
        return DiceTaskPair(
            id=task_id,
            first_frame_path=str(first_frame_path),
            final_frame_path=str(final_frame_path),
            prompt=prompt,
            dice_data=task_metadata,
            difficulty=difficulty
        )


def create_dataset(num_samples: int = 50,
                   balanced: bool = True,
                   seed: Optional[int] = None,
                   data_root: str = "data/questions") -> Dict[str, Any]:
    """
    Create a dataset of dice reasoning tasks

    Args:
        num_samples: Number of tasks to generate
        balanced: Whether to balance difficulty distribution
        seed: Random seed for reproducibility
        data_root: Root directory for data storage

    Returns:
        Dictionary containing task pairs and metadata
    """
    if seed is not None:
        random.seed(seed)

    generator = DiceTaskGenerator(data_root)

    # Determine difficulty distribution
    if balanced:
        difficulties = []
        samples_per_difficulty = num_samples // 3
        remainder = num_samples % 3

        difficulties.extend(['easy'] * samples_per_difficulty)
        difficulties.extend(['medium'] * samples_per_difficulty)
        difficulties.extend(['hard'] * samples_per_difficulty)

        # Add remainder
        for i in range(remainder):
            difficulties.append(['easy', 'medium', 'hard'][i])

        random.shuffle(difficulties)
    else:
        difficulties = [random.choice(['easy', 'medium', 'hard'])
                       for _ in range(num_samples)]

    # Generate tasks
    tasks = []
    for i, diff in enumerate(difficulties):
        task_id = f"dice_{i:04d}"
        task = generator.generate_single_task(task_id, difficulty=diff)
        tasks.append({
            'id': task.id,
            'first_frame': task.first_frame_path,
            'final_frame': task.final_frame_path,
            'prompt': task.prompt,
            'dice_data': task.dice_data,
            'difficulty': task.difficulty
        })

    # Create dataset dictionary
    dataset = {
        'pairs': tasks,
        'generation_info': {
            'num_samples': num_samples,
            'balanced': balanced,
            'seed': seed,
            'difficulty_distribution': {
                'easy': difficulties.count('easy'),
                'medium': difficulties.count('medium'),
                'hard': difficulties.count('hard')
            }
        }
    }

    return dataset


def create_single_task(task_id: str = "dice_0000",
                      shown_face: Optional[int] = None,
                      difficulty: Optional[str] = None,
                      data_root: str = "data/questions") -> DiceTaskPair:
    """
    Create a single dice task (convenience function)

    Args:
        task_id: Task identifier
        shown_face: Face to show (1-6, None for random)
        difficulty: Difficulty level (None for random)
        data_root: Data root directory

    Returns:
        DiceTaskPair object
    """
    generator = DiceTaskGenerator(data_root)
    return generator.generate_single_task(task_id, shown_face, difficulty)


if __name__ == "__main__":
    # Test code
    print("Testing Dice Reasoning Task Generator...")

    # Test single task
    task = create_single_task("test_dice_001", shown_face=3, difficulty="easy")
    print(f"✅ Created task: {task.id}")
    print(f"   Shown face: {task.dice_data['shown_face']}")
    print(f"   Answer face: {task.dice_data['answer_face']}")
    print(f"   Difficulty: {task.difficulty}")
    print(f"   Prompt: {task.prompt}")

    # Test dataset generation
    dataset = create_dataset(num_samples=30, balanced=True, seed=42)
    print(f"\n✅ Created dataset with {len(dataset['pairs'])} tasks")
    print(f"   Generation info: {dataset['generation_info']}")

    # Show sample tasks
    print("\nSample tasks:")
    for i, pair in enumerate(dataset['pairs'][:5]):
        data = pair['dice_data']
        print(f"  {i+1}. Shown={data['shown_face']}, Answer={data['answer_face']}, Difficulty={data['difficulty']}")
