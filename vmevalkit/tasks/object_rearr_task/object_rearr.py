import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import math

from PIL import Image, ImageDraw, ImageFont


IMAGE_SIZE = (1200, 1200)  # High resolution for better quality
DPI = 300  # High DPI for sharp images

# Object properties
# SHAPES = ['square', 'circle', 'triangle']
# COLORS = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']


@dataclass
class ObjectSpec:
    """Object specification for scene generation."""
    shape: str = "circle"
    color: str = "black"
    position: Tuple[float, float] = (0.5, 0.5)
    size: float = 0.1
    id: str = ""
    idx: int = 0


class ObjectRearrGenerator:
    """Generator for object rearrangement tasks."""
    
    def __init__(self):
        pass
    
    def generate_task(self, difficulty: str = 'easy') -> Dict[str, Any]:
        """Generate a single object rearrangement task with different number of objects based on difficulty."""
        # Set number of objects based on difficulty
        if difficulty == 'easy':
            num_objects = 2
        elif difficulty == 'medium':
            num_objects = 4
        elif difficulty == 'hard':
            num_objects = 9
        else:
            num_objects = 2
        
        # Generate objects - all circles with indices
        initial_objects = []
        
        # Generate random positions (normalized 0-1, with margin to avoid edge clipping)
        margin = 0.1
        positions = []
        
        # For 2 objects: split left/right
        # For 4 objects: 2x2 grid
        # For 9 objects: 3x3 grid
        if num_objects == 2:
            positions = [
                (random.uniform(margin, 0.5 - margin), random.uniform(margin, 1 - margin)),
                (random.uniform(0.5 + margin, 1 - margin), random.uniform(margin, 1 - margin))
            ]
        elif num_objects == 4:
            # 2x2 grid
            grid_positions = [
                (0.25, 0.25), (0.75, 0.25),
                (0.25, 0.75), (0.75, 0.75)
            ]
            for pos in grid_positions:
                positions.append((
                    random.uniform(pos[0] - 0.15, pos[0] + 0.15),
                    random.uniform(pos[1] - 0.15, pos[1] + 0.15)
                ))
        elif num_objects == 9:
            # 3x3 grid
            grid_positions = [
                (0.2, 0.2), (0.5, 0.2), (0.8, 0.2),
                (0.2, 0.5), (0.5, 0.5), (0.8, 0.5),
                (0.2, 0.8), (0.5, 0.8), (0.8, 0.8)
            ]
            for pos in grid_positions:
                positions.append((
                    random.uniform(pos[0] - 0.1, pos[0] + 0.1),
                    random.uniform(pos[1] - 0.1, pos[1] + 0.1)
                ))
        
        # Ensure positions are within bounds
        positions = [
            (max(margin, min(1 - margin, x)), max(margin, min(1 - margin, y)))
            for x, y in positions
        ]
        
        # Create initial objects - all circles with indices
        for i in range(num_objects):
            obj = ObjectSpec(
                shape="circle",
                color="black",
                position=positions[i],
                size=0.1,
                id=f"object{i+1}",
                idx=i
            )
            initial_objects.append(obj)
        
        # Create final objects by swapping positions of first two objects
        final_objects = []
        for i in range(num_objects):
            obj_initial = initial_objects[i]
            # Swap positions of first two objects, keep others in place
            if i == 0:
                new_position = initial_objects[1].position
            elif i == 1:
                new_position = initial_objects[0].position
            else:
                new_position = obj_initial.position
            
            obj_final = ObjectSpec(
                shape=obj_initial.shape,
                color=obj_initial.color,
                position=new_position,
                size=obj_initial.size,
                id=obj_initial.id,
                idx=obj_initial.idx
            )
            final_objects.append(obj_final)
        
        # Generate prompt - always use swap format for first two objects
        prompt = f"Swap the positions of circle {initial_objects[0].idx} and circle {initial_objects[1].idx}."
        
        return {
            'initial_objects': initial_objects,
            'final_objects': final_objects,
            'num_objects': num_objects,
            'num_steps': 1,
            'spatial_relations': ['swap'],
            'target_relation': {'type': 'swap', 'value': 'positions'},
            'prompt': prompt,
            'difficulty': difficulty
        }
    
    def _render_scene(self, objects: List[ObjectSpec], output_path: Path, is_final: bool = False, num_objects: int = 2):
        """Render scene with objects using PIL for better performance."""
        # Create image with white background
        img = Image.new('RGB', IMAGE_SIZE, color='white')
        draw = ImageDraw.Draw(img)
        
        # Use default font for text rendering
        font = ImageFont.load_default(size=40)
        
        # Calculate border width based on image size
        border_width = max(4, int(IMAGE_SIZE[0] / 200))
        
        # Draw each object as circle with index label
        for obj in objects:
            x, y = obj.position
            size = obj.size
            
            # Convert normalized coordinates to pixel coordinates
            pixel_x = x * IMAGE_SIZE[0]
            pixel_y = (1 - y) * IMAGE_SIZE[1]  # Invert y-axis for image coordinates
            pixel_size = size * IMAGE_SIZE[0]
            half_size = pixel_size / 2
            
            # Draw circle with black border
            left = pixel_x - half_size
            top = pixel_y - half_size
            right = pixel_x + half_size
            bottom = pixel_y + half_size
            draw.ellipse([left, top, right, bottom], fill='white', outline='black', width=border_width)
            
            # Draw index text in the center of the circle
            idx_text = str(obj.idx)
            # Get text bounding box to center it
            bbox = draw.textbbox((0, 0), idx_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = pixel_x - text_width / 2
            text_y = pixel_y - text_height / 2
            draw.text((text_x, text_y), idx_text, fill='black', font=font)
        
        # Save with high DPI for better quality
        img.save(output_path, format='PNG', dpi=(DPI, DPI))
    
    def generate_tasks(self, num_tasks: int) -> List[Dict[str, Any]]:
        """Generate multiple object rearrangement tasks."""
        tasks = []
        for i in range(num_tasks):
            difficulty = random.choice(['easy', 'medium', 'hard'])
            task = self.generate_task(difficulty=difficulty)
            tasks.append(task)
        return tasks


def create_task_pair(task_data: Dict[str, Any], task_id: str, base_dir: Path = None, generator: ObjectRearrGenerator = None) -> Dict[str, Any]:
    """Create a task pair in VMEvalKit format."""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent.parent
    
    # Create temp directory for images
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    
    # Generate images
    first_image_path = temp_dir / f"{task_id}_first.png"
    final_image_path = temp_dir / f"{task_id}_final.png"
    
    # Reuse generator if provided, otherwise create new one
    if generator is None:
        generator = ObjectRearrGenerator()
    
    # Convert ObjectSpec to list for rendering
    initial_objects = task_data['initial_objects']
    final_objects = task_data['final_objects']
    num_objects = task_data['num_objects']
    
    generator._render_scene(initial_objects, first_image_path, is_final=False, num_objects=num_objects)
    generator._render_scene(final_objects, final_image_path, is_final=True, num_objects=num_objects)
    
    # Create task pair
    return {
        "id": task_id,
        "prompt": task_data['prompt'],
        "first_image_path": str(first_image_path),
        "final_image_path": str(final_image_path),
        "task_category": "ObjectRearr",
        "scene_data": {
            "generation_method": "2d_simplified_visualization",
            "num_objects": task_data['num_objects'],
            "num_steps": task_data['num_steps'],
            "spatial_relations": task_data['spatial_relations'],
            "target_relation": task_data['target_relation']
        },
        "difficulty": task_data['difficulty'],
        "num_objects": task_data['num_objects'],
        "num_steps": task_data['num_steps'],
        "objects": [
            {
                "shape": obj.shape,
                "color": obj.color,
                "position": obj.position,
                "size": obj.size,
                "id": obj.id,
                "idx": obj.idx
            }
            for obj in initial_objects
        ],
        "spatial_relations": task_data['spatial_relations'],
        "manipulation_type": "telekinetic",  # Simplified for 2D visualization
        "created_at": datetime.now().isoformat()
    }


def create_dataset(num_samples: int = 30, difficulty_distribution: Dict[str, float] = None) -> Dict[str, Any]:
    """Create object rearrangement dataset - main entry point matching other domains."""
    
    print(f"🎯 Creating Object Rearrangement Dataset")
    print(f"   Total samples: {num_samples}")
    print(f"   Difficulty distribution: {difficulty_distribution}")
    
    start_time = datetime.now()
    
    # Generate tasks
    generator = ObjectRearrGenerator()
    tasks = generator.generate_tasks(num_samples)
    
    # Create task pairs (reuse generator for efficiency)
    pairs = []
    for i, task_data in enumerate(tasks):
        task_id = f"obj_rearrange_{i:04d}"
        pair = create_task_pair(task_data, task_id, generator=generator)
        pairs.append(pair)
        print(f"✅ Created task {task_id} ({pair['difficulty']})")
    
    # Create dataset
    dataset = {
        "name": "object_rearrangement_tasks",
        "description": f"Object rearrangement tasks with spatial manipulation instructions ({len(pairs)} pairs)",
        "pairs": pairs
    }
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ Dataset creation complete!")
    print(f"   Total tasks: {len(pairs)}")
    print(f"   Time elapsed: {elapsed:.1f}s")
    
    return dataset

