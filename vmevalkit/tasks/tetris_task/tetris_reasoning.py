import random
import time
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from PIL import Image, ImageDraw
import shutil
import json
from pathlib import Path
from datetime import datetime
# Import standardized prompts used across tasks
from .PROMPTS import PROMPTS

from dataclasses import dataclass, asdict, field


# Dataclass for serializing a single Tetris task pair (matches provided metadata)
@dataclass
class TetrisTaskPair: 
    id: str
    task_category: str 
    prompt: str 
    first_image_path: str
    final_image_path: str
    map_size: List[int]  # Moved before fields with defaults
    difficulty: str = ""
    initial_state_bottom3: List[List[int]] = field(default_factory=list)
    final_state_bottom3: List[List[int]] = field(default_factory=list)
    new_block: Optional[Dict[str, Any]] = None #will use in difficult tasks
    created_at: str = ""
    simulator_version: str = "tetris_sim.py@feature/add_tetris"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of this task pair."""
        return asdict(self)

    def write_json(self, path: Union[str, Path]):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

CELL_SIZE = 40  # VMEvalKit standard: 400x400 image size
COLORS = {
    "I": (0, 255, 255),     
    "O": (255, 255, 0),      
    "T": (128, 0, 128),      
    "S": (0, 255, 0),      
    "Z": (255, 0, 0),       
    "J": (0, 0, 255),       
    "L": (255, 165, 0),     
    0: (30, 30, 30),        
}

class TetrisShape(Enum):
    I = "I"  
    O = "O" 
    T = "T"  
    S = "S"  
    Z = "Z"  
    J = "J"  
    L = "L"  
    
@dataclass

    
class TetrisBlock:
    SHAPES = {
        TetrisShape.I: [
            [(0, 0), (0, 1), (0, 2), (0, 3)],  
            [(0, 0), (1, 0), (2, 0), (3, 0)],  
        ],
        TetrisShape.O: [
            [(0, 0), (0, 1), (1, 0), (1, 1)],  
        ],
        TetrisShape.T: [
            [(0, 1), (1, 0), (1, 1), (1, 2)],  
            [(0, 1), (1, 1), (1, 2), (2, 1)],  
            [(1, 0), (1, 1), (1, 2), (2, 1)],  
            [(0, 1), (1, 0), (1, 1), (2, 1)],  
        ],
        TetrisShape.S: [
            [(0, 1), (0, 2), (1, 0), (1, 1)],  
            [(0, 0), (1, 0), (1, 1), (2, 1)],  
        ],
        TetrisShape.Z: [
            [(0, 0), (0, 1), (1, 1), (1, 2)],  
            [(0, 1), (1, 0), (1, 1), (2, 0)],  
        ],
        TetrisShape.J: [
            [(0, 0), (1, 0), (1, 1), (1, 2)],  
            [(0, 1), (0, 2), (1, 1), (2, 1)],  
            [(1, 0), (1, 1), (1, 2), (2, 2)],  
            [(0, 1), (1, 1), (2, 0), (2, 1)],  
        ],
        TetrisShape.L: [
            [(0, 2), (1, 0), (1, 1), (1, 2)],  
            [(0, 1), (1, 1), (2, 1), (2, 2)],  
            [(1, 0), (1, 1), (1, 2), (2, 0)],  
            [(0, 0), (0, 1), (1, 1), (2, 1)],  
        ],
    }
    
    def __init__(self, shape: TetrisShape, x: int = 0, y: int = 0):
        self.shape = shape
        self.x = x
        self.y = y
        self.rotation = 0  
        
    def get_coordinates(self) -> List[Tuple[int, int]]:
        shape_coords = self.SHAPES[self.shape][self.rotation]
        return [(self.x + dx, self.y + dy) for dx, dy in shape_coords]
    
    def rotate_clockwise(self):
        max_rotation = len(self.SHAPES[self.shape])
        self.rotation = (self.rotation + 1) % max_rotation
    
    def rotate_counterclockwise(self):
        max_rotation = len(self.SHAPES[self.shape])
        self.rotation = (self.rotation - 1) % max_rotation
    
    def move(self, dx: int, dy: int):
        self.x += dx
        self.y += dy
    
    def copy(self):
        new_block = TetrisBlock(self.shape, self.x, self.y)
        new_block.rotation = self.rotation
        return new_block
    
    @staticmethod
    def get_random_shape() -> TetrisShape:
        return random.choice(list(TetrisShape))
    

class TetrisMap:
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.current_block: Optional[TetrisBlock] = None
        self.score = 0
        self.lines_cleared = 0
        
    def is_valid_position(self, block: TetrisBlock) -> bool:
        for x, y in block.get_coordinates():
            if x < 0 or x >= self.height or y < 0 or y >= self.width:
                return False
            if self.grid[x][y] != 0:
                return False
        return True
    
    def place_block(self, block: TetrisBlock):
        for x, y in block.get_coordinates():
            if 0 <= x < self.height and 0 <= y < self.width:
                self.grid[x][y] = block.shape.value
    
    def spawn_new_block(self) -> bool:
        shape = TetrisBlock.get_random_shape()
        self.current_block = TetrisBlock(shape, x=0, y=self.width // 2 - 1)
        
        if not self.is_valid_position(self.current_block):
            return False  
        return True
    
    def move_block_down(self) -> bool:
        if self.current_block is None:
            return False
        
        temp_block = self.current_block.copy()
        temp_block.move(1, 0)
        
        if self.is_valid_position(temp_block):
            self.current_block = temp_block
            return True
        else:
            self.place_block(self.current_block)
            self.clear_lines()
            return False
    
    def move_block_left(self) -> bool:
        if self.current_block is None:
            return False
        
        temp_block = self.current_block.copy()
        temp_block.move(0, -1)
        
        if self.is_valid_position(temp_block):
            self.current_block = temp_block
            return True
        return False
    
    def move_block_right(self) -> bool:
        if self.current_block is None:
            return False
        
        temp_block = self.current_block.copy()
        temp_block.move(0, 1)
        
        if self.is_valid_position(temp_block):
            self.current_block = temp_block
            return True
        return False
    
    def rotate_block(self) -> bool:
        if self.current_block is None:
            return False
        
        temp_block = self.current_block.copy()
        temp_block.rotate_clockwise()
        
        if self.is_valid_position(temp_block):
            self.current_block = temp_block
            return True
        return False
    
    def clear_lines(self) -> int:
        # Find all full lines
        lines_to_clear = []
        for i in range(self.height):
            if all(self.grid[i][j] != 0 for j in range(self.width)):
                lines_to_clear.append(i)
        
        # Remove full lines and keep non-full lines
        if lines_to_clear:
            new_grid = []
            for i in range(self.height):
                if i not in lines_to_clear:
                    new_grid.append(self.grid[i])
            
            # Add empty lines at the top
            num_cleared = len(lines_to_clear)
            for _ in range(num_cleared):
                new_grid.insert(0, [0 for _ in range(self.width)])
            
            self.grid = new_grid
            self.lines_cleared += num_cleared
            self.score += num_cleared * num_cleared * 100
            
            return num_cleared
        
        return 0
    
    def hard_drop(self):
        if self.current_block is None:
            return
        
        while self.move_block_down():
            pass
    
    def get_display_grid(self) -> List[List[str]]:
        display = [row[:] for row in self.grid]
        
        if self.current_block:
            for x, y in self.current_block.get_coordinates():
                if 0 <= x < self.height and 0 <= y < self.width:
                    display[x][y] = self.current_block.shape.value
        
        return display
    
    def print_map(self):
        print("\n" + "=" * (self.width * 2 + 2))
        display = self.get_display_grid()
        for row in display:
            print("|" + "".join(f"{cell if cell != 0 else '.':<2}" for cell in row) + "|")
        print("=" * (self.width * 2 + 2))
        print(f"Score: {self.score} | Lines cleared: {self.lines_cleared}")
    
    def render_to_image(self, save_path: str = "tetris.png"):
        display = self.get_display_grid()
        h = len(display)
        w = len(display[0])
        
        img = Image.new("RGB", (w * CELL_SIZE, h * CELL_SIZE), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        for x in range(h):
            for y in range(w):
                block = display[x][y]
                color = COLORS.get(block, (100, 100, 100))
                draw.rectangle(
                    [y * CELL_SIZE, x * CELL_SIZE, (y + 1) * CELL_SIZE, (x + 1) * CELL_SIZE],
                    fill=color,
                    outline=(50, 50, 50)
                )
        
        img.save(save_path)
        print(f"✔ Image saved: {save_path}")
    
    def is_game_over(self) -> bool:
        return any(self.grid[0][j] != 0 for j in range(self.width))
    
    def initialize_with_random_blocks(self, max_blocks: int = 7, max_height: int = 3, strategy: str = "mixed"):
        """
        Initialize the bottom of the map with blocks (simulate already-placed blocks).
        :param max_blocks: maximum number of blocks to generate (default 7)
        :param max_height: max number of bottom rows to use (default 3)
        :param strategy: generation strategy ("random", "flat", "mixed")
        """
        print(f"\nInitializing map: placing blocks in the bottom {max_height} rows...")

        # choose generation strategy
        if strategy == "mixed":
            # mixed strategy: 70% flat, 30% random
            use_flat = random.random() < 0.7
        elif strategy == "flat":
            use_flat = True
        else:
            use_flat = False

        if use_flat:
            print("  Strategy: flat fill (near-clearing state)")
            self._initialize_flat_bottom(max_height)
        else:
            print("  Strategy: random drop")
            self._initialize_random_drop(max_blocks, max_height)

        # show the initialized map
        self.print_map()
    
    def _initialize_flat_bottom(self, max_height: int = 3):
        """
        Flat-fill strategy: create bottom rows that are close to clearing.
        The bottom 1-2 rows are almost full, leaving 1-2 gaps per row.
        """
        # decide how many rows to fill (1 or 2)
        num_rows_to_fill = random.randint(1, min(2, max_height))

        print(f"  Filling bottom {num_rows_to_fill} rows, leaving 1-2 gaps per row")

        # record gap positions for each row
        gap_positions_per_row = []

        for row_offset in range(num_rows_to_fill):
            row = self.height - 1 - row_offset  # start from bottom

            if row_offset == 0:
                # bottom row: choose 1-2 gaps
                num_gaps = random.randint(1, 2)
                gap_positions = random.sample(range(self.width), num_gaps)
            else:
                # upper rows must include gaps from lower rows (to ensure support)
                previous_all_gaps = set()
                for prev_gaps in gap_positions_per_row:
                    previous_all_gaps.update(prev_gaps)

                gap_positions = list(previous_all_gaps)

                # optionally add one extra gap
                remaining_cols = [c for c in range(self.width) if c not in gap_positions]
                if remaining_cols and random.random() < 0.3:
                    extra_gaps = random.randint(1, min(1, len(remaining_cols)))
                    gap_positions.extend(random.sample(remaining_cols, extra_gaps))

            gap_positions_per_row.append(gap_positions)

            print(f"  Row {row}: leaving {len(gap_positions)} gaps at positions {gap_positions}")

            # fill this row except for gaps
            shapes = list(TetrisShape)
            for col in range(self.width):
                if col not in gap_positions:
                    self.grid[row][col] = random.choice(shapes).value

        # Place some scattered blocks on the third row only on fully supported columns
        if max_height >= 3:
            third_row = self.height - 3
            supported_positions = []
            for col in range(self.width):
                is_fully_supported = True
                for check_row in range(third_row + 1, self.height):
                    if self.grid[check_row][col] == 0:
                        is_fully_supported = False
                        break
                if is_fully_supported:
                    supported_positions.append(col)

            if supported_positions:
                num_blocks_third_row = random.randint(0, min(len(supported_positions), self.width // 2))
                if num_blocks_third_row > 0:
                    positions = random.sample(supported_positions, num_blocks_third_row)
                    shapes = list(TetrisShape)
                    for col in positions:
                        self.grid[third_row][col] = random.choice(shapes).value
                    print(f"  Row {third_row}: randomly placed {num_blocks_third_row} blocks among {len(supported_positions)} fully supported columns")

        print(f"  Flat-fill initialization complete")
    
    def _initialize_random_drop(self, max_blocks: int, max_height: int):
        """
        Random-drop strategy: simulate natural falling of pieces.
        """
        # decide how many blocks to attempt
        num_blocks = random.randint(1, max_blocks)
        placed_blocks = 0
        attempts = 0
        max_attempts = max_blocks * 20

        while placed_blocks < num_blocks and attempts < max_attempts:
            attempts += 1

            shape = TetrisBlock.get_random_shape()
            block = TetrisBlock(shape, x=0, y=0)

            # random rotation
            num_rotations = random.randint(0, len(TetrisBlock.SHAPES[shape]) - 1)
            for _ in range(num_rotations):
                block.rotate_clockwise()

            # random horizontal position
            y_pos = random.randint(0, self.width - 1)
            block.x = 0
            block.y = y_pos

            # adjust horizontal position to fit
            coords = block.get_coordinates()
            min_y = min(dy for _, dy in coords)
            max_y = max(dy for _, dy in coords)
            if min_y < 0:
                block.y -= min_y
            if max_y >= self.width:
                block.y -= (max_y - self.width + 1)

            # simulate fall to lowest valid position
            final_x = 0
            for x in range(self.height):
                block.x = x
                coords = block.get_coordinates()
                max_block_x = max(bx for bx, _ in coords)
                if max_block_x >= self.height:
                    continue

                collision = False
                for bx, by in coords:
                    if bx < 0 or bx >= self.height or by < 0 or by >= self.width:
                        collision = True
                        break
                    if self.grid[bx][by] != 0:
                        collision = True
                        break

                if not collision:
                    final_x = x
                else:
                    break

            block.x = final_x
            coords = block.get_coordinates()

            min_allowed_row = self.height - max_height
            if all(bx >= min_allowed_row for bx, _ in coords):
                valid = True
                for bx, by in coords:
                    if bx < 0 or bx >= self.height or by < 0 or by >= self.width:
                        valid = False
                        break
                    if self.grid[bx][by] != 0:
                        valid = False
                        break

                if valid:
                    self.place_block(block)
                    placed_blocks += 1
                    print(f"  Placed block #{placed_blocks}: {shape.value} (rotations={num_rotations})")

        print(f"  Random drop complete! Placed {placed_blocks} blocks")
        
        
def _smart_fill_bottom_rows(tetris_map: TetrisMap, num_rows: int, fill_ratio: float, guarantee_clear: bool):
    """
    Smart fill bottom rows (ensuring no floating blocks)
    
    Args:
        tetris_map: Tetris map object
        num_rows: Number of rows to fill
        fill_ratio: Fill ratio for each row (0.0-1.0)
        guarantee_clear: Whether to guarantee at least one line can be cleared
    """
    n = tetris_map.width
    shapes = list(TetrisShape)
    
    # Decide which rows will be full (for clearing)
    if guarantee_clear is True:
        # Guarantee at least one full line
        num_full_lines = random.randint(1, max(1, num_rows // 2))
        full_line_indices = random.sample(range(num_rows), num_full_lines)
        print(f"  Strategy: guarantee clear mode - will have {num_full_lines} full line(s)")
    elif guarantee_clear is False:
        # Guarantee no full lines
        full_line_indices = []
        print(f"  Strategy: guarantee no-clear mode - all rows will have at least 1 gap")
    else:
        # Random mode: naturally decide based on fill_ratio
        full_line_indices = []
        if fill_ratio >= 0.95 and random.random() < 0.5:
            # High fill ratio has 50% chance to generate full lines
            num_full_lines = random.randint(1, max(1, num_rows // 3))
            full_line_indices = random.sample(range(num_rows), num_full_lines)
            print(f"  Strategy: random mode - generating {num_full_lines} full line(s)")
        else:
            print(f"  Strategy: random mode - filling at {fill_ratio:.0%} ratio")
    
    # Fill from bottom up (ensuring no floating blocks)
    for i in range(num_rows):
        row_idx = tetris_map.height - 1 - i  # Start from bottom
        
        if i in full_line_indices:
            # Full line case
            if i == 0:
                # Bottom layer: can fill directly
                for col in range(n):
                    tetris_map.grid[row_idx][col] = random.choice(shapes).value
                print(f"  Row {row_idx}: full line (100%)")
            else:
                # Non-bottom layer: need to check support below
                row_below = tetris_map.height - i
                supported_cols = [col for col in range(n) if tetris_map.grid[row_below][col] != 0]
                
                if len(supported_cols) == n:
                    # Row below is full, safe to fill
                    for col in range(n):
                        tetris_map.grid[row_idx][col] = random.choice(shapes).value
                    print(f"  Row {row_idx}: full line (100%)")
                else:
                    # Row below not full, can only fill supported positions
                    # Downgrade to non-full line strategy
                    if len(supported_cols) > 0:
                        for col in supported_cols:
                            tetris_map.grid[row_idx][col] = random.choice(shapes).value
                        print(f"  Row {row_idx}: {len(supported_cols)}/{n} cells ({len(supported_cols)/n:.0%}) [planned full line but no complete support below]")
                    else:
                        print(f"  Row {row_idx}: skipped (no support below)")
        else:
            # Non-full line case
            # Fill according to fill_ratio
            if guarantee_clear is False:
                # Guarantee no clear: leave at least one gap
                num_filled = random.randint(int(n * fill_ratio * 0.7), n - 1)
            else:
                # Normal mode: fill by ratio
                num_filled = int(n * fill_ratio + random.uniform(-0.2, 0.2) * n)
                num_filled = max(0, min(n - 1, num_filled))
            
            if i == 0:
                # Bottom layer: can place randomly
                positions = random.sample(range(n), num_filled)
                for col in positions:
                    tetris_map.grid[row_idx][col] = random.choice(shapes).value
                print(f"  Row {row_idx}: {num_filled}/{n} cells ({num_filled/n:.0%})")
            else:
                # Non-bottom layer: can only place on supported positions
                row_below = tetris_map.height - i
                supported_cols = [col for col in range(n) if tetris_map.grid[row_below][col] != 0]
                
                if not supported_cols:
                    print(f"  Row {row_idx}: skipped (no support below)")
                    continue
                
                # Randomly select from supported positions
                actual_num_filled = min(num_filled, len(supported_cols))
                if actual_num_filled > 0:
                    positions = random.sample(supported_cols, actual_num_filled)
                    for col in positions:
                        tetris_map.grid[row_idx][col] = random.choice(shapes).value
                    print(f"  Row {row_idx}: {actual_num_filled}/{n} cells ({actual_num_filled/n:.0%}) [supported: {len(supported_cols)} cols]")
                else:
                    print(f"  Row {row_idx}: 0/{n} cells (0%) [supported: {len(supported_cols)} cols]")
    
    print(f"  Smart fill complete! Filled {num_rows} rows, ensuring no floating blocks")

class TetrisEasyTaskGenerator:
    def __init__(self):
        self.map_width = 5
        self.num_rows = 2  # For 5x5 map, use 2 rows to ensure interesting line-clearing scenarios
        self.fill_ratio = 0.8
        self.tetris_map = TetrisMap(width=self.map_width, height=self.map_width)
        
    def generate_single_task(self, task_id: str, guarantee_clear: Optional[bool] = None) -> TetrisTaskPair:
        """Generate a single task pair.

        guarantee_clear: pass True/False/None to _smart_fill_bottom_rows. If None, the function
        will run in random mode (as implemented in _smart_fill_bottom_rows).
        """
        _smart_fill_bottom_rows(
            self.tetris_map,
            self.num_rows,
            self.fill_ratio,
            guarantee_clear,  # allow True/False/None
        )
        bottom_rows_initial = []
        temp_dir = tempfile.mkdtemp()
        start_row = self.tetris_map.height - self.num_rows
        for r in range(start_row, self.tetris_map.height):
            bottom_rows_initial.append([cell for cell in self.tetris_map.grid[r]])
        first_temp_path = os.path.join(temp_dir, f"{task_id}_first.png")
        self.tetris_map.render_to_image(first_temp_path)
        
        lines_cleared = self.tetris_map.clear_lines()
        bottom_rows_final = []
        actual_rows = min(self.num_rows, self.tetris_map.height)
        start_row_final = self.tetris_map.height - actual_rows
        for r in range(start_row_final, self.tetris_map.height):
            bottom_rows_final.append([cell for cell in self.tetris_map.grid[r]])
        final_temp_path = os.path.join(temp_dir, f"{task_id}_final.png")
        self.tetris_map.render_to_image(final_temp_path)
        prompt = PROMPTS[0].format(difficulty="easy", n=self.map_width)
        task_pair = TetrisTaskPair(
            id=task_id,
            task_category="Tetris",
            prompt=prompt,
            first_image_path=first_temp_path,
            final_image_path=final_temp_path,
            difficulty="easy",
            initial_state_bottom3=bottom_rows_initial,
            final_state_bottom3=bottom_rows_final,
            map_size=[self.tetris_map.width, self.tetris_map.height],
            created_at=datetime.now().isoformat()
        )
        return task_pair
        
        
    def generate_dataset(self, num_samples: int = 10) -> Dict[str, Any]:
        tasks = []
        # Ensure that across generated samples we cover True, False and None at least once
        options = [True, False, None]
        if num_samples >= len(options):
            # include each option once, then fill remaining with random choices
            guaranteed_list = options.copy()
            remaining = num_samples - len(options)
            guaranteed_list += [random.choice(options) for _ in range(remaining)]
        else:
            # fewer samples than options: pick a subset without duplicates
            guaranteed_list = random.sample(options, k=num_samples)

        random.shuffle(guaranteed_list)

        for sample_idx in range(num_samples):
            task_id = f"tetris_{sample_idx:04d}"  # Format as string like "tetris_0001"
            guarantee_choice = guaranteed_list[sample_idx]
            task = self.generate_single_task(task_id, guarantee_clear=guarantee_choice)
            tasks.append(task)
        task_dicts = []
        for task in tasks:
            task_dict = {
                'id': task.id,
                'prompt': task.prompt,
                'task_category': task.task_category,
                'first_image_path': task.first_image_path,
                'final_image_path': task.final_image_path,
                'initial_state_bottom3': task.initial_state_bottom3,
                'final_state_bottom3': task.final_state_bottom3,
                'map_size': task.map_size,
                'difficulty': task.difficulty,
                'created_at': task.created_at
            }
            task_dicts.append(task_dict)
        dataset_dict = {
            'name': 'Tetris Easy Dataset',
            'description': 'A dataset of easy Tetris tasks generated by TetrisEasyTaskGenerator for video model reasoning evaluation.',
            'pairs': task_dicts  # Changed from 'tasks' to 'pairs' to match other tasks
        }
        return dataset_dict


class TetrisMediumTaskGenerator:
    def __init__(self):
        self.map_width = 10
        self.num_rows = 3  # Fixed width of 3 for medium
        self.fill_ratio = 0.8
        self.tetris_map = TetrisMap(width=self.map_width, height=self.map_width)
        
    def generate_single_task(self, task_id: str, guarantee_clear: Optional[bool] = None) -> TetrisTaskPair:
        """Generate a single medium task pair (10x10 map, 3-row initialization).

        guarantee_clear: pass True/False/None to _smart_fill_bottom_rows. If None, the function
        will run in random mode (as implemented in _smart_fill_bottom_rows).
        """
        _smart_fill_bottom_rows(
            self.tetris_map,
            self.num_rows,
            self.fill_ratio,
            guarantee_clear,  # allow True/False/None
        )
        bottom_rows_initial = []
        temp_dir = tempfile.mkdtemp()
        start_row = self.tetris_map.height - self.num_rows
        for r in range(start_row, self.tetris_map.height):
            bottom_rows_initial.append([cell for cell in self.tetris_map.grid[r]])
        first_temp_path = os.path.join(temp_dir, f"{task_id}_first.png")
        self.tetris_map.render_to_image(first_temp_path)
        
        lines_cleared = self.tetris_map.clear_lines()
        bottom_rows_final = []
        actual_rows = min(self.num_rows, self.tetris_map.height)
        start_row_final = self.tetris_map.height - actual_rows
        for r in range(start_row_final, self.tetris_map.height):
            bottom_rows_final.append([cell for cell in self.tetris_map.grid[r]])
        final_temp_path = os.path.join(temp_dir, f"{task_id}_final.png")
        self.tetris_map.render_to_image(final_temp_path)
        prompt = PROMPTS[0].format(difficulty="medium", n=self.map_width)
        task_pair = TetrisTaskPair(
            id=task_id,
            task_category="Tetris",
            prompt=prompt,
            first_image_path=first_temp_path,
            final_image_path=final_temp_path,
            difficulty="medium",
            initial_state_bottom3=bottom_rows_initial,
            final_state_bottom3=bottom_rows_final,
            map_size=[self.tetris_map.width, self.tetris_map.height],
            created_at=datetime.now().isoformat()
        )
        return task_pair
        
        
    def generate_dataset(self, num_samples: int = 10) -> Dict[str, Any]:
        tasks = []
        # Ensure that across generated samples we cover True, False and None at least once
        options = [True, False, None]
        if num_samples >= len(options):
            # include each option once, then fill remaining with random choices
            guaranteed_list = options.copy()
            remaining = num_samples - len(options)
            guaranteed_list += [random.choice(options) for _ in range(remaining)]
        else:
            # fewer samples than options: pick a subset without duplicates
            guaranteed_list = random.sample(options, k=num_samples)

        random.shuffle(guaranteed_list)

        for sample_idx in range(num_samples):
            task_id = f"tetris_medium_{sample_idx:04d}"  # Format as string like "tetris_medium_0001"
            guarantee_choice = guaranteed_list[sample_idx]
            task = self.generate_single_task(task_id, guarantee_clear=guarantee_choice)
            tasks.append(task)
        task_dicts = []
        for task in tasks:
            task_dict = {
                'id': task.id,
                'prompt': task.prompt,
                'task_category': task.task_category,
                'first_image_path': task.first_image_path,
                'final_image_path': task.final_image_path,
                'initial_state_bottom3': task.initial_state_bottom3,
                'final_state_bottom3': task.final_state_bottom3,
                'map_size': task.map_size,
                'difficulty': task.difficulty,
                'created_at': task.created_at
            }
            task_dicts.append(task_dict)
        dataset_dict = {
            'name': 'Tetris Medium Dataset',
            'description': 'A dataset of medium Tetris tasks (10x10 map, 3-row initialization) generated by TetrisMediumTaskGenerator for video model reasoning evaluation.',
            'pairs': task_dicts  # Changed from 'tasks' to 'pairs' to match other tasks
        }
        return dataset_dict

class TetrisHardTaskGenerator:
    """
    Hard difficulty Tetris task generator.
    - 10x10 map (same as medium)
    - Initial state must NOT have any clearable lines (guarantee_clear=False)
    - Spawns a new block at the top
    - Simulates block drop AND line clearing
    """
    def __init__(self):
        self.map_width = 10
        self.num_rows = 3  # Same as medium: 3 bottom rows pre-filled
        self.fill_ratio = 0.8
        self.tetris_map = TetrisMap(width=self.map_width, height=self.map_width)
        
    def generate_single_task(self, task_id: str) -> TetrisTaskPair:
        """Generate a single hard task pair (10x10 map, with new block falling from top).
        
        Initial state: bottom rows filled but NO complete lines (guarantee_clear=False)
        A new block is spawned at top and will drop down.
        Final state: after block lands and lines are cleared.
        """
        # Fill bottom rows but guarantee NO complete lines initially
        _smart_fill_bottom_rows(
            self.tetris_map,
            self.num_rows,
            self.fill_ratio,
            guarantee_clear=False,  # Force no initial clearable lines
        )
        
        # Verify no lines are clearable in initial state
        initial_lines = []
        for i in range(self.tetris_map.height):
            if all(self.tetris_map.grid[i][j] != 0 for j in range(self.tetris_map.width)):
                initial_lines.append(i)
        if initial_lines:
            print(f"  Warning: Initial state has {len(initial_lines)} full line(s), regenerating...")
            # Clear and retry
            self.tetris_map.grid = [[0 for _ in range(self.map_width)] for _ in range(self.map_width)]
            _smart_fill_bottom_rows(self.tetris_map, self.num_rows, self.fill_ratio, guarantee_clear=False)
        
        # Capture initial state (bottom 3 rows)
        bottom_rows_initial = []
        temp_dir = tempfile.mkdtemp()
        start_row = self.tetris_map.height - self.num_rows
        for r in range(start_row, self.tetris_map.height):
            bottom_rows_initial.append([cell for cell in self.tetris_map.grid[r]])
        
        # Spawn a new block at the top
        spawn_success = self.tetris_map.spawn_new_block()
        if not spawn_success:
            print(f"  Warning: Failed to spawn block (game over state), using fallback block")
            # Fallback: manually create a simple block
            fallback_shape = TetrisShape.I
            self.tetris_map.current_block = TetrisBlock(fallback_shape, x=0, y=self.map_width // 2 - 1)
        
        # Record the spawned block info
        new_block_info = {
            "shape": self.tetris_map.current_block.shape.value,
            "initial_x": self.tetris_map.current_block.x,
            "initial_y": self.tetris_map.current_block.y,
            "rotation": self.tetris_map.current_block.rotation
        }
        
        # Render first frame (with block at top)
        first_temp_path = os.path.join(temp_dir, f"{task_id}_first.png")
        self.tetris_map.render_to_image(first_temp_path)
        
        # Simulate block drop (hard drop to bottom)
        self.tetris_map.hard_drop()
        
        # After hard_drop, current_block is None (already placed and cleared)
        # Now check and clear any complete lines
        lines_cleared = self.tetris_map.clear_lines()
        
        # Capture final state (bottom 3 rows)
        bottom_rows_final = []
        actual_rows = min(self.num_rows, self.tetris_map.height)
        start_row_final = self.tetris_map.height - actual_rows
        for r in range(start_row_final, self.tetris_map.height):
            bottom_rows_final.append([cell for cell in self.tetris_map.grid[r]])
        
        # Render final frame (after drop and clear)
        final_temp_path = os.path.join(temp_dir, f"{task_id}_final.png")
        self.tetris_map.render_to_image(final_temp_path)
        
        # Generate prompt
        prompt = PROMPTS[0].format(difficulty="hard", n=self.map_width)
        
        # Create task pair
        task_pair = TetrisTaskPair(
            id=task_id,
            task_category="Tetris",
            prompt=prompt,
            first_image_path=first_temp_path,
            final_image_path=final_temp_path,
            difficulty="hard",
            initial_state_bottom3=bottom_rows_initial,
            final_state_bottom3=bottom_rows_final,
            new_block=new_block_info,  # Include new block info for hard difficulty
            map_size=[self.tetris_map.width, self.tetris_map.height],
            created_at=datetime.now().isoformat()
        )
        
        print(f"  ✅ Hard task generated: block {new_block_info['shape']} dropped, {lines_cleared} lines cleared")
        return task_pair
    
    def generate_dataset(self, num_samples: int = 10) -> Dict[str, Any]:
        """Generate a dataset of hard Tetris tasks."""
        tasks = []
        for sample_idx in range(num_samples):
            task_id = f"tetris_hard_{sample_idx:04d}"
            
            # Reset the map for each task
            self.tetris_map = TetrisMap(width=self.map_width, height=self.map_width)
            
            task = self.generate_single_task(task_id)
            tasks.append(task)
        
        # Convert tasks to dictionaries
        task_dicts = []
        for task in tasks:
            task_dict = {
                'id': task.id,
                'prompt': task.prompt,
                'task_category': task.task_category,
                'first_image_path': task.first_image_path,
                'final_image_path': task.final_image_path,
                'initial_state_bottom3': task.initial_state_bottom3,
                'final_state_bottom3': task.final_state_bottom3,
                'new_block': task.new_block,  # Include new block info
                'map_size': task.map_size,
                'difficulty': task.difficulty,
                'created_at': task.created_at
            }
            task_dicts.append(task_dict)
        
        dataset_dict = {
            'name': 'Tetris Hard Dataset',
            'description': 'A dataset of hard Tetris tasks (10x10 map, new block falling from top, simulate drop + clear) generated by TetrisHardTaskGenerator for video model reasoning evaluation.',
            'pairs': task_dicts
        }
        return dataset_dict
    

class TetrisTaskGenerator:
    def __init__(self, difficulty: str = "easy"):
        self.difficulty = difficulty
        if difficulty == "easy":
            self.generator = TetrisEasyTaskGenerator()
        elif difficulty == "medium":
            self.generator = TetrisMediumTaskGenerator()
        elif difficulty == "hard":
            self.generator = TetrisHardTaskGenerator()
        else:
            raise ValueError("Invalid difficulty level. Choose 'easy', 'medium', or 'hard'.")
        
    def generate_dataset(self, num_samples: int = 10) -> Dict[str, Any]:
        return self.generator.generate_dataset(num_samples)

def create_dataset(num_samples: int = 10, difficulty: str = "easy") -> Dict[str, Any]:
    """
    Create a Tetris dataset with specified difficulty level.
    
    Args:
        num_samples: Number of tasks to generate
        difficulty: Difficulty level - "easy" (5x5), "medium" (10x10), or "hard" (10x10 with block drop)
    
    Returns:
        Dataset dictionary with task pairs
    """
    generator = TetrisTaskGenerator(difficulty="hard")
    return generator.generate_dataset(num_samples)



    