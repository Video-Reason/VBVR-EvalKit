"""
Sequence Completion Reasoning Task for VMEvalKit

Sequence completion reasoning system for video model evaluation.
The task shows a sequence of elements with a pattern and asks the model
to complete the sequence by adding the next element.

Follows the same data format as other tasks with first/final frames and prompts.

Author: VMEvalKit Team
"""

import json
import random
import numpy as np
from itertools import permutations, product
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, RegularPolygon, FancyBboxPatch, Polygon, FancyArrowPatch, Arrow
import matplotlib.font_manager as fm

# Configure matplotlib to support Chinese characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Import prompts from centralized location
from .PROMPTS import TYPE_PROMPTS

# Shape mappings
SHAPE_MAP = {
    '○': 'circle',
    '□': 'square',
    '△': 'triangle',
    '◇': 'diamond',
    'star': 'star'
}

# Color list for validation and rendering
COLORS = ['red', 'blue', 'green', 'yellow', 'orange']

# Position list for validation and rendering (2D only, no 3D positions)
POSITIONS = ['top', 'bottom', 'left', 'right', 'center', 'top-left', 'top-right', 'bottom-left', 'bottom-right']


@dataclass
class SequenceCompletionTaskPair:
    """
    Data structure for sequence completion reasoning video model evaluation.
    
    Contains:
    - prompt: Instructions for the video model
    - first_image: The sequence with missing element
    - final_image: The sequence with completed element
    """
    id: str
    prompt: str                      # What to ask the video model to do
    first_image_path: str           # The initial sequence image (with ?)
    final_image_path: str           # The completed sequence image
    task_category: str              # "Sequence Completion"
    sequence_completion_data: Dict[str, Any] = None  # Metadata
    task_type: int = 0              # Type 1-8
    sequence: List[Any] = None      # The sequence elements
    answer: Any = None              # The correct answer
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class SequenceCompletionDataset:
    """Collection of SequenceCompletionTaskPair instances."""
    name: str
    description: str
    pairs: List[SequenceCompletionTaskPair]
    metadata: Dict[str, Any]
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> SequenceCompletionTaskPair:
        return self.pairs[idx]
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save dataset to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SequenceCompletionDataset':
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert dictionaries back to SequenceCompletionTaskPair objects
        pairs = []
        for pair_data in data['pairs']:
            pairs.append(SequenceCompletionTaskPair(**pair_data))
        
        data['pairs'] = pairs
        return cls(**data)


class SequenceRenderer:
    """Renderer for different types of sequence elements."""
    
    def __init__(self, figsize=(10, 10), dpi=150, output_size=(1024, 1024)):
        # Square image size
        # output_size: target output image size in pixels (width, height)
        self.figsize = figsize
        self.dpi = dpi
        self.output_size = output_size
        # Coordinate system: 0 to 10 for both x and y
        self.canvas_size = 10
        self.blank_cell_width = 1.2
        self.blank_cell_height = 1.2
        # Standardized sizes for consistent rendering (further reduced font sizes)
        self.shape_size = 0.55  # Reduced from 0.70 to 0.55 for smaller shape symbols
        self.color_circle_size = 0.55  # Reduced from 0.70 to 0.55 for smaller color symbols
        self.number_fontsize = 50  # Reduced from 64 to 50 for smaller numbers
        self.text_fontsize = 32  # Reduced from 42 to 32 for smaller symbol text
    
    def render_sequence(self, sequence: List[Any], show_blank: bool = False, 
                       output_path: Optional[str] = None) -> None:
        """
        Render a sequence of elements. Only the last cell is shown as a blank box if show_blank=True.
        
        Args:
            sequence: List of sequence elements (numbers, shapes, colors, positions, or mixed)
            show_blank: If True, show blank cell for the last element (for first_frame)
            output_path: Path to save the image
        """
        num_elements = len(sequence)
        
        # New logic: small margin, adaptive scaling based on sequence length
        # Small safe margin to keep elements close to edges but not touching
        safe_margin = 0.8  # Small margin (reduced from 2.5)
        
        # Target spacing for elements (consistent spacing for all types)
        # For Type 7 and Type 8 len7-8, use slightly larger spacing
        if getattr(self, '_type7_long_spacing', False) or getattr(self, '_type8_long_spacing', False):
            target_spacing = 2.5  # Increased spacing for Type 7/8 len7-8 (increased from 2.0)
        else:
            target_spacing = 1.5  # Standard spacing
        
        # Base element sizes (for short sequences, keep original size)
        base_shape_size = self.shape_size
        base_color_circle_size = self.color_circle_size
        base_number_fontsize = self.number_fontsize
        base_text_fontsize = self.text_fontsize
        
        # Calculate available width with small margin
        available_width = self.canvas_size - 2 * safe_margin
        
        # Base element radius (for calculation)
        max_element_radius_base = max(self.shape_size, self.color_circle_size, self.blank_cell_width / 2)
        
        # Try with original size first
        element_size_scale = 1.0
        element_spacing = target_spacing
        
        # For Type 7/8 len7-8, we want to maintain larger spacing even if we need to scale down
        is_type7_or8_long = getattr(self, '_type7_long_spacing', False) or getattr(self, '_type8_long_spacing', False)
        
        # Calculate total width needed with original size
        total_width_needed = (num_elements - 1) * element_spacing + max_element_radius_base * 2
        
        # If doesn't fit, need to scale down (for long sequences)
        if total_width_needed > available_width:
            # Calculate required scale to fit
            # We need: (num_elements - 1) * spacing + scaled_radius * 2 <= available_width
            # Solve for scale: available_width = (num_elements - 1) * spacing + (base_radius * scale) * 2
            # scale = (available_width - (num_elements - 1) * spacing) / (base_radius * 2)
            
            # Try with target spacing first
            max_radius_available = (available_width - (num_elements - 1) * target_spacing) / 2
            if max_radius_available > 0:
                element_size_scale = max_radius_available / max_element_radius_base
                element_size_scale = max(0.3, min(1.0, element_size_scale))  # Limit scale between 0.3 and 1.0
                element_spacing = target_spacing
            else:
                # Even with target spacing, need to reduce spacing too
                # For Type 7/8 len7-8, try to maintain larger spacing by using a higher min_spacing
                if is_type7_or8_long:
                    # For Type 7/8 len7-8, use a higher minimum spacing to maintain visual separation
                    min_spacing = 1.5  # Higher minimum for Type 7/8 len7-8
                else:
                    # Use minimum spacing and calculate scale
                    min_spacing = 1.0
                
                max_radius_available = (available_width - (num_elements - 1) * min_spacing) / 2
                if max_radius_available > 0:
                    element_size_scale = max_radius_available / max_element_radius_base
                    element_size_scale = max(0.3, min(1.0, element_size_scale))
                    element_spacing = min_spacing
                else:
                    # Very long sequence, need both smaller spacing and scale
                    if is_type7_or8_long:
                        # For Type 7/8 len7-8, still try to maintain larger spacing
                        min_spacing = 1.2  # Still higher than default 0.8
                    else:
                        min_spacing = 0.8
                    
                    max_radius_available = (available_width - (num_elements - 1) * min_spacing) / 2
                    element_size_scale = max(0.25, max_radius_available / max_element_radius_base) if max_radius_available > 0 else 0.25
                    element_size_scale = max(0.25, min(1.0, element_size_scale))
                    element_spacing = min_spacing
        
        max_element_radius = max_element_radius_base * element_size_scale
        
        # Check if sequence contains numbers - use edge-based spacing for numbers
        has_numbers = any(isinstance(elem, (int, float)) for elem in sequence if elem is not None)
        
        if has_numbers:
            # For number sequences: calculate positions based on edge-based spacing with fixed gap
            scale = element_size_scale
            fontsize = int(self.number_fontsize * scale)
            
            # Create a temporary figure to measure text widths
            temp_fig, temp_ax = plt.subplots(figsize=(1, 1))
            temp_ax.axis('off')
            
            # Calculate width of each number (edge-based)
            number_widths = []
            for element in sequence:
                if element is None:
                    # Question mark width
                    text_obj = temp_ax.text(0, 0, '？', fontsize=fontsize, ha='center', va='center',
                                           fontweight='bold')
                    bbox = text_obj.get_window_extent(renderer=temp_fig.canvas.get_renderer())
                    width = bbox.width / temp_fig.dpi * (self.canvas_size / self.figsize[0])
                    number_widths.append(width)
                    text_obj.remove()
                else:
                    text_obj = temp_ax.text(0, 0, str(element), fontsize=fontsize, ha='center', va='center',
                                           fontweight='bold')
                    bbox = text_obj.get_window_extent(renderer=temp_fig.canvas.get_renderer())
                    width = bbox.width / temp_fig.dpi * (self.canvas_size / self.figsize[0])
                    number_widths.append(width)
                    text_obj.remove()
            
            plt.close(temp_fig)
            
            # Use fixed spacing between number edges (consistent gap for all pairs)
            # Make spacing consistent with other element types (center-to-center spacing)
            # For numbers, we use edge-based spacing, so we need to calculate gap to match center-to-center spacing
            # If element_spacing is center-to-center, we need: gap = element_spacing - (width1/2 + width2/2)
            # But to keep it simple and consistent, use a fixed ratio
            gap_between_numbers = element_spacing * 0.5  # Reduced from 0.8 to make spacing more consistent
            
            x_positions = []
            current_x = safe_margin
            
            for i in range(len(number_widths)):
                # Center of current number
                x_positions.append(current_x + number_widths[i] / 2)
                
                if i < len(number_widths) - 1:
                    # Move to next position: current number's right edge + gap + next number's left edge
                    current_x += number_widths[i] / 2 + gap_between_numbers + number_widths[i + 1] / 2
            
            # Calculate the actual left and right edges of the sequence
            # First element's left edge
            first_number_left_edge = x_positions[0] - number_widths[0] / 2
            # Last element's right edge (this includes the question mark)
            last_number_right_edge = x_positions[-1] + number_widths[-1] / 2
            # Total width from first element's left edge to last question mark's right edge
            actual_total_width = last_number_right_edge - first_number_left_edge
            # Calculate the center of the actual sequence
            actual_sequence_center = first_number_left_edge + actual_total_width / 2
            # Center the sequence in the canvas
            canvas_center = self.canvas_size / 2
            # Calculate offset to center the sequence
            offset = canvas_center - actual_sequence_center
            x_positions = [x + offset for x in x_positions]
            
            # Final check: ensure small margins on both sides
            first_pos_left = x_positions[0] - number_widths[0] / 2
            last_pos_right = x_positions[-1] + number_widths[-1] / 2
            if first_pos_left < safe_margin:
                # Shift right to maintain left margin
                shift = safe_margin - first_pos_left
                x_positions = [x + shift for x in x_positions]
            elif last_pos_right > self.canvas_size - safe_margin:
                # Shift left to maintain right margin
                shift = (self.canvas_size - safe_margin) - last_pos_right
                x_positions = [x + shift for x in x_positions]
        else:
            # For non-number sequences: use center-based spacing (original method)
            # Calculate total width from first element's left edge to last element's right edge
            # This includes the question mark at the end
            total_width = (num_elements - 1) * element_spacing + max_element_radius * 2
            # Calculate the actual left and right edges of the sequence
            # First, calculate a temporary start position
            temp_start_x = max_element_radius
            first_element_left = temp_start_x - max_element_radius
            last_element_right = temp_start_x + (num_elements - 1) * element_spacing + max_element_radius
            actual_total_width = last_element_right - first_element_left
            # Calculate the center of the actual sequence
            actual_sequence_center = first_element_left + actual_total_width / 2
            # Center the sequence in the canvas
            canvas_center = self.canvas_size / 2
            # Calculate offset to center the sequence
            center_offset = canvas_center - actual_sequence_center
            # Calculate start_x with centering
            start_x = temp_start_x + center_offset
            x_positions = [start_x + i * element_spacing for i in range(num_elements)]
            
            # Final check: ensure small margins on both sides
            first_pos_left = x_positions[0] - max_element_radius
            last_pos_right = x_positions[-1] + max_element_radius
            if first_pos_left < safe_margin:
                # Shift right to maintain left margin
                shift = safe_margin - first_pos_left
                x_positions = [x + shift for x in x_positions]
            elif last_pos_right > self.canvas_size - safe_margin:
                # Shift left to maintain right margin
                shift = (self.canvas_size - safe_margin) - last_pos_right
                x_positions = [x + shift for x in x_positions]
        
        center_y = self.canvas_size / 2  # Vertical center
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.set_xlim(0, self.canvas_size)
        ax.set_ylim(0, self.canvas_size)
        ax.axis('off')
        
        for i, element in enumerate(sequence):
            x = x_positions[i]
            y = center_y
            
            if i == len(sequence) - 1 and show_blank:
                # Show question mark instead of blank cell
                scale = getattr(self, '_current_scale', 1.0)
                question_fontsize = int(self.number_fontsize * scale)
                ax.text(x, y, '？', fontsize=question_fontsize, ha='center', va='center',
                       fontweight='bold', color='black')
            else:
                # Render the element
                self._render_element(ax, element, x, y)
        
        plt.tight_layout(pad=0)
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=self.dpi, pad_inches=0)
            # Resize to exact output size if specified
            if self.output_size:
                from PIL import Image
                img = Image.open(output_path)
                img_resized = img.resize(self.output_size, Image.Resampling.LANCZOS)
                img_resized.save(output_path)
        plt.close()
    
    def _render_element(self, ax, element: Any, x: float, y: float) -> None:
        """Render a single element based on its type."""
        if element is None:
            # Blank element - do nothing (cell background already drawn)
            return
        
        if isinstance(element, (int, float)):
            # Number - standardized size, apply scale for long sequences
            scale = getattr(self, '_current_scale', 1.0)
            ax.text(x, y, str(element), fontsize=int(self.number_fontsize * scale), ha='center', va='center',
                   fontweight='bold', color='black')
        
        elif isinstance(element, str):
            # Check if it's a shape, color, position, or mixed
            if element in SHAPE_MAP:
                self._render_shape(ax, element, x, y)
            elif element in COLORS:
                # Color name (already in English)
                self._render_color(ax, element, x, y)
            elif element in POSITIONS:
                # Position name (already in English)
                self._render_position(ax, element, x, y)
            elif '+' in element or '○' in element or '□' in element or '△' in element or '◇' in element or 'star' in element:
                # Mixed element (e.g., color+shape or shape+position combination)
                self._render_mixed(ax, element, x, y)
            elif '-' in element and any(color in element for color in COLORS):
                # Mixed element: color+position (e.g., 'red-top', 'blue-bottom')
                # Check if it starts with a color and has a position after '-'
                self._render_mixed(ax, element, x, y)
            else:
                # Plain text with standardized size
                scale = getattr(self, '_current_scale', 1.0)
                ax.text(x, y, element, fontsize=int(self.text_fontsize * scale), ha='center', va='center',
                       fontweight='bold', color='black')
        
        elif isinstance(element, list) or isinstance(element, tuple):
            # Composite element (e.g., [color, shape])
            if len(element) == 2:
                self._render_mixed(ax, f"{element[0]}{element[1]}", x, y)
            else:
                scale = getattr(self, '_current_scale', 1.0)
                ax.text(x, y, str(element), fontsize=int(self.text_fontsize * scale), ha='center', va='center',
                       fontweight='bold', color='black')
    
    def _render_shape(self, ax, shape: str, x: float, y: float) -> None:
        """Render a shape with standardized size. All shapes have similar visual size."""
        shape_type = SHAPE_MAP[shape]
        # Use consistent bounding box size for all shapes
        # This ensures all shapes appear roughly the same size
        # Apply scale factor if sequence is long
        scale = getattr(self, '_current_scale', 1.0)
        bbox_size = self.shape_size * 2 * scale  # Total width/height of bounding box
        
        if shape_type == 'circle':
            # Circle: radius = bbox_size / 2
            circle = Circle((x, y), bbox_size / 2, facecolor='lightblue', edgecolor='black', linewidth=2)
            ax.add_patch(circle)
        elif shape_type == 'square':
            # Square: make it slightly smaller to match visual size of other shapes
            square_size = bbox_size * 0.85  # Reduce by 15% for better visual balance
            square = Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size,
                             facecolor='lightblue', edgecolor='black', linewidth=2)
            ax.add_patch(square)
        elif shape_type == 'triangle':
            # Triangle: use radius such that bounding box height ≈ bbox_size
            # For equilateral triangle, height = radius * sqrt(3), so radius = bbox_size / sqrt(3)
            triangle_radius = bbox_size / np.sqrt(3)
            triangle = RegularPolygon((x, y), 3, radius=triangle_radius,
                                    facecolor='lightblue', edgecolor='black', linewidth=2)
            ax.add_patch(triangle)
        elif shape_type == 'diamond':
            # Diamond: draw using Polygon with 4 vertices (top, right, bottom, left)
            # This creates a proper diamond shape like ▪️
            half_size = bbox_size / 2
            diamond_points = [
                (x, y - half_size),  # top
                (x + half_size, y),  # right
                (x, y + half_size),  # bottom
                (x - half_size, y),  # left
            ]
            diamond = Polygon(diamond_points, closed=True, facecolor='lightblue', 
                            edgecolor='black', linewidth=2)
            ax.add_patch(diamond)
        elif shape_type == 'star':
            # Star: use radius such that bounding box ≈ bbox_size
            # For 5-pointed star, approximate radius = bbox_size / 2.2
            star_radius = bbox_size / 2.2
            star = RegularPolygon((x, y), 5, radius=star_radius,
                               facecolor='lightblue', edgecolor='black', linewidth=2)
            ax.add_patch(star)
    
    def _render_color(self, ax, color: str, x: float, y: float) -> None:
        """Render a color as a colored circle with standardized size (no text)."""
        # Color is already in English format
        scale = getattr(self, '_current_scale', 1.0)
        circle = Circle((x, y), self.color_circle_size * scale, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        # No text - color circle is self-explanatory
    
    def _render_position(self, ax, position: str, x: float, y: float) -> None:
        """Render a position as an arrow pointing in the direction, centered at the position."""
        scale = getattr(self, '_current_scale', 1.0)
        # Increase arrow length for better visibility
        arrow_size = self.shape_size * scale * 2.2  # Increased from 1.8 to 2.2 for much larger arrows
        
        # Map position to arrow direction (2D only)
        position_2d = position
        
        # Draw arrow centered at position (arrow spans across the center point)
        # Use thicker line width for better visibility
        arrow_lw = 5 * scale  # Increased from 3 to 5 for thicker arrows
        center_arrow_lw = 4 * scale  # Increased from 2 to 4 for center arrows
        # Increase arrow head size for better visibility
        arrow_head_scale = 30 * scale  # Increased from 20 to 30 for more prominent arrow heads
        center_arrow_head_scale = 25 * scale  # Increased from 15 to 25 for center arrows
        
        if 'top' in position_2d and 'left' in position_2d:
            # top-left - arrow from bottom-right to top-left, centered
            arrow = FancyArrowPatch((x + arrow_size * 0.35, y + arrow_size * 0.35), 
                                   (x - arrow_size * 0.35, y - arrow_size * 0.35),
                                   arrowstyle='->', mutation_scale=arrow_head_scale, lw=arrow_lw, color='black')
            ax.add_patch(arrow)
        elif 'top' in position_2d and 'right' in position_2d:
            # top-right - arrow from bottom-left to top-right, centered
            arrow = FancyArrowPatch((x - arrow_size * 0.35, y + arrow_size * 0.35), 
                                   (x + arrow_size * 0.35, y - arrow_size * 0.35),
                                   arrowstyle='->', mutation_scale=arrow_head_scale, lw=arrow_lw, color='black')
            ax.add_patch(arrow)
        elif 'bottom' in position_2d and 'left' in position_2d:
            # bottom-left - arrow from top-right to bottom-left, centered
            arrow = FancyArrowPatch((x + arrow_size * 0.35, y - arrow_size * 0.35), 
                                   (x - arrow_size * 0.35, y + arrow_size * 0.35),
                                   arrowstyle='->', mutation_scale=arrow_head_scale, lw=arrow_lw, color='black')
            ax.add_patch(arrow)
        elif 'bottom' in position_2d and 'right' in position_2d:
            # bottom-right - arrow from top-left to bottom-right, centered
            arrow = FancyArrowPatch((x - arrow_size * 0.35, y - arrow_size * 0.35), 
                                   (x + arrow_size * 0.35, y + arrow_size * 0.35),
                                   arrowstyle='->', mutation_scale=arrow_head_scale, lw=arrow_lw, color='black')
            ax.add_patch(arrow)
        elif 'top' in position_2d:
            # top - arrow from bottom to top, centered
            arrow = FancyArrowPatch((x, y + arrow_size * 0.5), (x, y - arrow_size * 0.5),
                                   arrowstyle='->', mutation_scale=arrow_head_scale, lw=arrow_lw, color='black')
            ax.add_patch(arrow)
        elif 'bottom' in position_2d:
            # bottom - arrow from top to bottom, centered
            arrow = FancyArrowPatch((x, y - arrow_size * 0.5), (x, y + arrow_size * 0.5),
                                   arrowstyle='->', mutation_scale=arrow_head_scale, lw=arrow_lw, color='black')
            ax.add_patch(arrow)
        elif 'left' in position_2d:
            # left - arrow from right to left, centered
            arrow = FancyArrowPatch((x + arrow_size * 0.5, y), (x - arrow_size * 0.5, y),
                                   arrowstyle='->', mutation_scale=arrow_head_scale, lw=arrow_lw, color='black')
            ax.add_patch(arrow)
        elif 'right' in position_2d:
            # right - arrow from left to right, centered
            arrow = FancyArrowPatch((x - arrow_size * 0.5, y), (x + arrow_size * 0.5, y),
                                   arrowstyle='->', mutation_scale=arrow_head_scale, lw=arrow_lw, color='black')
            ax.add_patch(arrow)
        elif 'center' in position_2d:
            # center - draw arrows pointing inward from all four directions, centered
            small_arrow = arrow_size * 0.4
            # Up arrow
            arrow_up = FancyArrowPatch((x, y + small_arrow * 0.5), (x, y - small_arrow * 0.5),
                                      arrowstyle='->', mutation_scale=center_arrow_head_scale, lw=center_arrow_lw, color='black')
            ax.add_patch(arrow_up)
            # Down arrow
            arrow_down = FancyArrowPatch((x, y - small_arrow * 0.5), (x, y + small_arrow * 0.5),
                                        arrowstyle='->', mutation_scale=center_arrow_head_scale, lw=center_arrow_lw, color='black')
            ax.add_patch(arrow_down)
            # Left arrow
            arrow_left = FancyArrowPatch((x + small_arrow * 0.5, y), (x - small_arrow * 0.5, y),
                                        arrowstyle='->', mutation_scale=center_arrow_head_scale, lw=center_arrow_lw, color='black')
            ax.add_patch(arrow_left)
            # Right arrow
            arrow_right = FancyArrowPatch((x - small_arrow * 0.5, y), (x + small_arrow * 0.5, y),
                                         arrowstyle='->', mutation_scale=center_arrow_head_scale, lw=center_arrow_lw, color='black')
            ax.add_patch(arrow_right)
    
    def _render_mixed(self, ax, mixed: str, x: float, y: float) -> None:
        """Render a mixed element (e.g., color+shape, color+position, or shape+position)."""
        scale = getattr(self, '_current_scale', 1.0)
        
        shape_chars = ['○', '□', '△', '◇', 'star']
        
        # Priority 1: Check if it starts with a color (color+shape or color+position)
        color_part = None
        remaining = None
        
        # Check for English color (e.g., 'red', 'blue', etc.)
        if mixed:
            for color in COLORS:
                if mixed.startswith(color):
                    color_part = color
                    remaining = mixed[len(color):]
                    # Check if there's a separator (e.g., 'red-top' or 'red○')
                    if remaining.startswith('-'):
                        remaining = remaining[1:]  # Remove the separator
                    break
        
        if color_part:
            # Color is already in English format
            color_en = color_part
            
            # Check if remaining part is a shape
            if remaining and remaining in SHAPE_MAP:
                # Color + Shape: render colored shape
                shape_type = SHAPE_MAP[remaining]
                bbox_size = self.shape_size * 2 * scale
                
                if shape_type == 'circle':
                    circle = Circle((x, y), bbox_size / 2, facecolor=color_en, edgecolor='black', linewidth=2)
                    ax.add_patch(circle)
                elif shape_type == 'square':
                    square_size = bbox_size * 0.85
                    square = Rectangle((x - square_size / 2, y - square_size / 2), square_size, square_size,
                                     facecolor=color_en, edgecolor='black', linewidth=2)
                    ax.add_patch(square)
                elif shape_type == 'triangle':
                    triangle_radius = bbox_size / np.sqrt(3)
                    triangle = RegularPolygon((x, y), 3, radius=triangle_radius,
                                            facecolor=color_en, edgecolor='black', linewidth=2)
                    ax.add_patch(triangle)
                elif shape_type == 'diamond':
                    half_size = bbox_size / 2
                    diamond_points = [
                        (x, y - half_size),
                        (x + half_size, y),
                        (x, y + half_size),
                        (x - half_size, y),
                    ]
                    diamond = Polygon(diamond_points, closed=True, facecolor=color_en, 
                                    edgecolor='black', linewidth=2)
                    ax.add_patch(diamond)
                elif shape_type == 'star':
                    star_radius = bbox_size / 2.2
                    star = RegularPolygon((x, y), 5, radius=star_radius,
                                       facecolor=color_en, edgecolor='black', linewidth=2)
                    ax.add_patch(star)
                return
            
            # Check if remaining part is a position
            elif remaining and (remaining in POSITIONS or any(p in remaining for p in POSITIONS)):
                # Color + Position: render color circle, then position arrow
                circle = Circle((x, y), self.color_circle_size * scale, facecolor=color_en, edgecolor='black', linewidth=2)
                ax.add_patch(circle)
                self._render_position(ax, remaining, x, y)
                return
        
        # Priority 2: Check if it's shape+position (e.g., '○top', '△left')
        for shape_char in shape_chars:
            if mixed.startswith(shape_char):
                position_part = mixed[len(shape_char):]
                if position_part in POSITIONS or any(p in position_part for p in POSITIONS):
                    # Shape + Position: render shape first, then arrow
                    self._render_shape(ax, shape_char, x, y)
                    self._render_position(ax, position_part, x, y)
                    return
        
        # Fallback: render as text
        ax.text(x, y, mixed, fontsize=int(self.text_fontsize * scale), ha='center', va='center',
               fontweight='bold', color='black')


class SequenceCompletionTaskGenerator:
    """Generator for sequence completion tasks."""
    
    def __init__(self, output_size=(1024, 1024)):
        """
        Initialize the generator.
        
        Args:
            output_size: Target output image size in pixels (width, height). Default is (1024, 1024).
        """
        self.renderer = SequenceRenderer(output_size=output_size)
        # Load all task definitions from the exhaustive list
        # This will be populated by parsing the markdown file or hardcoding
        self._load_task_definitions()
    
    def _load_task_definitions(self):
        """Load all task definitions. For now, we'll generate them programmatically."""
        # Task definitions will be generated based on the patterns in the markdown file
        pass
    
    def generate_arithmetic_sequence(self, start: int, step: int, length: int) -> Tuple[List[int], int]:
        """Generate arithmetic sequence and answer."""
        sequence = [start + i * step for i in range(length - 1)]
        answer = start + (length - 1) * step
        return sequence, answer
    
    def generate_geometric_sequence(self, start: int, ratio: int, length: int) -> Tuple[List[int], int]:
        """Generate geometric sequence and answer."""
        sequence = [start * (ratio ** i) for i in range(length - 1)]
        answer = start * (ratio ** (length - 1))
        return sequence, answer
    
    def generate_power_sequence(self, base: int, power: int, length: int) -> Tuple[List[int], int]:
        """Generate power sequence (squares) and answer."""
        sequence = [(base + i) ** power for i in range(length - 1)]
        answer = (base + length - 1) ** power
        return sequence, answer
    
    def generate_fibonacci_sequence(self, first: int, second: int, length: int) -> Tuple[List[int], int]:
        """Generate Fibonacci sequence and answer."""
        sequence = [first, second]
        for i in range(2, length - 1):
            sequence.append(sequence[i-1] + sequence[i-2])
        answer = sequence[-1] + sequence[-2]
        return sequence, answer
    
    def generate_cycle_sequence(self, cycle: List[Any], length: int) -> Tuple[List[Any], Any]:
        """Generate cycle sequence and answer."""
        sequence = []
        for i in range(length - 1):
            sequence.append(cycle[i % len(cycle)])
        answer = cycle[(length - 1) % len(cycle)]
        return sequence, answer
    
    def format_sequence_str(self, sequence: List[Any]) -> str:
        """Format sequence as string for prompt."""
        return '[' + ', '.join(str(elem) for elem in sequence) + ', ?]'
    
    def generate_single_task(self, task_type: int, task_params: Dict[str, Any], 
                            task_id: str, output_dir: Path) -> SequenceCompletionTaskPair:
        """Generate a single task pair."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate sequence based on type
        if task_type == 1:  # Arithmetic
            sequence, answer = self.generate_arithmetic_sequence(
                task_params['start'], task_params['step'], task_params['length'])
        elif task_type == 2:  # Geometric
            sequence, answer = self.generate_geometric_sequence(
                task_params['start'], task_params['ratio'], task_params['length'])
        elif task_type == 3:  # Power
            sequence, answer = self.generate_power_sequence(
                task_params['base'], task_params['power'], task_params['length'])
        elif task_type == 4:  # Fibonacci
            sequence, answer = self.generate_fibonacci_sequence(
                task_params['first'], task_params['second'], task_params['length'])
        elif task_type in [5, 6, 7]:  # Cycles
            sequence, answer = self.generate_cycle_sequence(
                task_params['cycle'], task_params['length'])
        elif task_type == 8:  # Mixed
            sequence, answer = self.generate_cycle_sequence(
                task_params['cycle'], task_params['length'])
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Generate images
        first_frame_path = output_dir / "first_frame.png"
        final_frame_path = output_dir / "final_frame.png"
        
        # For Type 5, 6, and 7 with len7 or len8, reduce font sizes
        original_shape_size = self.renderer.shape_size
        original_color_circle_size = self.renderer.color_circle_size
        
        # For Type 7 and Type 8 with len7 or len8, increase spacing between elements
        if task_type == 7 and task_params.get('length', 0) in [7, 8]:
            # Set flag to increase spacing for Type 7 len7-8
            self.renderer._type7_long_spacing = True
            self.renderer._type8_long_spacing = False
        elif task_type == 8 and task_params.get('length', 0) in [7, 8]:
            # Set flag to increase spacing for Type 8 len7-8
            self.renderer._type8_long_spacing = True
            self.renderer._type7_long_spacing = False
        else:
            self.renderer._type7_long_spacing = False
            self.renderer._type8_long_spacing = False
        
        if (task_type in [5, 6, 7]) and (task_params.get('length', 0) in [7, 8]):
            # Reduce font sizes for Type 5, 6, and 7, len7 and len8
            self.renderer.shape_size = 0.45  # Reduced from 0.55 (affects Type 5 shapes and Type 7 arrows)
            self.renderer.color_circle_size = 0.45  # Reduced from 0.55 (affects Type 6 colors)
        
        # First frame: sequence with blank last cell
        # Add None as placeholder for the blank cell
        sequence_with_blank = sequence + [None]
        self.renderer.render_sequence(sequence_with_blank, show_blank=True, 
                                     output_path=str(first_frame_path))
        
        # Final frame: complete sequence with answer
        complete_sequence = sequence + [answer]
        self.renderer.render_sequence(complete_sequence, show_blank=False,
                                     output_path=str(final_frame_path))
        
        # Restore original sizes
        self.renderer.shape_size = original_shape_size
        self.renderer.color_circle_size = original_color_circle_size
        # Clear spacing flags
        self.renderer._type7_long_spacing = False
        self.renderer._type8_long_spacing = False
        
        # Generate prompt (no longer includes sequence string)
        prompt_template = TYPE_PROMPTS[task_type]
        prompt = prompt_template
        
        # Save prompt
        prompt_path = output_dir / "prompt.txt"
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        # Create task pair
        return SequenceCompletionTaskPair(
            id=task_id,
            prompt=prompt,
            first_image_path=str(first_frame_path),
            final_image_path=str(final_frame_path),
            task_category="Sequence Completion",
            sequence_completion_data=task_params,
            task_type=task_type,
            sequence=sequence,
            answer=answer
        )
    
    def generate_all_tasks(self, num_samples: Optional[int] = None) -> List[Tuple[str, int, Dict[str, Any]]]:
        """
        Generate all sequence completion task definitions.
        
        Args:
            num_samples: If provided, randomly sample this many tasks. Otherwise generate all 1242 tasks.
        
        Returns:
            List of tuples: (task_id, task_type, task_params)
        """
        all_tasks = []
        task_id_counter = 0
        
        # Type 1: Arithmetic Sequence (180 tasks)
        for start in range(1, 16):
            for step in [-5, -2, -1, 1, 2, 5]:
                for length in [5, 6, 7]:
                    # Check if sequence would be valid (no negative numbers for display)
                    test_seq, test_ans = self.generate_arithmetic_sequence(start, step, length)
                    if min(test_seq + [test_ans]) >= 0 or all(x >= 0 for x in test_seq):
                        task_id = f"sequence_completion_type1_{task_id_counter:04d}"
                        task_params = {'start': start, 'step': step, 'length': length}
                        all_tasks.append((task_id, 1, task_params))
                        task_id_counter += 1
        
        # Type 2: Geometric Sequence (95 tasks)
        for start in range(1, 31):
            for ratio in [2, 3, 4]:
                for length in [5, 6, 7]:
                    test_seq, test_ans = self.generate_geometric_sequence(start, ratio, length)
                    if test_ans <= 1000:  # Limit as per doc
                        task_id = f"sequence_completion_type2_{task_id_counter:04d}"
                        task_params = {'start': start, 'ratio': ratio, 'length': length}
                        all_tasks.append((task_id, 2, task_params))
                        task_id_counter += 1
        
        # Type 3: Power Sequence (20 tasks)
        for base in range(1, 11):
            for length in [5, 6]:
                test_seq, test_ans = self.generate_power_sequence(base, 2, length)
                if test_ans <= 100:  # Limit as per doc
                    task_id = f"sequence_completion_type3_{task_id_counter:04d}"
                    task_params = {'base': base, 'power': 2, 'length': length}
                    all_tasks.append((task_id, 3, task_params))
                    task_id_counter += 1
        
        # Type 4: Fibonacci Sequence (162 tasks)
        for first in range(1, 10):
            for second in range(1, 10):
                for length in [6, 7]:
                    task_id = f"sequence_completion_type4_{task_id_counter:04d}"
                    task_params = {'first': first, 'second': second, 'length': length}
                    all_tasks.append((task_id, 4, task_params))
                    task_id_counter += 1
        
        # Type 5: Shape Cycle (141 tasks)
        shapes = ['○', '□', '△', '◇', 'star']
        # Generate all 3-element cycles
        for cycle_len in [3, 4, 5]:
            if cycle_len == 3:
                # All permutations of 3 shapes from 5 shapes
                for combo in permutations(shapes, 3):
                    cycle = list(combo)
                    for length in [5, 6, 7]:
                        task_id = f"sequence_completion_type5_{task_id_counter:04d}"
                        task_params = {'cycle': cycle, 'length': length}
                        all_tasks.append((task_id, 5, task_params))
                        task_id_counter += 1
            elif cycle_len == 4:
                # All permutations of 4 shapes
                for combo in permutations(shapes, 4):
                    cycle = list(combo)
                    for length in [6, 7, 8]:
                        task_id = f"sequence_completion_type5_{task_id_counter:04d}"
                        task_params = {'cycle': cycle, 'length': length}
                        all_tasks.append((task_id, 5, task_params))
                        task_id_counter += 1
            elif cycle_len == 5:
                # All 5 shapes in different orders
                for combo in permutations(shapes, 5):
                    cycle = list(combo)
                    for length in [7, 8]:
                        task_id = f"sequence_completion_type5_{task_id_counter:04d}"
                        task_params = {'cycle': cycle, 'length': length}
                        all_tasks.append((task_id, 5, task_params))
                        task_id_counter += 1
        
        # Type 6: Color Cycle (110 tasks)
        colors = ['red', 'blue', 'green', 'yellow', 'orange']
        # Generate all 3-element cycles
        for combo in permutations(colors, 3):
            cycle = list(combo)
            for length in [5, 6, 7, 8]:
                task_id = f"sequence_completion_type6_{task_id_counter:04d}"
                task_params = {'cycle': cycle, 'length': length}
                all_tasks.append((task_id, 6, task_params))
                task_id_counter += 1
        # Generate all 4-element cycles
        for combo in permutations(colors, 4):
            cycle = list(combo)
            for length in [6, 7, 8]:
                task_id = f"sequence_completion_type6_{task_id_counter:04d}"
                task_params = {'cycle': cycle, 'length': length}
                all_tasks.append((task_id, 6, task_params))
                task_id_counter += 1
        
        # Type 7: Direction Cycle (2D directions only, no 3D directions)
        # Define position sets (2D only: top, bottom, left, right, center, and diagonals)
        position_sets_3 = [
            ['top', 'bottom', 'left'], ['left', 'right', 'top'], ['top', 'bottom', 'right'],
            ['top-left', 'bottom-right', 'top-right'], ['top-right', 'bottom-left', 'top-left'],
            ['left', 'right', 'bottom'], ['top-left', 'bottom-left', 'top-right']
        ]
        position_sets_4 = [
            ['top', 'bottom', 'left', 'right'], ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
            ['top', 'bottom', 'left', 'right'], ['left', 'right', 'top', 'bottom'],
            ['left', 'right', 'top', 'bottom'], ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
            ['top', 'left', 'bottom', 'right'], ['top-right', 'bottom-left', 'top-left', 'bottom-right']
        ]
        position_sets_5 = [
            ['top', 'bottom', 'left', 'right', 'top-left'], ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'top'],
            ['top', 'bottom', 'left', 'right', 'center'], ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'],
            ['top', 'left', 'bottom', 'right', 'top-left']
        ]
        
        for cycle in position_sets_3:
            for length in [5, 6, 7, 8]:
                task_id = f"sequence_completion_type7_{task_id_counter:04d}"
                task_params = {'cycle': cycle, 'length': length}
                all_tasks.append((task_id, 7, task_params))
                task_id_counter += 1
        
        for cycle in position_sets_4:
            for length in [6, 7, 8]:
                task_id = f"sequence_completion_type7_{task_id_counter:04d}"
                task_params = {'cycle': cycle, 'length': length}
                all_tasks.append((task_id, 7, task_params))
                task_id_counter += 1
        
        for cycle in position_sets_5:
            for length in [7, 8]:
                task_id = f"sequence_completion_type7_{task_id_counter:04d}"
                task_params = {'cycle': cycle, 'length': length}
                all_tasks.append((task_id, 7, task_params))
                task_id_counter += 1
        
        # Type 8: Mixed Sequence (Color + Shape only)
        # Generate combinations dynamically using itertools
        # Only color_shape combinations are generated (color+position and shape+position are removed)
        
        # color_shape combinations: generate all 3-element cycles of color+shape
        # Each cycle uses 3 different colors and 3 different shapes
        # Limit to avoid generating too many tasks (use first N combinations)
        color_shape_cycles = []
        limit_reached = False
        for color_combo in permutations(COLORS, 3):
            if limit_reached:
                break
            for shape_combo in permutations(SHAPE_MAP.keys(), 3):
                # Create cycle: [color1+shape1, color2+shape2, color3+shape3]
                cycle = [f"{color}{shape}" for color, shape in zip(color_combo, shape_combo)]
                color_shape_cycles.append(cycle)
                # Limit to approximately 48 cycles (similar to original hardcoded list)
                if len(color_shape_cycles) >= 48:
                    limit_reached = True
                    break
        
        for cycle in color_shape_cycles:
            for length in [6, 7, 8]:
                task_id = f"sequence_completion_type8_{task_id_counter:04d}"
                task_params = {'cycle': cycle, 'length': length, 'mixed_type': 'color_shape'}
                all_tasks.append((task_id, 8, task_params))
                task_id_counter += 1
        
        # Sample if needed
        if num_samples is not None and num_samples < len(all_tasks):
            all_tasks = random.sample(all_tasks, num_samples)
        
        return all_tasks


def create_dataset(num_samples: Optional[int] = None, 
                  output_base_dir: Union[str, Path] = "data/questions/sequence_completion_task",
                  output_size: tuple = (1024, 1024)) -> Dict[str, Any]:
    """
    Create sequence completion dataset.
    
    Args:
        num_samples: Number of tasks to generate. If None, generates all 1242 tasks.
        output_base_dir: Base directory for output
        output_size: Target output image size in pixels (width, height). Default is (1024, 1024).
    
    Returns:
        Dictionary with dataset information
    """
    output_base_dir = Path(output_base_dir)
    generator = SequenceCompletionTaskGenerator(output_size=output_size)
    
    # Generate all task definitions
    task_definitions = generator.generate_all_tasks(num_samples)
    
    # Generate actual tasks
    pairs = []
    for task_id, task_type, task_params in task_definitions:
        task_dir = output_base_dir / task_id
        try:
            task_pair = generator.generate_single_task(task_type, task_params, task_id, task_dir)
            pairs.append(task_pair)
        except Exception as e:
            print(f"Error generating task {task_id}: {e}")
            continue
    
    dataset = SequenceCompletionDataset(
        name="Sequence Completion",
        description="Sequence completion reasoning tasks for video model evaluation",
        pairs=pairs,
        metadata={
            'total_tasks': len(pairs),
            'task_types': list(range(1, 9)),
            'generated_at': datetime.now().isoformat()
        }
    )
    
    return {
        'dataset': dataset,
        'pairs': pairs
    }


def render_sequence(sequence: List[Any], show_blank: bool = False,
                   output_path: Optional[str] = None) -> None:
    """Utility function to render a sequence."""
    renderer = SequenceRenderer()
    renderer.render_sequence(sequence, show_blank, output_path)

