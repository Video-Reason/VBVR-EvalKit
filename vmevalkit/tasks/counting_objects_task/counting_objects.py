"""
Counting Objects Task - Combined circles, pentagons, and squares counting
Adapted from Tin's simple_task_video_reasoning

Original sources:
- Circles: https://github.com/tin-xai/simple_task_video_reasoning/blob/main/CountingCircles/create_circles.py
- Pentagons: https://github.com/tin-xai/simple_task_video_reasoning/blob/main/CountingCircles/create_pentagons.py
- Squares: https://github.com/tin-xai/simple_task_video_reasoning/blob/main/CountingSquares/create_squares.py

All generation logic is preserved from Tin's original implementations.
"""

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
from matplotlib import cm
import numpy as np
import random
import json
import os
import tempfile
from typing import Dict, Any, List, Optional, Sequence

def hue_to_rgb(hue):
    rgb = hsv_to_rgb([hue, 1, 1])
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def get_colors_from_colormap(colormap_name, num_colors):
    colormap = cm.get_cmap(colormap_name, num_colors)
    colors = [colormap(i) for i in range(num_colors)]
    return colors

def draw_circles(dpi, size, radius, centers, colors, thickness, add_text=False, 
                 total_count=None, text_position='top', filename=None, output_dir=None):
    """Tin's original draw_circles function."""
    
    assert len(centers) == len(colors)
    h=5
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, h)
    ax.set_ylim(0, h)
    ax.axis("off")

    for center, color in zip(centers, colors):
        circle1_plot = plt.Circle((center[0] * h, center[1] * h), radius * h, color=color, fill=False, linewidth=thickness)
        ax.add_artist(circle1_plot)

    # Add text if requested (for last frame)
    if add_text and total_count is not None:
        text_str = f"Total: {total_count}"
        fontsize = 20
        if text_position == 'top':
            ax.text(h/2, h * 0.95, text_str, fontsize=fontsize, ha='center', va='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        elif text_position == 'bottom':
            ax.text(h/2, h * 0.05, text_str, fontsize=fontsize, ha='center', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:  # middle
            ax.text(h/2, h/2, text_str, fontsize=fontsize, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    filepath = os.path.join(output_dir, filename + '.png')
    fig.savefig(filepath, bbox_inches='tight', dpi=dpi, pad_inches=0)
    plt.close(fig)
    return filename

def draw_pentagon(ax, center, diam, side, **kwargs):
    x_points = center[0] + np.array([0, diam*np.cos(np.pi/10), side/2, -side/2, -diam*np.cos(np.pi/10)])
    y_points = center[1] + np.array([diam, diam*np.sin(np.pi/10), -diam*np.cos(np.pi/5),-diam*np.cos(np.pi/5), diam*np.sin(np.pi/10)])
    ax.fill(x_points*5, y_points*5, **kwargs)

def draw_pentagons(centers, diam, side, dpi, colors, thickness, add_text=False, total_count=None, text_position='top', filename=None, output_dir=None):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)

    for center, color in zip(centers, colors):
        draw_pentagon(ax, center, diam, side, edgecolor=color, fill=False, linewidth=thickness)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal', adjustable='box')
    ax.axis("off")
    
    # Add text if requested (for last frame)
    if add_text and total_count is not None:
        text_str = f"Total: {total_count}"
        fontsize = 20
        if text_position == 'top':
            ax.text(2.5, 4.75, text_str, fontsize=fontsize, ha='center', va='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        elif text_position == 'bottom':
            ax.text(2.5, 0.25, text_str, fontsize=fontsize, ha='center', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:  # middle
            ax.text(2.5, 2.5, text_str, fontsize=fontsize, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    filepath = os.path.join(output_dir, filename + '.png')
    fig.savefig(filepath, bbox_inches='tight', dpi=dpi, pad_inches=0)
    plt.close(fig)
    return filename

def compute_squares(center, size, depth, reduction_factor, padding, squares_list):
    if depth == 0:
        return

    # Store the current square's details
    squares_list.append({"center": center, "size": size})

    # Calculate the size of the next square, reduced by the reduction factor and padding
    new_size = size * reduction_factor - padding

    # Ensure new_size is positive
    if new_size <= 0:
        return

    # Generate random offsets within bounds to ensure no overlap, adjusted for padding
    max_offset = (size - new_size - padding) / 2
    offset_x = random.uniform(-max_offset, max_offset)
    offset_y = random.uniform(-max_offset, max_offset)

    # Calculate the new center
    new_center = (center[0] + offset_x, center[1] + offset_y)

    # Recursive call to compute further nested squares
    compute_squares(
        new_center, new_size, depth - 1, reduction_factor, padding, squares_list
    )


def plot_squares(ax, squares_list, line_thickness, add_text=False, total_count=None, text_position='top'):
    for square in squares_list:
        center = square["center"]
        size = square["size"]
        # Create and add a square patch to the axes
        square_patch = patches.Rectangle(
            (center[0] - size / 2, center[1] - size / 2),
            size,
            size,
            fill=False,
            linewidth=line_thickness,
        )
        ax.add_patch(square_patch)
    
    # Add text if requested (for last frame)
    if add_text and total_count is not None:
        text_str = f"Total: {total_count}"
        fontsize = 20
        if text_position == 'top':
            ax.text(0, 14, text_str, fontsize=fontsize, ha='center', va='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        elif text_position == 'bottom':
            ax.text(0, -14, text_str, fontsize=fontsize, ha='center', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:  # middle
            ax.text(0, 0, text_str, fontsize=fontsize, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================
# CIRCLES Dataset Generation
# ============================================

def create_circles_dataset(num_samples: int = 10, temp_dir: str = None, difficulties: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """
    Generate counting circles dataset using Tin's original generation logic.
    
    Args:
        num_samples: Number of samples to generate
        temp_dir: Directory to save images (creates temp if None)
        difficulties: List of difficulty levels to generate. 
                     Options: ['easy', 'medium', 'hard']
                     If None, generates all difficulties
        
    Returns:
        List of sample dictionaries
    """
    
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    # Setup difficulties
    diffs = list(difficulties) if difficulties else ["easy", "medium", "hard"]
    
    # Determine which counts to generate based on difficulties
    num_circles = []
    if "easy" in diffs:
        num_circles.extend([4, 5])
    if "medium" in diffs:
        num_circles.extend([6, 7])
    if "hard" in diffs:
        num_circles.extend([8, 9])
    
    # If no circles to generate, return empty
    if not num_circles:
        return []
    
    # Tin's original parameters
    size = 500
    dpi = 300  # Use highest DPI
    thickness = 1  # Use highest thickness
    dist = 0.1
    
    test_samples = []
    text_positions = ['top', 'middle', 'bottom']
    
    # Create all possible configurations
    configurations = []
    for r in [5, 10]:
        for num in num_circles:
            for colors in [['black'] * num, get_colors_from_colormap('tab10', num)]:
                configurations.append((r, num, colors))
    
    # Generate num_samples by cycling through configurations
    for sample_idx in range(num_samples):
        # Select configuration (cycle through if needed)
        config_idx = sample_idx % len(configurations)
        r, num, colors = configurations[config_idx]
        
        rad = 0.5 / r
        
        # Determine difficulty
        if num in [4, 5]:
            difficulty = "easy"
        elif num in [8, 9]:
            difficulty = "hard"
        elif num in [6, 7]:
            difficulty = "medium"
        else:
            difficulty = "medium"  # fallback
        
        text_pos = text_positions[sample_idx % len(text_positions)]
        
        if num % 2 != 0:
            centers = []
            row_1 = (num + 1) // 2
            row_2 = row_1 - 1
            
            y = 0.6
            x = 0.5
            
            ratio = dist * rad
            min_dist = rad * 2.0 + ratio
            
            if row_1 * rad * 2 + row_2 * ratio >= 1:
                continue
            
            if row_1 == 3:
                centers.append([x, y])
                centers.append([x - min_dist, y])
                centers.append([x + min_dist, y])
                centers.append([x - rad - ratio/2, y - rad])
                centers.append([x + rad + ratio/2, y - rad])
            elif row_1 == 5:
                centers.append([x, y])
                centers.append([x - min_dist, y])
                centers.append([x + min_dist, y])
                centers.append([x - 2 * min_dist, y])
                centers.append([x + 2 * min_dist, y])
                centers.append([x - rad - ratio / 2, y - rad])
                centers.append([x + rad + ratio / 2, y - rad])
                centers.append([x - rad - ratio - min_dist, y - rad])
                centers.append([x + rad + ratio + min_dist, y - rad])
            elif row_1 == 2:
                centers.append([x - rad - ratio/2, y])
                centers.append([x + rad + ratio/2, y])
                centers.append([x, y - rad])
            else:
                centers.append([x - rad - ratio/2, y])
                centers.append([x + rad + ratio/2, y])
                centers.append([x - rad - ratio/2 - min_dist, y])
                centers.append([x + rad + ratio/2 + min_dist, y])
                centers.append([x, y - rad])
                centers.append([x + min_dist, y - rad])
                centers.append([x - min_dist, y - rad])
            
            # Generate frames using Tin's logic
            first_frame_id = draw_circles(dpi, size, rad, centers, colors, thickness, 
                                            add_text=False, 
                                            filename=f"circles_{sample_idx + 1}_first",
                                            output_dir=temp_dir)
            
            last_frame_id = draw_circles(dpi, size, rad, centers, colors, thickness, 
                                        add_text=True, 
                                        total_count=num, text_position=text_pos,
                                        filename=f"circles_{sample_idx + 1}_last",
                                        output_dir=temp_dir)
            
            test_sample = {
                "sample_id": f"circles_{sample_idx + 1:04d}",
                "prompt": f"Create a video to show how to count the number of circles",
                "first_frame": f"{first_frame_id}.png",
                "last_frame": f"{last_frame_id}.png",
                "ground_truth_count": num,
                "text_position": text_pos,
                "shape_type": "circles",
                "difficulty": difficulty,
                "metadata": {
                    "diameter": rad * 2,
                    "centers": centers,
                    "distance": dist,
                    "dpi": dpi,
                    "canvas_size": 5.0,
                    "linewidth": thickness,
                    "colors": [c if isinstance(c, str) else f"rgba{tuple(c)}" for c in colors]
                },
                # VMEvalKit required fields
                "id": f"counting_circles_{sample_idx:04d}",
                "domain": "counting_objects",
                "first_image_path": os.path.join(temp_dir, f"{first_frame_id}.png"),
                "final_image_path": os.path.join(temp_dir, f"{last_frame_id}.png"),
            }
            test_samples.append(test_sample)
        
        else:
            # Even number case
            row_1 = num // 2
            row_2 = row_1
            
            y = 0.6
            x = 0.5
            
            ratio = dist * rad
            min_dist = rad * 2.0 + ratio
            
            if row_2 * min_dist + 2 * rad >= 1:
                continue
            
            # Choose i based on sample_idx to have variation
            i = sample_idx % 2
            centers = []
            if row_1 == 3:
                centers.append([x, y])
                centers.append([x - min_dist, y])
                centers.append([x + min_dist, y])
                centers.append([x - rad - ratio/2, y - rad])
                centers.append([x + rad + ratio/2, y - rad])
                if i == 0:
                    centers.append([x - rad - ratio - min_dist, y - rad])
                else:
                    centers.append([x + rad + ratio + min_dist, y - rad])
            elif row_1 == 2:
                centers.append([x - rad - ratio/2, y])
                centers.append([x + rad + ratio/2, y])
                centers.append([x, y - rad])
                if i == 0:
                    centers.append([x + min_dist, y - rad])
                else:
                    centers.append([x - min_dist, y - rad])
            else:
                centers.append([x - rad - ratio/2, y])
                centers.append([x + rad + ratio/2, y])
                centers.append([x - rad - ratio/2 - min_dist, y])
                centers.append([x + rad + ratio/2 + min_dist, y])
                centers.append([x, y - rad])
                centers.append([x + min_dist, y - rad])
                centers.append([x - min_dist, y - rad])
                if i == 0:
                    centers.append([x + 2 * min_dist, y - rad])
                else:
                    centers.append([x - 2 * min_dist, y - rad])
            
            # Generate frames
            first_frame_id = draw_circles(dpi, size, rad, centers, colors, thickness, add_text=False,
                                            filename=f"circles_{sample_idx + 1}_first",
                                            output_dir=temp_dir)
            
            last_frame_id = draw_circles(dpi, size, rad, centers, colors, thickness, add_text=True,
                                        total_count=num, text_position=text_pos,
                                        filename=f"circles_{sample_idx + 1}_last",
                                        output_dir=temp_dir)
            
            test_sample = {
                "sample_id": f"circles_{sample_idx + 1:04d}",
                "prompt": f"Create a video to show how to count the number of circles",
                "first_frame": f"{first_frame_id}.png",
                "last_frame": f"{last_frame_id}.png",
                "ground_truth_count": num,
                "text_position": text_pos,
                "shape_type": "circles",
                "difficulty": difficulty,
                "metadata": {
                    "diameter": rad * 2,
                    "centers": centers,
                    "distance": dist,
                    "dpi": dpi,
                    "canvas_size": 5.0,
                    "linewidth": thickness,
                    "colors": [c if isinstance(c, str) else f"rgba{tuple(c)}" for c in colors]
                },
                # VMEvalKit required fields
                "id": f"counting_circles_{sample_idx:04d}",
                "domain": "counting_objects",
                "first_image_path": os.path.join(temp_dir, f"{first_frame_id}.png"),
                "final_image_path": os.path.join(temp_dir, f"{last_frame_id}.png"),
            }
            test_samples.append(test_sample)
    
    return test_samples

# ============================================
# PENTAGONS Dataset Generation
# ============================================

def create_pentagons_dataset(num_samples: int = 10, temp_dir: str = None, difficulties: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """
    Generate counting pentagons dataset using Tin's original generation logic.
    
    Args:
        num_samples: Number of samples to generate
        temp_dir: Directory to save images (creates temp if None)
        difficulties: List of difficulty levels to generate. 
                     Options: ['easy', 'medium', 'hard']
                     If None, generates all difficulties
        
    Returns:
        List of sample dictionaries
    """
    
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    # Setup difficulties
    diffs = list(difficulties) if difficulties else ["easy", "medium", "hard"]
    
    # Determine which counts to generate based on difficulties
    num_pentagons = []
    if "easy" in diffs:
        num_pentagons.extend([4, 5])
    if "medium" in diffs:
        num_pentagons.extend([6, 7])
    if "hard" in diffs:
        num_pentagons.extend([8, 9])
    
    # If no pentagons to generate, return empty
    if not num_pentagons:
        return []
    
    # Tin's original parameters
    dpi = 300  # Use highest DPI
    thickness = 1  # Use highest thickness
    dist = 0.1
    
    test_samples = []
    text_positions = ['top', 'middle', 'bottom']
    
    # Create all possible configurations
    configurations = []
    for r in [5, 10]:
        side = 0.5 / r
        diam = side * 0.5/np.sin(np.pi/5)
        for num in num_pentagons:
            for colors in [['black'] * num, get_colors_from_colormap('tab10', num)]:
                configurations.append((r, side, diam, num, colors))
    
    # Generate num_samples by cycling through configurations  
    for sample_idx in range(num_samples):
        # Select configuration (cycle through if needed)
        config_idx = sample_idx % len(configurations)
        r, side, diam, num, colors = configurations[config_idx]
        
        # Determine difficulty
        if num in [4, 5]:
            difficulty = "easy"
        elif num in [8, 9]:
            difficulty = "hard"
        elif num in [6, 7]:
            difficulty = "medium"
        else:
            difficulty = "medium"  # fallback
        
        text_pos = text_positions[sample_idx % len(text_positions)]
        
        if num % 2 != 0:
            centers = []
            row_1 = (num + 1) // 2
            row_2 = row_1 - 1

            y = 0.6
            x = 0.5

            ratio = dist * side
            min_dist = 2 * diam * np.cos(np.pi/10) + ratio

            if row_1 * min_dist * 2 + row_2 * ratio >= 1:
                continue

            if row_1 == 3:
                centers.append([x, y])
                centers.append([x - min_dist, y])
                centers.append([x + min_dist, y])
                centers.append([x - min_dist/2, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
                centers.append([x + min_dist/2, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
            elif row_1 == 5:
                centers.append([x, y])
                centers.append([x - min_dist, y])
                centers.append([x + min_dist, y])
                centers.append([x - 2 * min_dist, y])
                centers.append([x + 2 * min_dist, y])
                centers.append([x - min_dist/2, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
                centers.append([x + min_dist/2, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
                centers.append([x - min_dist - min_dist/2, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
                centers.append([x + min_dist + min_dist/2, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
            else:
                centers.append([x-min_dist/2, y])
                centers.append([x+min_dist/2, y])
                centers.append([x-min_dist/2 - min_dist, y])
                centers.append([x+min_dist/2 + min_dist, y])
                centers.append([x, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
                centers.append([x - min_dist, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
                centers.append([x + min_dist, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])

            # Generate first frame (without text)
            first_frame_id = draw_pentagons(centers, diam, side, dpi, colors, thickness, add_text=False,
                                         filename=f"pentagons_{sample_idx + 1}_first",
                                         output_dir=temp_dir)
            
            # Generate last frame (with text)
            last_frame_id = draw_pentagons(centers, diam, side, dpi, colors, thickness, add_text=True,
                                        total_count=num, text_position=text_pos,
                                        filename=f"pentagons_{sample_idx + 1}_last",
                                        output_dir=temp_dir)
            
            test_sample = {
                "sample_id": f"pentagons_{sample_idx + 1:04d}",
                "prompt": f"Create a video to show how to count the number of pentagons",
                "first_frame": f"{first_frame_id}.png",
                "last_frame": f"{last_frame_id}.png",
                "ground_truth_count": num,
                "text_position": text_pos,
                "shape_type": "pentagons",
                "difficulty": difficulty,
                "metadata": {
                    "side": side,
                    "centers": centers,
                    "dpi": dpi,
                    "canvas_size": 5.0,
                    "linewidth": thickness,
                    "colors": [c if isinstance(c, str) else f"rgba{tuple(c)}" for c in colors]
                },
                # VMEvalKit required fields
                "id": f"counting_pentagons_{sample_idx:04d}",
                "domain": "counting_objects",
                "first_image_path": os.path.join(temp_dir, f"{first_frame_id}.png"),
                "final_image_path": os.path.join(temp_dir, f"{last_frame_id}.png"),
            }
            test_samples.append(test_sample)

        else:
            # Even number case
            row_1 = num // 2
            row_2 = row_1

            y = 0.6
            x = 0.5

            ratio = dist * side
            min_dist = 2 * diam * np.cos(np.pi/10) + ratio

            if row_1 * diam + (row_1 - 1) * ratio + min_dist/2 >= 1:
                continue

            i = sample_idx % 2  # Use sample_idx for variation
            centers = []
            if row_1 == 3:
                centers.append([x, y])
                centers.append([x - min_dist, y])
                centers.append([x + min_dist, y])
                centers.append([x - min_dist/2, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
                centers.append([x + min_dist/2, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
                if i == 0:
                    centers.append([x - min_dist - min_dist/2, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
                else:
                    centers.append([x + min_dist + min_dist/2, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
            else:
                centers.append([x-min_dist/2, y])
                centers.append([x+min_dist/2, y])
                centers.append([x-min_dist/2 - min_dist, y])
                centers.append([x+min_dist/2 + min_dist, y])
                centers.append([x, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
                centers.append([x - min_dist, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
                centers.append([x + min_dist, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
                if i == 0:
                    centers.append([x - 2 * min_dist, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])
                else:
                    centers.append([x - 2 * min_dist, y - diam * (np.cos(np.pi/5) + np.sin(np.pi/10))])

            # Generate first frame (without text)
            first_frame_id = draw_pentagons(centers, diam, side, dpi, colors, thickness, add_text=False,
                                         filename=f"pentagons_{sample_idx + 1}_first",
                                         output_dir=temp_dir)
            
            # Generate last frame (with text)
            last_frame_id = draw_pentagons(centers, diam, side, dpi, colors, thickness, add_text=True,
                                        total_count=num, text_position=text_pos,
                                        filename=f"pentagons_{sample_idx + 1}_last",
                                        output_dir=temp_dir)
            
            test_sample = {
                "sample_id": f"pentagons_{sample_idx + 1:04d}",
                "prompt": f"Create a video to show how to count the number of pentagons",
                "first_frame": f"{first_frame_id}.png",
                "last_frame": f"{last_frame_id}.png",
                "ground_truth_count": num,
                "text_position": text_pos,
                "shape_type": "pentagons",
                "difficulty": difficulty,
                "metadata": {
                    "side": side,
                    "centers": centers,
                    "dpi": dpi,
                    "canvas_size": 5.0,
                    "linewidth": thickness,
                    "colors": [c if isinstance(c, str) else f"rgba{tuple(c)}" for c in colors]
                },
                # VMEvalKit required fields
                "id": f"counting_pentagons_{sample_idx:04d}",
                "domain": "counting_objects",
                "first_image_path": os.path.join(temp_dir, f"{first_frame_id}.png"),
                "final_image_path": os.path.join(temp_dir, f"{last_frame_id}.png"),
            }
            test_samples.append(test_sample)

    return test_samples

# ============================================
# SQUARES Dataset Generation
# ============================================

def create_squares_dataset(num_samples: int = 10, temp_dir: str = None, difficulties: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """
    Generate counting squares dataset using Tin's original generation logic.
    
    Args:
        num_samples: Number of samples to generate
        temp_dir: Directory to save images (creates temp if None)
        difficulties: List of difficulty levels to generate. 
                     Options: ['easy', 'medium', 'hard']
                     If None, generates all difficulties
        
    Returns:
        List of sample dictionaries
    """
    
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    
    # Setup difficulties
    diffs = list(difficulties) if difficulties else ["easy", "medium", "hard"]
    
    # Determine which depths to generate based on difficulties
    depths = []
    if "easy" in diffs:
        depths.append(2)
    if "medium" in diffs:
        depths.append(3)
    if "hard" in diffs:
        depths.extend([4, 5])
    
    # If no depths to generate, return empty
    if not depths:
        return []
    
    test_samples = []
    text_positions = ['top', 'middle', 'bottom']
    
    # Create all possible configurations
    configurations = []
    for depth in depths:
        for line_thickness in [2, 3, 4]:
            configurations.append((depth, line_thickness))
    
    # Generate num_samples by cycling through configurations
    for sample_idx in range(num_samples):
        # Select configuration (cycle through if needed)
        config_idx = sample_idx % len(configurations)
        depth, line_thickness = configurations[config_idx]
        
        # Determine difficulty
        if depth == 2:
            difficulty = "easy"
        elif depth == 3:
            difficulty = "medium"
        else:  # 4 or 5
            difficulty = "hard"
        
        # Generate random parameters
        center = (random.uniform(-5, 5), random.uniform(-5, 5))
        initial_size = random.uniform(8, 18)
        reduction_factor = 0.75
        padding = 0.75

        # Compute all squares first
        squares_list = []
        compute_squares(
            center, initial_size, depth, reduction_factor, padding, squares_list
        )

        # Calculate total number of nested squares
        total_count = len(squares_list)
        
        text_pos = text_positions[sample_idx % len(text_positions)]
        
        # Generate first frame (without text)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.axis("off")
        plot_squares(ax, squares_list, line_thickness, add_text=False)
        
        first_frame_name = f"squares_{sample_idx + 1}_first.png"
        plt.savefig(os.path.join(temp_dir, first_frame_name), format="png", bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        # Generate last frame (with text)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal")
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.axis("off")
        plot_squares(ax, squares_list, line_thickness, add_text=True, 
                   total_count=total_count, text_position=text_pos)
        
        last_frame_name = f"squares_{sample_idx + 1}_last.png"
        plt.savefig(os.path.join(temp_dir, last_frame_name), format="png", bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        test_sample = {
            "sample_id": f"squares_{sample_idx + 1:04d}",
            "prompt": "Create a video to show how to count the number of nested squares",
            "first_frame": first_frame_name,
            "last_frame": last_frame_name,
            "ground_truth_count": total_count,
            "text_position": text_pos,
            "shape_type": "squares",
            "difficulty": difficulty,
            "metadata": {
                "depth": depth,
                "center": center,
                "initial_size": initial_size,
                "reduction_factor": reduction_factor,
                "line_thickness": line_thickness,
                "padding": padding,
                "squares": squares_list,
            },
            # VMEvalKit required fields
            "id": f"counting_squares_{sample_idx:04d}",
            "domain": "counting_objects",
            "first_image_path": os.path.join(temp_dir, first_frame_name),
            "final_image_path": os.path.join(temp_dir, last_frame_name),
        }
        test_samples.append(test_sample)

    return test_samples

# ============================================
# Main Dataset Creation Function
# ============================================

def create_dataset(
    num_samples: int = 10,
    shape_types: Optional[List[str]] = None,
    difficulties: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """
    Generate counting objects dataset for specified shape types.
    
    Args:
        num_samples: Number of samples to generate per shape type
        shape_types: List of shape types to generate. 
                     Options: ['circles', 'pentagons', 'squares']
                     If None, generates all types
        difficulties: List of difficulty levels to generate.
                     Options: ['easy', 'medium', 'hard']
                     If None, generates all difficulties
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    
    if shape_types is None:
        shape_types = ['circles', 'pentagons', 'squares']
    
    # Create temp directory for all images
    temp_dir = tempfile.mkdtemp()
    
    all_samples = []
    
    if 'circles' in shape_types:
        circles_samples = create_circles_dataset(num_samples, temp_dir, difficulties)
        all_samples.extend(circles_samples)
    
    if 'pentagons' in shape_types:
        pentagons_samples = create_pentagons_dataset(num_samples, temp_dir, difficulties)
        all_samples.extend(pentagons_samples)
    
    if 'squares' in shape_types:
        squares_samples = create_squares_dataset(num_samples, temp_dir, difficulties)
        all_samples.extend(squares_samples)
    
    return {
        "name": "counting_objects_tasks",
        "pairs": all_samples,
        "source": "tin_tasks",
        "total_samples": len(all_samples),
        "shape_types": shape_types,
        "samples_per_type": num_samples,
        "difficulties": list(difficulties) if difficulties else ["easy", "medium", "hard"]
    }



if __name__ == "__main__":
    dataset = create_dataset(num_samples=10, shape_types=['circles', 'pentagons', 'squares'], difficulties=['easy', 'medium', 'hard'])
    print(dataset)
