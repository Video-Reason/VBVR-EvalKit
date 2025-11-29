"""
Counting Pentagons Task - Adapted from Tin's simple_task_video_reasoning
Original: https://github.com/tin-xai/simple_task_video_reasoning/blob/main/CountingCircles/create_pentagons.py

Minimal modifications to fit VMEvalKit interface.
All generation logic is preserved from Tin's original implementation.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import random
import json
from matplotlib import cm
import math
from matplotlib.patches import Polygon
import os
import tempfile
from typing import Dict, Any

# ============================================
# Tin's Original Functions (UNCHANGED)
# ============================================

def draw_pentagon(ax, center, diam, side, **kwargs):
    x_points = center[0] + np.array([0, diam*np.cos(np.pi/10), side/2, -side/2, -diam*np.cos(np.pi/10)])
    y_points = center[1] + np.array([diam, diam*np.sin(np.pi/10), -diam*np.cos(np.pi/5),-diam*np.cos(np.pi/5), diam*np.sin(np.pi/10)])
    ax.fill(x_points*5, y_points*5, **kwargs)

def draw_all(centers, diam, side, dpi, colors, thickness, add_text=False, total_count=None, text_position='top', filename=None, output_dir=None):
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
    
def get_colors_from_colormap(colormap_name, num_colors):
    colormap = matplotlib.colormaps.get_cmap(colormap_name)
    colors = [colormap(i) for i in range(num_colors)]
    return colors

# ============================================
# VMEvalKit Wrapper
# ============================================

def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Generate counting pentagons dataset using Tin's original generation logic.
    
    Args:
        num_samples: Number of samples to generate (None = generate all variations)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    
    # Create temp directory for images
    temp_dir = tempfile.mkdtemp()
    
    # Tin's original parameters
    dpi = [100, 200, 300]
    num_circles = [6]
    dist = 0.1
    
    test_samples = []
    text_positions = ['top', 'middle', 'bottom']
    sample_idx = 0

    # ============================================
    # Tin's Original Generation Logic (UNCHANGED)
    # ============================================

    for thickness in [0.5, 1]:
        for d in dpi:
            for r in [5, 10]:
                side = 0.5 / r
                diam = side * 0.5/np.sin(np.pi/5)

                for num in num_circles:
                    for colors in [['black'] * num, get_colors_from_colormap('tab10', num)]:
                        
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
                            text_pos = text_positions[sample_idx % len(text_positions)]
                            first_frame_id = draw_all(centers, diam, side, d, colors, thickness, add_text=False,
                                                     filename=f"{sample_idx + 1}_first",
                                                     output_dir=temp_dir)
                            
                            # Generate last frame (with text)
                            last_frame_id = draw_all(centers, diam, side, d, colors, thickness, add_text=True,
                                                    total_count=num, text_position=text_pos,
                                                    filename=f"{sample_idx + 1}_last",
                                                    output_dir=temp_dir)
                            
                            # Tin's original data structure + minimal VMEvalKit fields
                            test_sample = {
                                "sample_id": f"sample_{sample_idx + 1:04d}",
                                "prompt": f"Create a video to show how to count the number of pentagons",
                                "first_frame": f"{first_frame_id}.png",
                                "last_frame": f"{last_frame_id}.png",
                                "ground_truth_count": num,
                                "text_position": text_pos,
                                "metadata": {
                                    "side": side,
                                    "centers": centers,
                                    "dpi": d,
                                    "canvas_size": 5.0,
                                    "linewidth": thickness,
                                    "colors": [c if isinstance(c, str) else f"rgba{tuple(c)}" for c in colors]
                                },
                                # VMEvalKit required fields
                                "id": f"counting_pentagons_{sample_idx:04d}",
                                "domain": "counting_pentagons",
                                "first_image_path": os.path.join(temp_dir, f"{first_frame_id}.png"),
                                "final_image_path": os.path.join(temp_dir, f"{last_frame_id}.png"),
                            }
                            test_samples.append(test_sample)
                            sample_idx += 1
                            
                            if num_samples and len(test_samples) >= num_samples:
                                break

                        else:
                            row_1 = num // 2
                            row_2 = row_1

                            y = 0.6
                            x = 0.5

                            ratio = dist * side
                            min_dist = 2 * diam * np.cos(np.pi/10) + ratio

                            if row_1 * diam + (row_1 - 1) * ratio + min_dist/2 >= 1:
                                continue

                            i = random.choice([0, 1])
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
                            text_pos = text_positions[sample_idx % len(text_positions)]
                            first_frame_id = draw_all(centers, diam, side, d, colors, thickness, add_text=False,
                                                     filename=f"{sample_idx + 1}_first",
                                                     output_dir=temp_dir)
                            
                            # Generate last frame (with text)
                            last_frame_id = draw_all(centers, diam, side, d, colors, thickness, add_text=True,
                                                    total_count=num, text_position=text_pos,
                                                    filename=f"{sample_idx + 1}_last",
                                                    output_dir=temp_dir)
                            
                            # Tin's original data structure + minimal VMEvalKit fields
                            test_sample = {
                                "sample_id": f"sample_{sample_idx + 1:04d}",
                                "prompt": f"Create a video to show how to count the number of pentagons",
                                "first_frame": f"{first_frame_id}.png",
                                "last_frame": f"{last_frame_id}.png",
                                "ground_truth_count": num,
                                "text_position": text_pos,
                                "metadata": {
                                    "side": side,
                                    "centers": centers,
                                    "dpi": d,
                                    "canvas_size": 5.0,
                                    "linewidth": thickness,
                                    "colors": [c if isinstance(c, str) else f"rgba{tuple(c)}" for c in colors]
                                },
                                # VMEvalKit required fields
                                "id": f"counting_pentagons_{sample_idx:04d}",
                                "domain": "counting_pentagons",
                                "first_image_path": os.path.join(temp_dir, f"{first_frame_id}.png"),
                                "final_image_path": os.path.join(temp_dir, f"{last_frame_id}.png"),
                            }
                            test_samples.append(test_sample)
                            sample_idx += 1
                            
                            if num_samples and len(test_samples) >= num_samples:
                                break
                        
                        if num_samples and len(test_samples) >= num_samples:
                            break
                    if num_samples and len(test_samples) >= num_samples:
                        break
                if num_samples and len(test_samples) >= num_samples:
                    break
            if num_samples and len(test_samples) >= num_samples:
                break
        if num_samples and len(test_samples) >= num_samples:
            break

    return {
        "name": "counting_pentagons_tasks",
        "pairs": test_samples,
        "source": "tin_tasks",
        "total_samples": len(test_samples)
    }

