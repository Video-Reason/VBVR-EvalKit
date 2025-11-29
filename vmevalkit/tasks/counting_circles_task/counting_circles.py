"""
Counting Circles Task - Adapted from Tin's simple_task_video_reasoning
Original: https://github.com/tin-xai/simple_task_video_reasoning/blob/main/CountingCircles/create_circles.py

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
import os
import tempfile
from typing import Dict, Any

# ============================================
# Tin's Original Functions (UNCHANGED)
# ============================================

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

# ============================================
# VMEvalKit Wrapper
# ============================================

def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Generate counting circles dataset using Tin's original generation logic.
    
    Args:
        num_samples: Number of samples to generate (None = generate all variations)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    
    # Create temp directory for images
    temp_dir = tempfile.mkdtemp()
    
    # Tin's original parameters
    size = 500
    dpi = [100, 200, 300]
    num_circles = [5, 6, 7, 8, 9]
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
                rad = 0.5 / r
                for num in num_circles:
                    for colors in [['black'] * num, get_colors_from_colormap('tab10', num)]:
                        
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
                            text_pos = text_positions[sample_idx % len(text_positions)]
                            first_frame_id = draw_circles(d, size, rad, centers, colors, thickness, 
                                                         add_text=False, 
                                                         filename=f"{sample_idx + 1}_first",
                                                         output_dir=temp_dir)
                            
                            last_frame_id = draw_circles(d, size, rad, centers, colors, thickness, 
                                                        add_text=True, 
                                                        total_count=num, text_position=text_pos,
                                                        filename=f"{sample_idx + 1}_last",
                                                        output_dir=temp_dir)
                            
                            # Tin's original data structure + minimal VMEvalKit fields
                            test_sample = {
                                "sample_id": f"sample_{sample_idx + 1:04d}",
                                "prompt": f"Create a video to show how to count the number of circles",
                                "first_frame": f"{first_frame_id}.png",
                                "last_frame": f"{last_frame_id}.png",
                                "ground_truth_count": num,
                                "text_position": text_pos,
                                "metadata": {
                                    "diameter": rad * 2,
                                    "centers": centers,
                                    "distance": dist,
                                    "dpi": d,
                                    "canvas_size": 5.0,
                                    "linewidth": thickness,
                                    "colors": [c if isinstance(c, str) else f"rgba{tuple(c)}" for c in colors]
                                },
                                # VMEvalKit required fields
                                "id": f"counting_circles_{sample_idx:04d}",
                                "domain": "counting_circles",
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
                            
                            ratio = dist * rad
                            min_dist = rad * 2.0 + ratio
                            
                            if row_2 * min_dist + 2 * rad >= 1:
                                continue
                            
                            for i in range(2):
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
                                text_pos = text_positions[sample_idx % len(text_positions)]
                                first_frame_id = draw_circles(d, size, rad, centers, colors, thickness, add_text=False,
                                                             filename=f"{sample_idx + 1}_first",
                                                             output_dir=temp_dir)
                                
                                last_frame_id = draw_circles(d, size, rad, centers, colors, thickness, add_text=True,
                                                            total_count=num, text_position=text_pos,
                                                            filename=f"{sample_idx + 1}_last",
                                                            output_dir=temp_dir)
                                
                                # Tin's original data structure + minimal VMEvalKit fields
                                test_sample = {
                                    "sample_id": f"sample_{sample_idx + 1:04d}",
                                    "prompt": f"Create a video to show how to count the number of circles",
                                    "first_frame": f"{first_frame_id}.png",
                                    "last_frame": f"{last_frame_id}.png",
                                    "ground_truth_count": num,
                                    "text_position": text_pos,
                                    "metadata": {
                                        "diameter": rad * 2,
                                        "centers": centers,
                                        "distance": dist,
                                        "dpi": d,
                                        "canvas_size": 5.0,
                                        "linewidth": thickness,
                                        "colors": [c if isinstance(c, str) else f"rgba{tuple(c)}" for c in colors]
                                    },
                                    # VMEvalKit required fields
                                    "id": f"counting_circles_{sample_idx:04d}",
                                    "domain": "counting_circles",
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
        "name": "counting_circles_tasks",
        "pairs": test_samples,
        "source": "tin_tasks",
        "total_samples": len(test_samples)
    }

