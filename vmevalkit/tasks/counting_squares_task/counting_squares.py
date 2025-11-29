"""
Counting Squares Task - Adapted from Tin's simple_task_video_reasoning
Original: https://github.com/tin-xai/simple_task_video_reasoning/blob/main/CountingSquares/create_squares.py

Minimal modifications to fit VMEvalKit interface.
All generation logic is preserved from Tin's original implementation.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os
import json
import tempfile
from typing import Dict, Any

# ============================================
# Tin's Original Functions (UNCHANGED)
# ============================================

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
# VMEvalKit Wrapper
# ============================================

def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Generate counting squares dataset using Tin's original generation logic.
    
    Args:
        num_samples: Number of samples to generate (None = generate all variations)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    
    # Create temp directory for images
    temp_dir = tempfile.mkdtemp()
    
    # Tin's original parameters
    num_images_per_depth = 10
    depths = [2, 3, 4, 5]
    
    test_samples = []
    text_positions = ['top', 'middle', 'bottom']
    sample_idx = 0

    # ============================================
    # Tin's Original Generation Logic (UNCHANGED)
    # ============================================

    for depth in depths:
        for i in range(num_images_per_depth):
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

            # Generate first and last frames with different line thicknesses
            for line_thickness in [2, 3, 4]:
                text_pos = text_positions[sample_idx % len(text_positions)]
                
                # Generate first frame (without text)
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_aspect("equal")
                ax.set_xlim(-15, 15)
                ax.set_ylim(-15, 15)
                ax.axis("off")
                plot_squares(ax, squares_list, line_thickness, add_text=False)
                
                first_frame_name = f"{sample_idx + 1}_first.png"
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
                
                last_frame_name = f"{sample_idx + 1}_last.png"
                plt.savefig(os.path.join(temp_dir, last_frame_name), format="png", bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

                # Tin's original data structure + minimal VMEvalKit fields
                test_sample = {
                    "sample_id": f"sample_{sample_idx + 1:04d}",
                    "prompt": "Create a video to show how to count the number of nested squares",
                    "first_frame": first_frame_name,
                    "last_frame": last_frame_name,
                    "ground_truth_count": total_count,
                    "text_position": text_pos,
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
                    "domain": "counting_squares",
                    "first_image_path": os.path.join(temp_dir, first_frame_name),
                    "final_image_path": os.path.join(temp_dir, last_frame_name),
                }
                test_samples.append(test_sample)
                sample_idx += 1
                
                if num_samples and len(test_samples) >= num_samples:
                    break
            
            if num_samples and len(test_samples) >= num_samples:
                break
        
        if num_samples and len(test_samples) >= num_samples:
            break

    return {
        "name": "counting_squares_tasks",
        "pairs": test_samples,
        "source": "tin_tasks",
        "total_samples": len(test_samples)
    }

