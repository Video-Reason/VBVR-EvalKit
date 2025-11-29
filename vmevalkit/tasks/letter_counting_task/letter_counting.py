"""
Letter Counting Task - Adapted from Tin's simple_task_video_reasoning
Original: https://github.com/tin-xai/simple_task_video_reasoning/blob/main/FindingWords/create_strings.py

Minimal modifications to fit VMEvalKit interface.
All generation logic is preserved from Tin's original implementation.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import json
import os
import tempfile
from typing import Dict, Any

# ============================================
# Tin's Original Functions (UNCHANGED)
# ============================================

def draw_word(word, target_letter, dpi=100, add_circles=False, total_count=None, 
              text_position='top', filename=None, output_dir=None):
    """Draw a word with optional circles around target letters."""
    
    fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)
    ax.set_xlim(0, len(word) + 1)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Draw each letter
    letter_positions = []
    for i, letter in enumerate(word):
        x_pos = i + 1
        y_pos = 2
        
        # Draw the letter
        color = 'black'
        fontsize = 80
        ax.text(x_pos, y_pos, letter, fontsize=fontsize, ha='center', va='center', 
               color=color, fontweight='bold', family='monospace')
        
        # Store position if it's the target letter
        if letter.upper() == target_letter.upper():
            letter_positions.append((x_pos, y_pos))
    
    # Add circles around target letters if requested (for last frame)
    if add_circles:
        for x_pos, y_pos in letter_positions:
            circle = patches.Circle((x_pos, y_pos), 0.45, linewidth=4, 
                                   edgecolor='red', facecolor='none', zorder=10)
            ax.add_patch(circle)
    
    # Add text count if requested (for last frame)
    if add_circles and total_count is not None:
        text_str = f"Total: {total_count}"
        fontsize = 40
        if text_position == 'top':
            ax.text(len(word) / 2 + 0.5, 3.5, text_str, fontsize=fontsize, 
                   ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=2))
        elif text_position == 'bottom':
            ax.text(len(word) / 2 + 0.5, 0.5, text_str, fontsize=fontsize, 
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=2))
        else:  # middle
            ax.text(len(word) / 2 + 0.5, 3.2, text_str, fontsize=fontsize, 
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=2))
    
    filepath = os.path.join(output_dir, filename + '.png')
    fig.savefig(filepath, bbox_inches='tight', dpi=dpi, pad_inches=0.2)
    plt.close(fig)
    return filename


def count_letter_in_word(word, letter):
    """Count occurrences of a letter in a word (case-insensitive)."""
    return word.upper().count(letter.upper())

# ============================================
# VMEvalKit Wrapper
# ============================================

def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Generate letter counting dataset using Tin's original generation logic.
    
    Args:
        num_samples: Number of samples to generate (None = generate all variations)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    
    # Create temp directory for images
    temp_dir = tempfile.mkdtemp()
    
    # Tin's original words list
    words = [
        "STRAWBERRY", "MISSISSIPPI", "BANANA", "BOOKKEEPER", "COMMITTEE",
        "COCONUT", "PIZZA", "COFFEE", "CHOCOLATE", "RESTAURANT",
        "HIPPOPOTAMUS", "ALLIGATOR", "BUTTERFLY", "WATERMELON", "PINEAPPLE",
        "GRASSHOPPER", "TENNESSEE", "TOMORROW", "NECESSARY", "PARALLEL",
        "SUCCESSFUL", "ACCELERATION", "PROGRAMMING", "MASSACHUSETTS", "BUBBLE"
    ]
    
    test_samples = []
    text_positions = ['top', 'bottom', 'middle']
    sample_idx = 0
    
    dpis = [100, 150]

    # ============================================
    # Tin's Original Generation Logic (UNCHANGED)
    # ============================================
    
    for word in words:
        # Get unique letters in the word
        unique_letters = list(set(word.upper()))
        
        # For each word, create samples for letters that appear at least once
        for letter in unique_letters:
            count = count_letter_in_word(word, letter)
            
            # Skip if letter doesn't appear (shouldn't happen) or only create samples for interesting cases
            if count == 0:
                continue
            
            for dpi in dpis:
                text_pos = text_positions[sample_idx % len(text_positions)]
                
                # Generate first frame (without circles)
                first_frame_id = draw_word(
                    word, letter, dpi=dpi, add_circles=False,
                    filename=f"{sample_idx + 1}_first",
                    output_dir=temp_dir
                )
                
                # Generate last frame (with circles and count)
                last_frame_id = draw_word(
                    word, letter, dpi=dpi, add_circles=True, 
                    total_count=count, text_position=text_pos,
                    filename=f"{sample_idx + 1}_last",
                    output_dir=temp_dir
                )
                
                # Tin's original data structure + minimal VMEvalKit fields
                test_sample = {
                    "sample_id": f"sample_{sample_idx + 1:04d}",
                    "prompt": f"Create a video to show how to count the number of '{letter}' in {word}",
                    "first_frame": f"{first_frame_id}.png",
                    "last_frame": f"{last_frame_id}.png",
                    "word": word,
                    "target_letter": letter,
                    "ground_truth_count": count,
                    "text_position": text_pos,
                    "metadata": {
                        "word_length": len(word),
                        "dpi": dpi
                    },
                    # VMEvalKit required fields
                    "id": f"letter_counting_{sample_idx:04d}",
                    "domain": "letter_counting",
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

    return {
        "name": "letter_counting_tasks",
        "pairs": test_samples,
        "source": "tin_tasks",
        "total_samples": len(test_samples)
    }

