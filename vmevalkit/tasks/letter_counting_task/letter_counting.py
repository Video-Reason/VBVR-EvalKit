"""
Letter Counting Task - Adapted from Tin's simple_task_video_reasoning
Original: https://github.com/tin-xai/simple_task_video_reasoning/blob/main/FindingWords/create_strings.py

Minimal modifications to fit VMEvalKit interface.
All generation logic is preserved from Tin's original implementation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import json
import os
import tempfile
from typing import Dict, Any, Optional, Sequence

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

def create_dataset(num_samples: int = 10, difficulties: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """
    Generate letter counting dataset using Tin's original generation logic.
    
    Args:
        num_samples: Number of samples to generate
        difficulties: List of difficulty levels to generate.
                     Options: ['easy', 'medium', 'hard']
                     If None, generates all difficulties
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    
    # Create temp directory for images
    temp_dir = tempfile.mkdtemp()
    
    # Setup difficulties
    diffs = list(difficulties) if difficulties else ["easy", "medium", "hard"]
    
    # Words list categorized by difficulty
    # Easy: Short words (4-7 letters)
    easy_words = ["BANANA", "COCONUT", "PIZZA", "COFFEE", "BUBBLE"]
    
    # Medium: Medium words (8-11 letters)
    medium_words = [
        "CHOCOLATE", "RESTAURANT", "ALLIGATOR", "BUTTERFLY", "WATERMELON",
        "PINEAPPLE", "TENNESSEE", "TOMORROW", "NECESSARY", "COMMITTEE"
    ]
    
    # Hard: Long words (12+ letters)
    hard_words = [
        "STRAWBERRY", "MISSISSIPPI", "BOOKKEEPER", "HIPPOPOTAMUS", 
        "GRASSHOPPER", "PARALLEL", "SUCCESSFUL", "ACCELERATION", 
        "PROGRAMMING", "MASSACHUSETTS"
    ]
    
    # Combine words based on requested difficulties
    available_words = []
    if "easy" in diffs:
        available_words.extend(easy_words)
    if "medium" in diffs:
        available_words.extend(medium_words)
    if "hard" in diffs:
        available_words.extend(hard_words)
    
    # If no words available, return empty
    if not available_words:
        return {
            "name": "letter_counting_tasks",
            "pairs": [],
            "source": "tin_tasks",
            "total_samples": 0,
            "difficulties": list(diffs)
        }
    
    test_samples = []
    text_positions = ['top', 'bottom', 'middle']
    dpi = 150

    # Generate num_samples by randomly selecting words
    for sample_idx in range(num_samples):
        # Randomly select a word
        word = random.choice(available_words)
        
        # Get unique letters in the word
        unique_letters = list(set(word.upper()))
        
        # Randomly select a letter from the word
        letter = random.choice(unique_letters)
        
        # Count occurrences
        count = count_letter_in_word(word, letter)
        
        # Determine difficulty based on word length
        word_length = len(word)
        if word_length <= 7:
            difficulty = "easy"
        elif word_length <= 11:
            difficulty = "medium"
        else:
            difficulty = "hard"
        
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
        
        # minimal VMEvalKit fields
        test_sample = {
            "sample_id": f"sample_{sample_idx + 1:04d}",
            "prompt": f"Create a video to show how to count the number of '{letter}' in {word}",
            "first_frame": f"{first_frame_id}.png",
            "last_frame": f"{last_frame_id}.png",
            "word": word,
            "target_letter": letter,
            "ground_truth_count": count,
            "text_position": text_pos,
            "difficulty": difficulty,
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

    return {
        "name": "letter_counting_tasks",
        "pairs": test_samples,
        "source": "tin_tasks",
        "total_samples": len(test_samples),
        "difficulties": list(diffs)
    }


if __name__ == "__main__":
    dataset = create_dataset(num_samples=10, difficulties=['easy', 'medium', 'hard'])
    print(dataset)
