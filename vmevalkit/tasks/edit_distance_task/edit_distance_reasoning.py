"""
Edit Distance Reasoning Task for VMEvalKit

A visual calculation benchmark where video generation models must compute the
Levenshtein (edit) distance between two strings and display the result.

Task Flow:
1. Input: Static frame showing String_A (top), String_B (bottom), empty answer box (right)
2. Process: Model calculates edit distance between the strings
3. Output: Video where the calculated number appears in the answer box

Author: VMEvalKit Team
"""

import json
import random
import string
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import prompts from centralized location
from .PROMPTS import PROMPTS, DEFAULT_PROMPT_INDEX


# Word lists for generating realistic string pairs
WORD_LIST_SHORT = [
    "CAT", "DOG", "BAT", "RAT", "HAT", "MAT", "SAT", "FAT",
    "BIG", "DIG", "FIG", "PIG", "WIG", "JIG",
    "SUN", "RUN", "FUN", "BUN", "GUN", "NUN",
    "RED", "BED", "LED", "FED", "WED",
]

WORD_LIST_MEDIUM = [
    "KITTEN", "SITTING", "HORSE", "HOUSE", "MOUSE", "MOOSE",
    "BREAD", "BREAK", "GREAT", "GREET", "GREEN", "QUEEN",
    "APPLE", "APPLY", "SUPPLY", "SIMPLY",
    "WATER", "LATER", "HATER", "MATER",
    "NIGHT", "LIGHT", "RIGHT", "SIGHT", "FIGHT",
    "SOUND", "ROUND", "POUND", "FOUND", "BOUND",
]

WORD_LIST_LONG = [
    "ALGORITHM", "LOGARITHM", "ARITHMETIC",
    "ELEPHANT", "RELEVANT", "ELEMENTS",
    "BEAUTIFUL", "BOUNTIFUL",
    "COMPUTER", "COMMUTER", "COMPILER",
    "DISTANCE", "INSTANCE", "SUBSTANCE",
    "KEYBOARD", "CUPBOARD", "CARDBOARD",
]


def calculate_levenshtein_distance(str1: str, str2: str) -> int:
    """
    Calculate the Levenshtein (edit) distance between two strings.
    
    Uses dynamic programming to compute the minimum number of single-character
    edits (insertions, deletions, or substitutions) required to transform
    str1 into str2.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        The edit distance (integer >= 0)
    """
    len1, len2 = len(str1), len(str2)
    
    # Create DP table
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Initialize base cases
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )
    
    return dp[len1][len2]


def generate_random_string(length: int, rng: random.Random) -> str:
    """
    Generate a random uppercase alphabetic string.
    
    Args:
        length: Length of the string to generate
        rng: Random number generator
        
    Returns:
        Random uppercase string
    """
    return ''.join(rng.choices(string.ascii_uppercase, k=length))


def find_word_pair_with_distance(word_list: List[str], target_min: int, target_max: int,
                                  rng: random.Random, max_attempts: int = 100) -> Optional[Tuple[str, str]]:
    """
    Find a pair of words from the word list with edit distance in the target range.
    
    Args:
        word_list: List of words to choose from
        target_min: Minimum desired edit distance
        target_max: Maximum desired edit distance
        rng: Random number generator
        max_attempts: Maximum attempts to find a suitable pair
        
    Returns:
        Tuple of (word1, word2) if found, None otherwise
    """
    for _ in range(max_attempts):
        word1 = rng.choice(word_list)
        word2 = rng.choice(word_list)
        if word1 != word2:
            distance = calculate_levenshtein_distance(word1, word2)
            if target_min <= distance <= target_max:
                return word1, word2
    return None


def apply_random_edits(base_string: str, num_edits: int, rng: random.Random,
                       min_length: int = 1) -> str:
    """
    Apply random edits (insertions, deletions, substitutions) to a string.
    
    Args:
        base_string: String to modify
        num_edits: Number of edits to apply
        rng: Random number generator
        min_length: Minimum length to maintain (prevents empty strings)
        
    Returns:
        Modified string
    """
    result = list(base_string)
    
    for _ in range(num_edits):
        if len(result) == 0:
            # If string is empty, only insert
            result.insert(0, rng.choice(string.ascii_uppercase))
            continue
            
        edit_type = rng.choice(["substitute", "insert", "delete"])
        
        if edit_type == "substitute":
            pos = rng.randint(0, len(result) - 1)
            result[pos] = rng.choice(string.ascii_uppercase)
        elif edit_type == "insert":
            pos = rng.randint(0, len(result))
            result.insert(pos, rng.choice(string.ascii_uppercase))
        elif edit_type == "delete" and len(result) > min_length:
            pos = rng.randint(0, len(result) - 1)
            result.pop(pos)
    
    return ''.join(result)


def generate_string_pair(difficulty: str, rng: random.Random, 
                        use_words: bool = True) -> Tuple[str, str, int]:
    """
    Generate a pair of strings with specified difficulty level.
    
    Uses constraint-based generation:
    - Easy: distance 1-2, length 3-6
    - Medium: distance 3-5, length 6-10
    - Hard: distance 5-8, length 10-15
    
    Args:
        difficulty: "easy", "medium", or "hard"
        rng: Random number generator
        use_words: If True, use word lists; if False, use random strings
        
    Returns:
        Tuple of (string_a, string_b, edit_distance)
    """
    max_attempts = 50  # Maximum attempts to satisfy constraints
    
    if difficulty == "easy":
        # Constraints: 3-6 chars, distance 1-2
        target_min_dist, target_max_dist = 1, 2
        min_len, max_len = 3, 6
        word_list = WORD_LIST_SHORT
        
        if use_words:
            # Try to find word pair satisfying constraints
            word_pair = find_word_pair_with_distance(word_list, target_min_dist, target_max_dist, rng)
            if word_pair:
                str_a, str_b = word_pair
            else:
                # Fallback: generate by modifying a word
                base_word = rng.choice(word_list)
                str_a = base_word
                str_b = apply_random_edits(base_word, rng.randint(1, 2), rng, min_length=1)
        else:
            # Random strings with controlled edits
            length = rng.randint(min_len, max_len)
            str_a = generate_random_string(length, rng)
            str_b = apply_random_edits(str_a, rng.randint(1, 2), rng, min_length=1)
    
    elif difficulty == "medium":
        # Constraints: 6-10 chars, distance 3-5
        target_min_dist, target_max_dist = 3, 5
        min_len, max_len = 6, 10
        word_list = WORD_LIST_MEDIUM
        
        if use_words:
            # Try to find word pair satisfying constraints
            word_pair = find_word_pair_with_distance(word_list, target_min_dist, target_max_dist, rng)
            if word_pair:
                str_a, str_b = word_pair
            else:
                # Fallback: generate by modifying a word
                base_word = rng.choice(word_list)
                str_a = base_word
                str_b = apply_random_edits(base_word, rng.randint(3, 5), rng, min_length=2)
        else:
            # Random strings with controlled edits
            length = rng.randint(min_len, max_len)
            str_a = generate_random_string(length, rng)
            str_b = apply_random_edits(str_a, rng.randint(3, 5), rng, min_length=2)
    
    else:  # hard
        # Constraints: 10-15 chars, distance 5-8
        target_min_dist, target_max_dist = 5, 8
        min_len, max_len = 10, 15
        word_list = WORD_LIST_LONG
        
        if use_words:
            # Try to find word pair satisfying constraints
            word_pair = find_word_pair_with_distance(word_list, target_min_dist, target_max_dist, rng)
            if word_pair:
                str_a, str_b = word_pair
            else:
                # Fallback: generate by modifying a word
                base_word = rng.choice(word_list)
                str_a = base_word
                str_b = apply_random_edits(base_word, rng.randint(5, 8), rng, min_length=3)
        else:
            # Random strings - generate two independent strings for larger distance
            length = rng.randint(min_len, max_len)
            str_a = generate_random_string(length, rng)
            # For hard difficulty random strings, use completely different string
            str_b = generate_random_string(length, rng)
    
    # Calculate actual edit distance
    distance = calculate_levenshtein_distance(str_a, str_b)
    
    return str_a, str_b, distance


def render_edit_distance_frame(string_a: str, string_b: str, 
                               answer: Optional[int], 
                               output_path: Path) -> None:
    """
    Render a frame showing two strings and an answer box.
    
    Layout:
    - String_A at the top
    - String_B at the bottom
    - Answer box on the right (empty if answer is None, filled otherwise)
    
    Args:
        string_a: First string (shown at top)
        string_b: Second string (shown at bottom)
        answer: Edit distance to display (None for empty box)
        output_path: Path to save the image
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    
    # Set up coordinate system
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw String_A at top (y=7.5)
    ax.text(1, 7.5, "String_A:", fontsize=14, fontweight='bold', 
            ha='left', va='center', family='monospace')
    ax.text(5, 7.5, string_a, fontsize=18, fontweight='bold',
            ha='center', va='center', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                     edgecolor='black', linewidth=2))
    
    # Draw String_B at bottom (y=2.5)
    ax.text(1, 2.5, "String_B:", fontsize=14, fontweight='bold',
            ha='left', va='center', family='monospace')
    ax.text(5, 2.5, string_b, fontsize=18, fontweight='bold',
            ha='center', va='center', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen',
                     edgecolor='black', linewidth=2))
    
    # Draw answer box on the right (x=8.5, y=5)
    box_size = 1.5
    answer_box = patches.FancyBboxPatch(
        (8.5 - box_size/2, 5 - box_size/2), box_size, box_size,
        boxstyle="round,pad=0.1",
        facecolor='white' if answer is None else 'yellow',
        edgecolor='black',
        linewidth=3
    )
    ax.add_patch(answer_box)
    
    # Add "Answer:" label above the box
    ax.text(8.5, 6.5, "Answer:", fontsize=12, fontweight='bold',
            ha='center', va='center', family='monospace')
    
    # If answer is provided, display it in the box
    if answer is not None:
        ax.text(8.5, 5, str(answer), fontsize=32, fontweight='bold',
                ha='center', va='center', family='monospace', color='red')
    
    # Save figure
    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close(fig)


def create_dataset(num_samples: int = 15) -> Dict[str, Any]:
    """
    Create edit distance dataset - main entry point.
    
    Generates a dataset of string pairs where models must calculate the
    Levenshtein distance and display it in an answer box.
    
    Args:
        num_samples: Number of tasks to generate (default: 15)
        
    Returns:
        Dataset dictionary in standard VMEvalKit format
    """
    print(f"🎯 Creating Edit Distance Dataset")
    print(f"   Total samples: {num_samples}")
    
    start_time = datetime.now()
    rng = random.Random(42)  # Fixed seed for reproducibility
    
    pairs = []
    
    # Distribute samples across difficulty levels
    # Progressive difficulty: easier tasks first
    difficulty_distribution = []
    samples_per_difficulty = num_samples // 3
    remainder = num_samples % 3
    
    for diff_idx, difficulty in enumerate(["easy", "medium", "hard"]):
        count = samples_per_difficulty + (1 if diff_idx < remainder else 0)
        difficulty_distribution.extend([difficulty] * count)
    
    print(f"\n📊 Difficulty distribution:")
    print(f"   Easy: {difficulty_distribution.count('easy')}")
    print(f"   Medium: {difficulty_distribution.count('medium')}")
    print(f"   Hard: {difficulty_distribution.count('hard')}")
    
    # Generate tasks
    for i in range(num_samples):
        task_id = f"edit_distance_{i:04d}"
        difficulty = difficulty_distribution[i]
        
        # Alternate between word pairs and random strings
        use_words = (i % 2 == 0)
        
        try:
            # Generate string pair
            str_a, str_b, distance = generate_string_pair(difficulty, rng, use_words)
            
            # Create temporary directory for images
            temp_dir = tempfile.mkdtemp()
            
            # Render first frame (empty answer box)
            first_path = Path(temp_dir) / f"{task_id}_first.png"
            render_edit_distance_frame(str_a, str_b, None, first_path)
            
            # Render final frame (with answer)
            final_path = Path(temp_dir) / f"{task_id}_final.png"
            render_edit_distance_frame(str_a, str_b, distance, final_path)
            
            # Get prompt
            prompt = PROMPTS[DEFAULT_PROMPT_INDEX]
            
            # Create task pair dictionary
            pair = {
                "id": task_id,
                "prompt": prompt,
                "first_image_path": str(first_path),
                "final_image_path": str(final_path),
                "domain": "edit_distance",
                "task_category": "EditDistance",
                "difficulty": difficulty,
                "edit_distance_data": {
                    "string_a": str_a,
                    "string_b": str_b,
                    "distance": distance,
                    "is_word_pair": use_words,
                },
                "created_at": datetime.now().isoformat()
            }
            pairs.append(pair)
            
            if (i + 1) % 5 == 0:
                print(f"  Generated {i + 1}/{num_samples} tasks...")
                
        except Exception as e:
            print(f"❌ Error generating task {task_id}: {e}")
            continue
    
    # Create dataset dictionary
    dataset = {
        "name": "edit_distance_tasks",
        "description": f"Edit distance calculation tasks for video model evaluation ({len(pairs)} pairs)",
        "pairs": pairs,
        "metadata": {
            "total_tasks": len(pairs),
            "difficulty_levels": ["easy", "medium", "hard"],
            "generation_date": datetime.now().isoformat(),
            "random_seed": 42
        },
        "created_at": datetime.now().isoformat()
    }
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ Dataset creation complete!")
    print(f"   Total tasks: {len(pairs)}")
    print(f"   Time elapsed: {elapsed:.1f}s")
    
    return dataset

