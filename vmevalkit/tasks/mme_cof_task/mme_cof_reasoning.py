"""
MME-CoF Chain-of-Frame Reasoning Task

Implementation of the MME-CoF benchmark for evaluating video models as zero-shot reasoners.
This module handles the MME-CoF dataset which evaluates video models' ability to perform
chain-of-frame reasoning across 12 different cognitive domains.

Reference: "Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-COF Benchmark"
Dataset: https://huggingface.co/datasets/ZiyuG/MME-CoF
"""

from typing import Dict, Any, List
from .PROMPTS import get_prompt_for_category, get_category_description


def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Create MME-CoF dataset metadata.
    
    Note: MME-CoF is downloaded from HuggingFace, so this function primarily
    returns metadata about the dataset structure. The actual download happens
    in the download_hf_domain_to_folders function.
    
    Args:
        num_samples: Not used for HF datasets (controlled by dataset size)
    
    Returns:
        Dictionary with dataset metadata
    """
    
    dataset = {
        "name": "mme_cof",
        "description": "MME-CoF: Video Chain-of-Frame Reasoning Benchmark",
        "total_questions": 59,  # As per the HuggingFace dataset
        "categories": 12,
        "source": "https://huggingface.co/datasets/ZiyuG/MME-CoF",
        "reference": "https://github.com/ZiyuGuo99/MME-CoF",
        "pairs": []
    }
    
    return dataset


def process_mme_cof_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Process a single MME-CoF dataset item.
    
    Args:
        item: Raw item from HuggingFace dataset with 'image' and 'label' fields
        idx: Index of the item
    
    Returns:
        Processed task dictionary with generated prompt
    """
    
    # Extract category label
    category = item.get('label', 'unknown')
    
    # Generate appropriate prompt for this category
    prompt = get_prompt_for_category(category)
    
    task = {
        "id": f"mme_cof_{idx:04d}",
        "category": category,
        "category_description": get_category_description(category),
        "prompt": prompt,
        "image": item.get('image'),
        "reasoning_type": "chain_of_frame",
        "evaluation_type": "zero_shot_reasoning"
    }
    
    return task

