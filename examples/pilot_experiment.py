#!/usr/bin/env python3
"""
Pilot Experiment: Test representative models from each family on VMEvalKit v1.0 dataset.

This script runs inference on 10 task pairs from each task category (chess, maze, raven, rotation)
using one representative model from each commercial API model family:
- Luma Dream Machine: luma-ray-2
- Google Veo: veo-3.0-generate
- WaveSpeed WAN 2.2: wavespeed-wan-2.2-i2v-720p
- Runway ML: runway-gen4-turbo
- OpenAI Sora: openai-sora-2

Total: 40 tasks × 5 models = 200 video generations

Requirements:
- All necessary API keys configured in environment
- venv activated
- Output directory: ./data/outputs/pilot_experiment/
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vmevalkit.runner.inference import (
    run_inference, 
    AVAILABLE_MODELS, 
    MODEL_FAMILIES,
    InferenceRunner
)


# ========================================
# PILOT EXPERIMENT CONFIGURATION
# ========================================

# Limit pilot run to only the OpenAI Sora model
PILOT_MODELS = {
    "openai-sora-2": "OpenAI Sora",
}

# Number of tasks per category
TASKS_PER_CATEGORY = 10

# Task categories to test
TASK_CATEGORIES = ["chess", "maze", "raven", "rotation"]

# Dataset path
DATASET_PATH = Path("data/questions/vmeval_dataset_v1.json")

# Output directory
OUTPUT_DIR = Path("data/outputs/pilot_experiment")

# Fallback dataset for maze tasks (since v1 main dataset may not include maze pairs)
MAZE_FALLBACK_PATH = Path("data/questions/maze_tasks/combined_maze_tasks.json")


# ========================================
# DATASET LOADING
# ========================================

def load_dataset(dataset_path: Path) -> Dict[str, Any]:
    """Load the VMEvalKit dataset."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def select_tasks_by_category(
    dataset: Dict[str, Any], 
    category: str, 
    n_tasks: int = 10,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Select n_tasks from a specific category.
    
    Args:
        dataset: Full dataset dictionary
        category: Task category (chess, maze, raven, rotation)
        n_tasks: Number of tasks to select
        random_seed: Random seed for reproducibility
        
    Returns:
        List of selected task pairs
    """
    import random
    random.seed(random_seed)
    
    # Filter tasks by category
    pairs = dataset.get("pairs", [])
    
    # Category detection based on id prefix
    category_pairs = [
        pair for pair in pairs 
        if pair["id"].startswith(category)
    ]

    # Special-case fallback for maze: use combined maze tasks file if none in main dataset
    if category == "maze" and len(category_pairs) == 0:
        try:
            if MAZE_FALLBACK_PATH.exists():
                with open(MAZE_FALLBACK_PATH, 'r') as f:
                    maze_data = json.load(f)
                    # Expect same schema: id, prompt, first_image_path, final_image_path
                    fallback_pairs = maze_data.get("pairs", [])
                    # Filter pairs whose input image exists to avoid run-time errors
                    category_pairs = [
                        p for p in fallback_pairs
                        if p.get("first_image_path") and Path(p["first_image_path"]).exists()
                    ]
                    print(f"Using fallback maze dataset: found {len(category_pairs)} candidates")
            else:
                print(f"Maze fallback dataset not found at {MAZE_FALLBACK_PATH}")
        except Exception as e:
            print(f"Failed to load maze fallback dataset: {e}")
            category_pairs = []

    print(f"Found {len(category_pairs)} tasks in category '{category}'")
    
    # Select n_tasks randomly
    if len(category_pairs) >= n_tasks:
        selected = random.sample(category_pairs, n_tasks)
    else:
        print(f"⚠️  Warning: Only {len(category_pairs)} tasks available, using all")
        selected = category_pairs
    
    return selected


def get_pilot_tasks(dataset_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all tasks for pilot experiment.
    
    Returns:
        Dictionary mapping category to list of tasks
    """
    dataset = load_dataset(dataset_path)
    
    pilot_tasks = {}
    for category in TASK_CATEGORIES:
        tasks = select_tasks_by_category(dataset, category, TASKS_PER_CATEGORY)
        pilot_tasks[category] = tasks
        print(f"✅ Selected {len(tasks)} tasks from '{category}' category")
    
    return pilot_tasks


# ========================================
# INFERENCE EXECUTION
# ========================================

def create_output_structure(base_dir: Path) -> None:
    """Create organized output directory structure."""
    base_dir.mkdir(exist_ok=True, parents=True)
    
    # Create subdirectories for each model
    for model_name in PILOT_MODELS.keys():
        model_dir = base_dir / model_name
        model_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for each category
        for category in TASK_CATEGORIES:
            (model_dir / category).mkdir(exist_ok=True, parents=True)
    
    # Create logs directory (renamed from results)
    (base_dir / "logs").mkdir(exist_ok=True, parents=True)


def _ensure_real_png(image_path: str) -> bool:
    """If file is SVG mislabeled as .png, convert to real PNG in-place using CairoSVG."""
    try:
        # Quick check by trying to open as PNG
        Image.open(image_path).verify()
        return True
    except Exception:
        # Fallback: detect SVG text and convert
        try:
            with open(image_path, 'rb') as f:
                head = f.read(1024)
            # Heuristic: look for '<svg' in the head bytes
            if b"<svg" in head.lower():
                import cairosvg
                with open(image_path, 'rb') as f:
                    svg_bytes = f.read()
                cairosvg.svg2png(bytestring=svg_bytes, write_to=image_path)
                # Validate conversion
                Image.open(image_path).verify()
                print(f"   🔧 Converted SVG→PNG in-place: {image_path}")
                return True
        except Exception as e:
            print(f"   ⚠️  Image fix failed for {image_path}: {e}")
    return False


def run_single_inference(
    model_name: str,
    task: Dict[str, Any],
    category: str,
    output_dir: Path,
    **kwargs
) -> Dict[str, Any]:
    """
    Run inference for a single task-model pair.
    
    Args:
        model_name: Name of the model to use
        task: Task dictionary from dataset
        category: Task category
        output_dir: Base output directory
        **kwargs: Additional model parameters
        
    Returns:
        Result dictionary with metadata
    """
    task_id = task["id"]
    image_path = task["first_image_path"]
    prompt = task["prompt"]
    
    # Create output filename
    model_safe = model_name.replace("/", "_").replace("-", "_")
    output_filename = f"{task_id}_{model_safe}.mp4"
    
    # Model-specific output directory
    model_output_dir = output_dir / model_name / category
    
    print(f"\n  🎬 Generating: {task_id} with {model_name}")
    print(f"     Image: {image_path}")
    print(f"     Prompt: {prompt[:80]}...")
    
    start_time = datetime.now()
    
    try:
        # Check if image exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        # Validate image is a real PNG (auto-fix if it's actually SVG)
        if not _ensure_real_png(image_path):
            raise ValueError(f"Input image invalid or corrupt: {image_path}")
        
        # Run inference
        result = run_inference(
            model_name=model_name,
            image_path=image_path,
            text_prompt=prompt,
            output_dir=str(model_output_dir),
            output_filename=output_filename,
            **kwargs
        )
        
        # Add metadata
        result.update({
            "task_id": task_id,
            "category": category,
            "model_name": model_name,
            "model_family": PILOT_MODELS[model_name],
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "success": True
        })
        
        print(f"     ✅ Success! Video saved to: {result.get('video_path', 'N/A')}")
        
        return result
        
    except Exception as e:
        error_msg = f"Failed: {str(e)}"
        print(f"     ❌ {error_msg}")
        
        return {
            "task_id": task_id,
            "category": category,
            "model_name": model_name,
            "model_family": PILOT_MODELS[model_name],
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "success": False,
            "error": error_msg,
            "traceback": traceback.format_exc()
        }


def run_pilot_experiment(
    tasks_by_category: Dict[str, List[Dict[str, Any]]],
    models: Dict[str, str],
    output_dir: Path,
    skip_existing: bool = True
) -> Dict[str, Any]:
    """
    Run full pilot experiment.
    
    Args:
        tasks_by_category: Dictionary mapping category to task lists
        models: Dictionary of model names to test
        output_dir: Base output directory
        skip_existing: Skip tasks that already have outputs
        
    Returns:
        Dictionary with all results and statistics
    """
    print("=" * 80)
    print("🚀 VMEVAL KIT PILOT EXPERIMENT")
    print("=" * 80)
    print(f"\n📊 Experiment Configuration:")
    print(f"   Models: {len(models)}")
    print(f"   Categories: {len(tasks_by_category)}")
    print(f"   Tasks per category: {TASKS_PER_CATEGORY}")
    
    total_tasks = sum(len(tasks) for tasks in tasks_by_category.values())
    total_generations = total_tasks * len(models)
    print(f"   Total tasks: {total_tasks}")
    print(f"   Total generations: {total_generations}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   Skip existing: {skip_existing}\n")
    
    # Create output structure
    create_output_structure(output_dir)
    
    # Store all results
    all_results = []
    statistics = {
        "total_tasks": total_tasks,
        "total_generations": total_generations,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "by_model": {},
        "by_category": {}
    }
    
    # Initialize statistics
    for model in models.keys():
        statistics["by_model"][model] = {"completed": 0, "failed": 0, "skipped": 0}
    for category in tasks_by_category.keys():
        statistics["by_category"][category] = {"completed": 0, "failed": 0, "skipped": 0}
    
    experiment_start = datetime.now()
    
    # Run inference for each model-task combination
    for category, tasks in tasks_by_category.items():
        print(f"\n{'=' * 80}")
        print(f"📂 Category: {category.upper()} ({len(tasks)} tasks)")
        print(f"{'=' * 80}")
        
        for i, task in enumerate(tasks, 1):
            task_id = task["id"]
            print(f"\n[{i}/{len(tasks)}] Task: {task_id}")
            
            for model_name in models.keys():
                # Check if output already exists
                model_safe = model_name.replace("/", "_").replace("-", "_")
                output_filename = f"{task_id}_{model_safe}.mp4"
                model_output_dir = output_dir / model_name / category
                output_path = model_output_dir / output_filename
                
                if skip_existing and output_path.exists():
                    print(f"  ⏭️  Skipping {model_name}: output already exists")
                    statistics["skipped"] += 1
                    statistics["by_model"][model_name]["skipped"] += 1
                    statistics["by_category"][category]["skipped"] += 1
                    continue
                
                # Run inference
                result = run_single_inference(
                    model_name=model_name,
                    task=task,
                    category=category,
                    output_dir=output_dir
                )
                
                all_results.append(result)
                
                # Update statistics
                if result["success"]:
                    statistics["completed"] += 1
                    statistics["by_model"][model_name]["completed"] += 1
                    statistics["by_category"][category]["completed"] += 1
                else:
                    statistics["failed"] += 1
                    statistics["by_model"][model_name]["failed"] += 1
                    statistics["by_category"][category]["failed"] += 1
                
                # Save intermediate results
                save_results(all_results, statistics, output_dir, intermediate=True)
    
    experiment_end = datetime.now()
    duration = (experiment_end - experiment_start).total_seconds()
    
    # Final statistics
    statistics["experiment_start"] = experiment_start.isoformat()
    statistics["experiment_end"] = experiment_end.isoformat()
    statistics["duration_seconds"] = duration
    statistics["duration_formatted"] = format_duration(duration)
    
    return {
        "results": all_results,
        "statistics": statistics
    }


# ========================================
# RESULTS MANAGEMENT
# ========================================

def save_results(
    results: List[Dict[str, Any]],
    statistics: Dict[str, Any],
    output_dir: Path,
    intermediate: bool = False
) -> None:
    """Save results and statistics to JSON files."""
    results_dir = output_dir / "logs"
    
    # Save detailed results
    if intermediate:
        results_file = results_dir / "logs_intermediate.json"
    else:
        results_file = results_dir / "logs_final.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save statistics
    stats_file = results_dir / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    if not intermediate:
        print(f"\n✅ Logs saved to: {results_file}")
        print(f"✅ Statistics saved to: {stats_file}")


def generate_summary_report(
    statistics: Dict[str, Any],
    output_dir: Path
) -> None:
    """Generate human-readable summary report."""
    report_file = output_dir / "logs" / "SUMMARY.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VMEVAL KIT PILOT EXPERIMENT - SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Experiment Date: {statistics['experiment_start']}\n")
        f.write(f"Duration: {statistics['duration_formatted']}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total Generations: {statistics['total_generations']}\n")
        f.write(f"Completed: {statistics['completed']} ({statistics['completed']/statistics['total_generations']*100:.1f}%)\n")
        f.write(f"Failed: {statistics['failed']} ({statistics['failed']/statistics['total_generations']*100:.1f}%)\n")
        f.write(f"Skipped: {statistics['skipped']} ({statistics['skipped']/statistics['total_generations']*100:.1f}%)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("RESULTS BY MODEL FAMILY\n")
        f.write("=" * 80 + "\n")
        for model, stats in statistics['by_model'].items():
            f.write(f"\n{model}:\n")
            f.write(f"  Completed: {stats['completed']}\n")
            f.write(f"  Failed: {stats['failed']}\n")
            f.write(f"  Skipped: {stats['skipped']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("RESULTS BY TASK CATEGORY\n")
        f.write("=" * 80 + "\n")
        for category, stats in statistics['by_category'].items():
            f.write(f"\n{category.upper()}:\n")
            f.write(f"  Completed: {stats['completed']}\n")
            f.write(f"  Failed: {stats['failed']}\n")
            f.write(f"  Skipped: {stats['skipped']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✅ Summary report saved to: {report_file}")


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


# ========================================
# MAIN EXECUTION
# ========================================

def main():
    """Main execution function."""
    print("🔍 Loading VMEvalKit dataset...")
    
    # Check if dataset exists
    if not DATASET_PATH.exists():
        print(f"❌ Dataset not found at: {DATASET_PATH}")
        print("   Please ensure the dataset is available.")
        sys.exit(1)
    
    # Load and select tasks
    tasks_by_category = get_pilot_tasks(DATASET_PATH)
    
    # Verify models are available
    print(f"\n🔍 Verifying models...")
    for model_name in PILOT_MODELS.keys():
        if model_name in AVAILABLE_MODELS:
            print(f"   ✅ {model_name}: {AVAILABLE_MODELS[model_name]['description']}")
        else:
            print(f"   ❌ {model_name}: NOT FOUND")
            sys.exit(1)
    
    # Check for API keys (warn if missing)
    print(f"\n🔑 Checking API keys...")
    api_keys = {
        "LUMA_API_KEY": os.getenv("LUMA_API_KEY"),
        "GOOGLE_PROJECT_ID": os.getenv("GOOGLE_PROJECT_ID") or os.getenv("PROJECT_ID"),
        "WAVESPEED_API_KEY": os.getenv("WAVESPEED_API_KEY"),
        "RUNWAYML_API_SECRET": os.getenv("RUNWAYML_API_SECRET"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    }
    
    for key_name, key_value in api_keys.items():
        if key_value:
            print(f"   ✅ {key_name}: configured")
        else:
            print(f"   ⚠️  {key_name}: not configured (related models may fail)")
    
    print(f"\n{'=' * 80}")
    input("Press ENTER to start the pilot experiment (or Ctrl+C to cancel)...")
    
    # Run experiment
    experiment_results = run_pilot_experiment(
        tasks_by_category=tasks_by_category,
        models=PILOT_MODELS,
        output_dir=OUTPUT_DIR,
        skip_existing=True
    )
    
    # Save final results
    print(f"\n{'=' * 80}")
    print("💾 Saving final results...")
    save_results(
        results=experiment_results["results"],
        statistics=experiment_results["statistics"],
        output_dir=OUTPUT_DIR,
        intermediate=False
    )
    
    # Generate summary report
    generate_summary_report(
        statistics=experiment_results["statistics"],
        output_dir=OUTPUT_DIR
    )
    
    # Print final summary
    print(f"\n{'=' * 80}")
    print("🎉 PILOT EXPERIMENT COMPLETE!")
    print(f"{'=' * 80}")
    stats = experiment_results["statistics"]
    print(f"\n📊 Final Statistics:")
    print(f"   Total: {stats['total_generations']} generations")
    print(f"   Completed: {stats['completed']} ({stats['completed']/stats['total_generations']*100:.1f}%)")
    print(f"   Failed: {stats['failed']} ({stats['failed']/stats['total_generations']*100:.1f}%)")
    print(f"   Skipped: {stats['skipped']} ({stats['skipped']/stats['total_generations']*100:.1f}%)")
    print(f"   Duration: {stats['duration_formatted']}")
    print(f"\n📁 All outputs saved to: {OUTPUT_DIR}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Experiment failed with error:")
        print(f"   {str(e)}")
        traceback.print_exc()
        sys.exit(1)

