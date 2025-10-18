#!/usr/bin/env python3
"""
Quick Test Experiment: Test 6 representative models on 1 task per domain with sequential execution.

This script runs inference on 1 task from each domain (chess, maze, raven, rotation, sudoku)
for rapid testing and validation of all model integrations. Models are processed sequentially
(one model at a time), and for each model, tasks are processed one by one.

Models tested:
- Luma Dream Machine: luma-ray-2
- Google Veo 3.0: veo-3.0-generate  
- Google Veo 3.1 (via WaveSpeed): veo-3.1-720p
- Runway ML: runway-gen4-turbo
- OpenAI Sora: openai-sora-2
- WaveSpeed WAN 2.2: wavespeed-wan-2.2-i2v-720p

Total: 5 tasks × 6 models = 30 quick test generations (sequential)

Human Curation: Only tasks with existing folders are processed (deleted folders = rejected tasks)

Requirements:
- All necessary API keys configured in environment
- venv activated
- Output directory: ./data/outputs/pilot_experiment/
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import traceback
from PIL import Image
import time
import signal
import atexit

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

# Test 6 models including the new Veo 3.1
PILOT_MODELS = {
    "luma-ray-2": "Luma Dream Machine",
    "veo-3.0-generate": "Google Veo 3.0",
    "veo-3.1-720p": "Google Veo 3.1 (via WaveSpeed)",
    "runway-gen4-turbo": "Runway ML",
    "openai-sora-2": "OpenAI Sora",
    "wavespeed-wan-2.2-i2v-720p": "WaveSpeed WAN 2.2",
}

# Questions directory path
QUESTIONS_DIR = Path("data/questions")

# Output directory
OUTPUT_DIR = Path("data/outputs/pilot_experiment")

# Expected domains (for validation)
EXPECTED_DOMAINS = ["chess", "maze", "raven", "rotation", "sudoku"]


# ========================================
# PROGRESS TRACKING REMOVED (deprecated)
# ========================================


# ========================================
# FOLDER-BASED TASK DISCOVERY
# ========================================

def discover_all_tasks_from_folders(questions_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Discover all human-approved tasks by direct file detection.
    
    Scans for actual PNG and TXT files, then loads supplemental metadata from JSON.
    This approach is more robust as it relies on actual files rather than metadata.
    
    Args:
        questions_dir: Path to questions directory
        
    Returns:
        Dictionary mapping domain to list of task dictionaries
    """
    print(f"🔍 Discovering tasks by scanning actual files: {questions_dir}")
    
    tasks_by_domain = {}
    total_tasks = 0
    
    # Scan each domain folder
    for domain_dir in sorted(questions_dir.glob("*_task")):
        if not domain_dir.is_dir():
            continue
            
        domain = domain_dir.name.replace("_task", "")
        domain_tasks = []
        
        print(f"   📁 Scanning {domain_dir.name}/")
        
        # Scan each question folder in this domain  
        for question_dir in sorted(domain_dir.glob(f"{domain}_*")):
            if not question_dir.is_dir():
                continue
                
            task_id = question_dir.name
            
            # Look for actual files (PNG and TXT) - this is our primary source
            prompt_file = question_dir / "prompt.txt"
            first_image = question_dir / "first_frame.png"
            final_image = question_dir / "final_frame.png"
            
            # Required files check
            if not prompt_file.exists():
                print(f"      ⚠️  Skipping {task_id}: Missing prompt.txt")
                continue
                
            if not first_image.exists():
                print(f"      ⚠️  Skipping {task_id}: Missing first_frame.png")
                continue
            
            # Load prompt directly from actual file
            try:
                prompt_text = prompt_file.read_text().strip()
            except Exception as e:
                print(f"      ⚠️  Skipping {task_id}: Cannot read prompt.txt - {e}")
                continue
            
            # Create task dictionary from actual detected files
            task = {
                "id": task_id,
                "domain": domain,
                "prompt": prompt_text,
                "first_image_path": str(first_image.absolute()),
                "final_image_path": str(final_image.absolute()) if final_image.exists() else None
            }
            
            # Load supplemental metadata from JSON files if they exist
            metadata_file = question_dir / "question_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        supplemental_metadata = json.load(f)
                    
                    # Add supplemental data but don't override core fields
                    for key, value in supplemental_metadata.items():
                        if key not in task:  # Don't override our detected values
                            task[key] = value
                            
                except Exception as e:
                    print(f"      ⚠️  Warning: Could not load metadata for {task_id} - {e}")
            
            domain_tasks.append(task)
            
        print(f"      ✅ Found {len(domain_tasks)} approved tasks in {domain}")
        tasks_by_domain[domain] = domain_tasks
        total_tasks += len(domain_tasks)
    
    print(f"\n📊 Discovery Summary:")
    print(f"   Total approved tasks: {total_tasks}")
    for domain, tasks in tasks_by_domain.items():
        print(f"   {domain.title()}: {len(tasks)} tasks")
    
    return tasks_by_domain


# ========================================
# INFERENCE EXECUTION
# ========================================

def create_output_structure(base_dir: Path) -> None:
    """Create organized output directory structure for the new system."""
    base_dir.mkdir(exist_ok=True, parents=True)
    
    # With the new structured output, each inference creates its own folder
    # Logs are disabled; no top-level logs directory
    
    print(f"📁 Output directory structure ready at: {base_dir}")
    print(f"   Each inference will create a self-contained folder with:")
    print(f"   - video/: Generated video file")
    print(f"   - question/: Input images and prompt")
    print(f"   - metadata.json: Complete inference metadata")


def create_model_directories(base_dir: Path, models: Dict[str, str]) -> None:
    """Create a subfolder per model and mirror questions tree for visibility."""
    # Create per-model root
    for model_name in models.keys():
        model_root = base_dir / model_name
        model_root.mkdir(exist_ok=True, parents=True)
        # Mirror questions directory structure with empty domain/task folders
        for domain_dir in sorted(QUESTIONS_DIR.glob("*_task")):
            if not domain_dir.is_dir():
                continue
            domain_name = domain_dir.name  # e.g., rotation_task
            # Create domain folder under model
            model_domain_dir = model_root / domain_name
            model_domain_dir.mkdir(exist_ok=True, parents=True)
            # Create each task folder (without runs yet)
            for task_dir in sorted(domain_dir.glob("*") ):
                if task_dir.is_dir():
                    (model_domain_dir / task_dir.name).mkdir(exist_ok=True, parents=True)
    print(f"📁 Created per-model folders under: {base_dir} and mirrored question tasks")




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
    runner: Optional[InferenceRunner] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run inference for a single task-model pair using the new structured output system.
    
    Args:
        model_name: Name of the model to use
        task: Task dictionary from dataset
        category: Task category
        output_dir: Base output directory
        runner: Optional InferenceRunner instance (created if not provided)
        **kwargs: Additional model parameters
        
    Returns:
        Result dictionary with metadata
    """
    task_id = task["id"]
    image_path = task["first_image_path"]
    prompt = task["prompt"]
    
    # Create a unique run_id for this inference
    run_id = f"{model_name}_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
        
        # Use InferenceRunner for structured output
        if runner is None:
            runner = InferenceRunner(output_dir=str(output_dir))
        
        # Run inference with complete question data for structured output
        result = runner.run(
            model_name=model_name,
            image_path=image_path,
            text_prompt=prompt,
            run_id=run_id,
            question_data=task,  # Pass full task data for structured output
            **kwargs  # Clean! No API key filtering needed
        )
        
        # Add metadata
        result.update({
            "task_id": task_id,
            "category": category,
            "model_name": model_name,
            "model_family": PILOT_MODELS[model_name],
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "success": result.get("status") != "failed"
        })
        
        if result.get("status") != "failed":
            print(f"     ✅ Success! Structured output saved to: {result.get('inference_dir', 'N/A')}")
        else:
            print(f"     ❌ Failed: {result.get('error', 'Unknown error')}")
        
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
    tasks_by_domain: Dict[str, List[Dict[str, Any]]],
    models: Dict[str, str],
    output_dir: Path,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Run full pilot experiment with SEQUENTIAL execution on ALL human-approved tasks.
    
    Processes one model at a time, and for each model, one task at a time.
    
    Args:
        tasks_by_domain: Dictionary mapping domain to task lists
        models: Dictionary of model names to test
        output_dir: Base output directory
        skip_existing: Skip tasks that already have outputs
        
    Returns:
        Dictionary with all results and statistics
    """
    print("=" * 80)
    print("🚀 VMEVAL KIT EXPERIMENT - CLEAN EXECUTION")
    print("=" * 80)
    print(f"\n📊 Experiment Configuration:")
    print(f"   Models to run: {len(models)} - {', '.join(models.keys())}")
    print(f"   Domains: {len(tasks_by_domain)}")
    print(f"   🔄 Execution Mode: SEQUENTIAL")
    
    # Calculate totals
    total_tasks = sum(len(tasks) for tasks in tasks_by_domain.values())
    total_generations = total_tasks * len(models)
    
    print(f"\n📈 Task Distribution:")
    for domain, tasks in tasks_by_domain.items():
        print(f"   {domain.title()}: {len(tasks)} approved tasks")
    print(f"   Total approved tasks: {total_tasks}")
    print(f"   Total generations: {total_generations}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   Skip existing: {skip_existing}\n")
    
    # Create output structure
    create_output_structure(output_dir)
    # Create per-model directories
    create_model_directories(output_dir, models)
    
    # Results storage (no longer needs thread safety since we're sequential)
    all_results = []
    
    statistics = {
        "total_tasks": total_tasks,
        "total_generations": total_generations,
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "by_model": {},
        "by_domain": {}
    }
    
    # Initialize statistics
    for model in models.keys():
        statistics["by_model"][model] = {"completed": 0, "failed": 0, "skipped": 0}
    for domain in tasks_by_domain.keys():
        statistics["by_domain"][domain] = {"completed": 0, "failed": 0, "skipped": 0}
    
    experiment_start = datetime.now()
    
    # Count total jobs for progress tracking
    total_jobs = sum(len(tasks) for tasks in tasks_by_domain.values()) * len(models)
    print(f"📋 Total inference jobs to run: {total_jobs}")
    print("🚀 Starting sequential execution...\n")
    print("   Processing order: Model by model, task by task\n")
    
    # Track overall progress
    job_counter = 0
    
    # Sequential execution: model by model, task by task
    for model_name, model_display in models.items():
        print(f"\n{'='*60}")
        print(f"🤖 Processing Model: {model_display} ({model_name})")
        print(f"{'='*60}")
        # Use a model-specific output directory and runner (runner will further mirror domain/task)
        model_output_dir = output_dir / model_name
        runner = InferenceRunner(output_dir=str(model_output_dir))
        
        model_start_time = datetime.now()
        model_completed = 0
        model_failed = 0
        model_skipped = 0
        
        # Process all tasks for this model
        for domain, tasks in tasks_by_domain.items():
            print(f"\n  📚 Domain: {domain.title()}")
            
            for task in tasks:
                job_counter += 1
                task_id = task["id"]
                job_id = f"{model_name}_{task_id}"
                
                print(f"    [{job_counter}/{total_jobs}] Processing: {task_id}")
                
                # Check if inference folder already exists
                # Check inside mirrored domain/task folder for existing runs
                run_id_pattern = f"{model_name}_{task_id}_*"
                domain_dir_name = f"{domain}_task"
                task_folder = model_output_dir / domain_dir_name / task_id
                existing_dirs = list(task_folder.glob(run_id_pattern))
                
                if skip_existing and existing_dirs:
                    statistics["skipped"] += 1
                    statistics["by_model"][model_name]["skipped"] += 1
                    statistics["by_domain"][domain]["skipped"] += 1
                    model_skipped += 1
                    print(f"      ⏭️  Skipped (existing output: {existing_dirs[0].name})")
                    continue
                
                # Run inference with structured output
                result = run_single_inference(
                    model_name=model_name,
                    task=task,
                    category=domain,
                    output_dir=model_output_dir,  # Use model-specific output dir
                    runner=runner
                )
                
                # Update statistics and results
                all_results.append(result)
                
                if result["success"]:
                    statistics["completed"] += 1
                    statistics["by_model"][model_name]["completed"] += 1
                    statistics["by_domain"][domain]["completed"] += 1
                    model_completed += 1
                    print(f"      ✅ Completed successfully")
                else:
                    statistics["failed"] += 1
                    statistics["by_model"][model_name]["failed"] += 1
                    statistics["by_domain"][domain]["failed"] += 1
                    model_failed += 1
                    print(f"      ❌ Failed: {result.get('error', 'Unknown error')}")
                
                # Intermediate results saving removed
        
        # Model summary
        model_duration = (datetime.now() - model_start_time).total_seconds()
        print(f"\n  📊 Model {model_display} Summary:")
        print(f"     Completed: {model_completed}")
        print(f"     Failed: {model_failed}")
        print(f"     Skipped: {model_skipped}")
        print(f"     Duration: {format_duration(model_duration)}")
    
    experiment_end = datetime.now()
    duration = (experiment_end - experiment_start).total_seconds()
    
    # Final statistics
    statistics["experiment_start"] = experiment_start.isoformat()
    statistics["experiment_end"] = experiment_end.isoformat()
    statistics["duration_seconds"] = duration
    statistics["duration_formatted"] = format_duration(duration)
    
    # Final save to progress tracker removed
    
    print(f"\n⏱️  Sequential execution completed in {format_duration(duration)}")
    
    return {
        "results": all_results,
        "statistics": statistics,
    }


# ========================================
# RESULTS MANAGEMENT - JSON GENERATION REMOVED
# ========================================


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
    """Main execution function with resume support."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="VMEvalKit Quick Test Experiment")
    parser.add_argument("--all-tasks", action="store_true", help="Run all tasks, not just 1 per domain")
    parser.add_argument(
        "--only-model",
        nargs="*",
        default=None,
        help="Optional list of model names to run/resume (others will be skipped)"
    )
    args = parser.parse_args()
    
    print("🔍 Discovering human-approved tasks from folder structure...")
    
    # Check if questions directory exists
    if not QUESTIONS_DIR.exists():
        print(f"❌ Questions directory not found at: {QUESTIONS_DIR}")
        print("   Please ensure the questions directory exists with task folders.")
        sys.exit(1)
    
    # Discover all approved tasks from folders
    all_tasks_by_domain = discover_all_tasks_from_folders(QUESTIONS_DIR)
    
    # Choose task set based on args
    tasks_by_domain = {}
    if args.all_tasks:
        tasks_by_domain = all_tasks_by_domain
        print(f"   🎯 Running ALL approved tasks")
    else:
        # Limit to 1 task per domain for quick testing
        for domain, tasks in all_tasks_by_domain.items():
            if tasks:
                tasks_by_domain[domain] = [tasks[0]]  # Take only first task
                print(f"   🎯 Testing with 1 task from {domain}: {tasks[0]['id']}")
            else:
                tasks_by_domain[domain] = []
    
    # Verify models are available
    # Optionally restrict to a subset of models
    selected_models = PILOT_MODELS
    if args.only_model:
        selected_models = {
            k: v for k, v in PILOT_MODELS.items() if k in set(args.only_model)
        }
        if not selected_models:
            print("❌ No valid models selected with --only-model")
            sys.exit(1)
        skipped_models = set(PILOT_MODELS.keys()) - set(selected_models.keys())
        print(f"\n🎯 Running selected models: {', '.join(selected_models.keys())}")
        if skipped_models:
            print(f"⏭️  Skipping models: {', '.join(sorted(skipped_models))}")

    print(f"\n🔍 Verifying {len(selected_models)} model(s) for testing...")
    for model_name, family in selected_models.items():
        if model_name in AVAILABLE_MODELS:
            print(f"   ✅ {model_name}: {family}")
        else:
            print(f"   ❌ {model_name}: NOT FOUND in available models")
            print(f"      Please check model name or add it to AVAILABLE_MODELS")
            # Don't exit, just warn - some models might not be configured yet
    
    print(f"\n{'=' * 80}")
    # Removed interactive prompt for non-interactive execution
    
    # Verify we found tasks
    if not tasks_by_domain or sum(len(tasks) for tasks in tasks_by_domain.values()) == 0:
        print("❌ No approved tasks found. Please check the questions directory structure.")
        sys.exit(1)
    
    # Run experiment
    experiment_results = run_pilot_experiment(
        tasks_by_domain=tasks_by_domain,
        models=selected_models,
        output_dir=OUTPUT_DIR,
        skip_existing=True,
    )
    
    # Results saving removed - outputs are in video directories
    
    # Print final summary
    print(f"\n{'=' * 80}")
    print("🎉 QUICK TEST COMPLETE!")
    print(f"{'=' * 80}")
    stats = experiment_results["statistics"]
    
    # Calculate actual totals based on what was run
    actual_total_attempted = stats['completed'] + stats['failed'] + stats['skipped']
    
    print(f"\n📊 Final Statistics:")
    print(f"   Models tested: {len(selected_models)}")  # Use selected_models, not PILOT_MODELS
    print(f"   Approved tasks per model: {stats['total_tasks']}")
    print(f"   Total possible generations: {stats['total_generations']}")
    print(f"   Total attempted: {actual_total_attempted}")
    print(f"   Completed: {stats['completed']} ({stats['completed']/max(actual_total_attempted,1)*100:.1f}%)")
    print(f"   Failed: {stats['failed']} ({stats['failed']/max(actual_total_attempted,1)*100:.1f}%)")
    print(f"   Skipped: {stats['skipped']} ({stats['skipped']/max(actual_total_attempted,1)*100:.1f}%)")
    print(f"   ⏱️ Duration: {stats['duration_formatted']}")
    
    print(f"\n🎯 Results by Domain:")
    for domain, domain_stats in stats['by_domain'].items():
        domain_total = domain_stats['completed'] + domain_stats['failed'] + domain_stats['skipped']
        if domain_total > 0:
            c, f, s = domain_stats['completed'], domain_stats['failed'], domain_stats['skipped']
            print(f"   {domain.title()}: ✅ {c} completed | ❌ {f} failed | ⏭️  {s} skipped")
    
    print(f"\n🤖 Results by Model:")
    for model_name, model_stats in stats['by_model'].items():
        model_total = model_stats['completed'] + model_stats['failed'] + model_stats['skipped']
        if model_total > 0:  # Only show models that were actually run
            c, f, s = model_stats['completed'], model_stats['failed'], model_stats['skipped']
            print(f"   {model_name}: ✅ {c} | ❌ {f} | ⏭️  {s}")
    
    print(f"\n📁 All outputs saved to: {OUTPUT_DIR}")
    
    # Resume functionality removed
    
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
