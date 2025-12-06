#!/usr/bin/env python3
"""
VMEvalKit Video Generation - Flexible Model and Task Runner

This script provides flexible video generation with customizable model and task selection.
Run on specific models, domains, or individual tasks with full control over the process.

Key Features:
- Select specific domains or individual task IDs (all domains from TASK_CATALOG are supported)
- Control number of tasks per domain or run all available tasks
- Sequential execution with progress tracking and resume capability

Human Curation: Only tasks with existing folders are processed (deleted folders = rejected tasks)

Requirements:
- Relevant API keys configured in environment for selected models
- Questions available in: ./data/questions/
- Output directory: ./data/outputs/pilot_experiment/

Use --help for detailed usage examples and options.
"""

import sys
import json
import shutil
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vmevalkit.runner.inference import  InferenceRunner
from vmevalkit.runner.MODEL_CATALOG import AVAILABLE_MODELS, get_model_family
from vmevalkit.runner.TASK_CATALOG import TASK_REGISTRY


# Default models for quick testing (can be overridden with --model)
DEFAULT_TEST_MODELS = [
    "luma-ray-2",
    "veo-3.0-generate", 
    "veo-3.1-720p",
    "runway-gen4-turbo",
    "openai-sora-2",
    "wavespeed-wan-2.2-i2v-720p",
]

QUESTIONS_DIR = Path("data/questions")
OUTPUT_DIR = Path("data/outputs/pilot_experiment")

# Expected domains (dynamically loaded from TASK_CATALOG)
EXPECTED_DOMAINS = sorted(list(TASK_REGISTRY.keys()))

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
            prompt_text = prompt_file.read_text().strip()
            
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
                with open(metadata_file, 'r') as f:
                    supplemental_metadata = json.load(f)
                
                # Add supplemental data but don't override core fields
                for key, value in supplemental_metadata.items():
                    if key not in task:  # Don't override our detected values
                        task[key] = value
            
            domain_tasks.append(task)
            
        print(f"      ✅ Found {len(domain_tasks)} approved tasks in {domain}")
        tasks_by_domain[domain] = domain_tasks
        total_tasks += len(domain_tasks)
    
    print(f"\n📊 Discovery Summary:")
    print(f"   Total approved tasks: {total_tasks}")
    for domain, tasks in tasks_by_domain.items():
        print(f"   {domain.title()}: {len(tasks)} tasks")
    
    return tasks_by_domain


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

    run_id = f"{model_name}_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\n  🎬 Generating: {task_id} with {model_name}")
    print(f"     Image: {image_path}")
    print(f"     Prompt: {prompt[:80]}...")
    
    start_time = datetime.now()

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Validate image is a real PNG (auto-fix if it's actually SVG)
    if not _ensure_real_png(image_path):
        raise ValueError(f"Input image invalid or corrupt: {image_path}")
    

    if runner is None:
        runner = InferenceRunner(output_dir=str(output_dir))
    

    result = runner.run(
        model_name=model_name,
        image_path=image_path,
        text_prompt=prompt,
        run_id=run_id,
        question_data=task,  # Pass full task data for structured output
        **kwargs  # Clean! No API key filtering needed
    )
    
    result.update({
        "task_id": task_id,
        "category": category,
        "model_name": model_name,
        "model_family": get_model_family(model_name),
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "success": result.get("status") != "failed"
    })
    
    if result.get("status") != "failed":
        print(f"     ✅ Success! Structured output saved to: {result.get('inference_dir', 'N/A')}")
    else:
        print(f"     ❌ Failed: {result.get('error', 'Unknown error')}")
    
    return result


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
    print("🚀 VMEVAL KIT EXPERIMENT - CLEAN EXECUTION")
    print(f"\n📊 Experiment Configuration:")
    print(f"   Models to run: {len(models)} - {', '.join(models.keys())}")
    print(f"   Domains: {len(tasks_by_domain)}")
    print(f"   🔄 Execution Mode: SEQUENTIAL")
    

    total_tasks = sum(len(tasks) for tasks in tasks_by_domain.values())
    total_generations = total_tasks * len(models)
    
    print(f"\n📈 Task Distribution:")
    for domain, tasks in tasks_by_domain.items():
        print(f"   {domain.title()}: {len(tasks)} approved tasks")
    print(f"   Total approved tasks: {total_tasks}")
    print(f"   Total generations: {total_generations}")
    print(f"\n📁 Output directory: {output_dir}")
    print(f"   Skip existing: {skip_existing}\n")
    
    create_output_structure(output_dir)
    create_model_directories(output_dir, models)
    
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
    
    for model in models.keys():
        statistics["by_model"][model] = {"completed": 0, "failed": 0, "skipped": 0}
    for domain in tasks_by_domain.keys():
        statistics["by_domain"][domain] = {"completed": 0, "failed": 0, "skipped": 0}
    
    experiment_start = datetime.now()
    
    total_jobs = sum(len(tasks) for tasks in tasks_by_domain.values()) * len(models)
    print(f"📋 Total inference jobs to run: {total_jobs}")
    print("🚀 Starting sequential execution...\n")
    print("   Processing order: Model by model, task by task\n")
    
    job_counter = 0
    
    for model_name, model_display in models.items():
        print(f"🤖 Processing Model: {model_display} ({model_name})")
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
                
                # Check if inference folder already exists WITH actual video file
                # Check inside mirrored domain/task folder for existing runs
                run_id_pattern = f"{model_name}_{task_id}_*"
                domain_dir_name = f"{domain}_task"
                task_folder = model_output_dir / domain_dir_name / task_id
                existing_dirs = list(task_folder.glob(run_id_pattern))
                
                # Verify the run folder actually contains a video file
                has_valid_output = False
                if existing_dirs:
                    for run_dir in existing_dirs:
                        video_dir = run_dir / "video"
                        if video_dir.exists():
                            video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.webm"))
                            if video_files:
                                has_valid_output = True
                                break
                
                if skip_existing and has_valid_output:
                    statistics["skipped"] += 1
                    statistics["by_model"][model_name]["skipped"] += 1
                    statistics["by_domain"][domain]["skipped"] += 1
                    model_skipped += 1
                    print(f"      ⏭️  Skipped (existing output: {existing_dirs[0].name})")
                    continue
                
                result = run_single_inference(
                    model_name=model_name,
                    task=task,
                    category=domain,
                    output_dir=model_output_dir,  # Use model-specific output dir
                    runner=runner
                )
                
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
                
        
        model_duration = (datetime.now() - model_start_time).total_seconds()
        print(f"\n  📊 Model {model_display} Summary: {model_completed} completed, {model_failed} failed, {model_skipped} skipped in {format_duration(model_duration)}")
    
    experiment_end = datetime.now()
    duration = (experiment_end - experiment_start).total_seconds()
    
    statistics["experiment_start"] = experiment_start.isoformat()
    statistics["experiment_end"] = experiment_end.isoformat()
    statistics["duration_seconds"] = duration
    statistics["duration_formatted"] = format_duration(duration)
    
    print(f"\n⏱️  Sequential execution completed in {format_duration(duration)}")
    
    return {
        "results": all_results,
        "statistics": statistics,
    }


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def main():
    """Main execution function with resume support."""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="VMEvalKit Video Generation - Flexible model and task selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run 1 task per domain on default models
            python generate_videos.py
            
            # Run all tasks on default models  
            python generate_videos.py --all-tasks
            
            # Run on specific models
            python generate_videos.py --model luma-ray-2 openai-sora-2
            
            # Run specific tasks/domains
            python generate_videos.py --task chess maze --pairs-per-domain 3
            
            # Run specific task IDs
            python generate_videos.py --task-id chess_0001 maze_0005 --model luma-ray-2
        """
    )
    
    parser.add_argument(
        "--model", 
        nargs="+", 
        default=None,
        help=f"Specific model(s) to run. Available: {', '.join(list(AVAILABLE_MODELS.keys())[:10])}... (see --list-models for all)"
    )
    
    parser.add_argument(
        "--task",
        nargs="+",
        choices=sorted(list(TASK_REGISTRY.keys())),
        default=None,
        help=f"Specific task domain(s) to run. Available: {', '.join(sorted(list(TASK_REGISTRY.keys())))}. If not specified, runs all domains."
    )
    
    parser.add_argument(
        "--task-id",
        nargs="+", 
        default=None,
        help="Specific task ID(s) to run (e.g., chess_0001 maze_0005). Overrides other task selection."
    )
    
    parser.add_argument(
        "--pairs-per-domain", 
        type=int, 
        default=1, 
        help="Number of task pairs to run per domain (default: 1)"
    )
    
    parser.add_argument("--all-tasks", action="store_true", help="Run ALL available tasks (overrides --pairs-per-domain)")
    
    parser.add_argument("--list-models", action="store_true", help="List all available models and exit")
    
    parser.add_argument(
        "--override",
        dest="override",
        action="store_true",
        help="Delete data/outputs/pilot_experiment directory before running (override existing outputs)"
    )
    
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID to use (e.g., --gpu 1 for GPU 1). Sets CUDA_VISIBLE_DEVICES environment variable."
    )
    
    # Legacy parameter for compatibility
    parser.add_argument(
        "--only-model",
        nargs="*",
        default=None,
        help="(Legacy) Same as --model"
    )
    
    args = parser.parse_args()
    
    # Handle --gpu: Set CUDA_VISIBLE_DEVICES environment variable
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"🎮 Using GPU {args.gpu} (CUDA_VISIBLE_DEVICES={args.gpu})")
    
    if args.list_models:
        print("🎬 Available Models:")
        print("=" * 60)
        families = {}
        for model_name, model_info in AVAILABLE_MODELS.items():
            family = model_info.get('family', 'Unknown')
            if family not in families:
                families[family] = []
            families[family].append((model_name, model_info.get('description', '')))
        
        for family, models in sorted(families.items()):
            print(f"\n📁 {family}:")
            for model_name, description in sorted(models):
                print(f"   • {model_name:25} - {description}")
        
        print(f"\nTotal: {len(AVAILABLE_MODELS)} models across {len(families)} families")
        return 
    
    if args.override:
        if OUTPUT_DIR.exists():
            print(f"🗑️  Override mode: Deleting {OUTPUT_DIR}...")
            shutil.rmtree(OUTPUT_DIR)
            print(f"   ✅ Deleted {OUTPUT_DIR}")
        else:
            print(f"   ℹ️  Output directory does not exist: {OUTPUT_DIR}")
    
    print("🔍 Discovering human-approved tasks from folder structure...")
    
    if not QUESTIONS_DIR.exists():
        raise ValueError(f"Questions directory not found at: {QUESTIONS_DIR}. Please ensure the questions directory exists with task folders.")
    

    all_tasks_by_domain = discover_all_tasks_from_folders(QUESTIONS_DIR)
    
    # Select tasks based on arguments
    tasks_by_domain = {}
    
    if args.task_id:
        # Specific task IDs requested
        print(f"   🎯 Running specific task IDs: {', '.join(args.task_id)}")
        tasks_by_domain = {}
        for task_id in args.task_id:
            # Find which domain this task belongs to
            found = False
            for domain, tasks in all_tasks_by_domain.items():
                for task in tasks:
                    if task['id'] == task_id:
                        if domain not in tasks_by_domain:
                            tasks_by_domain[domain] = []
                        tasks_by_domain[domain].append(task)
                        found = True
                        break
            if not found:
                print(f"   ⚠️  Task ID '{task_id}' not found")
    
    elif args.all_tasks:
        # All tasks requested
        if args.task:
            # All tasks from specific domains
            tasks_by_domain = {domain: tasks for domain, tasks in all_tasks_by_domain.items() if domain in args.task}
            print(f"   🎯 Running ALL tasks from domains: {', '.join(args.task)}")
        else:
            # All tasks from all domains
            tasks_by_domain = all_tasks_by_domain
            print(f"   🎯 Running ALL approved tasks")
    
    else:
        # Limited number per domain
        if args.task:
            # Specific domains
            selected_domains = args.task
        else:
            # All domains
            selected_domains = list(all_tasks_by_domain.keys())
        
        for domain in selected_domains:
            if domain in all_tasks_by_domain and all_tasks_by_domain[domain]:
                num_tasks = min(args.pairs_per_domain, len(all_tasks_by_domain[domain]))
                tasks_by_domain[domain] = all_tasks_by_domain[domain][:num_tasks]
                task_names = [task['id'] for task in tasks_by_domain[domain]]
                print(f"   🎯 Running {num_tasks} task(s) from {domain}: {', '.join(task_names)}")
            else:
                tasks_by_domain[domain] = []
    
    model_names = []
    if args.model:
        model_names = args.model
    else:
        model_names = DEFAULT_TEST_MODELS
    
    selected_models = {}
    unavailable_models = []
    for model_name in model_names:
        if model_name in AVAILABLE_MODELS:
            selected_models[model_name] = AVAILABLE_MODELS[model_name].get('family', 'Unknown')
        else:
            unavailable_models.append(model_name)
    
    if unavailable_models:
        print(f"⚠️  Models not available: {', '.join(unavailable_models)}. Use --list-models to see all available models")
    
    if not selected_models:
        raise ValueError("No valid models selected")
        
    print(f"\n🎯 Selected {len(selected_models)} model(s): {', '.join(selected_models.keys())}")

    print(f"\n🔍 Verifying {len(selected_models)} model(s) for testing...")
    for model_name, family in selected_models.items():
        print(f"   ✅ {model_name}: {family}")
    
    
    if not tasks_by_domain or sum(len(tasks) for tasks in tasks_by_domain.values()) == 0:
        raise ValueError("No approved tasks found. Please check the questions directory structure.")
    
    experiment_results = run_pilot_experiment(
        tasks_by_domain=tasks_by_domain,
        models=selected_models,
        output_dir=OUTPUT_DIR,
        skip_existing=True,
    )
    
    print("🎉 VIDEO GENERATION COMPLETE!")
    stats = experiment_results["statistics"]
    
    actual_total_attempted = stats['completed'] + stats['failed'] + stats['skipped']
    
    print(f"\n📊 Final Statistics:")
    print(f"   Models tested: {len(selected_models)}")
    print(f"   Tasks per model: {stats['total_tasks']}")
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


if __name__ == "__main__":
    main()
