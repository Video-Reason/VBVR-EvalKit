"""
Command-line interface for VMEvalKit.

Provides direct access to inference functionality without separate scripts.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .inference import InferenceRunner, BatchInferenceRunner


def run_inference(args):
    """Run single inference."""
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"Error: Invalid resolution format: {args.resolution}")
        sys.exit(1)
    
    # Initialize runner
    runner = InferenceRunner(output_dir=str(args.output_dir))
    
    # Run inference
    if args.task_file:
        # Load task from file
        with open(args.task_file, 'r') as f:
            data = json.load(f)
        
        # Handle single task or dataset format
        if "pairs" in data:
            # Dataset format
            pairs = data["pairs"]
            if args.task_id:
                task = next((p for p in pairs if p.get("id") == args.task_id), None)
                if not task:
                    print(f"Error: Task ID not found: {args.task_id}")
                    sys.exit(1)
            else:
                if args.task_index >= len(pairs):
                    print(f"Error: Task index {args.task_index} out of range")
                    sys.exit(1)
                task = pairs[args.task_index]
        else:
            # Single task format
            task = data
        
        result = runner.run_from_task(
            model_name=args.model,
            task_data=task,
            duration=args.duration,
            fps=args.fps,
            resolution=resolution
        )
    else:
        # Direct image + prompt
        if not args.prompt:
            print("Error: --prompt is required when using --image")
            sys.exit(1)
        
        result = runner.run(
            model_name=args.model,
            image_path=args.image,
            text_prompt=args.prompt,
            duration=args.duration,
            fps=args.fps,
            resolution=resolution
        )
    
    # Print result
    if result.get("status") == "success":
        print(f"\n🎬 Video generated: {result['video_path']}")
    else:
        print(f"\n❌ Generation failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def run_batch(args):
    """Run batch inference."""
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"Error: Invalid resolution format: {args.resolution}")
        sys.exit(1)
    
    # Initialize batch runner
    runner = BatchInferenceRunner(
        output_dir=str(args.output_dir),
        max_workers=args.workers
    )
    
    # Run inference
    if len(args.models) == 1:
        # Single model batch run
        result = runner.run_dataset(
            model_name=args.models[0],
            dataset_path=args.dataset,
            task_ids=args.task_ids,
            max_tasks=args.max_tasks,
            duration=args.duration,
            fps=args.fps,
            resolution=resolution
        )
    else:
        # Multiple models comparison
        result = runner.run_models_comparison(
            model_names=args.models,
            dataset_path=args.dataset,
            task_ids=args.task_ids,
            duration=args.duration,
            fps=args.fps,
            resolution=resolution
        )
    
    print("\n✨ Batch inference complete!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VMEvalKit - Video Model Evaluation Kit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single inference from task file
  vmevalkit inference luma-dream-machine --task-file data/task.json

  # Direct inference with image and prompt
  vmevalkit inference luma-dream-machine --image maze.png --prompt "Solve the maze"

  # Batch inference on dataset
  vmevalkit batch luma-dream-machine --dataset data/tasks.json

  # Multi-model comparison
  vmevalkit batch luma-dream-machine google-veo --dataset data/tasks.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Inference command
    inference_parser = subparsers.add_parser(
        "inference",
        help="Run single video generation inference"
    )
    
    inference_parser.add_argument(
        "model",
        type=str,
        help="Model name (e.g., luma-dream-machine, google-veo-001)"
    )
    
    # Input specification
    input_group = inference_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--task-file",
        type=Path,
        help="Path to task JSON file with prompt and first_image_path"
    )
    input_group.add_argument(
        "--image",
        type=Path,
        help="Path to input image (use with --prompt)"
    )
    
    inference_parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt (required when using --image)"
    )
    
    # Task selection (when using --task-file)
    inference_parser.add_argument(
        "--task-id",
        type=str,
        help="Specific task ID to run from the dataset"
    )
    inference_parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Task index to run (0-based, default: 0)"
    )
    
    # Generation parameters
    inference_parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Video duration in seconds (default: 5.0)"
    )
    inference_parser.add_argument(
        "--resolution",
        type=str,
        default="1280x720",
        help="Resolution as WIDTHxHEIGHT (default: 1280x720)"
    )
    inference_parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second (default: 24)"
    )
    
    # Output
    inference_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Output directory (default: ./outputs)"
    )
    
    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Run batch inference on dataset"
    )
    
    # Model(s) selection
    batch_parser.add_argument(
        "models",
        type=str,
        nargs='+',
        help="Model name(s) to run (e.g., luma-dream-machine google-veo-001)"
    )
    
    # Dataset
    batch_parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset JSON file"
    )
    
    # Task selection
    batch_parser.add_argument(
        "--task-ids",
        type=str,
        nargs='*',
        help="Specific task IDs to run (default: all)"
    )
    batch_parser.add_argument(
        "--max-tasks",
        type=int,
        help="Maximum number of tasks to process"
    )
    
    # Generation parameters
    batch_parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Video duration in seconds (default: 5.0)"
    )
    batch_parser.add_argument(
        "--resolution",
        type=str,
        default="1280x720",
        help="Resolution as WIDTHxHEIGHT (default: 1280x720)"
    )
    batch_parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second (default: 24)"
    )
    
    # Parallelization
    batch_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )
    
    # Output
    batch_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs"),
        help="Output directory (default: ./outputs)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Execute command
    if args.command == "inference":
        run_inference(args)
    elif args.command == "batch":
        run_batch(args)


if __name__ == "__main__":
    main()
