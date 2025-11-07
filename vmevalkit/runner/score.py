"""Scoring runner for VMEvalKit.

This script runs various scoring methods on generated videos.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from vmevalkit.eval import HumanScorer, GPT4OScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scoring.log')
    ]
)
logger = logging.getLogger(__name__)


def run_human_scoring(
    experiment_name: str = "pilot_experiment",
    output_dir: str = "data/scorings",
    annotator_name: str = "anonymous",
    port: int = 7860,
    share: bool = False
):
    """Run human scoring interface.
    
    Args:
        experiment_name: Name of the experiment to score
        output_dir: Directory to save scoring results
        annotator_name: Name of the human annotator
        port: Port to run Gradio interface on
        share: Whether to create a public share link
    """
    logger.info(f"Starting human scoring for experiment: {experiment_name}")
    logger.info(f"Annotator: {annotator_name}")
    
    scorer = HumanScorer(
        output_dir=output_dir,
        experiment_name=experiment_name,
        annotator_name=annotator_name
    )
    
    # Launch the Gradio interface
    logger.info(f"Launching interface on port {port}")
    scorer.launch_interface(share=share, port=port)


def run_gpt4o_scoring(
    experiment_name: str = "pilot_experiment",
    output_dir: str = "data/scorings",
    max_frames: int = 8,
    temperature: float = 0.1
):
    """Run GPT-4O automatic scoring on entire experiment.
    
    Args:
        experiment_name: Name of the experiment to score
        output_dir: Directory to save scoring results
        max_frames: Maximum frames to extract per video
        temperature: Temperature for GPT-4O responses
    """
    logger.info(f"Starting GPT-4O scoring for experiment: {experiment_name}")
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        sys.exit(1)
    
    scorer = GPT4OScorer(
        output_dir=output_dir,
        experiment_name=experiment_name,
        max_frames=max_frames,
        temperature=temperature
    )
    
    # Score all models and tasks in the experiment
    logger.info("Scoring all models and tasks in experiment")
    results = scorer.score_all_models()
    logger.info("Completed scoring for all models")
    
    # Print basic counts
    for model_name, model_results in results.items():
        if "scorings" in model_results:
            total_tasks = 0
            scored_tasks = 0
            for task_type, tasks in model_results["scorings"].items():
                for task_id, result in tasks.items():
                    total_tasks += 1
                    if "error" not in result:
                        scored_tasks += 1
            
            logger.info(f"\n{model_name}:")
            logger.info(f"  Total tasks: {total_tasks}")
            logger.info(f"  Scored: {scored_tasks}")


def main():
    """Main entry point for scoring runner."""
    parser = argparse.ArgumentParser(
        description="Run scoring on VMEvalKit experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run human scoring
  python -m vmevalkit.runner.score human --annotator "John Doe" --port 7860
  
  # Run GPT-4O scoring on all models
  python -m vmevalkit.runner.score gpt4o
  
  # Run GPT-4O scoring on specific models
  python -m vmevalkit.runner.score gpt4o --models luma-ray-2 openai-sora-2
  
  # Run scoring on a different experiment
  python -m vmevalkit.runner.score gpt4o --experiment my_experiment
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='method', help='Scoring method')
    
    # Human scoring subcommand
    human_parser = subparsers.add_parser('human', help='Run human scoring interface')
    human_parser.add_argument(
        '--experiment', '-e',
        type=str,
        default='pilot_experiment',
        help='Name of the experiment to score'
    )
    human_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/scorings',
        help='Directory to save scoring results'
    )
    human_parser.add_argument(
        '--annotator', '-a',
        type=str,
        default='anonymous',
        help='Name of the human annotator'
    )
    human_parser.add_argument(
        '--port', '-p',
        type=int,
        default=7860,
        help='Port to run Gradio interface on'
    )
    human_parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public share link for the interface'
    )
    
    # GPT-4O scoring subcommand
    gpt4o_parser = subparsers.add_parser('gpt4o', help='Run GPT-4O automatic scoring')
    gpt4o_parser.add_argument(
        '--experiment', '-e',
        type=str,
        default='pilot_experiment',
        help='Name of the experiment to score'
    )
    gpt4o_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/scorings',
        help='Directory to save scoring results'
    )
    gpt4o_parser.add_argument(
        '--max-frames',
        type=int,
        default=8,
        help='Maximum frames to extract per video'
    )
    gpt4o_parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='Temperature for GPT-4O responses'
    )
    
    args = parser.parse_args()
    
    if not args.method:
        parser.print_help()
        sys.exit(1)
    
    # Run the appropriate scoring method
    if args.method == 'human':
        run_human_scoring(
            experiment_name=args.experiment,
            output_dir=args.output_dir,
            annotator_name=args.annotator,
            port=args.port,
            share=args.share
        )
    elif args.method == 'gpt4o':
        run_gpt4o_scoring(
            experiment_name=args.experiment,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            temperature=args.temperature
        )
    else:
        logger.error(f"Unknown scoring method: {args.method}")
        sys.exit(1)


if __name__ == "__main__":
    main()
