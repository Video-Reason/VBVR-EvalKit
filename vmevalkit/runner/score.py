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
    from vmevalkit.eval import HumanEvaluator

    logger.info(f"Starting human scoring for experiment: {experiment_name}")
    logger.info(f"Annotator: {annotator_name}")

    scorer = HumanEvaluator(
        output_dir=output_dir,
        experiment_name=experiment_name
    )
    scorer.annotator_name = annotator_name

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
    from vmevalkit.eval import GPT4OEvaluator

    logger.info(f"Starting GPT-4O scoring for experiment: {experiment_name}")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        sys.exit(1)

    scorer = GPT4OEvaluator(
        output_dir=output_dir,
        experiment_name=experiment_name,
        temperature=temperature
    )

    # Score all models and tasks in the experiment
    logger.info("Scoring all models and tasks in experiment")
    results = scorer.evaluate_all_models()
    logger.info("Completed scoring for all models")

    # Print basic counts
    for model_name, model_results in results.items():
        if "evaluations" in model_results:
            total_tasks = 0
            scored_tasks = 0
            for task_type, tasks in model_results["evaluations"].items():
                for task_id, result in tasks.items():
                    total_tasks += 1
                    if "error" not in result:
                        scored_tasks += 1

            logger.info(f"\n{model_name}:")
            logger.info(f"  Total tasks: {total_tasks}")
            logger.info(f"  Scored: {scored_tasks}")


def run_multiframe_gpt4o_scoring(
    experiment_name: str = "pilot_experiment",
    output_dir: str = "data/evaluations/multiframe-gpt4o-eval",
    n_frames: int = 5,
    last_seconds: float = 3.0,
    strategy: str = "hybrid",
    voting: str = "weighted_majority",
    metric: str = "histogram",
    temporal_weight: float = 0.3,
    temperature: float = 0.0
):
    """Run multi-frame GPT-4O evaluation with voting.

    This is a convenience wrapper that calls run_multiframe_vlm_scoring with gpt4o.
    For more options, use multiframe-vlm command directly.

    Args:
        experiment_name: Name of the experiment to score
        output_dir: Directory to save scoring results
        n_frames: Number of frames to sample per video
        last_seconds: Sample from last N seconds of video
        strategy: Sampling strategy (uniform/keyframe/hybrid)
        voting: Voting method for aggregation
        metric: Similarity metric for consistency analysis
        temporal_weight: Weight for temporal bias (0-1)
        temperature: GPT-4O temperature
    """
    run_multiframe_vlm_scoring(
        experiment_name=experiment_name,
        output_dir=output_dir,
        evaluator_type="gpt4o",
        n_frames=n_frames,
        last_seconds=last_seconds,
        strategy=strategy,
        voting=voting,
        metric=metric,
        temporal_weight=temporal_weight,
        temperature=temperature
    )


def run_multiframe_vlm_scoring(
    experiment_name: str = "pilot_experiment",
    output_dir: Optional[str] = None,
    evaluator_type: str = "gpt4o",
    n_frames: int = 5,
    last_seconds: float = 3.0,
    strategy: str = "hybrid",
    voting: str = "weighted_majority",
    metric: str = "histogram",
    temporal_weight: float = 0.3,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
):
    """Run multi-frame evaluation with any VLM backend (GPT-4O or InternVL).

    This is the generic multi-frame evaluator that wraps any single-frame
    evaluator and adds multi-frame capabilities with voting aggregation.

    Args:
        experiment_name: Name of the experiment to score
        output_dir: Directory to save scoring results (auto-generated if None)
        evaluator_type: Base evaluator type ("gpt4o" or "internvl")
        n_frames: Number of frames to sample per video
        last_seconds: Sample from last N seconds of video
        strategy: Sampling strategy (uniform/keyframe/hybrid)
        voting: Voting method for aggregation
        metric: Similarity metric for consistency analysis
        temporal_weight: Weight for temporal bias (0-1)
        temperature: VLM temperature
        api_key: API key (env var fallback: OPENAI_API_KEY or VISION_API_KEY)
        base_url: Base URL for VLM API (only for internvl)
    """
    from vmevalkit.eval import MultiFrameEvaluator, GPT4OEvaluator, InternVLEvaluator

    logger.info(f"Starting multi-frame {evaluator_type.upper()} scoring for experiment: {experiment_name}")
    logger.info(f"Config: n_frames={n_frames}, strategy={strategy}, voting={voting}")

    # Create base evaluator based on type
    if evaluator_type == "gpt4o":
        if not api_key and not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable not set!")
            sys.exit(1)

        base_evaluator = GPT4OEvaluator(
            experiment_name=experiment_name,
            api_key=api_key,
            temperature=temperature
        )
        default_output_dir = "data/evaluations/multiframe-gpt4o-eval"

    elif evaluator_type == "internvl":
        resolved_api_key = api_key or os.getenv("VISION_API_KEY", "YOUR_API_KEY")
        resolved_base_url = base_url or os.getenv("VISION_API_BASE", "http://0.0.0.0:23333/v1")

        base_evaluator = InternVLEvaluator(
            experiment_name=experiment_name,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            temperature=temperature
        )
        default_output_dir = "data/evaluations/multiframe-internvl-eval"

    else:
        logger.error(f"Unknown evaluator type: {evaluator_type}")
        sys.exit(1)

    # Create multi-frame evaluator wrapper
    evaluator = MultiFrameEvaluator(
        base_evaluator=base_evaluator,
        output_dir=output_dir or default_output_dir,
        n_frames=n_frames,
        last_seconds=last_seconds,
        sampling_strategy=strategy,
        similarity_metric=metric,
        temporal_weight=temporal_weight,
        voting_method=voting,
        confidence_threshold=0.6
    )

    # Evaluate all models
    results = evaluator.evaluate_all_models()

    # Print summary
    for model_name, model_results in results.items():
        if "evaluations" in model_results:
            total_tasks = 0
            scored_tasks = 0
            review_needed = 0

            for task_type, tasks in model_results["evaluations"].items():
                for task_id, result in tasks.items():
                    total_tasks += 1
                    if result.get("status") == "completed":
                        scored_tasks += 1
                        if result.get("needs_review"):
                            review_needed += 1

            logger.info(f"\n{model_name}:")
            logger.info(f"  Total tasks: {total_tasks}")
            logger.info(f"  Scored: {scored_tasks}")
            logger.info(f"  Needs review: {review_needed}")


def run_multiframe_scoring(
    experiment_name: str = "pilot_experiment",
    output_dir: str = "data/scorings/multiframe",
    n_frames: int = 5,
    last_seconds: float = 3.0,
    strategy: str = "hybrid",
    voting: str = "weighted_majority",
    metric: str = "histogram"
):
    """Run multi-frame evaluation with voting.

    Args:
        experiment_name: Name of the experiment to score
        output_dir: Directory to save scoring results
        n_frames: Number of frames to sample
        last_seconds: Duration from video end to sample
        strategy: Sampling strategy (uniform/keyframe/hybrid)
        voting: Voting method
        metric: Similarity metric for consistency
    """
    from vmevalkit.eval import (
        FrameSampler,
        FrameConsistencyAnalyzer,
        VotingAggregator,
        VotingMethod,
        FrameScore,
        GPT4OEvaluator
    )
    import json
    from datetime import datetime

    logger.info(f"Starting multi-frame scoring for experiment: {experiment_name}")
    logger.info(f"Config: n_frames={n_frames}, strategy={strategy}, voting={voting}")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        sys.exit(1)

    # Initialize components
    sampler = FrameSampler(n_frames=n_frames, last_seconds=last_seconds)
    analyzer = FrameConsistencyAnalyzer(metric=metric)
    voter = VotingAggregator(method=VotingMethod(voting))
    evaluator = GPT4OEvaluator(output_dir=output_dir, experiment_name=experiment_name)

    experiment_dir = Path("data/outputs") / experiment_name
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)

    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    for model_dir in experiment_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        logger.info(f"Processing model: {model_name}")

        for task_type_dir in model_dir.iterdir():
            if not task_type_dir.is_dir():
                continue

            for task_dir in task_type_dir.iterdir():
                if not task_dir.is_dir():
                    continue

                task_type = task_type_dir.name
                task_id = task_dir.name

                # Find video
                run_dirs = list(task_dir.iterdir())
                if not run_dirs:
                    continue
                video_files = list((run_dirs[0] / "video").glob("*.mp4"))
                if not video_files:
                    continue

                video_path = str(video_files[0])

                try:
                    # Sample frames
                    frames = sampler.sample(video_path, strategy=strategy)
                    if not frames:
                        continue

                    # Analyze consistency
                    consistency = analyzer.analyze([f.image for f in frames])

                    # Evaluate each frame
                    scores = []
                    for i, (frame, weight) in enumerate(zip(frames, consistency.weights)):
                        result = evaluator.evaluate_single(
                            model_name, task_type, f"{task_id}_f{i}", video_path
                        )
                        score = result.get("solution_correctness_score", 0)
                        scores.append(FrameScore(
                            score=score,
                            timestamp=frame.timestamp,
                            weight=weight,
                            is_keyframe=frame.is_keyframe
                        ))

                    # Aggregate
                    voting_result = voter.aggregate(scores, stability_score=consistency.stability_score)

                    # Save
                    task_out = output_path / model_name / task_type / task_id
                    task_out.mkdir(parents=True, exist_ok=True)

                    with open(task_out / "MultiFrameEvaluator.json", 'w') as f:
                        json.dump({
                            "metadata": {
                                "evaluator": "MultiFrameEvaluator",
                                "timestamp": datetime.now().isoformat()
                            },
                            "result": {
                                "final_score": voting_result.final_score,
                                "confidence": voting_result.confidence,
                                "agreement_ratio": voting_result.agreement_ratio,
                                "stability_score": voting_result.stability_score,
                                "needs_review": voting_result.needs_review
                            }
                        }, f, indent=2)

                    logger.info(f"  {task_id}: score={voting_result.final_score}, conf={voting_result.confidence:.2f}")

                except Exception as e:
                    logger.error(f"  {task_id}: Error - {e}")


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

  # Run multi-frame GPT-4O scoring (recommended)
  python -m vmevalkit.runner.score multiframe-gpt4o --n-frames 5

  # Run multi-frame InternVL scoring (local VLM)
  python -m vmevalkit.runner.score multiframe-vlm --evaluator internvl

  # Run multi-frame with custom settings
  python -m vmevalkit.runner.score multiframe-vlm --evaluator gpt4o --n-frames 7 --strategy hybrid

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

    # Multi-frame scoring subcommand (legacy - uses mock evaluator)
    multi_parser = subparsers.add_parser('multiframe', help='Run multi-frame scoring with voting')
    multi_parser.add_argument('--experiment', '-e', default='pilot_experiment')
    multi_parser.add_argument('--output-dir', '-o', default='data/scorings/multiframe')
    multi_parser.add_argument('--n-frames', type=int, default=5, help='Number of frames to sample')
    multi_parser.add_argument('--last-seconds', type=float, default=3.0)
    multi_parser.add_argument('--strategy', choices=['uniform', 'keyframe', 'hybrid'], default='hybrid')
    multi_parser.add_argument('--voting', default='weighted_majority',
                             choices=['majority', 'weighted_majority', 'weighted_average', 'max_score', 'median'])
    multi_parser.add_argument('--metric', choices=['histogram', 'ssim', 'combined'], default='histogram')

    # Multi-frame GPT-4O scoring subcommand (RECOMMENDED)
    mf_gpt4o_parser = subparsers.add_parser('multiframe-gpt4o',
        help='Run multi-frame GPT-4O evaluation with voting (recommended)')
    mf_gpt4o_parser.add_argument('--experiment', '-e', default='pilot_experiment',
        help='Name of the experiment to score')
    mf_gpt4o_parser.add_argument('--output-dir', '-o', default='data/evaluations/multiframe-gpt4o-eval',
        help='Directory to save evaluation results')
    mf_gpt4o_parser.add_argument('--n-frames', type=int, default=5,
        help='Number of frames to sample per video')
    mf_gpt4o_parser.add_argument('--last-seconds', type=float, default=3.0,
        help='Sample from last N seconds of video')
    mf_gpt4o_parser.add_argument('--strategy', choices=['uniform', 'keyframe', 'hybrid'], default='hybrid',
        help='Frame sampling strategy')
    mf_gpt4o_parser.add_argument('--voting', default='weighted_majority',
        choices=['majority', 'weighted_majority', 'weighted_average', 'max_score', 'median'],
        help='Voting method for score aggregation')
    mf_gpt4o_parser.add_argument('--metric', choices=['histogram', 'ssim', 'combined'], default='histogram',
        help='Similarity metric for consistency analysis')
    mf_gpt4o_parser.add_argument('--temporal-weight', type=float, default=0.3,
        help='Weight for temporal bias (0-1, higher prefers later frames)')
    mf_gpt4o_parser.add_argument('--temperature', type=float, default=0.0,
        help='GPT-4O temperature (0.0 = deterministic)')

    # Multi-frame VLM scoring subcommand (GENERIC - supports GPT-4O and InternVL)
    mf_vlm_parser = subparsers.add_parser('multiframe-vlm',
        help='Run multi-frame evaluation with any VLM (GPT-4O or InternVL)')
    mf_vlm_parser.add_argument('--experiment', '-e', default='pilot_experiment',
        help='Name of the experiment to score')
    mf_vlm_parser.add_argument('--output-dir', '-o', default=None,
        help='Directory to save evaluation results (auto-generated based on evaluator)')
    mf_vlm_parser.add_argument('--evaluator', choices=['gpt4o', 'internvl'], default='gpt4o',
        help='Base evaluator type: gpt4o (OpenAI) or internvl (local VLM)')
    mf_vlm_parser.add_argument('--n-frames', type=int, default=5,
        help='Number of frames to sample per video')
    mf_vlm_parser.add_argument('--last-seconds', type=float, default=3.0,
        help='Sample from last N seconds of video')
    mf_vlm_parser.add_argument('--strategy', choices=['uniform', 'keyframe', 'hybrid'], default='hybrid',
        help='Frame sampling strategy')
    mf_vlm_parser.add_argument('--voting', default='weighted_majority',
        choices=['majority', 'weighted_majority', 'weighted_average', 'max_score', 'median'],
        help='Voting method for score aggregation')
    mf_vlm_parser.add_argument('--metric', choices=['histogram', 'ssim', 'combined'], default='histogram',
        help='Similarity metric for consistency analysis')
    mf_vlm_parser.add_argument('--temporal-weight', type=float, default=0.3,
        help='Weight for temporal bias (0-1, higher prefers later frames)')
    mf_vlm_parser.add_argument('--temperature', type=float, default=0.0,
        help='VLM temperature (0.0 = deterministic)')
    mf_vlm_parser.add_argument('--api-key', default=None,
        help='API key (fallback: OPENAI_API_KEY or VISION_API_KEY)')
    mf_vlm_parser.add_argument('--base-url', default=None,
        help='Base URL for VLM API (only for internvl, fallback: VISION_API_BASE)')

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
    elif args.method == 'multiframe':
        run_multiframe_scoring(
            experiment_name=args.experiment,
            output_dir=args.output_dir,
            n_frames=args.n_frames,
            last_seconds=args.last_seconds,
            strategy=args.strategy,
            voting=args.voting,
            metric=args.metric
        )
    elif args.method == 'multiframe-gpt4o':
        run_multiframe_gpt4o_scoring(
            experiment_name=args.experiment,
            output_dir=args.output_dir,
            n_frames=args.n_frames,
            last_seconds=args.last_seconds,
            strategy=args.strategy,
            voting=args.voting,
            metric=args.metric,
            temporal_weight=args.temporal_weight,
            temperature=args.temperature
        )
    elif args.method == 'multiframe-vlm':
        run_multiframe_vlm_scoring(
            experiment_name=args.experiment,
            output_dir=args.output_dir,
            evaluator_type=args.evaluator,
            n_frames=args.n_frames,
            last_seconds=args.last_seconds,
            strategy=args.strategy,
            voting=args.voting,
            metric=args.metric,
            temporal_weight=args.temporal_weight,
            temperature=args.temperature,
            api_key=args.api_key,
            base_url=args.base_url
        )
    else:
        logger.error(f"Unknown scoring method: {args.method}")
        sys.exit(1)


if __name__ == "__main__":
    main()
