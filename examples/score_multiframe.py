#!/usr/bin/env python3
"""
Multi-frame evaluation example for VMEvalKit.

This script demonstrates the multi-frame evaluation system that:
1. Samples multiple frames from video tail
2. Analyzes frame consistency for weight computation
3. Evaluates each frame with VLM (GPT-4O or InternVL)
4. Aggregates scores via weighted voting

Usage:
    # Evaluate inference results with GPT-4O (default)
    python examples/score_multiframe.py --inference-dir ./outputs --eval-output-dir ./evaluations

    # Evaluate with InternVL (local VLM)
    python examples/score_multiframe.py --inference-dir ./outputs --eval-output-dir ./evaluations --evaluator internvl

    # Custom settings
    python examples/score_multiframe.py --inference-dir ~/experiments/run1 --eval-output-dir ~/experiments/run1_scores --n-frames 7 --temporal-weight 0.5

    # Test on specific video (without API calls)
    python examples/score_multiframe.py --test-only --video path/to/video.mp4
"""

import os
import sys
import logging
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_multiframe_evaluation(
    inference_dir: str,
    eval_output_dir: str = "./evaluations/multiframe",
    evaluator_type: str = "gpt4o",
    n_frames: int = 5,
    last_seconds: float = 3.0,
    strategy: str = "hybrid",
    voting: str = "weighted_majority",
    metric: str = "histogram",
    temporal_weight: float = 0.3,
    temperature: float = 0.0
):
    """Run multi-frame evaluation on inference results.

    Args:
        inference_dir: Path to inference outputs to evaluate
        eval_output_dir: Path for evaluation results
        evaluator_type: "gpt4o" or "internvl"
        n_frames: Number of frames to sample per video
        last_seconds: Sample from last N seconds of video
        strategy: Sampling strategy (uniform/keyframe/hybrid)
        voting: Voting method for aggregation
        metric: Similarity metric for consistency analysis
        temporal_weight: Weight for temporal bias (0-1, higher prefers later frames)
        temperature: VLM temperature (0.0 = deterministic)
    """
    from vmevalkit.eval import MultiFrameEvaluator, GPT4OEvaluator, InternVLEvaluator

    print("\n" + "=" * 60)
    print(f"MULTI-FRAME {evaluator_type.upper()} EVALUATION")
    print("=" * 60)
    print(f"Inference Dir: {inference_dir}")
    print(f"Eval Output Dir: {eval_output_dir}")
    print(f"Config:")
    print(f"  - evaluator: {evaluator_type}")
    print(f"  - n_frames: {n_frames}")
    print(f"  - last_seconds: {last_seconds}")
    print(f"  - strategy: {strategy}")
    print(f"  - voting: {voting}")
    print(f"  - metric: {metric}")
    print(f"  - temporal_weight: {temporal_weight}")
    print(f"  - temperature: {temperature}")

    # Check inference directory exists
    inference_path = Path(inference_dir)
    if not inference_path.exists():
        print(f"\nError: Inference directory not found: {inference_path}")
        print("Please run inference first to generate videos.")
        sys.exit(1)

    # Create base evaluator based on type
    if evaluator_type == "gpt4o":
        if not os.getenv("OPENAI_API_KEY"):
            print("\nError: OPENAI_API_KEY environment variable not set!")
            print("Please set it with: export OPENAI_API_KEY=your_api_key")
            sys.exit(1)
        base_evaluator = GPT4OEvaluator(
            inference_dir=inference_dir,
            eval_output_dir=f"{eval_output_dir}/gpt4o",
            temperature=temperature
        )
    else:  # internvl
        api_key = os.getenv("VISION_API_KEY", "YOUR_API_KEY")
        base_url = os.getenv("VISION_API_BASE", "http://0.0.0.0:23333/v1")
        base_evaluator = InternVLEvaluator(
            inference_dir=inference_dir,
            eval_output_dir=f"{eval_output_dir}/internvl",
            api_key=api_key,
            base_url=base_url,
            temperature=temperature
        )

    # Initialize multi-frame evaluator
    evaluator = MultiFrameEvaluator(
        base_evaluator=base_evaluator,
        output_dir=eval_output_dir,
        n_frames=n_frames,
        last_seconds=last_seconds,
        sampling_strategy=strategy,
        voting_method=voting,
        similarity_metric=metric,
        temporal_weight=temporal_weight
    )

    # Check for existing evaluations
    eval_dir = Path(eval_output_dir)
    if eval_dir.exists():
        existing_files = list(eval_dir.rglob(f"{evaluator.evaluator_name}.json"))
        if existing_files:
            print(f"\nFound {len(existing_files)} existing evaluations - will resume from where left off")

    print("\nStarting evaluation...")
    print("Tip: You can interrupt (Ctrl+C) and resume later - progress is saved after each task")
    print("=" * 60)

    try:
        all_results = evaluator.evaluate_all_models()

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        total_all = 0
        completed_all = 0
        review_all = 0

        for model_name, results in all_results.items():
            if "evaluations" in results:
                total_tasks = 0
                evaluated_tasks = 0
                needs_review = 0

                for task_type, tasks in results["evaluations"].items():
                    for task_id, result in tasks.items():
                        total_tasks += 1
                        if result.get("status") == "completed":
                            evaluated_tasks += 1
                            if result.get("needs_review"):
                                needs_review += 1

                total_all += total_tasks
                completed_all += evaluated_tasks
                review_all += needs_review

                status = "Complete" if evaluated_tasks == total_tasks else f"{evaluated_tasks}/{total_tasks}"
                review_status = f" ({needs_review} need review)" if needs_review > 0 else ""
                print(f"  {model_name}: {status}{review_status}")

        print(f"\nTotal: {completed_all}/{total_all} tasks evaluated")
        if review_all > 0:
            print(f"Tasks needing review: {review_all}")
        print(f"\nResults saved to: {eval_output_dir}/")

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted!")
        print("Progress has been saved. Run the same command again to resume.")


def test_multiframe_pipeline(video_path: str, output_dir: str = "test_output"):
    """Test multi-frame pipeline without API calls (uses mock evaluator).

    This is useful for testing frame sampling and consistency analysis.
    """
    from vmevalkit.eval.frame_sampler import FrameSampler
    from vmevalkit.eval.consistency import FrameConsistencyAnalyzer
    from vmevalkit.eval.voting import VotingAggregator, VotingMethod, FrameScore
    from PIL import Image

    print("\n" + "=" * 60)
    print("MULTI-FRAME PIPELINE TEST (No API calls)")
    print("=" * 60)
    print(f"Video: {video_path}")

    # Initialize components
    sampler = FrameSampler(n_frames=5, last_seconds=3.0)
    analyzer = FrameConsistencyAnalyzer(metric="histogram", temporal_weight=0.3)
    voter = VotingAggregator(method=VotingMethod.WEIGHTED_MAJORITY)

    # Get video info
    try:
        info = sampler.get_video_info(video_path)
        print(f"\nVideo Info:")
        print(f"  - Duration: {info['duration']:.2f}s")
        print(f"  - Frames: {info['total_frames']}")
        print(f"  - FPS: {info['fps']:.2f}")
        print(f"  - Resolution: {info['width']}x{info['height']}")
    except Exception as e:
        print(f"Error getting video info: {e}")
        return

    # Sample frames
    print("\n--- Frame Sampling ---")
    frames = sampler.sample(video_path, strategy="hybrid")
    print(f"Sampled {len(frames)} frames:")
    for i, f in enumerate(frames):
        kf = " [KF]" if f.is_keyframe else ""
        print(f"  Frame {i+1}: idx={f.frame_index}, t={f.timestamp:.2f}s{kf}")

    # Save frames
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        img = Image.fromarray(f.image)
        img.save(out_path / f"frame_{i+1}.png")
    print(f"Saved frames to {out_path}/")

    # Analyze consistency
    print("\n--- Consistency Analysis ---")
    consistency = analyzer.analyze([f.image for f in frames])
    print(f"Stability Score: {consistency.stability_score:.3f}")
    print(f"Mean Similarity: {consistency.mean_similarity:.3f}")
    print(f"Outliers: {consistency.outlier_indices if consistency.outlier_indices else 'None'}")
    print(f"Weights: [{', '.join(f'{w:.3f}' for w in consistency.weights)}]")

    # Simulate voting with mock scores
    print("\n--- Voting Simulation ---")
    mock_scores = [4, 4, 3, 4, 5]  # Simulated GPT-4O scores
    print(f"Mock scores: {mock_scores}")

    frame_scores = [
        FrameScore(
            score=s,
            timestamp=f.timestamp,
            weight=w,
            is_keyframe=f.is_keyframe
        )
        for s, f, w in zip(mock_scores, frames, consistency.weights)
    ]

    result = voter.aggregate(frame_scores, stability_score=consistency.stability_score)

    print(f"\nVoting Result:")
    print(f"  Final Score: {result.final_score}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Agreement: {result.agreement_ratio:.1%}")
    print(f"  Needs Review: {result.needs_review}")
    print(f"  Vote Distribution: {result.vote_distribution}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-frame evaluation for VMEvalKit (supports GPT-4O and InternVL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with GPT-4O (default paths)
  python examples/score_multiframe.py --inference-dir ./outputs --eval-output-dir ./evaluations

  # Evaluate with InternVL (local VLM)
  python examples/score_multiframe.py --inference-dir ~/experiments/run1 --eval-output-dir ~/experiments/run1_scores --evaluator internvl

  # Custom evaluation settings
  python examples/score_multiframe.py --inference-dir ./outputs --eval-output-dir ./evaluations --n-frames 7 --temporal-weight 0.5

  # Test pipeline on single video (no API calls)
  python examples/score_multiframe.py --test-only --video path/to/video.mp4
        """
    )

    parser.add_argument(
        '--inference-dir',
        type=str,
        default='./outputs',
        help='Path to inference outputs to evaluate (default: ./outputs)'
    )
    parser.add_argument(
        '--eval-output-dir',
        type=str,
        default='./evaluations/multiframe',
        help='Path for evaluation results (default: ./evaluations/multiframe)'
    )
    parser.add_argument(
        '--evaluator',
        choices=['gpt4o', 'internvl'],
        default='gpt4o',
        help='VLM evaluator type (default: gpt4o)'
    )
    parser.add_argument(
        '--n-frames',
        type=int,
        default=5,
        help='Number of frames to sample per video (default: 5)'
    )
    parser.add_argument(
        '--last-seconds',
        type=float,
        default=3.0,
        help='Sample from last N seconds of video (default: 3.0)'
    )
    parser.add_argument(
        '--strategy',
        choices=['uniform', 'keyframe', 'hybrid'],
        default='hybrid',
        help='Frame sampling strategy (default: hybrid)'
    )
    parser.add_argument(
        '--voting',
        choices=['majority', 'weighted_majority', 'weighted_average', 'max_score', 'median'],
        default='weighted_majority',
        help='Voting method for aggregation (default: weighted_majority)'
    )
    parser.add_argument(
        '--metric',
        choices=['histogram', 'ssim', 'combined'],
        default='histogram',
        help='Similarity metric for consistency (default: histogram)'
    )
    parser.add_argument(
        '--temporal-weight',
        type=float,
        default=0.3,
        help='Temporal bias weight, 0-1 (default: 0.3, higher prefers later frames)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='VLM temperature (default: 0.0 for deterministic)'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Run pipeline test without API calls'
    )
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Video path for --test-only mode (required when --test-only is used)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_output',
        help='Output directory for --test-only mode'
    )

    args = parser.parse_args()

    if args.test_only:
        # Test mode - no API calls
        if not args.video:
            print("Error: --video is required when using --test-only")
            sys.exit(1)
        if not Path(args.video).exists():
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)
        test_multiframe_pipeline(args.video, args.output)
    else:
        # Full evaluation mode
        run_multiframe_evaluation(
            inference_dir=args.inference_dir,
            eval_output_dir=args.eval_output_dir,
            evaluator_type=args.evaluator,
            n_frames=args.n_frames,
            last_seconds=args.last_seconds,
            strategy=args.strategy,
            voting=args.voting,
            metric=args.metric,
            temporal_weight=args.temporal_weight,
            temperature=args.temperature
        )


if __name__ == "__main__":
    main()
