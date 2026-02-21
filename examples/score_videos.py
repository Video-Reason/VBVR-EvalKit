#!/usr/bin/env python3
"""
VBVR-EvalKit Video Scoring Script.

Runs VBVR-Bench rule-based evaluation on generated videos using 100+ task-specific
evaluators. Produces deterministic, reproducible 0-1 continuous scores with no API
calls required.

Usage:
    # Basic
    python examples/score_videos.py --inference-dir ./outputs

    # With GT data and full 5-dimension scoring
    python examples/score_videos.py --inference-dir ./outputs --gt-base-path /path/to/gt --full-score

    # CPU mode with custom output directory
    python examples/score_videos.py --inference-dir ./outputs --device cpu --eval-output-dir ./my_evals
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_rubrics_evaluation(
    inference_dir: str,
    eval_output_dir: str = "./evaluations/rubrics",
    gt_base_path: str = None,
    device: str = "cuda",
    task_specific_only: bool = True,
):
    """Run VBVR-Bench rule-based evaluation.

    Args:
        inference_dir: Directory containing inference outputs to evaluate.
        eval_output_dir: Directory to save evaluation results.
        gt_base_path: Optional path to VBVR-Bench GT data (for ground_truth.mp4).
        device: Computation device ('cuda' or 'cpu').
        task_specific_only: If True, score only the task-specific dimension.
    """
    from vbvrevalkit.eval.vbvr_bench_eval import VBVRBenchEvaluator

    print("\n" + "=" * 60)
    print("VBVR-BENCH EVALUATION")
    print("=" * 60)
    print(f"Inference Dir : {inference_dir}")
    print(f"Output Dir    : {eval_output_dir}")
    print(f"Device        : {device}")
    print(f"Scoring Mode  : {'task_specific only' if task_specific_only else 'full 5-dimension weighted'}")
    if gt_base_path:
        print(f"GT Base Path  : {gt_base_path}")
    print("=" * 60)

    scorer = VBVRBenchEvaluator(
        inference_dir=inference_dir,
        eval_output_dir=eval_output_dir,
        gt_base_path=gt_base_path,
        device=device,
        task_specific_only=task_specific_only,
    )

    print("\nStarting evaluation...")
    print("Tip: You can interrupt (Ctrl+C) and resume later — progress is saved after each task")
    print("-" * 60)

    results = scorer.evaluate_all_models()

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for model_name, model_data in results.get("models", {}).items():
        stats = model_data.get("model_statistics", {})
        print(f"\n  {model_name}:")
        print(f"    Samples : {stats.get('total_samples', 0)}")
        print(f"    Mean    : {stats.get('mean_score', 'N/A')}")

        by_split = model_data.get("by_split", {})
        for split_name, split_data in by_split.items():
            print(f"    {split_name}: {split_data.get('mean_score', 'N/A')} ({split_data.get('num_samples', 0)} samples)")

        by_cat = model_data.get("by_category", {})
        if by_cat:
            print("    Categories:")
            for cat, cat_data in by_cat.items():
                print(f"      {cat}: {cat_data.get('mean_score', 'N/A')} ({cat_data.get('num_samples', 0)} samples)")

    global_stats = results.get("global_statistics", {})
    print(f"\nGlobal: {global_stats.get('total_samples', 0)} samples, mean={global_stats.get('mean_score', 'N/A')}")
    print(f"Results saved to: {eval_output_dir}/")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="VBVR-Bench Rule-Based Video Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python examples/score_videos.py --inference-dir ./outputs

  # With GT data and device selection
  python examples/score_videos.py --inference-dir ./outputs --gt-base-path /path/to/gt --device cuda

  # Full 5-dimension weighted score (default: task_specific only)
  python examples/score_videos.py --inference-dir ./outputs --full-score

  # Custom output directory
  python examples/score_videos.py --inference-dir ./outputs --eval-output-dir ./evaluations/rubrics

Scoring Dimensions (with --full-score):
  first_frame_consistency (15%%): First frame alignment with GT
  final_frame_accuracy    (35%%): Final frame correctness
  temporal_smoothness     (15%%): Temporal coherence
  visual_quality          (10%%): Visual fidelity
  task_specific           (25%%): Task-specific reasoning logic

Default mode (without --full-score) returns only the task_specific dimension,
focusing on reasoning correctness.
        """
    )

    parser.add_argument(
        '--inference-dir', '-i',
        type=str,
        required=True,
        help='Directory containing inference outputs to evaluate'
    )
    parser.add_argument(
        '--eval-output-dir', '-o',
        type=str,
        default='./evaluations/rubrics',
        help='Directory to save evaluation results (default: ./evaluations/rubrics)'
    )
    parser.add_argument(
        '--gt-base-path', '-g',
        type=str,
        default=None,
        help='Path to VBVR-Bench GT data (optional, for ground_truth.mp4)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Computation device (default: cuda)'
    )
    parser.add_argument(
        '--full-score',
        action='store_true',
        help='Use full 5-dimension weighted score instead of task_specific only'
    )

    args = parser.parse_args()

    inference_path = Path(args.inference_dir)
    if not inference_path.exists():
        print(f"Error: Inference directory not found: {inference_path}")
        print("Please run inference first to generate videos.")
        sys.exit(1)

    run_rubrics_evaluation(
        inference_dir=args.inference_dir,
        eval_output_dir=args.eval_output_dir,
        gt_base_path=args.gt_base_path,
        device=args.device,
        task_specific_only=not args.full_score,
    )


if __name__ == "__main__":
    main()
