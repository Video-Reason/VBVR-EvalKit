"""Scoring runner for VBVR-EvalKit.

Runs VBVR-Bench rule-based evaluation on generated videos.
"""

import argparse
import logging
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('scoring.log')
    ]
)
logger = logging.getLogger(__name__)


def run_rubrics_scoring(
    inference_dir: str,
    eval_output_dir: str = "./evaluations/rubrics",
    gt_base_path: Optional[str] = None,
    device: str = "cuda",
    task_specific_only: bool = True,
):
    """Run VBVR-Bench rule-based evaluation (no API calls needed).

    Args:
        inference_dir: Directory containing inference outputs to score.
        eval_output_dir: Directory to save scoring results.
        gt_base_path: Optional path to VBVR-Bench GT data (for ground_truth.mp4).
        device: Device for computation ('cuda' or 'cpu').
        task_specific_only: If True, use only task-specific dimension score.
    """
    from vbvrevalkit.eval.vbvr_bench_eval import VBVRBenchEvaluator

    logger.info(f"Starting VBVR-Bench rubrics scoring for: {inference_dir}")
    logger.info(f"Device: {device}, task_specific_only: {task_specific_only}")

    scorer = VBVRBenchEvaluator(
        inference_dir=inference_dir,
        eval_output_dir=eval_output_dir,
        gt_base_path=gt_base_path,
        device=device,
        task_specific_only=task_specific_only,
    )

    results = scorer.evaluate_all_models()
    logger.info("Completed rubrics scoring for all models")

    for model_name, model_data in results.get("models", {}).items():
        stats = model_data.get("model_statistics", {})
        logger.info(f"\n{model_name}:")
        logger.info(f"  Total samples: {stats.get('total_samples', 0)}")
        logger.info(f"  Mean score: {stats.get('mean_score', 'N/A')}")

        by_split = model_data.get("by_split", {})
        for split_name, split_data in by_split.items():
            logger.info(f"  {split_name}: {split_data.get('mean_score', 'N/A')} ({split_data.get('num_samples', 0)} samples)")

        by_cat = model_data.get("by_category", {})
        for cat, cat_data in by_cat.items():
            logger.info(f"  {cat}: {cat_data.get('mean_score', 'N/A')} ({cat_data.get('num_samples', 0)} samples)")


def main():
    """Main entry point for scoring runner."""
    parser = argparse.ArgumentParser(
        description="VBVR-Bench Rule-Based Video Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python -m vbvrevalkit.runner.score --inference-dir ./outputs

  # With GT data and full 5-dimension score
  python -m vbvrevalkit.runner.score --inference-dir ./outputs --gt-base-path /path/to/gt --full-score

  # CPU mode with custom output directory
  python -m vbvrevalkit.runner.score --inference-dir ./outputs --device cpu --eval-output-dir ./my_evals

Scoring Dimensions (with --full-score):
  first_frame_consistency (15%): First frame alignment
  final_frame_accuracy    (35%): Final frame correctness
  temporal_smoothness     (15%): Temporal coherence
  visual_quality          (10%): Visual fidelity
  task_specific           (25%): Task-specific reasoning logic

Default mode returns only the task_specific dimension score.
        """
    )

    parser.add_argument(
        '--inference-dir', '-i',
        type=str,
        required=True,
        help='Directory containing inference outputs to score'
    )
    parser.add_argument(
        '--eval-output-dir', '-o',
        type=str,
        default='./evaluations/rubrics',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--gt-base-path', '-g',
        type=str,
        default=None,
        help='Path to VBVR-Bench GT data (for ground_truth.mp4, optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device for computation'
    )
    parser.add_argument(
        '--full-score',
        action='store_true',
        help='Use full 5-dimension weighted score (default: task_specific only)'
    )

    args = parser.parse_args()

    run_rubrics_scoring(
        inference_dir=args.inference_dir,
        eval_output_dir=args.eval_output_dir,
        gt_base_path=args.gt_base_path,
        device=args.device,
        task_specific_only=not args.full_score,
    )


if __name__ == "__main__":
    main()
