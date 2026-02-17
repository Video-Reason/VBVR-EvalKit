#!/usr/bin/env python3
"""
VMEvalKit Unified Video Scoring Script.

Supports all evaluation methods:
- human: Gradio-based human annotation interface
- gpt4o: Single-frame GPT-4O evaluation
- internvl: Single-frame InternVL evaluation  
- multiframe_gpt4o: Multi-frame GPT-4O with voting
- multiframe_internvl: Multi-frame InternVL with voting

Usage:
    python examples/score_videos.py --eval-config eval_config.json

    # Test multi-frame pipeline without API calls
    python examples/score_videos.py --test-multiframe --video path/to/video.mp4
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SamplingStrategy(str, Enum):
    """Frame sampling strategies for video evaluation."""
    LAST_FRAME = "last_frame"      # Single-frame evaluation (last frame)
    UNIFORM = "uniform"             # Multi-frame: uniform sampling
    KEYFRAME = "keyframe"           # Multi-frame: keyframe detection
    HYBRID = "hybrid"               # Multi-frame: hybrid strategy


class Evaluator(str, Enum):
    """VLM evaluators for video assessment."""
    GPT4O = "gpt4o"
    INTERNVL = "internvl"
    QWEN = "qwen"
    HUMAN = "human"


class VotingMethod(str, Enum):
    """Voting methods for multi-frame aggregation."""
    MAJORITY = "majority"
    WEIGHTED_MAJORITY = "weighted_majority"
    WEIGHTED_AVERAGE = "weighted_average"
    MAX_SCORE = "max_score"
    MIN_SCORE = "min_score"
    MEDIAN = "median"


class SimilarityMetric(str, Enum):
    """Similarity metrics for frame consistency analysis."""
    HISTOGRAM = "histogram"
    SSIM = "ssim"
    MSE = "mse"
    COMBINED = "combined"


class MultiFrameConfig(BaseModel):
    """Configuration for multi-frame evaluation."""
    n_frames: int = Field(default=5, ge=1, le=20, description="Number of frames to sample per video")
    last_seconds: float = Field(default=3.0, gt=0, description="Sample from last N seconds of video")
    strategy: SamplingStrategy = Field(default=SamplingStrategy.HYBRID, description="Frame sampling strategy")
    voting: VotingMethod = Field(default=VotingMethod.WEIGHTED_MAJORITY, description="Voting method for aggregation")
    metric: SimilarityMetric = Field(default=SimilarityMetric.HISTOGRAM, description="Similarity metric for consistency")
    temporal_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Temporal bias weight (0-1, higher prefers later frames)")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Below this confidence, results are flagged for review")


class EvalConfig(BaseModel):
    """Main evaluation configuration."""
    # New architecture: separate sampling strategy and evaluator
    sampling_strategy: Optional[SamplingStrategy] = Field(default=None, description="Frame sampling strategy")
    evaluator: Optional[Evaluator] = Field(default=None, description="VLM evaluator to use")
    
    inference_dir: str = Field(default="./outputs", description="Path to inference outputs to evaluate")
    eval_output_dir: str = Field(default="./evaluations", description="Path for evaluation results")
    
    # VLM settings
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="VLM temperature (0.0 = deterministic)")
    api_key: Optional[str] = Field(default=None, description="API key (falls back to env var if not set)")
    base_url: Optional[str] = Field(default=None, description="Base URL for InternVL server")
    
    # Human evaluator settings
    port: int = Field(default=7860, description="Port for Gradio interface (human evaluation)")
    share: bool = Field(default=True, description="Whether to create public Gradio link")
    
    # Multi-frame settings (only used for multiframe_* methods)
    multiframe: MultiFrameConfig = Field(default_factory=MultiFrameConfig, description="Multi-frame evaluation settings")

    @field_validator('inference_dir', 'eval_output_dir')
    @classmethod
    def expand_path(cls, v: str) -> str:
        """Expand user home directory in paths."""
        return str(Path(v).expanduser())


def run_human_evaluation(config: EvalConfig):
    """Run human evaluation with Gradio interface."""
    from vmevalkit.eval import HumanEvaluator
    
    print("\n" + "=" * 60)
    print("HUMAN EVALUATION")
    print("=" * 60)
    print(f"Inference Dir: {config.inference_dir}")
    print(f"Eval Output Dir: {config.eval_output_dir}")
    print("Tasks with existing scorings will be automatically skipped")
    
    scorer = HumanEvaluator(
        inference_dir=config.inference_dir,
        eval_output_dir=config.eval_output_dir
    )
    
    print(f"\nLaunching human scoring interface on port {config.port}...")
    print("Enter your annotator name in the interface")
    scorer.launch_interface(port=config.port, share=config.share)


def run_gpt4o_evaluation(config: EvalConfig):
    """Run single-frame GPT-4O evaluation."""
    from vmevalkit.eval import GPT4OEvaluator
    
    print("\n" + "=" * 60)
    print("GPT-4O EVALUATION")
    print("=" * 60)
    print(f"Inference Dir: {config.inference_dir}")
    print(f"Eval Output Dir: {config.eval_output_dir}")
    print(f"Temperature: {config.temperature}")
    print("Note: This will make API calls to OpenAI")
    
    api_key = config.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nError: OPENAI_API_KEY not set in config or environment!")
        sys.exit(1)
    
    scorer = GPT4OEvaluator(
        inference_dir=config.inference_dir,
        eval_output_dir=config.eval_output_dir,
        temperature=config.temperature
    )
    
    _check_existing_evaluations(config.eval_output_dir)
    _run_vlm_evaluation(scorer, "GPT-4O", config.eval_output_dir)


def run_internvl_evaluation(config: EvalConfig):
    """Run single-frame InternVL evaluation."""
    from vmevalkit.eval import InternVLEvaluator
    
    print("\n" + "=" * 60)
    print("INTERNVL EVALUATION")
    print("=" * 60)
    print(f"Inference Dir: {config.inference_dir}")
    print(f"Eval Output Dir: {config.eval_output_dir}")
    print(f"Temperature: {config.temperature}")
    
    api_key = config.api_key or os.getenv("VISION_API_KEY", "YOUR_API_KEY")
    base_url = config.base_url or os.getenv("VISION_API_BASE", "http://0.0.0.0:23333/v1")
    
    print(f"Base URL: {base_url}")
    
    if api_key == "YOUR_API_KEY":
        print("Warning: Using default API key. Set VISION_API_KEY if needed.")
    
    scorer = InternVLEvaluator(
        inference_dir=config.inference_dir,
        eval_output_dir=config.eval_output_dir,
        api_key=api_key,
        base_url=base_url,
        temperature=config.temperature
    )
    
    _check_existing_evaluations(config.eval_output_dir)
    _run_vlm_evaluation(scorer, "InternVL", config.eval_output_dir)


def run_qwen_evaluation(config: EvalConfig):
    """Run single-frame Qwen3-VL evaluation."""
    from vmevalkit.eval import Qwen3VLEvaluator
    
    print("\n" + "=" * 60)
    print("QWEN3-VL EVALUATION")
    print("=" * 60)
    print(f"Inference Dir: {config.inference_dir}")
    print(f"Eval Output Dir: {config.eval_output_dir}")
    print(f"Temperature: {config.temperature}")
    
    api_key = config.api_key or os.getenv("QWEN_API_KEY", "EMPTY")
    base_url = config.base_url or os.getenv("QWEN_API_BASE", "http://localhost:8000/v1")
    
    print(f"Base URL: {base_url}")
    
    if api_key == "EMPTY":
        print("Warning: Using default API key. Set QWEN_API_KEY if needed.")
    
    scorer = Qwen3VLEvaluator(
        inference_dir=config.inference_dir,
        eval_output_dir=config.eval_output_dir,
        api_key=api_key,
        base_url=base_url,
        temperature=config.temperature
    )
    
    _check_existing_evaluations(config.eval_output_dir)
    _run_vlm_evaluation(scorer, "Qwen3-VL", config.eval_output_dir)


def run_multiframe_evaluation(config: EvalConfig, evaluator_type: str):
    """Run multi-frame evaluation with voting aggregation."""
    from vmevalkit.eval import MultiFrameEvaluator, GPT4OEvaluator, InternVLEvaluator, Qwen3VLEvaluator
    
    mf = config.multiframe
    
    print("\n" + "=" * 60)
    print(f"MULTI-FRAME {evaluator_type.upper()} EVALUATION")
    print("=" * 60)
    print(f"Inference Dir: {config.inference_dir}")
    print(f"Eval Output Dir: {config.eval_output_dir}")
    print(f"Config:")
    print(f"  - evaluator: {evaluator_type}")
    print(f"  - n_frames: {mf.n_frames}")
    print(f"  - last_seconds: {mf.last_seconds}")
    print(f"  - strategy: {mf.strategy.value}")
    print(f"  - voting: {mf.voting.value}")
    print(f"  - metric: {mf.metric.value}")
    print(f"  - temporal_weight: {mf.temporal_weight}")
    print(f"  - temperature: {config.temperature}")
    
    # Create base evaluator
    if evaluator_type == "gpt4o":
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("\nError: OPENAI_API_KEY not set in config or environment!")
            sys.exit(1)
        base_evaluator = GPT4OEvaluator(
            inference_dir=config.inference_dir,
            eval_output_dir=config.eval_output_dir,
            temperature=config.temperature
        )
    elif evaluator_type == "internvl":
        api_key = config.api_key or os.getenv("VISION_API_KEY", "YOUR_API_KEY")
        base_url = config.base_url or os.getenv("VISION_API_BASE", "http://0.0.0.0:23333/v1")
        print(f"  - base_url: {base_url}")
        base_evaluator = InternVLEvaluator(
            inference_dir=config.inference_dir,
            eval_output_dir=config.eval_output_dir,
            api_key=api_key,
            base_url=base_url,
            temperature=config.temperature
        )
    elif evaluator_type == "qwen":
        api_key = config.api_key or os.getenv("QWEN_API_KEY", "EMPTY")
        base_url = config.base_url or os.getenv("QWEN_API_BASE", "http://localhost:8000/v1")
        print(f"  - base_url: {base_url}")
        base_evaluator = Qwen3VLEvaluator(
            inference_dir=config.inference_dir,
            eval_output_dir=config.eval_output_dir,
            api_key=api_key,
            base_url=base_url,
            temperature=config.temperature
        )
    else:
        print(f"\nError: Unknown evaluator type: {evaluator_type}")
        sys.exit(1)
    
    # Initialize multi-frame evaluator
    evaluator = MultiFrameEvaluator(
        base_evaluator=base_evaluator,
        output_dir=config.eval_output_dir,
        n_frames=mf.n_frames,
        last_seconds=mf.last_seconds,
        sampling_strategy=mf.strategy.value,
        voting_method=mf.voting.value,
        similarity_metric=mf.metric.value,
        temporal_weight=mf.temporal_weight,
        confidence_threshold=mf.confidence_threshold
    )
    
    # Check for existing evaluations
    eval_dir = Path(config.eval_output_dir)
    if eval_dir.exists():
        existing_files = list(eval_dir.rglob(f"{evaluator.evaluator_name}.json"))
        if existing_files:
            print(f"\nFound {len(existing_files)} existing evaluations - will resume from where left off")
    
    print("\nStarting evaluation...")
    print("Tip: You can interrupt (Ctrl+C) and resume later - progress is saved after each task")
    print("=" * 60)
    
    _run_multiframe_evaluation_loop(evaluator, config.eval_output_dir)


def _check_existing_evaluations(eval_output_dir: str):
    """Check and report existing evaluation files."""
    eval_dir = Path(eval_output_dir)
    if eval_dir.exists():
        existing_files = list(eval_dir.rglob("*.json"))
        if existing_files:
            print(f"\nFound {len(existing_files)} existing scorings - will resume from where left off")


def _count_results_from_model_data(model_data: dict) -> tuple:
    """Count total and evaluated tasks from model result data.

    Supports two return formats:
    - Enhanced format: model_data["tasks"][task_type]["samples"][task_id]
    - Legacy format: model_data["evaluations"][task_type][task_id]

    Returns:
        Tuple of (total_tasks, evaluated_tasks)
    """
    total_tasks = 0
    evaluated_tasks = 0

    tasks = model_data.get("tasks", {})
    if tasks:
        for task_data in tasks.values():
            for sample_data in task_data.get("samples", {}).values():
                total_tasks += 1
                if "error" not in sample_data and sample_data.get("status") != "failed":
                    evaluated_tasks += 1
    else:
        for task_samples in model_data.get("evaluations", {}).values():
            for result in task_samples.values():
                total_tasks += 1
                if "error" not in result and result.get("status") != "failed":
                    evaluated_tasks += 1

    return total_tasks, evaluated_tasks


def _run_vlm_evaluation(scorer, name: str, eval_output_dir: str):
    """Run VLM evaluation with progress tracking."""
    print(f"\nStarting {name} scoring...")
    print("Tip: You can interrupt (Ctrl+C) and resume later - progress is saved after each task")
    print("=" * 60)

    try:
        all_results = scorer.evaluate_all_models()

        print(f"\n{name} EVALUATION RESULTS:")
        total_all = 0
        completed_all = 0

        # Enhanced format nests models under "models" key; legacy format is flat
        models_data = all_results.get("models", all_results)

        for model_name, model_data in models_data.items():
            total_tasks, evaluated_tasks = _count_results_from_model_data(model_data)
            total_all += total_tasks
            completed_all += evaluated_tasks

            status = "Complete" if evaluated_tasks == total_tasks else f"{evaluated_tasks}/{total_tasks}"
            print(f"  {model_name}: {status}")

        print(f"\nEVALUATION COMPLETE!")
        print(f"Total: {completed_all}/{total_all} tasks evaluated successfully")
        print(f"Results saved to: {eval_output_dir}")

    except KeyboardInterrupt:
        print(f"\n{name} scoring interrupted!")
        print("Progress has been saved. Run the same command again to resume.")
        print(f"Partial results available in: {eval_output_dir}")


def _run_multiframe_evaluation_loop(evaluator, eval_output_dir: str):
    """Run multi-frame evaluation with progress tracking."""
    try:
        all_results = evaluator.evaluate_all_models()
        
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
    from vmevalkit.eval.voting import VotingAggregator, VotingMethod as VMethod, FrameScore
    from PIL import Image

    print("\n" + "=" * 60)
    print("MULTI-FRAME PIPELINE TEST (No API calls)")
    print("=" * 60)
    print(f"Video: {video_path}")

    # Initialize components
    sampler = FrameSampler(n_frames=5, last_seconds=3.0)
    analyzer = FrameConsistencyAnalyzer(metric="histogram", temporal_weight=0.3)
    voter = VotingAggregator(method=VMethod.WEIGHTED_MAJORITY)

    # Get video info
    info = sampler.get_video_info(video_path)
    print(f"\nVideo Info:")
    print(f"  - Duration: {info['duration']:.2f}s")
    print(f"  - Frames: {info['total_frames']}")
    print(f"  - FPS: {info['fps']:.2f}")
    print(f"  - Resolution: {info['width']}x{info['height']}")

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
    mock_scores = [4, 4, 3, 4, 5]  # Simulated VLM scores
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
        description="VMEvalKit Unified Video Scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with config file
  python examples/score_videos.py --eval-config eval_config.json

  # Test multi-frame pipeline (no API calls)
  python examples/score_videos.py --test-multiframe --video path/to/video.mp4

Config file example (eval_config.json):
{
    "method": "multiframe_gpt4o",
    "inference_dir": "./outputs",
    "eval_output_dir": "./evaluations",
    "temperature": 0.0,
    "multiframe": {
        "n_frames": 5,
        "last_seconds": 3.0,
        "strategy": "hybrid",
        "voting": "weighted_majority",
        "metric": "histogram",
        "temporal_weight": 0.3
    }
}

Available methods:
  - human              : Gradio-based human annotation
  - gpt4o              : Single-frame GPT-4O evaluation
  - internvl           : Single-frame InternVL evaluation
  - multiframe_gpt4o   : Multi-frame GPT-4O with voting
  - multiframe_internvl: Multi-frame InternVL with voting
        """
    )
    
    parser.add_argument(
        '--eval-config',
        type=str,
        default=None,
        help='Path to evaluation config JSON file'
    )
    parser.add_argument(
        '--evaluator',
        type=str,
        default=None,
        choices=['gpt4o', 'internvl', 'qwen'],
        help='VLM evaluator to use (overrides config file if specified)'
    )
    parser.add_argument(
        '--test-multiframe',
        action='store_true',
        help='Test multi-frame pipeline without API calls'
    )
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Video path for --test-multiframe mode'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_output',
        help='Output directory for --test-multiframe mode'
    )
    
    args = parser.parse_args()
    
    # Test multi-frame pipeline
    if args.test_multiframe:
        if not args.video:
            print("Error: --video is required when using --test-multiframe")
            sys.exit(1)
        if not Path(args.video).exists():
            print(f"Error: Video not found: {args.video}")
            sys.exit(1)
        test_multiframe_pipeline(args.video, args.output)
        return
    
    # Main evaluation mode
    if not args.eval_config:
        print("Error: --eval-config is required for evaluation")
        print("Use --help for more options")
        sys.exit(1)
    
    config_path = Path(args.eval_config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load and validate config
    with open(config_path) as f:
        config_dict = json.load(f)
    
    config = EvalConfig(**config_dict)
    
    # New architecture: Infer sampling strategy and evaluator
    logger.info("=== Configuration Processing ===")
    
    # 1. Determine sampling strategy
    if config.sampling_strategy:
        # Explicitly specified in config
        sampling_strategy = config.sampling_strategy
        logger.info(f"Sampling strategy from config: {sampling_strategy.value}")
    elif 'multiframe' in config_dict:
        # Multi-frame configuration detected
        multiframe_strategy = config_dict['multiframe'].get('strategy', 'uniform')
        sampling_strategy = SamplingStrategy(multiframe_strategy)
        logger.info(f"Inferred multi-frame sampling strategy: {sampling_strategy.value}")
    else:
        # Default to single-frame (last frame)
        sampling_strategy = SamplingStrategy.LAST_FRAME
        logger.info("Defaulting to last-frame sampling strategy")
    
    # 2. Determine evaluator
    if args.evaluator:
        # Command line override
        try:
            evaluator = Evaluator(args.evaluator)
            logger.info(f"Evaluator from command line: {evaluator.value}")
        except ValueError:
            print(f"Error: Unknown evaluator: {args.evaluator}")
            print(f"  Available: {[e.value for e in Evaluator]}")
            sys.exit(1)
    elif config.evaluator:
        # Explicitly specified in config
        evaluator = config.evaluator
        logger.info(f"Evaluator from config: {evaluator.value}")
    else:
        print("Error: No evaluator specified!")
        print("  Please specify 'evaluator' in config file OR use --evaluator flag")
        print("  Example: python score_videos.py --eval-config config.json --evaluator gpt4o")
        sys.exit(1)
    
    # 3. Update config with resolved values
    config.sampling_strategy = sampling_strategy
    config.evaluator = evaluator
    
    # 4. Resolve dynamic paths
    eval_method_name = config_path.stem  # e.g., 'last_frame'
    
    replacements = {
        '{evaluator}': evaluator.value,
        '{method}': eval_method_name,
        '{strategy}': sampling_strategy.value,
    }
    
    for placeholder, value in replacements.items():
        if placeholder in config.eval_output_dir:
            config.eval_output_dir = config.eval_output_dir.replace(placeholder, value)
            logger.info(f"Replaced {placeholder} with {value} in output path")
    
    logger.info(f"Final configuration:")
    logger.info(f"  Sampling Strategy: {sampling_strategy.value}")
    logger.info(f"  Evaluator: {evaluator.value}")
    logger.info(f"  Output Directory: {config.eval_output_dir}")
    logger.info("=" * 40)
    
    # Check inference directory
    inference_path = Path(config.inference_dir)
    if not inference_path.exists():
        print(f"Error: Inference directory not found: {inference_path}")
        print("Please run inference first to generate videos.")
        sys.exit(1)
    
    # Run evaluation based on sampling strategy and evaluator
    logger.info(f"Starting evaluation: {sampling_strategy.value} + {evaluator.value}")
    
    if sampling_strategy == SamplingStrategy.LAST_FRAME:
        # Single-frame evaluation
        if evaluator == Evaluator.HUMAN:
            run_human_evaluation(config)
        elif evaluator == Evaluator.GPT4O:
            run_gpt4o_evaluation(config)
        elif evaluator == Evaluator.INTERNVL:
            run_internvl_evaluation(config)
        elif evaluator == Evaluator.QWEN:
            run_qwen_evaluation(config)
    
    else:
        # Multi-frame evaluation (UNIFORM, KEYFRAME, HYBRID)
        if evaluator == Evaluator.HUMAN:
            print("Error: Human evaluation is not supported for multi-frame sampling")
            sys.exit(1)
        run_multiframe_evaluation(config, evaluator.value)


if __name__ == "__main__":
    main()
