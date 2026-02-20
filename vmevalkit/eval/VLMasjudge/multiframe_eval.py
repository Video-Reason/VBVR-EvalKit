"""
Generic Multi-frame evaluation for VMEvalKit.

This module provides a base multi-frame evaluator that can wrap any single-frame
evaluator (GPT4OEvaluator, InternVLEvaluator, etc.) and add multi-frame capabilities:
- Samples multiple frames from video tail using hybrid strategy
- Analyzes frame-to-frame consistency for stability detection
- Evaluates each frame independently using the base evaluator
- Aggregates scores via weighted voting

Usage:
    # With GPT-4O
    from vmevalkit.eval import MultiFrameEvaluator, GPT4OEvaluator
    base = GPT4OEvaluator(inference_dir="./outputs", eval_output_dir="./evaluations/gpt4o")
    evaluator = MultiFrameEvaluator(base_evaluator=base, n_frames=5)

    # With InternVL
    from vmevalkit.eval import MultiFrameEvaluator, InternVLEvaluator
    base = InternVLEvaluator(inference_dir="./outputs", eval_output_dir="./evaluations/internvl")
    evaluator = MultiFrameEvaluator(base_evaluator=base, n_frames=5)
"""

import json
import logging
import asyncio
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Protocol
from datetime import datetime
from abc import ABC, abstractmethod

import numpy as np

from .frame_sampler import FrameSampler, SampledFrame
from .consistency import FrameConsistencyAnalyzer, ConsistencyResult
from .voting import VotingAggregator, VotingMethod, FrameScore, VotingResult
from .run_selector import select_latest_run

logger = logging.getLogger(__name__)


class BaseEvaluatorProtocol(Protocol):
    """Protocol that base evaluators must implement."""

    def encode_image(self, image: Union[np.ndarray, str]) -> str:
        """Encode image to base64."""
        ...

    def create_prompt(self, task_type: str) -> str:
        """Create evaluation prompt for task type."""
        ...

    async def call_gpt4o(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call the VLM API (GPT-4O compatible)."""
        ...


class MultiFrameEvaluator:
    """
    Generic multi-frame evaluation with voting aggregation.

    This evaluator wraps any single-frame evaluator (GPT4OEvaluator, InternVLEvaluator)
    and adds multi-frame capabilities. It samples multiple frames from a video,
    evaluates each frame independently, then aggregates the results using weighted voting.

    Attributes:
        base_evaluator: Underlying single-frame evaluator
        sampler: Frame sampling component
        consistency_analyzer: Frame consistency analysis component
        voting_aggregator: Score aggregation component
        n_frames: Number of frames to sample per video

    Example:
        >>> from vmevalkit.eval import GPT4OEvaluator, InternVLEvaluator
        >>>
        >>> # With GPT-4O
        >>> base = GPT4OEvaluator(inference_dir="./outputs", eval_output_dir="./evaluations/gpt4o")
        >>> evaluator = MultiFrameEvaluator(base_evaluator=base, n_frames=5)
        >>>
        >>> # With InternVL
        >>> base = InternVLEvaluator(inference_dir="./outputs", eval_output_dir="./evaluations/internvl")
        >>> evaluator = MultiFrameEvaluator(base_evaluator=base, n_frames=5)
        >>>
        >>> result = evaluator.evaluate_single(
        ...     model_name="luma-ray-2",
        ...     task_type="maze_task",
        ...     task_id="maze_001",
        ...     video_path="output.mp4"
        ... )
    """

    def __init__(
        self,
        base_evaluator: Any,
        output_dir: Optional[str] = None,
        n_frames: int = 5,
        last_seconds: float = 3.0,
        sampling_strategy: str = "hybrid",
        similarity_metric: str = "histogram",
        temporal_weight: float = 0.3,
        voting_method: str = "weighted_majority",
        confidence_threshold: float = 0.6
    ):
        """
        Initialize generic multi-frame evaluator.

        Args:
            base_evaluator: Any evaluator with encode_image, create_prompt methods
                           (GPT4OEvaluator, InternVLEvaluator, etc.)
            output_dir: Directory to save evaluation results (default: uses base_evaluator's)
            n_frames: Number of frames to sample from each video
            last_seconds: Sample frames from the last N seconds of video
            sampling_strategy: "uniform", "keyframe", or "hybrid" (recommended)
            similarity_metric: "histogram", "ssim", "mse", or "combined"
            temporal_weight: Weight for temporal bias (0-1, higher = prefer later frames)
            voting_method: "majority", "weighted_majority", "weighted_average", etc.
            confidence_threshold: Results below this confidence are flagged for review
        """
        self.base_evaluator = base_evaluator

        # Use base evaluator's name directly (evaluation method is indicated by directory path)
        self.evaluator_name = base_evaluator.__class__.__name__

        # Multi-frame configuration
        self.n_frames = n_frames
        self.last_seconds = last_seconds
        self.sampling_strategy = sampling_strategy
        self.similarity_metric = similarity_metric
        self.temporal_weight = temporal_weight
        self.voting_method = voting_method
        self.confidence_threshold = confidence_threshold

        # Initialize multi-frame components
        self.sampler = FrameSampler(
            n_frames=n_frames,
            last_seconds=last_seconds
        )

        self.consistency_analyzer = FrameConsistencyAnalyzer(
            metric=similarity_metric,
            temporal_weight=temporal_weight
        )

        self.voting_aggregator = VotingAggregator(
            method=VotingMethod(voting_method),
            confidence_threshold=confidence_threshold
        )

        # Output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Create parallel directory structure
            base_output = Path(base_evaluator.eval_output_dir)
            self.output_dir = base_output.parent / f"multiframe-{base_output.name}"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inference_dir = base_evaluator.inference_dir

    def _has_evaluation(self, model_name: str, task_type: str, task_id: str) -> bool:
        """Check if task has already been evaluated with multi-frame method."""
        eval_path = self.output_dir / model_name / task_type / task_id
        eval_file = eval_path / f"{self.evaluator_name}.json"
        return eval_file.exists()

    async def _call_vlm(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Call the VLM API using the base evaluator's method."""
        # Try different API call methods based on evaluator type
        if hasattr(self.base_evaluator, 'call_gpt4o'):
            return await self.base_evaluator.call_gpt4o(messages)
        elif hasattr(self.base_evaluator, 'call_vlm'):
            return await self.base_evaluator.call_vlm(messages)
        else:
            raise NotImplementedError(
                f"Base evaluator {self.base_evaluator.__class__.__name__} "
                "must have either call_gpt4o() or call_vlm() method"
            )

    async def evaluate_frame_async(
        self,
        frame: np.ndarray,
        task_type: str,
        first_frame_path: Path,
        final_frame_path: Path,
        prompt_text: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single frame using the base evaluator's VLM.

        Args:
            frame: RGB numpy array of the frame to evaluate
            task_type: Type of task for prompt generation
            first_frame_path: Path to input/first frame image
            final_frame_path: Path to expected final frame image
            prompt_text: Task prompt text

        Returns:
            Evaluation result dictionary with score and explanation
        """
        # Encode images
        frame_b64 = self.base_evaluator.encode_image(frame)
        first_frame_b64 = self.base_evaluator.encode_image(str(first_frame_path))
        final_frame_b64 = self.base_evaluator.encode_image(str(final_frame_path))

        messages = [
            {"role": "system", "content": self.base_evaluator.create_prompt(task_type)},
            {"role": "user", "content": [
                {"type": "text", "text": f"Task: {task_type}\nPrompt: {prompt_text}\n\n1. Input image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{first_frame_b64}"}},
                {"type": "text", "text": "\n2. Expected final frame:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{final_frame_b64}"}},
                {"type": "text", "text": "\n3. Actual frame from video:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_b64}"}},
                {"type": "text", "text": "\nProvide your evaluation."}
            ]}
        ]

        response = await self._call_vlm(messages)
        content = response["choices"][0]["message"]["content"]

        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            eval_data = json.loads(json_match.group())
            return {
                "solution_correctness_score": eval_data.get("solution_correctness_score", 0),
                "explanation": eval_data.get("explanation", ""),
                "status": "completed"
            }

        raise ValueError("Could not parse JSON from VLM response")

    async def evaluate_single_async(
        self,
        model_name: str,
        task_type: str,
        task_id: str,
        video_path: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single video using multi-frame sampling and voting.

        Pipeline:
        1. Sample N frames from video tail using configured strategy
        2. Analyze frame consistency to compute weights
        3. Evaluate each frame independently with the base evaluator
        4. Aggregate scores via weighted voting

        Args:
            model_name: Name of the model being evaluated
            task_type: Type of task (maze_task, sudoku_task, etc.)
            task_id: Unique task identifier
            video_path: Path to the generated video file

        Returns:
            Comprehensive result dictionary containing:
            - final_score: Aggregated score from voting
            - confidence: Voting confidence (0-1)
            - agreement_ratio: Fraction of frames agreeing with final score
            - stability_score: Frame consistency score (0-1)
            - needs_review: Whether human review is recommended
            - frame_results: Per-frame evaluation details
        """
        # Get paths to ground truth
        task_dir = Path(video_path).parent.parent
        first_frame_path = task_dir / "question" / "first_frame.png"
        final_frame_path = task_dir / "question" / "final_frame.png"
        prompt_path = task_dir / "question" / "prompt.txt"

        if not final_frame_path.exists():
            return {
                "error": "No ground truth final frame available",
                "status": "skipped"
            }

        prompt_text = prompt_path.read_text() if prompt_path.exists() else ""

        # Step 1: Sample frames
        try:
            sampled_frames = self.sampler.sample(
                video_path,
                n_frames=self.n_frames,
                last_seconds=self.last_seconds,
                strategy=self.sampling_strategy
            )
        except Exception as e:
            logger.error(f"Frame sampling failed: {e}")
            return {"error": f"Frame sampling failed: {e}", "status": "failed"}

        if not sampled_frames:
            return {"error": "No frames could be sampled", "status": "failed"}

        # Step 2: Analyze frame consistency
        frame_images = [f.image for f in sampled_frames]
        consistency_result = self.consistency_analyzer.analyze(frame_images)

        # Step 3: Evaluate each frame
        frame_scores: List[FrameScore] = []
        frame_results: List[Dict[str, Any]] = []

        for i, (sampled_frame, weight) in enumerate(zip(sampled_frames, consistency_result.weights)):
            try:
                logger.debug(f"Evaluating frame {i+1}/{len(sampled_frames)} at t={sampled_frame.timestamp:.2f}s")

                eval_result = await self.evaluate_frame_async(
                    frame=sampled_frame.image,
                    task_type=task_type,
                    first_frame_path=first_frame_path,
                    final_frame_path=final_frame_path,
                    prompt_text=prompt_text
                )

                score = eval_result.get("solution_correctness_score", 0)

                frame_scores.append(FrameScore(
                    score=score,
                    timestamp=sampled_frame.timestamp,
                    weight=weight,
                    is_keyframe=sampled_frame.is_keyframe,
                    explanation=eval_result.get("explanation")
                ))

                frame_results.append({
                    "frame_index": sampled_frame.frame_index,
                    "timestamp": sampled_frame.timestamp,
                    "is_keyframe": sampled_frame.is_keyframe,
                    "weight": weight,
                    "score": score,
                    "explanation": eval_result.get("explanation", ""),
                    "status": "completed"
                })

            except Exception as e:
                logger.warning(f"Failed to evaluate frame {i}: {e}")
                frame_results.append({
                    "frame_index": sampled_frame.frame_index,
                    "timestamp": sampled_frame.timestamp,
                    "is_keyframe": sampled_frame.is_keyframe,
                    "weight": weight,
                    "error": str(e),
                    "status": "failed"
                })

        if not frame_scores:
            return {
                "error": "All frame evaluations failed",
                "status": "failed",
                "frame_results": frame_results,
                "stability_score": consistency_result.stability_score
            }

        # Step 4: Aggregate votes
        voting_result = self.voting_aggregator.aggregate(
            frame_scores,
            stability_score=consistency_result.stability_score
        )

        return {
            "final_score": voting_result.final_score,
            "solution_correctness_score": voting_result.final_score,  # Compatibility
            "confidence": voting_result.confidence,
            "agreement_ratio": voting_result.agreement_ratio,
            "stability_score": voting_result.stability_score,
            "needs_review": voting_result.needs_review,
            "voting_method": voting_result.method_used,
            "vote_distribution": voting_result.vote_distribution,
            "frame_results": frame_results,
            "consistency": {
                "mean_similarity": consistency_result.mean_similarity,
                "std_similarity": consistency_result.std_similarity,
                "outlier_indices": consistency_result.outlier_indices
            },
            "config": {
                "n_frames": self.n_frames,
                "last_seconds": self.last_seconds,
                "sampling_strategy": self.sampling_strategy,
                "similarity_metric": self.similarity_metric,
                "temporal_weight": self.temporal_weight,
                "voting_method": self.voting_method,
                "base_evaluator": self.base_evaluator.__class__.__name__
            },
            "evaluation_type": "multi_frame_voting",
            "status": "completed"
        }

    def evaluate_single(
        self,
        model_name: str,
        task_type: str,
        task_id: str,
        video_path: str
    ) -> Dict[str, Any]:
        """Evaluate a single video (sync wrapper)."""
        return asyncio.run(self.evaluate_single_async(model_name, task_type, task_id, video_path))

    async def evaluate_model_async(self, model_name: str) -> Dict[str, Any]:
        """Evaluate all tasks for a specific model using multi-frame method."""
        model_dir = self.inference_dir / model_name
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")

        results = {"model_name": model_name, "evaluations": {}}
        total_tasks = 0
        skipped_tasks = 0
        evaluated_tasks = 0
        failed_tasks = 0

        for task_type_dir in model_dir.iterdir():
            if not task_type_dir.is_dir():
                continue
            task_type = task_type_dir.name
            results["evaluations"][task_type] = {}

            for task_dir in task_type_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                task_id = task_dir.name
                total_tasks += 1

                # Check if already evaluated (RESUME MECHANISM)
                if self._has_evaluation(model_name, task_type, task_id):
                    logger.debug(f"Skipping {model_name}/{task_type}/{task_id} - already evaluated")
                    skipped_tasks += 1
                    continue

                run_dir = select_latest_run(task_dir)
                if not run_dir:
                    continue

                video_files = sorted((run_dir / "video").glob("*.mp4"))
                if not video_files:
                    continue

                try:
                    logger.info(f"Multi-frame evaluating {model_name}/{task_type}/{task_id}")
                    eval_result = await self.evaluate_single_async(
                        model_name, task_type, task_id, str(video_files[0])
                    )
                    results["evaluations"][task_type][task_id] = eval_result

                    # Save immediately (RESUME SUPPORT)
                    self._save_single_result(model_name, task_type, task_id, eval_result)
                    evaluated_tasks += 1

                except Exception as e:
                    logger.error(f"Error evaluating {model_name}/{task_type}/{task_id}: {e}")
                    results["evaluations"][task_type][task_id] = {"error": str(e), "status": "failed"}
                    failed_tasks += 1

        logger.info(f"Multi-Frame Evaluation Summary for {model_name}:")
        logger.info(f"  - Total tasks: {total_tasks}")
        logger.info(f"  - Already completed (skipped): {skipped_tasks}")
        logger.info(f"  - Newly evaluated: {evaluated_tasks}")
        logger.info(f"  - Failed: {failed_tasks}")

        return results

    def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """Evaluate all tasks for a model (sync wrapper)."""
        return asyncio.run(self.evaluate_model_async(model_name))

    async def evaluate_all_models_async(self) -> Dict[str, Any]:
        """Evaluate all models in the experiment."""
        all_results = {}
        for model_dir in self.inference_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                logger.info(f"Multi-frame evaluating model: {model_name}")
                all_results[model_name] = await self.evaluate_model_async(model_name)

        # Save combined results
        output_path = self.output_dir / f"{self.evaluator_name}_summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                "metadata": {
                    "evaluator": self.evaluator_name,
                    "base_evaluator": self.base_evaluator.__class__.__name__,
                    "timestamp": datetime.now().isoformat(),
                    "config": {
                        "n_frames": self.n_frames,
                        "last_seconds": self.last_seconds,
                        "sampling_strategy": self.sampling_strategy,
                        "similarity_metric": self.similarity_metric,
                        "temporal_weight": self.temporal_weight,
                        "voting_method": self.voting_method
                    }
                },
                "results": all_results
            }, f, indent=2)

        return all_results

    def evaluate_all_models(self) -> Dict[str, Any]:
        """Evaluate all models in the experiment (sync wrapper)."""
        return asyncio.run(self.evaluate_all_models_async())

    def _save_single_result(
        self,
        model_name: str,
        task_type: str,
        task_id: str,
        eval_result: Dict[str, Any]
    ):
        """Save a single evaluation result immediately (for resume support)."""
        task_output_dir = self.output_dir / model_name / task_type / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        with open(task_output_dir / f"{self.evaluator_name}.json", 'w') as f:
            json.dump({
                "metadata": {
                    "evaluator": self.evaluator_name,
                    "base_evaluator": self.base_evaluator.__class__.__name__,
                    "timestamp": datetime.now().isoformat(),
                    "model_name": model_name,
                    "task_type": task_type,
                    "task_id": task_id
                },
                "result": eval_result
            }, f, indent=2)

        logger.debug(f"Saved multi-frame evaluation for {model_name}/{task_type}/{task_id}")
