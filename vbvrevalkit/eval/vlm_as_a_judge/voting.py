"""
Voting aggregator for multi-frame evaluation.

Aggregates multiple frame-level evaluation scores into a single result
using various voting strategies with optional consistency-based weighting.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class VotingMethod(Enum):
    """Available voting methods for score aggregation."""
    MAJORITY = "majority"                    # Simple majority vote
    WEIGHTED_MAJORITY = "weighted_majority"  # Majority with weights
    WEIGHTED_AVERAGE = "weighted_average"    # Weighted average score
    MAX_SCORE = "max_score"                  # Optimistic: take highest score
    MIN_SCORE = "min_score"                  # Pessimistic: take lowest score
    MEDIAN = "median"                        # Median score


@dataclass
class FrameScore:
    """Container for a single frame's evaluation result."""
    score: int                    # Evaluation score (typically 1-5)
    timestamp: float              # Frame timestamp in seconds
    weight: float = 1.0           # Weight for voting
    is_keyframe: bool = False     # Whether this is a detected keyframe
    explanation: Optional[str] = None  # Optional explanation from evaluator


@dataclass
class VotingResult:
    """Container for voting aggregation result."""
    final_score: int              # Aggregated final score
    confidence: float             # Confidence in the result (0-1)
    agreement_ratio: float        # Ratio of frames agreeing with final score
    stability_score: float        # Frame consistency score (0-1)
    needs_review: bool            # Flag for low-confidence results
    method_used: str              # Voting method that was used
    vote_distribution: Dict[int, float]  # Score -> weighted vote count
    frame_details: List[Dict[str, Any]] = field(default_factory=list)  # Per-frame breakdown


class VotingAggregator:
    """
    Aggregator for combining multiple frame-level scores into a final result.

    Supports multiple voting strategies with optional consistency-based weighting.
    Can flag low-confidence results for human review.

    Attributes:
        method: Default voting method
        confidence_threshold: Below this, results are flagged for review
        min_agreement: Minimum agreement ratio required
        score_range: Valid score range (min, max)
    """

    def __init__(
        self,
        method: VotingMethod = VotingMethod.WEIGHTED_MAJORITY,
        confidence_threshold: float = 0.6,
        min_agreement: float = 0.4,
        score_range: Tuple[int, int] = (1, 5)
    ):
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.min_agreement = min_agreement
        self.score_range = score_range

    def aggregate(
        self,
        frame_scores: List[FrameScore],
        method: Optional[VotingMethod] = None,
        stability_score: Optional[float] = None
    ) -> VotingResult:
        """
        Aggregate multiple frame scores into a single result.

        Args:
            frame_scores: List of FrameScore objects with scores and weights
            method: Override default voting method
            stability_score: Pre-computed stability score (0-1)

        Returns:
            VotingResult with final score and metadata
        """
        if not frame_scores:
            return VotingResult(
                final_score=0,
                confidence=0.0,
                agreement_ratio=0.0,
                stability_score=0.0,
                needs_review=True,
                method_used="none",
                vote_distribution={},
                frame_details=[]
            )

        method = method or self.method
        scores = [fs.score for fs in frame_scores]
        weights = [fs.weight for fs in frame_scores]

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)

        # Compute vote distribution
        vote_dist = self._compute_vote_distribution(scores, weights)

        # Apply voting method
        if method == VotingMethod.MAJORITY:
            final_score = self._majority_vote(scores)
        elif method == VotingMethod.WEIGHTED_MAJORITY:
            final_score = self._weighted_majority_vote(scores, weights)
        elif method == VotingMethod.WEIGHTED_AVERAGE:
            final_score = self._weighted_average(scores, weights)
        elif method == VotingMethod.MAX_SCORE:
            final_score = max(scores)
        elif method == VotingMethod.MIN_SCORE:
            final_score = min(scores)
        elif method == VotingMethod.MEDIAN:
            final_score = int(np.median(scores))
        else:
            raise ValueError(f"Unknown voting method: {method}")

        # Compute agreement ratio
        agreement_ratio = scores.count(final_score) / len(scores)

        # Compute confidence
        confidence = self._compute_confidence(vote_dist, final_score, weights, scores)

        # Determine if review is needed
        needs_review = (
            confidence < self.confidence_threshold or
            agreement_ratio < self.min_agreement
        )

        # Build frame details
        frame_details = [
            {
                "timestamp": fs.timestamp,
                "score": fs.score,
                "weight": w,
                "is_keyframe": fs.is_keyframe,
                "agrees_with_final": fs.score == final_score
            }
            for fs, w in zip(frame_scores, weights)
        ]

        return VotingResult(
            final_score=final_score,
            confidence=confidence,
            agreement_ratio=agreement_ratio,
            stability_score=stability_score or 0.0,
            needs_review=needs_review,
            method_used=method.value,
            vote_distribution=vote_dist,
            frame_details=frame_details
        )

    def aggregate_from_raw(
        self,
        scores: List[int],
        weights: Optional[List[float]] = None,
        timestamps: Optional[List[float]] = None,
        method: Optional[VotingMethod] = None,
        stability_score: Optional[float] = None
    ) -> VotingResult:
        """
        Convenience method to aggregate from raw score lists.

        Args:
            scores: List of integer scores
            weights: Optional list of weights (default: equal weights)
            timestamps: Optional list of timestamps
            method: Voting method override
            stability_score: Pre-computed stability score

        Returns:
            VotingResult with final score and metadata
        """
        n = len(scores)
        if weights is None:
            weights = [1.0] * n
        if timestamps is None:
            timestamps = list(range(n))

        frame_scores = [
            FrameScore(score=s, timestamp=t, weight=w)
            for s, t, w in zip(scores, timestamps, weights)
        ]

        return self.aggregate(frame_scores, method, stability_score)

    def _compute_vote_distribution(
        self,
        scores: List[int],
        weights: List[float]
    ) -> Dict[int, float]:
        """Compute weighted vote counts for each score value."""
        distribution = defaultdict(float)
        for score, weight in zip(scores, weights):
            distribution[score] += weight
        return dict(distribution)

    def _majority_vote(self, scores: List[int]) -> int:
        """Simple majority voting - most frequent score wins."""
        from collections import Counter
        counter = Counter(scores)
        return counter.most_common(1)[0][0]

    def _weighted_majority_vote(self, scores: List[int], weights: List[float]) -> int:
        """
        Weighted majority voting.

        Each score's vote is weighted by its corresponding weight.
        The score with highest weighted vote count wins.
        """
        weighted_counts = defaultdict(float)
        for score, weight in zip(scores, weights):
            weighted_counts[score] += weight

        return max(weighted_counts, key=weighted_counts.get)

    def _weighted_average(self, scores: List[int], weights: List[float]) -> int:
        """
        Weighted average score, rounded to nearest integer.

        Produces a score that reflects the weighted mean of all evaluations.
        """
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        avg = weighted_sum / sum(weights) if sum(weights) > 0 else 0

        # Round and clamp to valid range
        result = int(round(avg))
        return max(self.score_range[0], min(self.score_range[1], result))

    def _compute_confidence(
        self,
        vote_dist: Dict[int, float],
        final_score: int,
        weights: List[float],
        scores: List[int]
    ) -> float:
        """
        Compute confidence in the voting result.

        Based on:
        - Winner's vote share (higher = more confident)
        - Margin over second place (larger margin = more confident)
        - Overall agreement level
        """
        if not vote_dist:
            return 0.0

        total_votes = sum(vote_dist.values())
        winner_votes = vote_dist.get(final_score, 0)

        # Winner's share
        winner_share = winner_votes / total_votes if total_votes > 0 else 0

        # Margin over second place
        sorted_votes = sorted(vote_dist.values(), reverse=True)
        if len(sorted_votes) >= 2:
            margin = (sorted_votes[0] - sorted_votes[1]) / total_votes
        else:
            margin = 1.0

        # Combine into confidence score
        confidence = 0.6 * winner_share + 0.4 * margin

        return float(np.clip(confidence, 0, 1))


class MultiFrameEvaluationPipeline:
    """
    Complete pipeline for multi-frame video evaluation.

    Combines frame sampling, consistency analysis, and voting aggregation
    into a single unified workflow.

    Example:
        >>> from vbvrevalkit.eval.frame_sampler import FrameSampler
        >>> from vbvrevalkit.eval.consistency import FrameConsistencyAnalyzer
        >>> from vbvrevalkit.eval.voting import MultiFrameEvaluationPipeline
        >>>
        >>> pipeline = MultiFrameEvaluationPipeline(
        ...     sampler=FrameSampler(n_frames=5),
        ...     consistency_analyzer=FrameConsistencyAnalyzer(),
        ...     voting_aggregator=VotingAggregator()
        ... )
        >>>
        >>> # Use with any frame evaluator function
        >>> def evaluate_frame(frame, expected, prompt):
        ...     # Your evaluation logic here
        ...     return {"score": 4, "explanation": "..."}
        >>>
        >>> result = pipeline.evaluate(
        ...     video_path="output.mp4",
        ...     frame_evaluator=evaluate_frame,
        ...     expected_frame=expected_img,
        ...     prompt="Navigate through maze"
        ... )
    """

    def __init__(
        self,
        sampler=None,
        consistency_analyzer=None,
        voting_aggregator: Optional[VotingAggregator] = None
    ):
        # Lazy imports to avoid circular dependencies
        self.sampler = sampler
        self.consistency_analyzer = consistency_analyzer
        self.voting_aggregator = voting_aggregator or VotingAggregator()

    def evaluate(
        self,
        video_path: str,
        frame_evaluator: callable,
        n_frames: int = 5,
        last_seconds: float = 3.0,
        **evaluator_kwargs
    ) -> Dict[str, Any]:
        """
        Run complete multi-frame evaluation pipeline.

        Args:
            video_path: Path to video file
            frame_evaluator: Function(frame, **kwargs) -> {"score": int, ...}
            n_frames: Number of frames to sample
            last_seconds: Duration from video end to sample
            **evaluator_kwargs: Additional arguments passed to frame_evaluator

        Returns:
            Dictionary containing:
            - final_score: Aggregated score
            - confidence: Voting confidence
            - stability_score: Frame consistency score
            - needs_review: Whether human review is recommended
            - frame_results: Per-frame evaluation details
            - voting_details: Full voting result
        """
        # Lazy import sampler and consistency analyzer
        if self.sampler is None:
            from .frame_sampler import FrameSampler
            self.sampler = FrameSampler()

        if self.consistency_analyzer is None:
            from .consistency import FrameConsistencyAnalyzer
            self.consistency_analyzer = FrameConsistencyAnalyzer()

        # Step 1: Sample frames
        sampled_frames = self.sampler.sample(
            video_path,
            n_frames=n_frames,
            last_seconds=last_seconds,
            strategy="hybrid"
        )

        if not sampled_frames:
            return {
                "final_score": 0,
                "confidence": 0.0,
                "stability_score": 0.0,
                "needs_review": True,
                "error": "No frames could be sampled",
                "frame_results": [],
                "voting_details": None
            }

        # Step 2: Analyze frame consistency
        frame_images = [f.image for f in sampled_frames]
        consistency_result = self.consistency_analyzer.analyze(frame_images)

        # Step 3: Evaluate each frame
        frame_scores = []
        frame_results = []

        for i, (sampled_frame, weight) in enumerate(zip(sampled_frames, consistency_result.weights)):
            try:
                eval_result = frame_evaluator(sampled_frame.image, **evaluator_kwargs)
                score = eval_result.get("score", eval_result.get("solution_correctness_score", 0))

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
                    "evaluation": eval_result
                })

            except Exception as e:
                logger.warning(f"Failed to evaluate frame {i}: {e}")
                frame_results.append({
                    "frame_index": sampled_frame.frame_index,
                    "timestamp": sampled_frame.timestamp,
                    "error": str(e)
                })

        if not frame_scores:
            return {
                "final_score": 0,
                "confidence": 0.0,
                "stability_score": consistency_result.stability_score,
                "needs_review": True,
                "error": "All frame evaluations failed",
                "frame_results": frame_results,
                "voting_details": None
            }

        # Step 4: Aggregate votes
        voting_result = self.voting_aggregator.aggregate(
            frame_scores,
            stability_score=consistency_result.stability_score
        )

        return {
            "final_score": voting_result.final_score,
            "confidence": voting_result.confidence,
            "agreement_ratio": voting_result.agreement_ratio,
            "stability_score": voting_result.stability_score,
            "needs_review": voting_result.needs_review,
            "method_used": voting_result.method_used,
            "vote_distribution": voting_result.vote_distribution,
            "frame_results": frame_results,
            "voting_details": {
                "frame_details": voting_result.frame_details,
                "outlier_indices": consistency_result.outlier_indices,
                "mean_similarity": consistency_result.mean_similarity
            }
        }


def aggregate_scores(
    scores: List[int],
    weights: Optional[List[float]] = None,
    method: str = "weighted_majority"
) -> Dict[str, Any]:
    """
    Convenience function for quick score aggregation.

    Args:
        scores: List of integer scores
        weights: Optional weights (default: equal weights)
        method: Voting method name

    Returns:
        Dictionary with final_score, confidence, agreement_ratio
    """
    aggregator = VotingAggregator(method=VotingMethod(method))
    result = aggregator.aggregate_from_raw(scores, weights)

    return {
        "final_score": result.final_score,
        "confidence": result.confidence,
        "agreement_ratio": result.agreement_ratio,
        "needs_review": result.needs_review,
        "vote_distribution": result.vote_distribution
    }
