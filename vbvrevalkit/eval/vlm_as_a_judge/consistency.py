"""
Frame consistency analyzer for multi-frame evaluation.

Computes inter-frame similarity metrics to:
- Measure video output stability
- Assign weights to frames based on consistency
- Detect anomalous/noisy frames
"""

import logging
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """Available similarity metrics for frame comparison."""
    HISTOGRAM = "histogram"      # Color histogram correlation
    SSIM = "ssim"               # Structural Similarity Index
    MSE = "mse"                 # Mean Squared Error (inverted to similarity)
    COMBINED = "combined"        # Weighted combination of multiple metrics


@dataclass
class ConsistencyResult:
    """Container for consistency analysis results."""
    stability_score: float           # Overall stability (0-1, higher = more stable)
    weights: List[float]             # Per-frame weights (sum to 1)
    pairwise_similarity: np.ndarray  # NxN similarity matrix
    outlier_indices: List[int]       # Indices of detected outlier frames
    mean_similarity: float           # Average pairwise similarity
    std_similarity: float            # Standard deviation of similarities


class FrameConsistencyAnalyzer:
    """
    Analyzer for computing frame-to-frame consistency in sampled video frames.

    Used to:
    1. Assess overall video output stability
    2. Compute weights for voting (consistent frames get higher weight)
    3. Identify outlier frames that may be noise or transition artifacts

    Attributes:
        metric: Similarity metric to use
        outlier_threshold: Z-score threshold for outlier detection
        weights_method: Method for computing frame weights
        temporal_weight: Weight factor for temporal bias (0=none, 1=strong preference for later frames)
    """

    def __init__(
        self,
        metric: Union[str, SimilarityMetric] = SimilarityMetric.HISTOGRAM,
        outlier_threshold: float = 2.0,
        weights_method: str = "mean_similarity",
        temporal_weight: float = 0.3
    ):
        if isinstance(metric, str):
            metric = SimilarityMetric(metric)
        self.metric = metric
        self.outlier_threshold = outlier_threshold
        self.weights_method = weights_method
        self.temporal_weight = temporal_weight  # 0-1, higher = prefer later frames

    def analyze(self, frames: List[np.ndarray]) -> ConsistencyResult:
        """
        Perform full consistency analysis on a list of frames.

        Args:
            frames: List of RGB image arrays (H, W, 3)

        Returns:
            ConsistencyResult containing all analysis metrics
        """
        if len(frames) < 2:
            # Single frame or empty - return trivial result
            return ConsistencyResult(
                stability_score=1.0,
                weights=[1.0] if frames else [],
                pairwise_similarity=np.array([[1.0]]) if frames else np.array([]),
                outlier_indices=[],
                mean_similarity=1.0,
                std_similarity=0.0
            )

        # Compute pairwise similarity matrix
        sim_matrix = self.compute_pairwise_similarity(frames)

        # Compute statistics
        # Use upper triangle (excluding diagonal) for statistics
        upper_tri = sim_matrix[np.triu_indices(len(frames), k=1)]
        mean_sim = float(np.mean(upper_tri))
        std_sim = float(np.std(upper_tri))

        # Compute per-frame average similarity (excluding self)
        frame_avg_sims = []
        for i in range(len(frames)):
            other_sims = [sim_matrix[i, j] for j in range(len(frames)) if i != j]
            frame_avg_sims.append(np.mean(other_sims))
        frame_avg_sims = np.array(frame_avg_sims)

        # Detect outliers using z-score on frame average similarities
        outlier_indices = self._detect_outliers(frame_avg_sims)

        # Compute weights based on consistency
        weights = self._compute_weights(frame_avg_sims, outlier_indices)

        # Overall stability score
        stability_score = self._compute_stability_score(mean_sim, std_sim, len(outlier_indices), len(frames))

        return ConsistencyResult(
            stability_score=stability_score,
            weights=weights,
            pairwise_similarity=sim_matrix,
            outlier_indices=outlier_indices,
            mean_similarity=mean_sim,
            std_similarity=std_sim
        )

    def compute_pairwise_similarity(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Compute NxN similarity matrix between all frame pairs.

        Args:
            frames: List of RGB image arrays

        Returns:
            NxN numpy array where entry (i,j) is similarity between frame i and j
        """
        n = len(frames)
        sim_matrix = np.eye(n)  # Diagonal is 1 (self-similarity)

        for i in range(n):
            for j in range(i + 1, n):
                sim = self._compute_similarity(frames[i], frames[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim  # Symmetric

        return sim_matrix

    def compute_stability_score(self, frames: List[np.ndarray]) -> float:
        """
        Compute overall stability score for a sequence of frames.

        Higher score indicates more consistent/stable video output.

        Args:
            frames: List of RGB image arrays

        Returns:
            Stability score between 0 and 1
        """
        result = self.analyze(frames)
        return result.stability_score

    def compute_weights(self, frames: List[np.ndarray]) -> List[float]:
        """
        Compute per-frame weights based on consistency.

        Frames that are more consistent with others get higher weights.
        Outlier frames get reduced weights.

        Args:
            frames: List of RGB image arrays

        Returns:
            List of weights that sum to 1
        """
        result = self.analyze(frames)
        return result.weights

    def _compute_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute similarity between two frames using configured metric."""
        if self.metric == SimilarityMetric.HISTOGRAM:
            return self._histogram_similarity(frame1, frame2)
        elif self.metric == SimilarityMetric.SSIM:
            return self._ssim_similarity(frame1, frame2)
        elif self.metric == SimilarityMetric.MSE:
            return self._mse_similarity(frame1, frame2)
        elif self.metric == SimilarityMetric.COMBINED:
            return self._combined_similarity(frame1, frame2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _histogram_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute color histogram correlation between frames.

        Uses 3D color histogram (8 bins per channel) and computes
        correlation coefficient between the two histograms.
        """
        # Convert to BGR for OpenCV
        bgr1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
        bgr2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)

        # Compute 3D color histogram
        hist1 = cv2.calcHist([bgr1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([bgr2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        # Normalize
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        # Compute correlation (range: -1 to 1, we map to 0-1)
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return (correlation + 1) / 2  # Map from [-1, 1] to [0, 1]

    def _ssim_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute Structural Similarity Index (SSIM) between frames.

        SSIM measures perceived quality considering luminance, contrast, and structure.
        """
        # Ensure same size
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # Convert to grayscale for SSIM
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

        # SSIM constants
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # Convert to float
        img1 = gray1.astype(np.float64)
        img2 = gray2.astype(np.float64)

        # Compute means
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Compute variances and covariance
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(np.mean(ssim_map))

    def _mse_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute similarity based on Mean Squared Error.

        MSE is inverted and normalized to produce similarity in [0, 1].
        """
        # Ensure same size
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # Compute MSE
        mse = np.mean((frame1.astype(np.float64) - frame2.astype(np.float64)) ** 2)

        # Convert to similarity (exponential decay)
        # When MSE=0, similarity=1; as MSE increases, similarity decreases
        similarity = np.exp(-mse / 1000)  # 1000 is a scaling factor
        return float(similarity)

    def _combined_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compute weighted combination of multiple similarity metrics.

        Combines histogram (global color), SSIM (structure), for robust comparison.
        """
        hist_sim = self._histogram_similarity(frame1, frame2)
        ssim_sim = self._ssim_similarity(frame1, frame2)

        # Weighted average (SSIM weighted higher as it's more perceptually meaningful)
        return 0.4 * hist_sim + 0.6 * ssim_sim

    def _detect_outliers(self, frame_similarities: np.ndarray) -> List[int]:
        """
        Detect outlier frames using z-score method.

        Frames with average similarity significantly below the mean are flagged.
        """
        if len(frame_similarities) < 3:
            return []

        mean_sim = np.mean(frame_similarities)
        std_sim = np.std(frame_similarities)

        if std_sim < 1e-6:  # All frames very similar
            return []

        z_scores = (mean_sim - frame_similarities) / std_sim  # Inverted: low similarity = high z-score

        outliers = [i for i, z in enumerate(z_scores) if z > self.outlier_threshold]
        return outliers

    def _compute_weights(
        self,
        frame_avg_sims: np.ndarray,
        outlier_indices: List[int]
    ) -> List[float]:
        """
        Compute per-frame weights based on average similarity and temporal position.

        Weight factors:
        1. Consistency weight: Higher average similarity = higher weight
        2. Temporal weight: Later frames get higher weight (prefer final state)
        3. Outlier penalty: Detected outliers get reduced weight
        """
        n = len(frame_avg_sims)

        if n == 0:
            return []

        if n == 1:
            return [1.0]

        # 1. Consistency-based weights (normalized)
        consistency_weights = frame_avg_sims.copy()
        if np.sum(consistency_weights) > 0:
            consistency_weights = consistency_weights / np.sum(consistency_weights)
        else:
            consistency_weights = np.ones(n) / n

        # 2. Temporal weights (linear increase toward later frames)
        # e.g., for 5 frames: [0.2, 0.4, 0.6, 0.8, 1.0] normalized
        temporal_weights = np.linspace(0.2, 1.0, n)
        temporal_weights = temporal_weights / np.sum(temporal_weights)

        # 3. Combine consistency and temporal weights
        # temporal_weight=0 -> only consistency
        # temporal_weight=1 -> only temporal
        combined_weights = (
            (1 - self.temporal_weight) * consistency_weights +
            self.temporal_weight * temporal_weights
        )

        # 4. Apply outlier penalty
        for idx in outlier_indices:
            combined_weights[idx] *= 0.5

        # Normalize to sum to 1
        total = np.sum(combined_weights)
        if total > 0:
            combined_weights = combined_weights / total
        else:
            combined_weights = np.ones(n) / n

        return combined_weights.tolist()

    def _compute_stability_score(
        self,
        mean_similarity: float,
        std_similarity: float,
        n_outliers: int,
        n_frames: int
    ) -> float:
        """
        Compute overall stability score.

        Based on:
        - Mean similarity (higher = more stable)
        - Standard deviation (lower = more consistent)
        - Outlier ratio (fewer outliers = more stable)
        """
        # Mean similarity contribution (0-1)
        mean_score = mean_similarity

        # Consistency contribution (low std = high score)
        # Using exponential decay: std=0 -> 1, std=0.5 -> 0.37
        consistency_score = np.exp(-std_similarity * 4)

        # Outlier penalty
        outlier_ratio = n_outliers / n_frames if n_frames > 0 else 0
        outlier_score = 1 - outlier_ratio

        # Weighted combination
        stability = 0.5 * mean_score + 0.3 * consistency_score + 0.2 * outlier_score

        return float(np.clip(stability, 0, 1))


def compute_frame_weights(frames: List[np.ndarray], metric: str = "histogram") -> List[float]:
    """
    Convenience function to compute frame weights.

    Args:
        frames: List of RGB image arrays
        metric: Similarity metric ("histogram", "ssim", "mse", "combined")

    Returns:
        List of weights that sum to 1
    """
    analyzer = FrameConsistencyAnalyzer(metric=metric)
    return analyzer.compute_weights(frames)


def compute_stability_score(frames: List[np.ndarray], metric: str = "histogram") -> float:
    """
    Convenience function to compute stability score.

    Args:
        frames: List of RGB image arrays
        metric: Similarity metric ("histogram", "ssim", "mse", "combined")

    Returns:
        Stability score between 0 and 1
    """
    analyzer = FrameConsistencyAnalyzer(metric=metric)
    return analyzer.compute_stability_score(frames)
