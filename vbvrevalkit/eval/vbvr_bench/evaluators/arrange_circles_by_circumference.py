"""Evaluator for G-174_arrange_circles_by_circumference_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-174_arrange_circles_by_circumference_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class ArrangeCirclesByCircumferenceEvaluator(BaseEvaluator):
    """
    G-174: Arrange circles by circumference (large to small).
    
    Rule-based evaluation:
    - Sorting correctness (40%): Circles ordered by size (descending left to right)
    - Layout accuracy (30%): Horizontal alignment, even spacing
    - Object fidelity (20%): Circle properties preserved
    - Completeness (10%): All circles present
    """
    
    TASK_WEIGHTS = {
        'sorting_correctness': 0.40,
        'layout_accuracy': 0.30,
        'object_fidelity': 0.20,
        'completeness': 0.10
    }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        if len(video_frames) < 2:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        final_frame = video_frames[-1]
        
        # 1. Sorting correctness (40%)
        scores['sorting_correctness'] = self._evaluate_sorting(final_frame)
        
        # 2. Layout accuracy (30%)
        scores['layout_accuracy'] = self._evaluate_layout(final_frame)
        
        # 3. Object fidelity (20%)
        scores['object_fidelity'] = self._evaluate_fidelity(first_frame, final_frame)
        
        # 4. Completeness (10%)
        scores['completeness'] = self._evaluate_completeness(first_frame, final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_sorting(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check if circles are sorted by circumference (large to small)."""
        circles = self._detect_circles_with_size(final_frame)
        
        if len(circles) < 2:
            return 0.0  # STRICT: Not enough circles detected
        
        # Sort by x-position (left to right)
        sorted_by_x = sorted(circles, key=lambda c: c[0])
        radii = [c[2] for c in sorted_by_x]
        
        # Count inversions (smaller before larger - should be descending)
        inversions = sum(1 for i in range(len(radii)-1) if radii[i] < radii[i+1])
        max_inversions = len(radii) - 1
        
        if max_inversions == 0:
            return 1.0
        
        score = 1.0 - inversions / max_inversions
        return score
    
    def _evaluate_layout(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate horizontal alignment and spacing."""
        circles = self._detect_circles_with_size(final_frame)
        
        if len(circles) < 2:
            return 0.0  # STRICT: Not enough circles for layout eval
        
        # Check Y-coordinate alignment
        y_coords = [c[1] for c in circles]
        y_variance = np.var(y_coords)
        alignment_score = 1.0 / (1.0 + y_variance / 100)
        
        # Check spacing uniformity
        sorted_by_x = sorted(circles, key=lambda c: c[0])
        spacings = []
        for i in range(1, len(sorted_by_x)):
            spacing = sorted_by_x[i][0] - sorted_by_x[i-1][0]
            spacings.append(spacing)
        
        if len(spacings) > 1:
            spacing_variance = np.var(spacings) / (np.mean(spacings) + 1)
            spacing_score = 1.0 / (1.0 + spacing_variance)
        else:
            spacing_score = 1.0
        
        return 0.6 * alignment_score + 0.4 * spacing_score
    
    def _evaluate_fidelity(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if circle properties are preserved."""
        first_circles = self._detect_circles_with_size(first_frame)
        final_circles = self._detect_circles_with_size(final_frame)
        
        if len(first_circles) == 0 or len(final_circles) == 0:
            return 0.0  # STRICT: No circles detected
        
        # Compare radii distributions
        first_radii = sorted([c[2] for c in first_circles])
        final_radii = sorted([c[2] for c in final_circles])
        
        if len(first_radii) != len(final_radii):
            return 0.0  # STRICT: Circle count changed
        
        # Calculate radius similarity
        radius_diffs = [abs(fr - gr) for fr, gr in zip(first_radii, final_radii)]
        avg_diff = np.mean(radius_diffs)
        
        return max(0.0, 1.0 - avg_diff / 30)
    
    def _evaluate_completeness(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if all circles are present."""
        first_circles = self._detect_circles_with_size(first_frame)
        final_circles = self._detect_circles_with_size(final_frame)
        
        if len(first_circles) == 0:
            return 0.0  # STRICT: No circles in first frame
        
        completeness = min(1.0, len(final_circles) / len(first_circles))
        
        if len(final_circles) > len(first_circles):
            completeness *= 0.9  # Penalize extra circles
        
        return completeness
    
    def _detect_circles_with_size(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect circles with their x, y, radius."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 30,
            param1=50, param2=30, minRadius=15, maxRadius=100
        )
        
        result = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                result.append((int(x), int(y), int(r)))
        
        return result
