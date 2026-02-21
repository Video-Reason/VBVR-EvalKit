"""Evaluator for O-43_object_subtraction_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-43_object_subtraction_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class ObjectSubtractionEvaluator(BaseEvaluator):
    """
    O-43: Object subtraction (deletion) evaluator.
    
    Rule-based evaluation:
    - Object identification accuracy (40%): Correct objects identified
    - Deletion completeness (30%): Objects fully removed
    - Preserved object fidelity (20%): Remaining objects unchanged
    - Selective deletion accuracy (10%): No extra deletions
    """
    
    TASK_WEIGHTS = {
        'identification': 0.40,
        'deletion_completeness': 0.30,
        'preservation': 0.20,
        'selective_accuracy': 0.10
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
        
        # Count objects
        first_count = self._count_objects(first_frame)
        final_count = self._count_objects(final_frame)
        
        # Get expected final count from GT if available
        expected_final_count = None
        if gt_final_frame is not None:
            expected_final_count = self._count_objects(gt_final_frame)
        
        # CRITICAL: Check if correct number of objects remain
        # Should remove exactly the right number, not all objects
        if final_count == 0 and first_count > 1:
            # All objects removed - WRONG
            self._last_task_details = {
                'identification': 0.0,
                'deletion_completeness': 0.0,
                'preservation': 0.0,
                'selective_accuracy': 0.0,
                'all_objects_removed': True,
                'first_count': first_count,
                'final_count': final_count
            }
            return 0.0
        
        if final_count > first_count:
            # Objects added - WRONG
            self._last_task_details = {
                'identification': 0.0,
                'deletion_completeness': 0.0,
                'preservation': 0.0,
                'selective_accuracy': 0.0,
                'objects_added': True,
                'first_count': first_count,
                'final_count': final_count
            }
            return 0.0
        
        # Check if final count matches expected
        if expected_final_count is not None:
            if final_count != expected_final_count:
                # Wrong number of objects remaining
                self._last_task_details = {
                    'identification': 0.0,
                    'deletion_completeness': 0.0,
                    'preservation': 0.0,
                    'selective_accuracy': 0.0,
                    'wrong_count': True,
                    'first_count': first_count,
                    'final_count': final_count,
                    'expected_count': expected_final_count
                }
                return 0.0
        
        scores['identification'] = self._evaluate_identification(first_frame, final_frame)
        scores['deletion_completeness'] = self._evaluate_deletion(first_frame, final_frame)
        scores['preservation'] = self._evaluate_preservation(first_frame, final_frame)
        scores['selective_accuracy'] = self._evaluate_selective(first_frame, final_frame)
        
        self._last_task_details = scores
        self._last_task_details['first_count'] = first_count
        self._last_task_details['final_count'] = final_count
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_identification(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if correct objects are identified for deletion."""
        first_count = self._count_objects(first_frame)
        final_count = self._count_objects(final_frame)
        
        # Some objects should be deleted
        if final_count < first_count:
            return 1.0
        elif final_count == first_count:
            return 0.3  # Nothing deleted
        else:
            return 0.2  # Objects added (wrong)
    
    def _evaluate_deletion(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if objects are completely deleted."""
        first_count = self._count_objects(first_frame)
        final_count = self._count_objects(final_frame)
        
        deleted = first_count - final_count
        
        if deleted >= 1:
            return 1.0
        else:
            return 0.3
    
    def _evaluate_preservation(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check if remaining objects are preserved."""
        # Compare color distributions
        first_colors = self._get_object_colors(first_frame)
        final_colors = self._get_object_colors(final_frame)
        
        if len(final_colors) == 0:
            return 0.5
        
        # Check if remaining colors exist in original
        preserved = 0
        for color in final_colors:
            for orig_color in first_colors:
                if self._colors_similar(color, orig_color):
                    preserved += 1
                    break
        
        return preserved / max(len(final_colors), 1)
    
    def _evaluate_selective(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Check for correct selective deletion."""
        first_count = self._count_objects(first_frame)
        final_count = self._count_objects(final_frame)
        
        # Should have fewer objects but not zero
        if 0 < final_count < first_count:
            return 1.0
        elif final_count == 0:
            return 0.3  # All deleted
        else:
            return 0.4
    
    def _count_objects(self, frame: np.ndarray) -> int:
        """Count colored objects."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect saturated (colored) regions
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return sum(1 for cnt in contours if cv2.contourArea(cnt) > 500)
    
    def _get_object_colors(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Get colors of objects."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        colors = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if 0 <= cy < frame.shape[0] and 0 <= cx < frame.shape[1]:
                        colors.append(tuple(int(c) for c in frame[cy, cx]))
        
        return colors
    
    def _colors_similar(self, c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> bool:
        """Check if two colors are similar."""
        diff = np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
        return diff < 60
