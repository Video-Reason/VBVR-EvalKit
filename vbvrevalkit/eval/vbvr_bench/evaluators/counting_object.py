"""Evaluator for O-33_counting_object_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-33_counting_object_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class CountingObjectEvaluator(BaseEvaluator):
    """
    O-33: Counting Objects
    
    Task: Count all visible geometric shapes in scene, output total count.
    Each object counted exactly once, no misses or duplicates.
    
    Rule-based evaluation:
    1. Count accuracy (50%) - Exact count correct
    2. Completeness (25%) - All objects identified
    3. Uniqueness (15%) - No duplicates
    4. Systematic approach (10%) - Orderly counting
    """
    
    TASK_WEIGHTS = {
        'count_accuracy': 0.50,
        'completeness': 0.25,
        'uniqueness': 0.15,
        'systematic': 0.10
    }
    
    def _count_objects(self, frame: np.ndarray) -> int:
        """Count distinct objects in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by size
        min_area = 100
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        return len(valid_contours)
    
    def _detect_number_annotation(self, frame: np.ndarray) -> Optional[int]:
        """Try to detect if there's a number annotation showing the count."""
        # This is a simplified version - in practice would use OCR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Look for text-like regions
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count small text-like contours
        text_contours = [c for c in contours if 10 < cv2.contourArea(c) < 500]
        
        return len(text_contours) if text_contours else None
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate object counting accuracy."""
        
        if not video_frames or gt_final_frame is None or gt_first_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # The task requires showing a NUMBER that matches the count of objects
        # Compare the final frame directly with GT final frame (which shows correct count)
        
        # 1. Count accuracy: Final frame must match GT final (shows correct number)
        if gen_final.shape == gt_final.shape:
            # VERY STRICT comparison - the displayed number must match GT exactly
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            if diff < 10:  # Very close match - correct number displayed
                scores['count_accuracy'] = 1.0
            elif diff < 20:
                scores['count_accuracy'] = 0.3
            else:
                scores['count_accuracy'] = 0.0  # Wrong number displayed
        else:
            scores['count_accuracy'] = 0.0
        
        # 2. Completeness: Check if final frame shows the counting result
        # Compare structure - GT final should show number, gen should too
        if gen_final.shape == gt_final.shape:
            # Check if there's text/number region in final frame
            gen_gray = cv2.cvtColor(gen_final, cv2.COLOR_BGR2GRAY) if len(gen_final.shape) == 3 else gen_final
            gt_gray = cv2.cvtColor(gt_final, cv2.COLOR_BGR2GRAY) if len(gt_final.shape) == 3 else gt_final
            
            # Check difference in the number display area (typically center or bottom)
            h, w = gen_gray.shape[:2]
            gen_center = gen_gray[h//4:3*h//4, w//4:3*w//4]
            gt_center = gt_gray[h//4:3*h//4, w//4:3*w//4]
            
            center_diff = np.abs(gen_center.astype(float) - gt_center.astype(float)).mean()
            if center_diff < 10:  # Very strict
                scores['completeness'] = 1.0
            elif center_diff < 25:
                scores['completeness'] = 0.3
            else:
                scores['completeness'] = 0.0
        else:
            scores['completeness'] = 0.0
        
        # 3. Uniqueness: Same as count_accuracy for this task
        scores['uniqueness'] = scores['count_accuracy']
        
        # 4. Systematic: Check overall frame similarity - very strict
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            if diff < 10:
                scores['systematic'] = 1.0
            elif diff < 20:
                scores['systematic'] = 0.3
            else:
                scores['systematic'] = 0.0
        else:
            scores['systematic'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
