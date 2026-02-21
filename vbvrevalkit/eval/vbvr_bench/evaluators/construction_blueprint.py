"""Evaluator for O-21_construction_blueprint_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-21_construction_blueprint_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class ConstructionBlueprintEvaluator(BaseEvaluator):
    """
    O-21: Construction Blueprint (Missing Piece)
    
    Task: Select the correct piece from 4 candidates to fill the highlighted 
    gap in a block structure.
    
    Rule-based evaluation:
    1. Piece selection correctness (40%) - Right piece chosen
    2. Shape matching accuracy (30%) - Exact fit to gap
    3. Placement precision (20%) - No gaps or overlaps
    4. Structure integrity (10%) - Complete, connected result
    """
    
    TASK_WEIGHTS = {
        'piece_selection': 0.40,
        'shape_matching': 0.30,
        'placement': 0.20,
        'integrity': 0.10
    }
    
    def _detect_gap_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect red-outlined gap region."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        if hsv is None:
            return None
        
        # Red detection
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(largest)
        
        return None
    
    def _check_gap_filled(self, frame: np.ndarray) -> float:
        """Check if gap is properly filled (no red outline remaining)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        if hsv is None:
            return 0.5
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        red_pixels = red_mask.sum() / 255
        
        if red_pixels < 100:
            return 1.0
        elif red_pixels < 500:
            return 0.7
        else:
            return 0.3
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate blueprint piece selection and placement."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # 1. Piece selection: Compare green regions with GT (STRICT)
        gen_green = self._detect_green_filled_region(gen_final)
        gt_green = self._detect_green_filled_region(gt_final)
        
        if gen_green is not None and gt_green is not None:
            # Check if green regions are in similar locations
            genx, geny, genw, genh = gen_green
            gtx, gty, gtw, gth = gt_green
            
            # Calculate position difference
            pos_diff = np.sqrt((genx - gtx)**2 + (geny - gty)**2)
            size_ratio = min(genw * genh, gtw * gth) / max(genw * genh, gtw * gth, 1)
            
            if pos_diff < 50 and size_ratio > 0.5:
                scores['piece_selection'] = 1.0
            elif pos_diff < 100 or size_ratio > 0.3:
                scores['piece_selection'] = 0.3
            else:
                scores['piece_selection'] = 0.0
        elif gen_green is None and gt_green is None:
            scores['piece_selection'] = 1.0  # Both have no green
        else:
            scores['piece_selection'] = 0.0  # Mismatch
        
        # 2. Shape matching: Compare with GT final (STRICT)
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            if diff < 20:
                scores['shape_matching'] = 1.0
            elif diff < 40:
                scores['shape_matching'] = 0.3
            else:
                scores['shape_matching'] = 0.0
        else:
            scores['shape_matching'] = 0.0
        
        # 3. Placement: Check if gap is filled (no red outline remaining)
        scores['placement'] = self._check_gap_filled(gen_final)
        
        # 4. Integrity: Overall frame similarity
        scores['integrity'] = scores['shape_matching']
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_green_filled_region(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect green filled region."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        if hsv is None:
            return None
        
        # Green detection
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:  # Significant green region
                return cv2.boundingRect(largest)
        
        return None
