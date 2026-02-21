"""Evaluator for O-14_shape_scale_then_outline_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-14_shape_scale_then_outline_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class ShapeScaleThenOutlineEvaluator(BaseEvaluator):
    """
    O-14: Shape Scale Then Outline (Two-step transformation)
    
    Task: A→B→C :: D→?→? format - first change scale, then change 
    from filled to outline.
    
    Rule-based evaluation:
    1. Two-step rule identification (30%) - Recognize scale then style
    2. First step accuracy (25%) - Scale change, style maintained
    3. Second step accuracy (25%) - Style change to outline
    4. Sequence consistency (20%) - Correct order maintained
    """
    
    TASK_WEIGHTS = {
        'two_step_rule': 0.30,
        'first_step': 0.25,
        'second_step': 0.25,
        'sequence': 0.20
    }
    
    def _get_shape_area(self, frame: np.ndarray) -> float:
        """Get area of main shape."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            return cv2.contourArea(largest)
        
        return 0
    
    def _is_outline_style(self, frame: np.ndarray) -> bool:
        """Check if shape is outline style."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        largest = max(contours, key=cv2.contourArea)
        
        M = cv2.moments(largest)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if 0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1]:
                center_val = gray[cy, cx]
                return center_val > 200
        
        return False
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate two-step scale-then-outline transformation."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Get properties
        gen_area = self._get_shape_area(gen_final)
        gt_area = self._get_shape_area(gt_final)
        gen_outline = self._is_outline_style(gen_final)
        gt_outline = self._is_outline_style(gt_final)
        
        # 1. Two-step rule: Compare final state with GT (STRICT)
        if gen_area > 0 and gt_area > 0:
            area_ratio = min(gen_area, gt_area) / max(gen_area, gt_area)
            style_match = gen_outline == gt_outline
            
            # STRICT: Both area and style must match
            if area_ratio > 0.7 and style_match:
                scores['two_step_rule'] = 1.0
            elif area_ratio > 0.5 or style_match:
                scores['two_step_rule'] = 0.3
            else:
                scores['two_step_rule'] = 0.0
        else:
            scores['two_step_rule'] = 0.0
        
        # 2. First step: Compare with GT final (STRICT)
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            if diff < 20:
                scores['first_step'] = 1.0
            elif diff < 40:
                scores['first_step'] = 0.3
            else:
                scores['first_step'] = 0.0
        else:
            scores['first_step'] = 0.0
        
        # 3. Second step: Outline style must match GT
        scores['second_step'] = 1.0 if gen_outline == gt_outline else 0.0
        
        # 4. Sequence consistency
        scores['sequence'] = min(scores['first_step'], scores['second_step'])
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
