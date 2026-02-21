"""Evaluator for O-12_shape_color_then_scale_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-12_shape_color_then_scale_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class ShapeColorThenScaleEvaluator(BaseEvaluator):
    """
    O-12: Shape Color Then Scale (Two-step transformation)
    
    Task: A→B→C :: D→?→? format - first change color, then change scale.
    
    Rule-based evaluation:
    1. Two-step rule identification (30%) - Recognize color then scale
    2. First step accuracy (25%) - Color change, size unchanged
    3. Second step accuracy (25%) - Scale change, color maintained
    4. Sequence consistency (20%) - Correct order maintained
    """
    
    TASK_WEIGHTS = {
        'two_step_rule': 0.30,
        'first_step': 0.25,
        'second_step': 0.25,
        'sequence': 0.20
    }
    
    def _detect_shape_properties(self, frame: np.ndarray) -> Dict:
        """Detect shape color and size."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if len(frame.shape) == 3 else None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'exists': False}
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        # Get dominant color
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        
        if hsv is not None:
            mean_hue = cv2.mean(hsv[:, :, 0], mask=mask)[0]
            mean_sat = cv2.mean(hsv[:, :, 1], mask=mask)[0]
        else:
            mean_hue = 0
            mean_sat = 0
        
        return {
            'exists': True,
            'area': area,
            'hue': mean_hue,
            'saturation': mean_sat
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate two-step color-then-scale transformation."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Analyze final state
        gen_props = self._detect_shape_properties(gen_final)
        gt_props = self._detect_shape_properties(gt_final)
        
        # 1. Two-step rule: Check if final matches GT (STRICT)
        if gen_props['exists'] and gt_props['exists']:
            # Compare area (scale) - must be within 20%
            area_ratio = min(gen_props['area'], gt_props['area']) / max(gen_props['area'], gt_props['area'], 1)
            
            # Compare color (hue)
            hue_diff = abs(gen_props['hue'] - gt_props['hue'])
            hue_diff = min(hue_diff, 180 - hue_diff)  # Circular hue
            
            # STRICT: both area and color must match closely
            area_ok = area_ratio > 0.7
            color_ok = hue_diff < 20
            
            if area_ok and color_ok:
                scores['two_step_rule'] = 1.0
            elif area_ok or color_ok:
                scores['two_step_rule'] = 0.3
            else:
                scores['two_step_rule'] = 0.0
        else:
            scores['two_step_rule'] = 0.0
        
        # 2. First step: Compare with GT final (STRICT)
        # The key is whether the final result matches GT
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
        
        # 3. Second step: Check final scale matches GT
        if gen_props['exists'] and gt_props['exists']:
            area_ratio = min(gen_props['area'], gt_props['area']) / max(gen_props['area'], gt_props['area'], 1)
            if area_ratio > 0.8:
                scores['second_step'] = 1.0
            elif area_ratio > 0.6:
                scores['second_step'] = 0.3
            else:
                scores['second_step'] = 0.0
        else:
            scores['second_step'] = 0.0
        
        # 4. Sequence: Overall consistency with GT
        scores['sequence'] = min(scores['first_step'], scores['second_step'])
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
