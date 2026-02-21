"""Evaluator for O-13_shape_outline_then_move_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-13_shape_outline_then_move_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class ShapeOutlineThenMoveEvaluator(BaseEvaluator):
    """
    O-13: Shape Outline Then Move (Two-step transformation)
    
    Task: A→B→C :: D→?→? format - first change fill/outline style, 
    then move vertically.
    
    Rule-based evaluation:
    1. Two-step rule identification (30%) - Recognize style then position
    2. First step accuracy (25%) - Style change, position at center
    3. Second step accuracy (25%) - Position change, style maintained
    4. Sequence consistency (20%) - Correct order maintained
    """
    
    TASK_WEIGHTS = {
        'two_step_rule': 0.30,
        'first_step': 0.25,
        'second_step': 0.25,
        'sequence': 0.20
    }
    
    def _get_shape_centroid(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Get centroid of main shape in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                return (cx, cy)
        
        return None
    
    def _is_outline_style(self, frame: np.ndarray) -> bool:
        """Check if shape is outline (not filled)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        # Check interior
        M = cv2.moments(largest)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if 0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1]:
                center_val = gray[cy, cx]
                # Outline style: center is bright (not filled)
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
        """Evaluate two-step outline-then-move transformation."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Get centroids
        gen_centroid = self._get_shape_centroid(gen_final)
        gt_centroid = self._get_shape_centroid(gt_final)
        
        # 1. Two-step rule: Compare final position with GT
        if gen_centroid is not None and gt_centroid is not None:
            dist = np.sqrt((gen_centroid[0] - gt_centroid[0])**2 + 
                          (gen_centroid[1] - gt_centroid[1])**2)
            scores['two_step_rule'] = max(0, 1.0 - dist / 100.0)
        else:
            scores['two_step_rule'] = 0.2  # Detection failed
        
        # 2. First step: Style change (outline detection)
        gen_is_outline = self._is_outline_style(gen_final)
        gt_is_outline = self._is_outline_style(gt_final)
        
        scores['first_step'] = 1.0 if gen_is_outline == gt_is_outline else 0.3
        
        # 3. Second step: Vertical movement
        if len(video_frames) >= 2:
            first_centroid = self._get_shape_centroid(video_frames[0])
            if first_centroid is not None and gen_centroid is not None:
                dy = abs(gen_centroid[1] - first_centroid[1])
                dx = abs(gen_centroid[0] - first_centroid[0])
                
                # Vertical movement: dy should be significant, dx minimal
                if dy > 20 and dx < dy:
                    scores['second_step'] = min(1.0, dy / 50.0)
                else:
                    scores['second_step'] = 0.3
            else:
                scores['second_step'] = 0.2  # Detection failed
        else:
            scores['second_step'] = 0.2  # Detection failed
        
        # 4. Sequence consistency
        scores['sequence'] = (scores['first_step'] + scores['second_step']) / 2
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
