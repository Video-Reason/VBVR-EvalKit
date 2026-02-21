"""Evaluator for O-10_shape_outline_fill_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-10_shape_outline_fill_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class ShapeOutlineFillEvaluator(BaseEvaluator):
    """
    O-10: Shape Outline Fill (Visual Analogy)
    
    CRITICAL RULES:
    1. First row (top left, top right) objects must remain UNCHANGED
    2. Second row: bottom right should have same shape as bottom left
    3. Bottom right shape should be either filled or have thicker lines (follow pattern)
    """
    
    TASK_WEIGHTS = {
        'first_row_preserved': 0.45,    # Top row unchanged
        'bottom_right_correct': 0.35,   # D has correct shape and style
        'shape_preserved': 0.20         # D shape matches C
    }
    
    def _detect_shapes_in_quadrant(self, frame: np.ndarray, quadrant: str) -> Dict:
        """Detect shape in specified quadrant and determine if filled or outline."""
        h, w = frame.shape[:2]
        
        if quadrant == 'top_left':
            region = frame[:h//2, :w//2]
        elif quadrant == 'top_right':
            region = frame[:h//2, w//2:]
        elif quadrant == 'bottom_left':
            region = frame[h//2:, :w//2]
        else:  # bottom_right
            region = frame[h//2:, w//2:]
        
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        
        # Detect shape
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'exists': False}
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        
        # Determine shape type
        approx = cv2.approxPolyDP(largest, 0.04 * perimeter, True)
        vertices = len(approx)
        
        if vertices == 3:
            shape_type = 'triangle'
        elif vertices == 4:
            shape_type = 'rectangle'
        elif vertices >= 8:
            shape_type = 'circle'
        else:
            shape_type = 'polygon'
        
        # Determine if filled or outline
        # Filled shapes have higher area-to-perimeter ratio
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter * perimeter)
        else:
            compactness = 0
        
        # Check interior fill by sampling center
        M = cv2.moments(largest)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # Check if center is dark (filled)
            if 0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1]:
                center_val = gray[cy, cx]
                is_filled = center_val < 128
            else:
                is_filled = area > 1000
        else:
            is_filled = area > 1000
        
        return {
            'exists': True,
            'shape_type': shape_type,
            'is_filled': is_filled,
            'area': area,
            'vertices': vertices
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate visual analogy transformation.
        
        CRITICAL RULES:
        1. First row (A, B) must remain unchanged
        2. Bottom right (D) should have same shape as bottom left (C)
        3. D should be filled or have thicker lines (follow pattern)
        """
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        scores = {}
        h, w = gen_final.shape[:2]
        
        # 1. CRITICAL: First row (top) must be preserved
        first_top_left = self._detect_shapes_in_quadrant(first_frame, 'top_left')
        first_top_right = self._detect_shapes_in_quadrant(first_frame, 'top_right')
        final_top_left = self._detect_shapes_in_quadrant(gen_final, 'top_left')
        final_top_right = self._detect_shapes_in_quadrant(gen_final, 'top_right')
        
        # Check if top row shapes are preserved (area should be similar)
        if first_top_left['exists'] and first_top_right['exists']:
            tl_change = abs(final_top_left.get('area', 0) - first_top_left.get('area', 0)) / max(first_top_left.get('area', 1), 1)
            tr_change = abs(final_top_right.get('area', 0) - first_top_right.get('area', 0)) / max(first_top_right.get('area', 1), 1)
            
            if tl_change > 0.5 or tr_change > 0.5:
                # First row changed significantly
                scores['first_row_preserved'] = 0.0
                scores['bottom_right_correct'] = 0.0
                scores['shape_preserved'] = 0.0
                self._last_task_details = scores
                self._last_task_details['first_row_changed'] = True
                return 0.0
            else:
                scores['first_row_preserved'] = max(0, 1.0 - (tl_change + tr_change) / 2)
        else:
            scores['first_row_preserved'] = 0.0  # STRICT: No shapes detected in first row
        
        # Analyze bottom row
        gen_c = self._detect_shapes_in_quadrant(gen_final, 'bottom_left')
        gen_d = self._detect_shapes_in_quadrant(gen_final, 'bottom_right')
        gt_d = self._detect_shapes_in_quadrant(gt_final, 'bottom_right')
        
        # 2. Bottom right (D) should match GT
        gen_d_region = gen_final[h//2:, w//2:]
        gt_d_region = gt_final[h//2:, w//2:]
        
        if gen_d_region.shape == gt_d_region.shape:
            diff = np.abs(gen_d_region.astype(float) - gt_d_region.astype(float)).mean()
            
            if diff < 15:
                scores['bottom_right_correct'] = 1.0
            elif diff < 30:
                scores['bottom_right_correct'] = 0.5
            else:
                scores['bottom_right_correct'] = 0.1
        else:
            scores['bottom_right_correct'] = 0.0
        
        # 3. Shape preservation: D should have same shape type as C
        if gen_d['exists'] and gen_c['exists']:
            if gen_d['shape_type'] == gen_c['shape_type']:
                scores['shape_preserved'] = 1.0
            else:
                scores['shape_preserved'] = 0.3
        else:
            scores['shape_preserved'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
