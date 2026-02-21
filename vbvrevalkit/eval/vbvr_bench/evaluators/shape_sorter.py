"""Evaluator for O-46_shape_sorter_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-46_shape_sorter_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class ShapeSorterEvaluator(BaseEvaluator):
    """
    O-46: Shape sorter evaluator.
    
    RULE: Colored shapes on left should be moved to cover the outlines on right.
    - Outlines are line drawings (low saturation) on the right side
    - Colored shapes (high saturation) start on left and should end on right
    - Final frame: outlines covered by matching colored shapes
    - No new objects should appear
    """
    
    TASK_WEIGHTS = {
        'shapes_moved_to_right': 0.50,  # Colored shapes should be on right
        'left_side_cleared': 0.30,       # Left side should have no colored shapes
        'no_new_shapes': 0.20            # No new shapes created
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
        h, w = first_frame.shape[:2]
        
        # Detect colored shapes (high saturation) in first and final frames
        first_left_shapes = self._detect_colored_shapes(first_frame[:, :w//2])
        first_right_shapes = self._detect_colored_shapes(first_frame[:, w//2:])
        final_left_shapes = self._detect_colored_shapes(final_frame[:, :w//2])
        final_right_shapes = self._detect_colored_shapes(final_frame[:, w//2:])
        
        # Total colored shapes in first frame (on left side)
        first_colored_count = len(first_left_shapes)
        
        # 1. CRITICAL: Colored shapes should be on right in final frame
        if first_colored_count == 0:
            scores['shapes_moved_to_right'] = 0.5
        else:
            # Check if all shapes moved to right
            if len(final_right_shapes) >= first_colored_count:
                scores['shapes_moved_to_right'] = 1.0
            elif len(final_right_shapes) >= first_colored_count - 1:
                scores['shapes_moved_to_right'] = 0.7
            else:
                scores['shapes_moved_to_right'] = len(final_right_shapes) / first_colored_count
        
        # 2. CRITICAL: Left side should be cleared of colored shapes
        if first_colored_count == 0:
            scores['left_side_cleared'] = 0.5
        else:
            if len(final_left_shapes) == 0:
                scores['left_side_cleared'] = 1.0
            else:
                # Penalize for shapes remaining on left
                remaining_ratio = len(final_left_shapes) / first_colored_count
                scores['left_side_cleared'] = max(0, 1.0 - remaining_ratio)
        
        # 3. No new shapes should be created
        total_first = first_colored_count + len(first_right_shapes)
        total_final = len(final_left_shapes) + len(final_right_shapes)
        
        if total_final <= total_first + 1:  # Allow 1 shape tolerance
            scores['no_new_shapes'] = 1.0
        else:
            # Penalize for new shapes
            new_shapes = total_final - total_first
            scores['no_new_shapes'] = max(0, 1.0 - new_shapes * 0.3)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _detect_colored_shapes(self, region: np.ndarray) -> List[Dict]:
        """Detect colored (high saturation) shapes in region."""
        if region.size == 0:
            return []
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # High saturation indicates colored shapes (not outlines)
        mask = hsv[:, :, 1] > 80
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    shapes.append({'center': (cx, cy), 'area': area})
        
        return shapes
    
    def _detect_shapes_old(self, region: np.ndarray) -> List[Tuple[int, int]]:
        """Detect shapes with their centers."""
        if region.size == 0:
            return []
        
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = hsv[:, :, 1] > 50
        mask = mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 300:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shapes.append((cx, cy))
        
        return shapes
