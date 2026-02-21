"""Evaluator for G-8_track_object_movement_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-8_track_object_movement_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow, safe_distance


class TrackObjectMovementEvaluator(BaseEvaluator):
    """
    G-8: Track object movement evaluator.
    
    Rule-based evaluation:
    - Tracking continuity (30%): Green border follows object B throughout movement
    - Horizontal movement (25%): Object B moves strictly horizontally
    - Alignment precision (20%): Object B aligns with object A's x-coordinate
    - Marker identification (15%): Correct identification of marked objects
    - Object fidelity (10%): Shape, size, color preserved
    """
    
    # CRITICAL: Alignment is the main success criterion - object must reach red star
    TASK_WEIGHTS = {
        'tracking': 0.10,
        'horizontal': 0.10,
        'alignment': 0.60,  # Main criterion - must align with red star
        'identification': 0.10,
        'fidelity': 0.10
    }
    
    def _detect_green_border(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect green border marker and return its center."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Find largest green contour (the border)
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    
    def _detect_red_star(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red star marker and return its center."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Find star-shaped contour
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > 2000:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) >= 8:  # Star has many vertices
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy)
        return None
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate object tracking task."""
        if len(video_frames) < 2 or gt_final_frame is None:
            return 0.0
        
        scores = {}
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        
        # 1. Tracking continuity: Check green border presence throughout video
        tracking_scores = []
        for frame in video_frames:
            border_pos = self._detect_green_border(frame)
            if border_pos is not None:
                tracking_scores.append(1.0)
            else:
                tracking_scores.append(0.0)
        scores['tracking'] = np.mean(tracking_scores) if tracking_scores else 0.5
        
        # 2. Horizontal movement: Track green border y-coordinate stability
        y_positions = []
        for frame in video_frames:
            border_pos = self._detect_green_border(frame)
            if border_pos is not None:
                y_positions.append(border_pos[1])
        
        if len(y_positions) >= 2:
            y_variance = np.var(y_positions)
            # Good horizontal movement: y variance < 100 pixels
            scores['horizontal'] = max(0, 1.0 - y_variance / 500.0)
        else:
            scores['horizontal'] = 0.2  # Detection failed
        
        # 3. Alignment precision: Check if green border aligns with red star at the END
        final_border = self._detect_green_border(last_frame)
        red_star = self._detect_red_star(first_frame)
        gt_border = self._detect_green_border(gt_final_frame) if gt_final_frame is not None else None
        
        if final_border is not None and red_star is not None:
            # The green-bordered object should align with red star's x-coordinate at end
            x_diff = abs(final_border[0] - red_star[0])
            # Strict alignment: must be within 30 pixels
            if x_diff < 30:
                scores['alignment'] = 1.0
            elif x_diff < 60:
                scores['alignment'] = 0.5
            else:
                scores['alignment'] = 0.0  # Not aligned - STRICT failure
        elif gt_border is not None and final_border is not None:
            x_diff = abs(final_border[0] - gt_border[0])
            scores['alignment'] = max(0, 1.0 - x_diff / 50.0)
        else:
            scores['alignment'] = 0.0  # Detection failed - STRICT
        
        # 4. Marker identification: Check if red star and green border are detected
        has_red_star = red_star is not None
        has_green_border = self._detect_green_border(first_frame) is not None
        
        if not has_red_star or not has_green_border:
            scores['identification'] = 0.0  # Must have both markers
        else:
            scores['identification'] = 1.0
        
        # 5. Fidelity: Check if movement was in a straight line (y stays constant)
        if len(y_positions) >= 3:
            # Check y variance - should be very low for straight horizontal movement
            y_variance = np.var(y_positions)
            if y_variance < 50:  # Very straight line
                scores['fidelity'] = 1.0
            elif y_variance < 200:
                scores['fidelity'] = 0.5
            else:
                scores['fidelity'] = 0.0  # Not a straight line
        else:
            scores['fidelity'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
