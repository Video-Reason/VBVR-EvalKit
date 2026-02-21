"""Evaluator for O-53_clock_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-53_clock_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import safe_distance


class ClockTimeEvaluator(BaseEvaluator):
    """
    O-53: Clock Time Reasoning
    
    Task: Given clock with hour hand only, calculate and show position 
    after k hours (using 12-hour modulo).
    
    Key evaluation criteria:
    1. Time calculation accuracy (50%) - Correct (initial + k) % 12
    2. Hand position accuracy (30%) - Correct angle
    3. Rotation direction (15%) - Clockwise
    4. Clock fidelity (5%) - Face preserved
    """
    
    def __init__(self, device: str = 'cuda', task_name: str = ''):
        super().__init__(device, task_name)
        self.DEFAULT_WEIGHTS = {
            'time_calculation_accuracy': 0.50,
            'hand_position_accuracy': 0.30,
            'rotation_direction': 0.15,
            'clock_fidelity': 0.05
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate clock time reasoning."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # Resize if needed
        if gen_final.shape != gt_final.shape:
            gt_final = cv2.resize(gt_final, (gen_final.shape[1], gen_final.shape[0]))
        
        scores = {}
        
        # 1. Time calculation (50%): Check if hour hand angle matches GT
        time_score = self._evaluate_time_calculation(gen_final, gt_final)
        scores['time_calculation_accuracy'] = time_score
        
        # 2. Hand position (30%): Check angle accuracy
        position_score = self._evaluate_hand_position(gen_final, gt_final)
        scores['hand_position_accuracy'] = position_score
        
        # 3. Rotation direction (15%): Check clockwise rotation
        direction_score = self._evaluate_rotation_direction(video_frames)
        scores['rotation_direction'] = direction_score
        
        # 4. Clock fidelity (5%): Check clock face preserved
        fidelity_score = self._evaluate_clock_fidelity(gen_final, gt_final)
        scores['clock_fidelity'] = fidelity_score
        
        self._last_task_details = scores
        return sum(scores[k] * self.DEFAULT_WEIGHTS[k] for k in self.DEFAULT_WEIGHTS)
    
    def _detect_hand_angle(self, frame: np.ndarray) -> Optional[float]:
        """Detect hour hand angle."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find clock center (assume center of frame)
        h, w = gray.shape
        center = (w // 2, h // 2)
        
        # Detect lines
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=10)
        
        if lines is None:
            return None
        
        # Find line from center (hour hand)
        best_line = None
        best_dist = float('inf')
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line passes near center
            dist1 = np.sqrt((x1 - center[0])**2 + (y1 - center[1])**2)
            dist2 = np.sqrt((x2 - center[0])**2 + (y2 - center[1])**2)
            
            min_dist = min(dist1, dist2)
            if min_dist < best_dist and min_dist < 30:
                best_dist = min_dist
                best_line = line[0]
        
        if best_line is None:
            return None
        
        x1, y1, x2, y2 = best_line
        
        # Calculate angle from center
        if np.sqrt((x1 - center[0])**2 + (y1 - center[1])**2) < \
           np.sqrt((x2 - center[0])**2 + (y2 - center[1])**2):
            # x1, y1 is closer to center
            dx, dy = x2 - center[0], y2 - center[1]
        else:
            dx, dy = x1 - center[0], y1 - center[1]
        
        # Angle from 12 o'clock position (top)
        angle = np.degrees(np.arctan2(dx, -dy))  # Negative dy because y increases downward
        if angle < 0:
            angle += 360
        
        return angle
    
    def _evaluate_time_calculation(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if time calculation is correct."""
        gen_angle = self._detect_hand_angle(gen_frame)
        gt_angle = self._detect_hand_angle(gt_frame)
        
        if gen_angle is None or gt_angle is None:
            return 0.5
        
        # Compare angles (each hour = 30 degrees)
        diff = abs(gen_angle - gt_angle)
        if diff > 180:
            diff = 360 - diff
        
        # Convert to hours
        hour_diff = diff / 30
        
        if hour_diff < 0.5:
            return 1.0
        elif hour_diff < 1:
            return 0.8
        elif hour_diff < 2:
            return 0.5
        else:
            return max(0.1, 1.0 - hour_diff / 6)
    
    def _evaluate_hand_position(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate hand position accuracy."""
        gen_angle = self._detect_hand_angle(gen_frame)
        gt_angle = self._detect_hand_angle(gt_frame)
        
        if gen_angle is None or gt_angle is None:
            return 0.5
        
        diff = abs(gen_angle - gt_angle)
        if diff > 180:
            diff = 360 - diff
        
        if diff < 5:
            return 1.0
        elif diff < 15:
            return 0.8
        elif diff < 30:
            return 0.5
        else:
            return max(0.1, 1.0 - diff / 90)
    
    def _evaluate_rotation_direction(self, frames: List[np.ndarray]) -> float:
        """Evaluate if rotation is clockwise."""
        if len(frames) < 3:
            return 0.5
        
        angles = []
        for frame in frames[::max(1, len(frames)//5)]:
            angle = self._detect_hand_angle(frame)
            if angle is not None:
                angles.append(angle)
        
        if len(angles) < 2:
            return 0.5
        
        # Check if angles increase (clockwise)
        increasing = 0
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i-1]
            # Handle wrap-around
            if diff < -180:
                diff += 360
            elif diff > 180:
                diff -= 360
            
            if diff > 0:
                increasing += 1
        
        return increasing / (len(angles) - 1)
    
    def _evaluate_clock_fidelity(self, gen_frame: np.ndarray, gt_frame: np.ndarray) -> float:
        """Evaluate if clock face is preserved."""
        # Detect circular clock face
        gen_gray = cv2.cvtColor(gen_frame, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
        
        gen_circles = cv2.HoughCircles(gen_gray, cv2.HOUGH_GRADIENT, 1, 50,
                                       param1=50, param2=30, minRadius=50, maxRadius=200)
        gt_circles = cv2.HoughCircles(gt_gray, cv2.HOUGH_GRADIENT, 1, 50,
                                      param1=50, param2=30, minRadius=50, maxRadius=200)
        
        if gen_circles is not None and gt_circles is not None:
            return 1.0
        elif gen_circles is not None or gt_circles is not None:
            return 0.5
        else:
            return 0.3
