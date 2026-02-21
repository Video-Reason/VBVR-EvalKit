"""Evaluator for O-31_ball_eating_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-31_ball_eating_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class BallEatingEvaluator(BaseEvaluator):
    """
    O-31: Ball Eating (Greedy Algorithm)
    
    CRITICAL RULES:
    1. Black dot must move to ALL red dots
    2. Black ball must gradually become BIGGER after eating
    3. All red balls should be eaten (final count = 0)
    """
    
    TASK_WEIGHTS = {
        'all_eaten': 0.50,         # All red balls eaten
        'growth': 0.30,            # Black ball grows significantly
        'animation': 0.20          # Smooth movement
    }
    
    def _count_red_balls(self, frame: np.ndarray) -> int:
        """Count red balls in frame."""
        if len(frame.shape) != 3:
            return 0
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 80])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        return int(np.sum(red_mask > 0))
    
    def _get_black_ball_size(self, frame: np.ndarray) -> float:
        """Get size of black ball."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Black ball detection
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.medianBlur(thresh, 5)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0
        
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        
        if area <= 0:
            return 0
        
        return np.sqrt(area / np.pi)
    
    def _count_red_ball_objects(self, frame: np.ndarray) -> int:
        """Count number of red ball objects (not pixels)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 80, 80])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 80, 80])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len([c for c in contours if cv2.contourArea(c) > 100])
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate ball eating behavior.
        
        CRITICAL RULES:
        1. ALL red balls must be eaten (final count = 0)
        2. Black ball must grow significantly
        """
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        first_frame = video_frames[0]
        gen_final = video_frames[-1]
        
        # 1. CRITICAL: All red balls must be eaten
        first_red_count = self._count_red_ball_objects(first_frame)
        final_red_count = self._count_red_ball_objects(gen_final)
        
        if first_red_count > 0:
            if final_red_count == 0:
                scores['all_eaten'] = 1.0
            else:
                eaten_ratio = max(0, 1 - (final_red_count / first_red_count))
                scores['all_eaten'] = eaten_ratio * 0.5  # Partial credit, max 0.5
        else:
            scores['all_eaten'] = 0.0
        
        # 2. Black ball must grow significantly
        first_size = self._get_black_ball_size(first_frame)
        last_size = self._get_black_ball_size(gen_final)
        
        if first_size > 0:
            growth_ratio = last_size / first_size
            # Should grow at least 1.4x per ball eaten
            # If 3 balls, expected growth = 1.4^3 = 2.74
            expected_growth = 1.4 ** first_red_count if first_red_count > 0 else 2.0
            
            if growth_ratio >= expected_growth * 0.8:
                scores['growth'] = 1.0
            elif growth_ratio >= 1.5:
                scores['growth'] = 0.5
            elif growth_ratio >= 1.2:
                scores['growth'] = 0.3
            else:
                scores['growth'] = 0.1
        else:
            scores['growth'] = 0.0
        
        # 3. Animation: Check for smooth movement
        if len(video_frames) >= 3:
            motion_scores = []
            for i in range(1, min(len(video_frames), 10)):
                diff = cv2.absdiff(video_frames[i], video_frames[i-1])
                motion = np.mean(diff)
                motion_scores.append(motion)
            
            if motion_scores:
                avg_motion = np.mean(motion_scores)
                if avg_motion > 1:  # Some movement detected
                    scores['animation'] = min(1.0, avg_motion / 10.0)
                else:
                    scores['animation'] = 0.0
            else:
                scores['animation'] = 0.0
        else:
            scores['animation'] = 0.0
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
