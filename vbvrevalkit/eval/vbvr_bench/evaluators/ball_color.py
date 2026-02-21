"""Evaluator for O-29_ballcolor_data-generator.

Repo: https://github.com/VBVR-DataFactory/O-29_ballcolor_data-generator
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class BallColorEvaluator(BaseEvaluator):
    """
    O-29: Ball Color (Cluster Merging)
    
    Task: Red cluster A moves and merges with other clusters. When A's 
    count >= target, target disappears and A absorbs all balls.
    
    Rule-based evaluation:
    1. Red A dominance (30%) - Only A moves, stays red
    2. Merge rule accuracy (35%) - Correct merging logic
    3. Ball count conservation (25%) - Total unchanged
    4. Iteration completeness (10%) - Continue until only A remains
    """
    
    TASK_WEIGHTS = {
        'red_dominance': 0.30,
        'merge_rule': 0.35,
        'conservation': 0.25,
        'completeness': 0.10
    }
    
    def _count_color_clusters(self, frame: np.ndarray) -> Dict[str, int]:
        """Count clusters by color."""
        if len(frame.shape) != 3:
            return {'red': 0, 'blue': 0, 'green': 0}
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        colors = {}
        
        # Red
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        colors['red'] = int(np.sum(red_mask > 0))
        
        # Blue
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        colors['blue'] = int(np.sum(blue_mask > 0))
        
        # Green
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        colors['green'] = int(np.sum(green_mask > 0))
        
        return colors
    
    def _count_balls(self, frame: np.ndarray) -> int:
        """Count circular objects (balls)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10,
                                   param1=50, param2=30, minRadius=3, maxRadius=30)
        
        if circles is not None:
            return len(circles[0])
        return 0
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate ball cluster merging behavior."""
        
        if not video_frames or gt_final_frame is None:
            return 0.0
        
        scores = {}
        
        gen_final = video_frames[-1]
        gt_final = gt_final_frame
        
        # 1. Red dominance: Check if red is dominant in final
        gen_colors = self._count_color_clusters(gen_final)
        gt_colors = self._count_color_clusters(gt_final)
        
        gen_red_ratio = gen_colors['red'] / max(sum(gen_colors.values()), 1)
        gt_red_ratio = gt_colors['red'] / max(sum(gt_colors.values()), 1)
        
        ratio_diff = abs(gen_red_ratio - gt_red_ratio)
        scores['red_dominance'] = max(0, 1.0 - ratio_diff * 2)
        
        # 2. Merge rule: Compare final state with GT
        if gen_final.shape == gt_final.shape:
            diff = np.abs(gen_final.astype(float) - gt_final.astype(float)).mean()
            scores['merge_rule'] = max(0, 1.0 - diff / 100.0)
        else:
            scores['merge_rule'] = 0.2  # Detection failed
        
        # 3. Conservation: Compare ball counts
        if len(video_frames) >= 2:
            first_count = self._count_balls(video_frames[0])
            last_count = self._count_balls(video_frames[-1])
            
            if first_count > 0:
                ratio = min(first_count, last_count) / max(first_count, last_count)
                scores['conservation'] = ratio
            else:
                scores['conservation'] = 0.2  # Detection failed
        else:
            scores['conservation'] = 0.2  # Detection failed
        
        # 4. Completeness: Check if process completed
        # Final should have mostly one color (red)
        total_colored = sum(gen_colors.values())
        if total_colored > 0:
            dominant_ratio = max(gen_colors.values()) / total_colored
            scores['completeness'] = dominant_ratio
        else:
            scores['completeness'] = 0.2  # Detection failed
        
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
