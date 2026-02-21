"""Evaluator for G-189_draw_midpoint_perpendicular_line_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-189_draw_midpoint_perpendicular_line_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator


class DrawMidpointPerpendicularEvaluator(BaseEvaluator):
    """
    G-189: Draw midpoint perpendicular line evaluator.
    
    Rule-based evaluation:
    - Midpoint identification accuracy (40%): Correct midpoint found
    - Perpendicular line position accuracy (30%): Line at x=width/2
    - Perpendicular line length/range (20%): Line spans between parallel lines
    - Visual quality (10%): Red line proper
    """
    
    TASK_WEIGHTS = {
        'midpoint': 0.40,
        'position': 0.30,
        'range': 0.20,
        'visual': 0.10
    }
    
    def _detect_red_line(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect red vertical line."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Find bounding box of red pixels
        points = np.where(red_mask > 0)
        if len(points[0]) < 10:
            return None
        
        y_min, y_max = points[0].min(), points[0].max()
        x_min, x_max = points[1].min(), points[1].max()
        
        # Check if vertical (height > width)
        height = y_max - y_min
        width = x_max - x_min
        
        x_center = (x_min + x_max) // 2
        
        return {
            'x_center': x_center,
            'y_min': y_min,
            'y_max': y_max,
            'length': height,
            'is_vertical': height > width * 2
        }
    
    def _evaluate_task_specific(
        self,
        video_frames: List[np.ndarray],
        gt_frames: List[np.ndarray],
        gt_first_frame: Optional[np.ndarray],
        gt_final_frame: Optional[np.ndarray],
        eval_info: Dict
    ) -> float:
        """Evaluate draw midpoint perpendicular task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect red lines
        gen_line = self._detect_red_line(last_frame)
        gt_line = self._detect_red_line(gt_last)
        
        # 1. Midpoint accuracy: Compare x-position
        if gen_line is not None and gt_line is not None:
            x_diff = abs(gen_line['x_center'] - gt_line['x_center'])
            scores['midpoint'] = max(0, 1.0 - x_diff / 30.0)
        elif gen_line is not None:
            # Check if at image center
            frame_center = last_frame.shape[1] // 2
            x_diff = abs(gen_line['x_center'] - frame_center)
            scores['midpoint'] = max(0, 1.0 - x_diff / 50.0)
        else:
            scores['midpoint'] = 0.0
        
        # 2. Position accuracy: Line should be vertical
        if gen_line is not None:
            scores['position'] = 1.0 if gen_line['is_vertical'] else 0.5
        else:
            scores['position'] = 0.0
        
        # 3. Range: Line length comparison
        if gen_line is not None and gt_line is not None:
            length_ratio = min(gen_line['length'], gt_line['length']) / max(gen_line['length'], gt_line['length'], 1)
            scores['range'] = length_ratio
        elif gen_line is not None:
            # Check if reasonable length
            scores['range'] = min(1.0, gen_line['length'] / 100.0)
        else:
            scores['range'] = 0.0
        
        # 4. Visual quality: Red pixel IoU
        hsv_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        hsv_gt = cv2.cvtColor(gt_last, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask_gen = cv2.inRange(hsv_gen, lower_red1, upper_red1) | cv2.inRange(hsv_gen, lower_red2, upper_red2)
        red_mask_gt = cv2.inRange(hsv_gt, lower_red1, upper_red1) | cv2.inRange(hsv_gt, lower_red2, upper_red2)
        
        red_overlap = np.sum((red_mask_gen > 0) & (red_mask_gt > 0))
        red_union = np.sum((red_mask_gen > 0) | (red_mask_gt > 0))
        
        scores['visual'] = red_overlap / red_union if red_union > 0 else 0.5
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
