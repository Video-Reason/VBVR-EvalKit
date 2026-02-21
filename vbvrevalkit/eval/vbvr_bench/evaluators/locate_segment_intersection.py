"""Evaluator for G-169_locate_intersection_of_segments_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-169_locate_intersection_of_segments_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator


class LocateSegmentIntersectionEvaluator(BaseEvaluator):
    """
    G-169: Locate intersection of segments evaluator.
    
    Rule-based evaluation:
    - Intersection calculation accuracy (60%): Precise intersection point
    - Marking position accuracy (25%): Circle centered on intersection
    - Visual annotation quality (10%): Red circle proper
    - Marking uniqueness (5%): Only one point marked
    """
    
    TASK_WEIGHTS = {
        'calculation': 0.60,
        'position': 0.25,
        'annotation': 0.10,
        'uniqueness': 0.05
    }
    
    def _detect_lines(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect line segments in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return []
        
        return [(l[0][0], l[0][1], l[0][2], l[0][3]) for l in lines]
    
    def _line_intersection(self, line1: Tuple, line2: Tuple) -> Optional[Tuple[float, float]]:
        """Calculate intersection point of two line segments."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        
        return (px, py)
    
    def _detect_red_marking(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red circle marking."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        M = cv2.moments(red_mask)
        if M['m00'] > 0:
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
        """Evaluate locate segment intersection task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect markings
        gen_marking = self._detect_red_marking(last_frame)
        gt_marking = self._detect_red_marking(gt_last)
        
        # 1. Calculation accuracy: Compare marking positions
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['calculation'] = max(0, 1.0 - dist / 30.0)  # Tight tolerance
        else:
            scores['calculation'] = 0.5 if gen_marking is None and gt_marking is None else 0.0
        
        # 2. Position accuracy: Same as calculation but with looser tolerance
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['position'] = max(0, 1.0 - dist / 50.0)
        else:
            scores['position'] = 0.2  # Detection failed
        
        # 3. Annotation quality: Red pixel IoU
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
        
        scores['annotation'] = red_overlap / red_union if red_union > 0 else 0.5
        
        # 4. Uniqueness: Only one marking
        contours_gen, _ = cv2.findContours(red_mask_gen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = [c for c in contours_gen if cv2.contourArea(c) > 50]
        
        if len(significant_contours) == 1:
            scores['uniqueness'] = 1.0
        elif len(significant_contours) == 0:
            scores['uniqueness'] = 0.0
        else:
            scores['uniqueness'] = max(0, 1.0 - (len(significant_contours) - 1) * 0.3)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
