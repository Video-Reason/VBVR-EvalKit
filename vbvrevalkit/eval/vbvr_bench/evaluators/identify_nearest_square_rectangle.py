"""Evaluator for G-168_identify_nearest_to_square_rectangle_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-168_identify_nearest_to_square_rectangle_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from .base_evaluator import BaseEvaluator


class IdentifyNearestSquareRectangleEvaluator(BaseEvaluator):
    """
    G-168: Identify nearest to square rectangle evaluator.
    
    Rule-based evaluation:
    - Aspect ratio judgment accuracy (50%): Correct rectangle selected
    - Marking uniqueness (20%): Only one rectangle marked
    - Marking position accuracy (20%): Circle accurately surrounds rectangle
    - Visual annotation quality (10%): Red circle proper
    """
    
    TASK_WEIGHTS = {
        'aspect_ratio': 0.50,
        'uniqueness': 0.20,
        'position': 0.20,
        'annotation': 0.10
    }
    
    def _detect_rectangles(self, frame: np.ndarray) -> List[Dict]:
        """Detect rectangles and calculate their aspect ratios."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            
            # Approximate polygon
            approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
            
            if len(approx) == 4:  # Rectangle
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = min(w, h) / max(w, h)  # 1.0 = perfect square
                
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                rectangles.append({
                    'center': (cx, cy),
                    'aspect_ratio': aspect_ratio,
                    'area': area,
                    'bounds': (x, y, w, h)
                })
        
        return rectangles
    
    def _detect_red_marking(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect red circle marking."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
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
        """Evaluate identify nearest to square rectangle task."""
        scores = {}
        
        last_frame = video_frames[-1] if len(video_frames) > 0 else None
        gt_last = gt_final_frame
        
        if last_frame is None or gt_last is None:
            return 0.0
        
        # Detect rectangles and markings
        gen_rects = self._detect_rectangles(last_frame)
        gt_rects = self._detect_rectangles(gt_last)
        
        gen_marking = self._detect_red_marking(last_frame)
        gt_marking = self._detect_red_marking(gt_last)
        
        # 1. Aspect ratio judgment: Check if marking is near the most square rectangle
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['aspect_ratio'] = max(0, 1.0 - dist / 100.0)
        elif gen_marking is not None and gen_rects:
            # Check if marked rectangle has highest aspect ratio
            marked_rect = None
            for rect in gen_rects:
                dist = np.sqrt((gen_marking[0] - rect['center'][0])**2 + 
                              (gen_marking[1] - rect['center'][1])**2)
                if dist < 100:
                    marked_rect = rect
                    break
            
            if marked_rect is not None:
                # Check if this is the most square one
                max_ratio = max(r['aspect_ratio'] for r in gen_rects)
                if marked_rect['aspect_ratio'] >= max_ratio - 0.1:
                    scores['aspect_ratio'] = 0.8
                else:
                    scores['aspect_ratio'] = 0.3
            else:
                scores['aspect_ratio'] = 0.3
        else:
            scores['aspect_ratio'] = 0.2  # Detection failed
        
        # 2. Uniqueness: Only one marking
        hsv_gen = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask_gen = cv2.inRange(hsv_gen, lower_red1, upper_red1) | cv2.inRange(hsv_gen, lower_red2, upper_red2)
        contours_gen, _ = cv2.findContours(red_mask_gen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        significant_contours = [c for c in contours_gen if cv2.contourArea(c) > 100]
        
        if len(significant_contours) == 1:
            scores['uniqueness'] = 1.0
        elif len(significant_contours) == 0:
            scores['uniqueness'] = 0.0
        else:
            scores['uniqueness'] = max(0, 1.0 - (len(significant_contours) - 1) * 0.3)
        
        # 3. Position accuracy: Compare marking positions
        if gen_marking is not None and gt_marking is not None:
            dist = np.sqrt((gen_marking[0] - gt_marking[0])**2 + 
                          (gen_marking[1] - gt_marking[1])**2)
            scores['position'] = max(0, 1.0 - dist / 50.0)
        else:
            scores['position'] = 0.2  # Detection failed
        
        # 4. Annotation quality: Red pixel presence
        scores['annotation'] = min(1.0, np.sum(red_mask_gen > 0) / 500.0)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
