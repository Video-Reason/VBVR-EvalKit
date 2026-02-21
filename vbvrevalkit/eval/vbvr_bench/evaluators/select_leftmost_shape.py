"""Evaluator for G-219_select_leftmost_shape_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-219_select_leftmost_shape_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class SelectLeftmostShapeEvaluator(BaseEvaluator):
    """
    G-219: Select leftmost shape evaluator.
    
    Rule-based evaluation:
    - Position identification correctness (45%): Find shape with smallest x
    - Marking precision (30%): Circle accurately marks target
    - Marking quality (15%): Red circle quality
    - Scene preservation (10%): Original shapes unchanged
    """
    
    TASK_WEIGHTS = {
        'position_identification': 0.45,
        'marking_precision': 0.30,
        'marking_quality': 0.15,
        'scene_preservation': 0.10
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
        
        # 1. Position identification (45%)
        scores['position_identification'] = self._evaluate_position_id(
            first_frame, final_frame
        )
        
        # 2. Marking precision (30%)
        scores['marking_precision'] = self._evaluate_marking_precision(
            first_frame, final_frame
        )
        
        # 3. Marking quality (15%)
        scores['marking_quality'] = self._evaluate_marking_quality(final_frame)
        
        # 4. Scene preservation (10%)
        scores['scene_preservation'] = self._evaluate_scene_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_position_id(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if leftmost shape (smallest x) is identified."""
        # Find leftmost shape
        leftmost = self._find_leftmost_shape(first_frame)
        
        # Detect circle marking
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        if leftmost is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - leftmost[0])**2 + (circle[1] - leftmost[1])**2)
        
        if dist < 40:
            return 1.0
        elif dist < 80:
            return 0.7
        elif dist < 120:
            return 0.4
        else:
            return 0.0
    
    def _evaluate_marking_precision(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Evaluate marking precision."""
        circle = self._detect_red_circle(final_frame)
        leftmost = self._find_leftmost_shape(first_frame)
        
        if circle is None:
            return 0.0
        if leftmost is None:
            return 0.0
        
        dist = np.sqrt((circle[0] - leftmost[0])**2 + (circle[1] - leftmost[1])**2)
        return max(0.0, 1.0 - dist / 60)
    
    def _evaluate_marking_quality(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate red circle quality."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        if 30 < r < 150:
            return 1.0
        else:
            return 0.5
    
    def _evaluate_scene_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if shapes are preserved."""
        first_count = self._count_shapes(first_frame)
        
        # Remove red marking
        hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        final_no_red = final_frame.copy()
        final_no_red[red_mask > 0] = [255, 255, 255]
        
        final_count = self._count_shapes(final_no_red)
        
        if abs(first_count - final_count) <= 1:
            return 1.0
        elif abs(first_count - final_count) <= 2:
            return 0.7
        else:
            return 0.1
    
    def _find_leftmost_shape(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the leftmost shape (smallest x)."""
        shapes = self._detect_shapes(frame)
        
        if len(shapes) == 0:
            return None
        
        leftmost = min(shapes, key=lambda s: s[0])
        return leftmost
    
    def _detect_shapes(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect shapes with their centers."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 300:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shapes.append((cx, cy))
        
        return shapes
    
    def _count_shapes(self, frame: np.ndarray) -> int:
        """Count shapes."""
        return len(self._detect_shapes(frame))
    
    def _detect_red_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect red circle."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                (x, y), r = cv2.minEnclosingCircle(largest)
                return (int(x), int(y), int(r))
        
        return None
