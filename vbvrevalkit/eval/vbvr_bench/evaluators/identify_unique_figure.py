"""Evaluator for G-147_identify_unique_figure_in_uniform_set_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-147_identify_unique_figure_in_uniform_set_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class IdentifyUniqueFigureEvaluator(BaseEvaluator):
    """
    G-147: Identify unique figure in uniform set.
    
    Rule-based evaluation:
    - Shape recognition (40%): Find the one shape that differs from others
    - Marking precision (35%): Red circle accurately marks the unique figure
    - Marking quality (15%): Circle color, size, line width
    - Scene preservation (10%): Original shapes unchanged
    """
    
    TASK_WEIGHTS = {
        'shape_recognition': 0.40,
        'marking_precision': 0.35,
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
        
        # 1. Shape recognition (40%)
        scores['shape_recognition'] = self._evaluate_recognition(
            first_frame, final_frame
        )
        
        # 2. Marking precision (35%)
        scores['marking_precision'] = self._evaluate_marking_position(final_frame)
        
        # 3. Marking quality (15%)
        scores['marking_quality'] = self._evaluate_marking_quality(final_frame)
        
        # 4. Scene preservation (10%)
        scores['scene_preservation'] = self._evaluate_scene_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_recognition(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if unique shape is identified."""
        # Find the unique shape from first frame
        unique_shape_center = self._find_unique_shape(first_frame)
        
        # Detect marking in final frame
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        if unique_shape_center is None:
            # Can't verify, give partial credit
            return 0.5
        
        # Check if circle marks the unique shape
        dist = np.sqrt((circle[0] - unique_shape_center[0])**2 + 
                      (circle[1] - unique_shape_center[1])**2)
        
        if dist < 30:
            return 1.0
        elif dist < 60:
            return 0.7
        elif dist < 100:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_marking_position(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate marking circle position accuracy."""
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # Check if circle is in a reasonable position (not at edges)
        h, w = final_frame.shape[:2]
        x, y, r = circle
        
        if 50 < x < w - 50 and 50 < y < h - 50:
            return 1.0
        elif 20 < x < w - 20 and 20 < y < h - 20:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_marking_quality(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate marking circle quality."""
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        # Check if circle is reasonable size
        if 20 < r < 150:
            size_score = 1.0
        else:
            size_score = 0.5
        
        # Check color (should be red)
        hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        roi_y = max(0, y - r - 10)
        roi_x = max(0, x - r - 10)
        h, w = final_frame.shape[:2]
        roi = hsv[roi_y:min(h, y+r+10), roi_x:min(w, x+r+10)]
        
        if roi.size > 0:
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask = cv2.inRange(roi, lower_red1, upper_red1) | cv2.inRange(roi, lower_red2, upper_red2)
            red_ratio = np.sum(mask > 0) / max(1, mask.size)
            color_score = min(1.0, red_ratio * 10)
        else:
            color_score = 0.5
        
        return 0.6 * size_score + 0.4 * color_score
    
    def _evaluate_scene_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if original shapes are preserved."""
        # Count shapes in first frame
        first_shapes = self._count_shapes(first_frame)
        
        # Count shapes in final frame (excluding red marking)
        final_shapes = self._count_shapes_excluding_red(final_frame)
        
        if first_shapes == 0:
            return 0.5
        
        # Shapes should be preserved
        if abs(first_shapes - final_shapes) <= 1:
            return 1.0
        elif abs(first_shapes - final_shapes) <= 2:
            return 0.7
        else:
            return 0.4
    
    def _find_unique_shape(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the unique shape that differs from others."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 3:
            return None
        
        # Analyze shape features
        shapes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                # Compute shape descriptor (approximate vertices)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                num_vertices = len(approx)
                
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    shapes.append((cx, cy, num_vertices, cv2.contourArea(cnt)))
        
        if len(shapes) < 3:
            return None
        
        # Find the outlier (different vertex count or significantly different area)
        vertex_counts = [s[2] for s in shapes]
        areas = [s[3] for s in shapes]
        
        # Check for vertex count outlier
        from collections import Counter
        vertex_counter = Counter(vertex_counts)
        
        for shape in shapes:
            if vertex_counter[shape[2]] == 1:  # Only one shape with this vertex count
                return (shape[0], shape[1])
        
        # Check for area outlier
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        
        for shape in shapes:
            if abs(shape[3] - mean_area) > 2 * std_area:
                return (shape[0], shape[1])
        
        return None
    
    def _count_shapes(self, frame: np.ndarray) -> int:
        """Count number of shapes in frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(1 for cnt in contours if cv2.contourArea(cnt) > 200)
    
    def _count_shapes_excluding_red(self, frame: np.ndarray) -> int:
        """Count shapes excluding red marking."""
        # Mask out red
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Remove red from frame
        frame_no_red = frame.copy()
        frame_no_red[red_mask > 0] = [255, 255, 255]
        
        return self._count_shapes(frame_no_red)
    
    def _detect_marking_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect the marking circle (red)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(largest)
            return (int(x), int(y), int(r))
        
        return None
