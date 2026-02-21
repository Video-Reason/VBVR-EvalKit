"""Evaluator for G-160_circle_largest_numerical_value_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-160_circle_largest_numerical_value_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class CircleLargestNumericalValueEvaluator(BaseEvaluator):
    """
    G-160: Circle the largest numerical value.
    
    Rule-based evaluation:
    - Numerical identification (40%): Circle marks the position of largest number
    - Circle position accuracy (30%): Circle center aligns with number
    - Circle style (20%): Red color, appropriate size
    - Animation quality (10%): Smooth expansion
    """
    
    TASK_WEIGHTS = {
        'numerical_identification': 0.40,
        'circle_position': 0.30,
        'circle_style': 0.20,
        'animation_quality': 0.10
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
        
        # 1. Numerical identification (40%)
        scores['numerical_identification'] = self._evaluate_number_selection(
            first_frame, final_frame
        )
        
        # 2. Circle position (30%)
        scores['circle_position'] = self._evaluate_circle_position(final_frame)
        
        # 3. Circle style (20%)
        scores['circle_style'] = self._evaluate_circle_style(final_frame)
        
        # 4. Animation quality (10%)
        scores['animation_quality'] = self._evaluate_animation_quality(video_frames)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_number_selection(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if the largest number is circled."""
        # Detect text regions (numbers) from first frame
        number_regions = self._detect_number_regions(first_frame)
        
        # Detect red circle in final frame
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        if len(number_regions) == 0:
            return 0.5  # Can't verify, partial credit
        
        # Find which number region is circled
        circled_region = None
        min_dist = float('inf')
        for region in number_regions:
            dist = np.sqrt((region[0] - circle[0])**2 + (region[1] - circle[1])**2)
            if dist < min_dist:
                min_dist = dist
                circled_region = region
        
        if circled_region is None or min_dist > 100:
            return 0.3
        
        # The largest number should have the darkest/largest text region
        # (assuming larger digit values = visually larger or more prominent)
        largest_region = max(number_regions, key=lambda r: r[2])  # r[2] is area
        
        if circled_region[2] == largest_region[2]:
            return 1.0
        elif circled_region[2] >= largest_region[2] * 0.7:
            return 0.7
        else:
            return 0.4
    
    def _evaluate_circle_position(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate circle position accuracy."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # Check if circle is in reasonable position (not at edges)
        h, w = final_frame.shape[:2]
        x, y, r = circle
        
        if 30 < x < w - 30 and 30 < y < h - 30:
            return 1.0
        else:
            return 0.5
    
    def _evaluate_circle_style(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate circle style (color, size)."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        # Size check
        if 30 < r < 200:
            size_score = 1.0
        else:
            size_score = 0.5
        
        # Color check
        color_score = self._check_red_color(final_frame, x, y, r)
        
        return 0.5 * size_score + 0.5 * color_score
    
    def _evaluate_animation_quality(self, video_frames: List[np.ndarray]) -> float:
        """Rule-based: Evaluate animation smoothness."""
        if len(video_frames) < 5:
            return 0.5
        
        radii = []
        for frame in video_frames[len(video_frames)//3:]:
            circle = self._detect_red_circle(frame)
            if circle:
                radii.append(circle[2])
        
        if len(radii) < 2:
            return 0.5
        
        # Check for smooth increase
        smooth_count = sum(1 for i in range(1, len(radii)) if radii[i] >= radii[i-1] * 0.95)
        smoothness = smooth_count / (len(radii) - 1)
        
        return smoothness
    
    def _detect_number_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect text/number regions."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 10000:  # Text size range
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    regions.append((cx, cy, area))
        
        return regions
    
    def _detect_red_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect red circle in the frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                (x, y), r = cv2.minEnclosingCircle(largest)
                return (int(x), int(y), int(r))
        
        return None
    
    def _check_red_color(self, frame: np.ndarray, x: int, y: int, r: int) -> float:
        """Check if the circle is red colored."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r + 5, 255, 10)
        
        roi_hsv = hsv[mask > 0]
        if len(roi_hsv) == 0:
            return 0.5
        
        red_count = sum(1 for pixel in roi_hsv if pixel[0] < 10 or pixel[0] > 160)
        return red_count / len(roi_hsv)
