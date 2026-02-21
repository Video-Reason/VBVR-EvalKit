"""Evaluator for G-212_find_incorrect_arrow_direction_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-212_find_incorrect_arrow_direction_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class FindIncorrectArrowDirectionEvaluator(BaseEvaluator):
    """
    G-212: Find incorrect arrow direction evaluator.
    
    Rule-based evaluation:
    - Arrow identification accuracy (50%): Correctly identify reversed arrow
    - Marking standardization (30%): Red circle marking
    - Marking precision (15%): Circle position and size
    - Element preservation (5%): Original elements unchanged
    """
    
    TASK_WEIGHTS = {
        'arrow_identification': 0.50,
        'marking_standardization': 0.30,
        'marking_precision': 0.15,
        'element_preservation': 0.05
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
        
        # 1. Arrow identification (50%)
        scores['arrow_identification'] = self._evaluate_arrow_identification(
            first_frame, final_frame, gt_final_frame
        )
        
        # 2. Marking standardization (30%)
        scores['marking_standardization'] = self._evaluate_marking_standard(final_frame)
        
        # 3. Marking precision (15%)
        scores['marking_precision'] = self._evaluate_marking_precision(final_frame)
        
        # 4. Element preservation (5%)
        scores['element_preservation'] = self._evaluate_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_arrow_identification(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        gt_final_frame: Optional[np.ndarray] = None
    ) -> float:
        """Rule-based: Check if the incorrect arrow is identified."""
        # Detect red circle marking
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # If GT final frame available, compare with GT marking position
        if gt_final_frame is not None:
            gt_circle = self._detect_red_circle(gt_final_frame)
            if gt_circle is not None:
                dist = np.sqrt((circle[0] - gt_circle[0])**2 + 
                              (circle[1] - gt_circle[1])**2)
                if dist < 40:
                    return 1.0
                elif dist < 80:
                    return 0.8
                elif dist < 120:
                    return 0.5
                else:
                    return 0.1
        
        # Fallback: Find the different arrow direction
        different_arrow_pos = self._find_different_arrow(first_frame)
        
        if different_arrow_pos is None:
            return 0.0
        
        # Check if circle marks the different arrow
        dist = np.sqrt((circle[0] - different_arrow_pos[0])**2 + 
                      (circle[1] - different_arrow_pos[1])**2)
        
        if dist < 40:
            return 1.0
        elif dist < 80:
            return 0.7
        elif dist < 120:
            return 0.4
        else:
            return 0.1
    
    def _evaluate_marking_standard(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check if red circle marking is used."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        # Check reasonable size
        if 15 < r < 100:
            size_score = 1.0
        else:
            size_score = 0.5
        
        # Check color is red
        color_score = self._check_red_color(final_frame, x, y, r)
        
        return 0.5 * size_score + 0.5 * color_score
    
    def _evaluate_marking_precision(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate circle position and size."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # Check if circle is in reasonable position
        h, w = final_frame.shape[:2]
        x, y, r = circle
        
        if 30 < x < w - 30 and 30 < y < h - 30:
            return 1.0
        else:
            return 0.6
    
    def _evaluate_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if original elements are preserved."""
        # Count arrows in first frame
        first_arrows = self._count_arrows(first_frame)
        
        # Count arrows in final (excluding red marking area)
        hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        final_no_red = final_frame.copy()
        final_no_red[red_mask > 0] = [255, 255, 255]
        
        final_arrows = self._count_arrows(final_no_red)
        
        if abs(first_arrows - final_arrows) <= 1:
            return 1.0
        elif abs(first_arrows - final_arrows) <= 2:
            return 0.7
        else:
            return 0.1
    
    def _find_different_arrow(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the arrow pointing in a different direction."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        arrows = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 10000:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Estimate arrow direction using bounding box
                    x, y, w, h = cv2.boundingRect(cnt)
                    direction = 1 if w > h else 0  # Simplified direction
                    
                    arrows.append((cx, cy, direction))
        
        if len(arrows) < 3:
            return None
        
        # Find the outlier direction
        from collections import Counter
        directions = [a[2] for a in arrows]
        direction_counts = Counter(directions)
        
        for arrow in arrows:
            if direction_counts[arrow[2]] == 1:
                return (arrow[0], arrow[1])
        
        return None
    
    def _count_arrows(self, frame: np.ndarray) -> int:
        """Count number of arrows."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return sum(1 for cnt in contours if 500 < cv2.contourArea(cnt) < 10000)
    
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
            if cv2.contourArea(largest) > 50:
                (x, y), r = cv2.minEnclosingCircle(largest)
                return (int(x), int(y), int(r))
        
        return None
    
    def _check_red_color(self, frame: np.ndarray, x: int, y: int, r: int) -> float:
        """Check if the marking is red."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), r + 5, 255, 10)
        
        roi_hsv = hsv[mask > 0]
        if len(roi_hsv) == 0:
            return 0.5
        
        red_count = sum(1 for pixel in roi_hsv if pixel[0] < 10 or pixel[0] > 160)
        return red_count / len(roi_hsv)
