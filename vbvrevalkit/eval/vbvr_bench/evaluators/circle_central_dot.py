"""Evaluator for G-217_circle_central_dot_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-217_circle_central_dot_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class CircleCentralDotEvaluator(BaseEvaluator):
    """
    G-217: Circle central dot evaluator.
    
    Rule-based evaluation:
    - Center identification accuracy (50%): Find the central dot (y≈512)
    - Marking accuracy (30%): Circle centered on dot
    - Marking appearance (15%): Red circle, proper size (~36px radius)
    - Scene preservation (5%): Original dots unchanged
    """
    
    TASK_WEIGHTS = {
        'center_identification': 0.50,
        'marking_accuracy': 0.30,
        'marking_appearance': 0.15,
        'scene_preservation': 0.05
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
        
        # Check if a red circle/marking was added
        # Count red pixels in first vs final frame
        first_red_count = self._count_red_pixels(first_frame)
        final_red_count = self._count_red_pixels(final_frame)
        red_increase = final_red_count - first_red_count
        
        # Need at least 500 new red pixels for a marking
        if red_increase < 500:
            # No red marking added - task not completed
            self._last_task_details = {
                'center_identification': 0.0,
                'marking_accuracy': 0.0,
                'marking_appearance': 0.0,
                'scene_preservation': 1.0,
                'no_red_marking': True,
                'red_pixel_increase': int(red_increase)
            }
            return 0.05  # Very low score for no marking
        
        # 1. Center identification (50%)
        scores['center_identification'] = self._evaluate_center_identification(
            first_frame, final_frame
        )
        
        # 2. Marking accuracy (30%)
        scores['marking_accuracy'] = self._evaluate_marking_accuracy(
            first_frame, final_frame
        )
        
        # 3. Marking appearance (15%)
        scores['marking_appearance'] = self._evaluate_marking_appearance(first_frame, final_frame)
        
        # 4. Scene preservation (5%)
        scores['scene_preservation'] = self._evaluate_scene_preservation(
            first_frame, final_frame
        )
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_center_identification(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if the central dot (y≈512) is identified."""
        # Find central dot from first frame
        central_dot = self._find_central_dot(first_frame)
        
        # Detect circle marking in final frame
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        if central_dot is None:
            return 0.5
        
        # Check if circle is centered on the central dot
        dist = np.sqrt((circle[0] - central_dot[0])**2 + (circle[1] - central_dot[1])**2)
        
        if dist < 25:
            return 1.0
        elif dist < 50:
            return 0.7
        elif dist < 100:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_marking_accuracy(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Evaluate if circle is properly centered."""
        circle = self._detect_red_circle(final_frame)
        central_dot = self._find_central_dot(first_frame)
        
        if circle is None:
            return 0.0
        if central_dot is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - central_dot[0])**2 + (circle[1] - central_dot[1])**2)
        return max(0.0, 1.0 - dist / 50)
    
    def _evaluate_marking_appearance(self, first_frame: np.ndarray, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate red circle appearance - size should match black dots."""
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        # Get expected size from black dots in first frame
        dots = self._detect_black_dots_with_size(first_frame)
        if dots:
            avg_dot_size = np.mean([d['size'] for d in dots])
            expected_radius = avg_dot_size / 2  # Radius should be half of dot diameter
            
            # Check if red circle radius matches expected (~26 for 51px dots)
            size_ratio = r / expected_radius if expected_radius > 0 else 0
            
            # Perfect match: ratio close to 1.0-1.4 (circle can be slightly larger)
            if 0.8 < size_ratio < 1.5:
                size_score = 1.0
            elif 0.6 < size_ratio < 2.0:
                size_score = 0.6
            else:
                # Circle is way too large or too small
                size_score = 0.1
        else:
            # Fallback: check size (should be ~36 pixels radius)
            if 25 < r < 50:
                size_score = 1.0
            elif 15 < r < 60:
                size_score = 0.6
            else:
                size_score = 0.1
        
        return size_score
    
    def _evaluate_scene_preservation(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if all dots are preserved - NO new dots should appear."""
        first_dots = self._detect_black_dots(first_frame)
        
        # Detect dots excluding red marking
        hsv = cv2.cvtColor(final_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        final_no_red = final_frame.copy()
        final_no_red[red_mask > 0] = [255, 255, 255]
        
        final_dots = self._detect_black_dots(final_no_red)
        
        if len(first_dots) == 0:
            return 0.5
        
        # Strict check: dots should be preserved exactly (or -1 if central dot is covered)
        # If dots are removed or added, penalize heavily
        if len(final_dots) > len(first_dots):
            # New dots appeared - bad
            return 0.0
        elif len(final_dots) == len(first_dots):
            # All dots preserved
            return 1.0
        elif len(final_dots) == len(first_dots) - 1:
            # One dot might be covered by red marking - acceptable
            return 0.9
        elif len(final_dots) == len(first_dots) - 2:
            # Two dots missing - questionable
            return 0.5
        else:
            # Many dots removed - very bad
            return 0.0
    
    def _find_central_dot(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the central dot (y ≈ center of frame)."""
        dots = self._detect_black_dots(frame)
        
        if len(dots) == 0:
            return None
        
        h, w = frame.shape[:2]
        center_y = h // 2
        
        # Find dot closest to vertical center
        central_dot = min(dots, key=lambda d: abs(d[1] - center_y))
        return central_dot
    
    def _detect_black_dots(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect black dots in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 5000:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dots.append((cx, cy))
        
        return dots
    
    def _detect_black_dots_with_size(self, frame: np.ndarray) -> List[Dict]:
        """Detect black dots with size information."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 5000:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(cnt)
                    dots.append({'center': (cx, cy), 'area': area, 'size': max(w, h)})
        
        return dots
    
    def _count_red_pixels(self, frame: np.ndarray) -> int:
        """Count red pixels in frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        return int(np.sum(mask > 0))
    
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
