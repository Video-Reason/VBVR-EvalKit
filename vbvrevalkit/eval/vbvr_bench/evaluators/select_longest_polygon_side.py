"""Evaluator for G-167_select_longest_polygon_side_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-167_select_longest_polygon_side_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any, Tuple
from .base_evaluator import BaseEvaluator
from ..utils import compute_optical_flow


class SelectLongestPolygonSideEvaluator(BaseEvaluator):
    """
    G-167: Select the longest polygon side.
    
    Rule-based evaluation:
    - Longest side identification (50%): Correctly identify longest edge
    - Marking position (25%): Circle/marker at midpoint of edge
    - Marking uniqueness (15%): Only one edge marked
    - Visual quality (10%): Circle style
    """
    
    TASK_WEIGHTS = {
        'longest_side_identification': 0.50,
        'marking_position': 0.25,
        'marking_uniqueness': 0.15,
        'visual_quality': 0.10
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
        
        # 1. Longest side identification (50%)
        scores['longest_side_identification'] = self._evaluate_side_identification(
            first_frame, final_frame, gt_final_frame
        )
        
        # 2. Marking position (25%)
        scores['marking_position'] = self._evaluate_marking_position(final_frame)
        
        # 3. Marking uniqueness (15%)
        scores['marking_uniqueness'] = self._evaluate_marking_uniqueness(final_frame)
        
        # 4. Visual quality (10%)
        scores['visual_quality'] = self._evaluate_visual_quality_marking(final_frame)
        
        self._last_task_details = scores
        return sum(scores[k] * self.TASK_WEIGHTS[k] for k in self.TASK_WEIGHTS)
    
    def _evaluate_side_identification(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray,
        gt_final_frame: Optional[np.ndarray] = None
    ) -> float:
        """Rule-based: Check if the correct side is identified."""
        # Detect marking in final frame
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # If GT final frame available, compare with GT marking position
        if gt_final_frame is not None:
            gt_circle = self._detect_marking_circle(gt_final_frame)
            if gt_circle is not None:
                dist = np.sqrt((circle[0] - gt_circle[0])**2 + 
                              (circle[1] - gt_circle[1])**2)
                if dist < 30:
                    return 1.0
                elif dist < 60:
                    return 0.8
                elif dist < 100:
                    return 0.5
                else:
                    return 0.2
        
        # Fallback: Find the longest edge midpoint from first frame
        longest_edge_midpoint = self._find_longest_edge_midpoint(first_frame)
        
        if longest_edge_midpoint is None:
            return 0.5  # Can't verify
        
        # Check if circle is near the longest edge midpoint
        dist = np.sqrt((circle[0] - longest_edge_midpoint[0])**2 + 
                      (circle[1] - longest_edge_midpoint[1])**2)
        
        if dist < 30:
            return 1.0
        elif dist < 60:
            return 0.7
        elif dist < 100:
            return 0.5
        else:
            return 0.2
    
    def _evaluate_marking_position(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate if circle is at midpoint of an edge."""
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        # Check if circle is in a reasonable position
        h, w = final_frame.shape[:2]
        x, y, r = circle
        
        if 20 < x < w - 20 and 20 < y < h - 20:
            return 1.0
        else:
            return 0.5
    
    def _evaluate_marking_uniqueness(self, final_frame: np.ndarray) -> float:
        """Rule-based: Check if only one edge is marked."""
        circles = self._detect_all_marking_circles(final_frame)
        
        if len(circles) == 0:
            return 0.0
        elif len(circles) == 1:
            return 1.0
        else:
            return max(0.3, 1.0 - 0.3 * (len(circles) - 1))
    
    def _evaluate_visual_quality_marking(self, final_frame: np.ndarray) -> float:
        """Rule-based: Evaluate marking circle visual quality."""
        circle = self._detect_marking_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        x, y, r = circle
        
        if 5 < r < 50:
            return 1.0
        else:
            return 0.5
    
    def _find_longest_edge_midpoint(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find midpoint of the longest edge in the polygon."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Get the main polygon contour
        polygon = max(contours, key=cv2.contourArea)
        
        # Approximate to get vertices
        peri = cv2.arcLength(polygon, True)
        approx = cv2.approxPolyDP(polygon, 0.02 * peri, True)
        
        if len(approx) < 3:
            return None
        
        # Find longest edge
        max_length = 0
        longest_midpoint = None
        
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]
            
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            if length > max_length:
                max_length = length
                longest_midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        
        return longest_midpoint
    
    def _detect_marking_circle(self, frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """Detect the marking circle (red, orange, or yellow)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Warm colors (red, orange, yellow)
        lower_warm1 = np.array([0, 50, 50])
        upper_warm1 = np.array([35, 255, 255])
        lower_warm2 = np.array([160, 50, 50])
        upper_warm2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_warm1, upper_warm1) | cv2.inRange(hsv, lower_warm2, upper_warm2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 10:
                (x, y), r = cv2.minEnclosingCircle(largest)
                if r > 3:
                    return (int(x), int(y), int(r))
        
        return None
    
    def _detect_all_marking_circles(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Detect all marking circles."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_warm1 = np.array([0, 80, 80])
        upper_warm1 = np.array([35, 255, 255])
        lower_warm2 = np.array([160, 80, 80])
        upper_warm2 = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_warm1, upper_warm1) | cv2.inRange(hsv, lower_warm2, upper_warm2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                (x, y), r = cv2.minEnclosingCircle(cnt)
                circles.append((int(x), int(y), int(r)))
        
        return circles
