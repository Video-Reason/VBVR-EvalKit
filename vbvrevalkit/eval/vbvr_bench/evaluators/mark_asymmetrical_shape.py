"""Evaluator for G-248_mark_asymmetrical_shape_data-generator.

Repo: https://github.com/VBVR-DataFactory/G-248_mark_asymmetrical_shape_data-generator
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from .base_evaluator import BaseEvaluator


class MarkAsymmetricalShapeEvaluator(BaseEvaluator):
    """
    G-248: Mark asymmetrical shape evaluator.
    
    Rule-based evaluation:
    - Symmetry identification correctness (45%): Find asymmetrical shape
    - Marking precision (30%): Circle accurately marks target
    - Marking quality (15%): Red circle quality
    - Scene preservation (10%): Original shapes unchanged
    """
    
    TASK_WEIGHTS = {
        'symmetry_identification': 0.45,
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
        
        # 1. Symmetry identification (45%)
        scores['symmetry_identification'] = self._evaluate_symmetry_id(
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
    
    def _evaluate_symmetry_id(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Check if asymmetrical shape is identified."""
        # Find asymmetrical shape
        asymmetric = self._find_asymmetrical_shape(first_frame)
        
        # Detect circle marking
        circle = self._detect_red_circle(final_frame)
        
        if circle is None:
            return 0.0
        
        if asymmetric is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - asymmetric[0])**2 + (circle[1] - asymmetric[1])**2)
        
        if dist < 50:
            return 1.0
        elif dist < 100:
            return 0.7
        elif dist < 150:
            return 0.4
        else:
            return 0.2
    
    def _evaluate_marking_precision(
        self, 
        first_frame: np.ndarray,
        final_frame: np.ndarray
    ) -> float:
        """Rule-based: Evaluate marking precision."""
        circle = self._detect_red_circle(final_frame)
        asymmetric = self._find_asymmetrical_shape(first_frame)
        
        if circle is None:
            return 0.0
        if asymmetric is None:
            return 0.5
        
        dist = np.sqrt((circle[0] - asymmetric[0])**2 + (circle[1] - asymmetric[1])**2)
        return max(0.0, 1.0 - dist / 80)
    
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
        else:
            return 0.6
    
    def _find_asymmetrical_shape(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find the asymmetrical shape (odd-sided polygon or lowest symmetry)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Get vertex count
                    perimeter = cv2.arcLength(cnt, True)
                    epsilon = 0.02 * perimeter
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    vertices = len(approx)
                    
                    # Calculate symmetry
                    symmetry = self._calculate_symmetry(cnt)
                    
                    shapes.append((cx, cy, symmetry, vertices))
        
        if len(shapes) == 0:
            return None
        
        # First, look for odd-sided polygons (they have no line of bilateral symmetry)
        odd_sided = [s for s in shapes if s[3] % 2 == 1 and s[3] >= 5]  # 5, 7, 9, etc.
        if odd_sided:
            # Return the odd-sided polygon (asymmetrical by definition)
            return (odd_sided[0][0], odd_sided[0][1])
        
        # Otherwise, find shape with lowest symmetry score
        most_asymmetric = min(shapes, key=lambda s: s[2])
        return (most_asymmetric[0], most_asymmetric[1])
    
    def _detect_shapes_with_symmetry(self, frame: np.ndarray) -> List[Tuple[int, int, float]]:
        """Detect shapes with their symmetry score."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate symmetry score
                    symmetry = self._calculate_symmetry(cnt)
                    shapes.append((cx, cy, symmetry))
        
        return shapes
    
    def _calculate_symmetry(self, contour: np.ndarray) -> float:
        """Calculate symmetry score for a contour using multiple methods."""
        # Get shape properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0 or area == 0:
            return 0.5
        
        # Circularity (how close to a circle)
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Convex hull for solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Approximate polygon to count vertices
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # Symmetry score based on multiple factors:
        # - High circularity = more symmetric
        # - High solidity = more regular
        # - Even vertices = more symmetric
        # - Odd vertices (especially prime like 7) = less symmetric
        
        vertex_symmetry = 1.0 if vertices % 2 == 0 else 0.5
        if vertices in [3, 5, 7, 11, 13]:  # Odd-sided polygons are less symmetric
            vertex_symmetry = 0.3
        
        # Also do mirror symmetry check
        x, y, w, h = cv2.boundingRect(contour)
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.drawContours(mask, [contour - [x-1, y-1]], -1, 255, -1)
        
        mid = w // 2
        left = mask[:, :mid]
        right = cv2.flip(mask[:, mid:], 1)
        
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]
        
        if left.size > 0 and right.size > 0:
            intersection = np.sum((left > 0) & (right > 0))
            union = np.sum((left > 0) | (right > 0))
            mirror_symmetry = intersection / union if union > 0 else 0.5
        else:
            mirror_symmetry = 0.5
        
        # Combine factors: lower score = more asymmetric
        symmetry = 0.3 * circularity + 0.2 * solidity + 0.2 * vertex_symmetry + 0.3 * mirror_symmetry
        
        return symmetry
    
    def _count_shapes(self, frame: np.ndarray) -> int:
        """Count shapes."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return sum(1 for cnt in contours if cv2.contourArea(cnt) > 300)
    
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
